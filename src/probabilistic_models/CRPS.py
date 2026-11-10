from .ProbabilisticModel import ProbabilisticModel
import torch
import torch.nn as nn
from torchnaut import crps
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy


class CRPS(ProbabilisticModel):
    """ CRPS (energy score) model for probabilistic prediction.

        An implicit generative network: random noise is injected after the first
        hidden layer (torchnaut EpsilonSampler), so every forward pass returns a set
        of samples from the predictive distribution. The network is trained with the
        multivariate CRPS (energy score). There is no explicit density, so all
        statistics (mean, median, variance) are estimated from samples.

        Forward mode  (inverse=False): input = parameters q,  output = QoIs.
        Inverse mode  (inverse=True):  input = QoIs,           output = parameters q.

        Attributes
        ----------
        Q : object
            Variable set of the model parameters.
        QoI_names : list of str
            Names of the quantities of interest.
        layers : list of dict
            Hidden layer configuration ('neurons', 'activation', 'dropout').
        n_epsilon : int
            Dimension of the injected noise vector (default 30).
        inverse : bool
            Direction of the mapping (see above).
        point_estimate : {'mean', 'median'}
            Default point estimate returned by predict() (default 'mean').
        n_eval_samples : int
            Default number of samples used by predict() (default 2000).
        eval_seed : int or None
            Default seed used by predict(); a fixed seed makes predictions and
            evaluation metrics reproducible (default 0).
    """

    POINT_ESTIMATES = ('mean', 'median')

    def __init__(self, Q, QoI_names, layers, n_epsilon=30, inverse=False):
        """ Initialize the CRPS model.

            Parameters
            ----------
            Q : object
                Variable set of the model parameters.
            QoI_names : list of str
                Names of the quantities of interest.
            layers : list of dict
                Hidden layer configuration; at least one layer is required because
                the noise is injected after the first hidden layer.
            n_epsilon : int, optional
                Dimension of the injected noise vector (default is 30).
            inverse : bool, optional
                If True, learn p(q | QoI) instead of p(QoI | q) (default is False).
        """
        if len(layers) == 0:
            raise ValueError("CRPS model needs at least one hidden layer (noise is injected after the first one).")

        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.n_epsilon = n_epsilon
        self.inverse = inverse
        self.point_estimate = 'mean'
        self.n_eval_samples = 2000
        self.eval_seed = 0

        if inverse:
            self.input_names, self.output_names = list(QoI_names), list(Q.variable_names())
        else:
            self.input_names, self.output_names = list(Q.variable_names()), list(QoI_names)
        self.input_size, self.output_size = len(self.input_names), len(self.output_names)

        self.model = self._build_inner()

    def _build_inner(self):
        """ Create a freshly initialized inner network. """
        return self.CRPSInner(input_size=self.input_size, output_size=self.output_size,
                              layers=self.layers, n_epsilon=self.n_epsilon)

    # ------------------------------------------------------------------ #
    #  Sampling and prediction                                             #
    # ------------------------------------------------------------------ #
    def _scaled_input(self, features):
        """ Scale the input features and convert them to a tensor on the model device. """
        return torch.tensor(self.q_scaler.transform(features), dtype=torch.float32, device=self.model.device)

    def _rng_devices(self):
        return [torch.cuda.current_device()] if self.model.device.type == 'cuda' else []

    def _sample_chunks(self, features, n_samples, seed, chunk_size):
        """ Yield scaled samples chunk by chunk, shape (chunk, n_samples, n_outputs).

            Inputs are processed in chunks because the activations grow as
            n_rows * n_samples * width. With a seed, the global RNG state is
            restored afterwards, so other random code is not affected.
        """
        X = self._scaled_input(features)
        self.model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=self._rng_devices()):
            if seed is not None:
                torch.manual_seed(seed)
            for start in range(0, X.shape[0], chunk_size):
                yield self.model(X[start:start + chunk_size], n_samples=n_samples)

    def predict(self, features, point=None, n_samples=None, seed=None, chunk_size=256):
        """ Point prediction estimated from samples of the predictive distribution.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
                Input features.
            point : {'mean', 'median'}, optional
                'mean'   : sample mean; estimates the conditional mean (minimizes MSE).
                'median' : per-output sample median (minimizes MAE).
                Defaults to self.point_estimate.
            n_samples : int, optional
                Samples per input row (defaults to self.n_eval_samples). The Monte Carlo
                error of the mean is sigma / sqrt(n_samples).
            seed : int, optional
                Random seed (defaults to self.eval_seed). Fixed seed -> reproducible output.
            chunk_size : int, optional
                Number of input rows processed at once (default 256).

            Returns
            -------
            np.ndarray, shape (n, n_outputs)
                Point predictions in physical units.
        """
        point = point or getattr(self, 'point_estimate', 'mean')
        n_samples = n_samples or getattr(self, 'n_eval_samples', 2000)
        seed = getattr(self, 'eval_seed', 0) if seed is None else seed

        if point == 'mean':
            reduce = lambda s: s.mean(dim=1)
        elif point == 'median':
            reduce = lambda s: s.median(dim=1).values
        else:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        z = torch.cat([reduce(s) for s in self._sample_chunks(features, n_samples, seed, chunk_size)])
        # StandardScaler is a per-dimension affine map: sample mean and median commute with it.
        return self.y_scaler.inverse_transform(z.cpu().numpy())

    def predict_mean_and_var(self, features, n_samples=None, seed=None, chunk_size=256):
        """ Sample estimates of the predictive mean and variance per input row.

            Returns
            -------
            mean : np.ndarray, shape (n, n_outputs)
            var : np.ndarray, shape (n, n_outputs)
                Predictive (marginal) variance in physical units.
        """
        n_samples = n_samples or getattr(self, 'n_eval_samples', 2000)
        seed = getattr(self, 'eval_seed', 0) if seed is None else seed
        means, variances = [], []
        for s in self._sample_chunks(features, n_samples, seed, chunk_size):
            means.append(s.mean(dim=1))
            variances.append(s.var(dim=1))
        mean = self.y_scaler.inverse_transform(torch.cat(means).cpu().numpy())
        var = torch.cat(variances).cpu().numpy() * self.y_scaler.scale_ ** 2
        return mean, var

    def sample(self, features, n_samples=2000, seed=None, chunk_size=256):
        """ Draw samples from the predictive distribution for every input row.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
            n_samples : int, optional
                Samples per input row (default is 2000).
            seed : int, optional
                Random seed (default None: not reproducible).
            chunk_size : int, optional
                Input rows processed at once (default 256).

            Returns
            -------
            np.ndarray, shape (n, n_samples, n_outputs)
                Samples in physical units.
        """
        z = torch.cat(list(self._sample_chunks(features, n_samples, seed, chunk_size))).cpu().numpy()
        return z * self.y_scaler.scale_ + self.y_scaler.mean_

    def inference(self, features, n_samples=2000, seed=None, chunk_size=256):
        """ Draw samples from the predictive distribution.

            Returns
            -------
            np.ndarray
                Shape (n_samples, n_outputs) for a single input row (as before),
                shape (n, n_samples, n_outputs) for several rows.
        """
        out = self.sample(features, n_samples=n_samples, seed=seed, chunk_size=chunk_size)
        return out[0] if out.shape[0] == 1 else out

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #
    def evaluate_model(self, y_train, X_test, y_test, verbose=False, point='mean', n_samples=2000, seed=0,
                       levels=(0.5, 0.9, 0.95), forward_model=None):
        """ Evaluate the CRPS model on a test set; the direction of the model selects the method.

            Forward model (inverse=False): point metrics of _evaluate_model
                (X_test = parameters, y_test / y_train = QoIs).
            Inverse model (inverse=True): _evaluate_inverse_model, i.e. point metrics,
                CRPS / energy score, coverage and optional forward consistency
                (X_test = observed QoIs, y_test / y_train = parameters).
                No NLL: the model has no explicit density.

            Parameters
            ----------
            y_train : pd.DataFrame
                Training targets (QoIs in forward mode, parameters in inverse mode).
            X_test : pd.DataFrame
                Test inputs.
            y_test : pd.DataFrame
                Test targets.
            verbose : bool, optional
                Print the metrics (default is False).
            point : {'mean', 'median'}, optional
                Point estimate used for the point metrics (default 'mean').
            n_samples : int, optional
                Samples per test row, for the point estimate and the scores (default 2000).
            seed : int, optional
                Random seed, fixed for reproducible metrics (default 0).
            levels, forward_model :
                Inverse mode only, see ProbabilisticModel._evaluate_inverse_model.

            Returns
            -------
            list
                Forward: [per-output DataFrame, aggregated dict].
                Inverse: [per-parameter DataFrame, summary dict, forward DataFrame or None].
        """
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        saved = (getattr(self, 'point_estimate', 'mean'), getattr(self, 'n_eval_samples', 2000), getattr(self, 'eval_seed', 0))
        self.point_estimate, self.n_eval_samples, self.eval_seed = point, n_samples, seed
        try:
            # if self.inverse:
            #     return self._evaluate_inverse_model(y_train, X_test, y_test, verbose=verbose, n_samples=n_samples,
            #                                         seed=seed, levels=levels, forward_model=forward_model)
            return self._evaluate_model(y_train, X_test, y_test, verbose=verbose)
        finally:
            self.point_estimate, self.n_eval_samples, self.eval_seed = saved

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, q, y, k_fold=None, **params):
        """ Train the CRPS model.

            Parameters
            ----------
            q : array-like
                Model inputs (parameters in forward mode, QoIs in inverse mode).
            y : array-like
                Model outputs (QoIs in forward mode, parameters in inverse mode).
            k_fold : int, optional
                If given, run k-fold cross-validation on the training part first
                (losses stored in self.cv_train_losses / self.cv_val_losses), then
                fit the final model.
            **params : dict
                Passed to CRPSInner.train_and_validate.
        """
        q_train, q_val, y_train, y_val = self.train_test_split(q, y, train_test_ratio=0.8, random_seed=1997, split_type='no_shuffle')

        self.train_config = params

        self.q_scaler = StandardScaler().fit(q_train)
        self.y_scaler = StandardScaler().fit(y_train)

        q_train_tensor = torch.tensor(self.q_scaler.transform(q_train), dtype=torch.float32)
        q_val_tensor = torch.tensor(self.q_scaler.transform(q_val), dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_scaler.transform(y_train), dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)

        print(f'----- Training started for \'CRPSModel\' model -----')

        if k_fold is not None:
            self.cv_train_losses, self.cv_val_losses = self.cross_validation(q_train_tensor, y_train_tensor, k_fold, **params)
            self.model = self._build_inner()      # fresh network for the final fit

        _, self.history = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor, **params)
        print(f'----- Training ended for \'CRPSModel\' model -----')

    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        """ Fit a freshly initialized network on one cross-validation fold.

            Returns
            -------
            tr_loss : float
                Training energy score of the last epoch.
            vl_loss : float
                Best validation energy score.
        """
        self.model = self._build_inner()
        _, history = self.model.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
        return history["train_loss"][-1], min(history["val_loss"])

    class CRPSInner(nn.Module):
        """ Inner network: MLP with noise injected after the first hidden layer.

            Parameters
            ----------
            input_size : int
                Number of input features.
            output_size : int
                Number of output features.
            layers : list of dict
                Hidden layer configuration ('neurons', 'activation', 'dropout').
            n_epsilon : int
                Dimension of the injected noise vector.
        """

        def __init__(self, input_size, output_size, layers, n_epsilon):
            super().__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            default_neurons = 32
            default_activation = 'relu'
            default_dropout = 0.0

            # Factories, so every layer gets its own module instance.
            # Softmax acts on the feature axis (-1): after the noise injection the
            # tensors are (batch, n_samples, features), so dim=1 would be the sample axis.
            activation_mapping = {
                'gelu': nn.GELU,
                'relu': nn.ReLU,
                'sigmoid': nn.Sigmoid,
                'tanh': nn.Tanh,
                'leakyrelu': nn.LeakyReLU,
                'softmax': lambda: nn.Softmax(dim=-1),
                'none': nn.Identity
            }

            hidden_layers = []
            prev_size = input_size

            for i, layer in enumerate(layers):
                neurons = default_neurons if layer['neurons'] is None else layer['neurons']
                activation = default_activation if layer['activation'] is None else layer['activation']
                dropout = default_dropout if layer['dropout'] is None else layer['dropout']

                if activation.lower() not in activation_mapping:
                    raise ValueError(f"Unknown activation '{activation}'")
                hidden_layers.append(nn.Linear(prev_size, neurons))
                hidden_layers.append(nn.LayerNorm(neurons))
                hidden_layers.append(activation_mapping[activation.lower()]())
                if dropout > 0:
                    hidden_layers.append(nn.Dropout(dropout))
                prev_size = neurons

                if i == 0:
                    hidden_layers.append(crps.EpsilonSampler(n_epsilon))
                    prev_size = neurons + n_epsilon

            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, output_size))

            self.to(self.device)

        def forward(self, x, n_samples=100):
            """ Forward pass.

                Returns
                -------
                torch.Tensor, shape (batch, n_samples, output_size)
                    Samples in the scaled output space.
            """
            with crps.EpsilonSampler.n_samples(n_samples):
                return self.all_layers(x.to(self.device))

        def loss_fn(self, xb, yb, n_samples=100):
            """ Mean multivariate CRPS (energy score) of a batch. """
            return crps.crps_loss_mv(self(xb, n_samples=n_samples), yb).mean()

        def get_batch_ixs(self, like_tensor, batch_size=16, permute=True):
            if like_tensor.shape[0] <= batch_size:
                return torch.arange(like_tensor.shape[0]).unsqueeze(0)
            ixs = torch.randperm(like_tensor.shape[0]) if permute else torch.arange(like_tensor.shape[0])
            return torch.tensor_split(ixs, ixs.shape[0] // batch_size)

        def predict(self, X, n_samples=2000):
            """ Draw samples for each input row.

                Returns
                -------
                torch.Tensor, shape (batch, n_samples, output_size)
                    Samples in the scaled output space.
            """
            self.eval()
            with torch.no_grad():
                return self(X, n_samples=n_samples)

        def train_and_validate(self, X_train, y_train, X_val, y_val, optimizer='adam', max_epoch=5, batch_size=64, lr=1e-3, weight_decay=1e-6, max_patience=2, val_seed=0, verbose=False):
            """ Train with early stopping on the validation energy score.

                Parameters
                ----------
                val_seed : int or None, optional
                    Seed for the noise used in the validation loss. A fixed seed
                    removes sampling noise from the early-stopping criterion, so
                    epochs are compared on the same random draws (default 0).

                Returns
                -------
                self
                    The trained network (best validation state).
                dict
                    History with keys 'epoch', 'train_loss', 'val_loss'.
            """
            self.batch_size = batch_size

            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

            optimizers = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
            if optimizer.lower() not in optimizers:
                raise ValueError(f"Unknown optimizer '{optimizer}'")
            optimizer = optimizers[optimizer.lower()](self.parameters(), lr=lr, weight_decay=weight_decay)

            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
            rng_devices = [torch.cuda.current_device()] if self.device.type == 'cuda' else []

            best_val, best_state, bad_epochs = float("inf"), deepcopy(self.state_dict()), 0
            history = {"epoch": [], "train_loss": [], "val_loss": []}

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    optimizer.zero_grad()
                    loss = self.loss_fn(X_train[ixs], y_train[ixs])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    optimizer.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
                    if val_seed is not None:
                        torch.manual_seed(val_seed)
                    val_loss = self.loss_fn(X_val, y_val).item()

                history["epoch"].append(epoch)
                history["train_loss"].append(float(np.mean(losses)))
                history["val_loss"].append(val_loss)

                if verbose and epoch % 20 == 0:
                    print(f"epoch {epoch:4d} | train {np.mean(losses):.4f} | val {val_loss:.4f}")

                if val_loss < best_val - 1e-6:
                    best_val, best_state, bad_epochs = val_loss, deepcopy(self.state_dict()), 0
                else:
                    bad_epochs += 1
                if bad_epochs > max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                    break

            self.load_state_dict(best_state)
            self.eval()
            return self, history
