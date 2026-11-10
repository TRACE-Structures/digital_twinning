from .ProbabilisticModel import ProbabilisticModel
import torch
import torch.nn as nn
from torchnaut.mdn import MDNMV
from sklearn.preprocessing import StandardScaler
import numpy as np
from copy import deepcopy


class MDN(ProbabilisticModel):
    """ Mixture Density Network (MDN) model for probabilistic modeling.

        Models the full conditional density p(output | input) as a Gaussian mixture
        whose parameters are produced by a neural network (Bishop, 1994).

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
        n_components : int
            Number of mixture components.
        inverse : bool
            Direction of the mapping (see above).
        point_estimate : {'mean', 'median', 'branch'}
            Default point estimate returned by predict() (default 'mean').
    """

    POINT_ESTIMATES = ('mean', 'median', 'branch')


    def __init__(self, Q, QoI_names, layers, n_components=5, inverse=False):
        """ Initialize the MDN model.

            Parameters
            ----------
            Q : object
                Variable set of the model parameters.
            QoI_names : list of str
                Names of the quantities of interest.
            layers : list of dict
                Hidden layer configuration ('neurons', 'activation', 'dropout').
            n_components : int, optional
                Number of mixture components (default is 5).
            inverse : bool, optional
                If True, learn p(q | QoI) instead of p(QoI | q) (default is False).
        """
        self.Q = Q
        self.QoI_names = QoI_names
        self.layers = layers
        self.n_components = n_components
        self.inverse = inverse
        self.point_estimate = 'mean'

        if inverse:
            self.input_names, self.output_names = list(QoI_names), list(Q.variable_names())
        else:
            self.input_names, self.output_names = list(Q.variable_names()), list(QoI_names)

        self.model = self._build_inner()

    def _build_inner(self):
        """ Create a freshly initialized inner network. """
        return self.MDNInner(input_size=len(self.input_names), output_size=len(self.output_names),
                             layers=self.layers, n_components=self.n_components)

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def _scaled_input(self, features):
        """ Scale the input features and convert them to a tensor on the model device. """
        return torch.tensor(self.q_scaler.transform(features), dtype=torch.float32, device=self.model.device)

    def predict_distribution(self, features):
        """ Conditional mixture distribution for each input row.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)

            Returns
            -------
            torch.distributions.MixtureSameFamily
                Batch of n mixtures, defined in the SCALED output space.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.mdn_mv_util.get_dist(self.model(self._scaled_input(features)))

    def predict(self, features, point=None):
        """ Point prediction computed analytically from the mixture (no sampling).

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
                Input features.
            point : {'mean', 'median', 'branch'}, optional
                'mean'   : conditional mean sum_i alpha_i mu_i (Bishop eq. 45); minimizes MSE.
                'median' : per-output median of the marginal mixture; minimizes MAE.
                'branch' : centre of the component with the largest mixing
                           coefficient (Bishop eq. 49); most probable branch.
                Defaults to self.point_estimate ('mean').

            Returns
            -------
            np.ndarray, shape (n, n_outputs)
                Point predictions in physical units.
        """
        point = point or getattr(self, 'point_estimate', 'mean')
        dist = self.predict_distribution(features)

        if point == 'mean':
            z = dist.mean
        elif point == 'median':
            z = self._marginal_median(dist)
        elif point == 'branch':
            k = dist.mixture_distribution.probs.argmax(dim=-1)
            z = dist.component_distribution.mean[torch.arange(k.shape[0], device=k.device), k]
        else:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        # StandardScaler is a per-dimension affine map: mean, median and component
        # centres all commute with it, so transform the scaled estimate back.
        return self.y_scaler.inverse_transform(z.cpu().numpy())

    def predict_mean_and_var(self, features):
        """ Conditional mean and (marginal) variance of the outputs (Bishop eq. 45, 47).

            Returns
            -------
            mean : np.ndarray, shape (n, n_outputs)
            var : np.ndarray, shape (n, n_outputs)
                Predictive variance = average component variance + spread of the
                component means, in physical units.
        """
        dist = self.predict_distribution(features)
        mean = self.y_scaler.inverse_transform(dist.mean.cpu().numpy())
        var = dist.variance.cpu().numpy() * self.y_scaler.scale_ ** 2
        return mean, var

    @staticmethod
    def _marginal_median(dist, n_iter=60):
        """ Per-dimension median of the marginal mixture via vectorized bisection on its CDF. """
        w = dist.mixture_distribution.probs                    # (B, K)
        mu = dist.component_distribution.mean                  # (B, K, D)
        sd = dist.component_distribution.variance.sqrt()       # (B, K, D)
        comp = torch.distributions.Normal(mu, sd)
        lo = (mu - 10 * sd).min(dim=1).values                  # (B, D)
        hi = (mu + 10 * sd).max(dim=1).values
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            cdf = (w.unsqueeze(-1) * comp.cdf(mid.unsqueeze(1))).sum(dim=1)
            below = cdf < 0.5
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def inference(self, features, n_samples=2000):
        """ Draw samples from the predicted conditional distribution.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
            n_samples : int, optional
                Number of samples per input row (default is 2000).

            Returns
            -------
            np.ndarray
                Shape (n_samples, n_outputs) for a single input row (as before),
                shape (n, n_samples, n_outputs) for several rows.
        """
        samples = self.model.predict(self._scaled_input(features), n_samples=n_samples).cpu().numpy()
        out = np.stack([self.y_scaler.inverse_transform(s) for s in samples])
        return out[0] if out.shape[0] == 1 else out


    # ------------------------------------------------------------------ #
    #  Evaluation                                                        #
    # ------------------------------------------------------------------ #
    def evaluate_model(self, y_train, X_test, y_test, verbose=False, point='mean'):
        """ Evaluate a FORWARD MDN with the same metrics as PredictiveModel.evaluate_model.

            Parameters
            ----------
            y_train : pd.DataFrame
                Training QoI values (used for normalization of the metrics).
            X_test : pd.DataFrame
                Test parameters (model inputs).
            y_test : pd.DataFrame
                Test QoI values.
            verbose : bool, optional
                Print the metrics (default is False).
            point : {'mean', 'median', 'branch'}, optional
                Point estimate to evaluate (default 'mean').

            Returns
            -------
            list
                [DataFrame of per-output metrics, dict of aggregated metrics]
        """
        if self.inverse:
            raise NotImplementedError("evaluate_model is only available for forward MDNs (inverse=False).")
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        previous = getattr(self, 'point_estimate', 'mean')
        self.point_estimate = point
        try:
            return self._evaluate_model(y_train, X_test, y_test, verbose=verbose)
        finally:
            self.point_estimate = previous


    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, q, y, k_fold=None, **params):
        """ Train the MDN model.

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
                Passed to MDNInner.train_and_validate.
        """
        q_train, q_val, y_train, y_val = self.train_test_split(q, y, train_test_ratio=0.8, random_seed=1997, split_type='no_shuffle')

        self.train_config = params

        self.q_scaler = StandardScaler().fit(q_train)
        self.y_scaler = StandardScaler().fit(y_train)

        q_train_tensor = torch.tensor(self.q_scaler.transform(q_train), dtype=torch.float32)
        q_val_tensor = torch.tensor(self.q_scaler.transform(q_val), dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_scaler.transform(y_train), dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)

        print(f'----- Training started for \'MDN\' model -----')

        if k_fold is not None:
            self.cv_train_losses, self.cv_val_losses = self.cross_validation(q_train_tensor, y_train_tensor, k_fold, **params)
            self.model = self._build_inner()      # fresh network for the final fit

        _, self.history = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor, **params)
        print(f'----- Training ended for \'MDN\' model -----')

    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        """ Fit a freshly initialized network on one cross-validation fold.

            Returns
            -------
            tr_loss : float
                Training NLL of the last epoch.
            vl_loss : float
                Best validation NLL.
        """
        self.model = self._build_inner()
        _, history = self.model.train_and_validate(xi_train, yt_train, xi_val, yt_val, **params)
        return history["train_loss"][-1], min(history["val_loss"])

    class MDNInner(nn.Module):
        """ Inner class defining the architecture of the MDN model.

            Parameters
            ----------
            input_size : int
                The number of input features.
            output_size : int
                The number of output features.
            layers : list of dict
                A list specifying the configuration of each hidden layer. Each dictionary should contain keys 'neurons', 'activation', and 'dropout'.
            n_components : int
                The number of mixture components in the MDN.
        """
        def __init__(self, input_size, output_size, layers, n_components):
            super().__init__()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.mdn_mv_util = MDNMV(n_components, output_size)

            default_neurons = 32
            default_activation = 'relu'
            default_dropout = 0.0

            activation_mapping = {
                'gelu': nn.GELU,
                'relu': nn.ReLU,
                'sigmoid': nn.Sigmoid,
                'tanh': nn.Tanh,
                'leakyrelu': nn.LeakyReLU,
                'softmax': lambda: nn.Softmax(dim=1),
                'none': nn.Identity
            }

            hidden_layers = []
            prev_size = input_size

            for layer in layers:
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

            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, self.mdn_mv_util.network_output_dim))

            self.to(self.device)

        def forward(self, x):
            return self.all_layers(x)

        def loss_fn(self, outputs, labels):
            """ Negative log-likelihood of the mixture (Bishop eq. 28-29). """
            return -self.mdn_mv_util.log_likelihood(outputs, labels, min_log_proba=-20).mean()

        def get_batch_ixs(self, like_tensor, batch_size=16, permute=True):
            if like_tensor.shape[0] <= batch_size:
                return torch.arange(like_tensor.shape[0]).unsqueeze(0)
            ixs = torch.randperm(like_tensor.shape[0]) if permute else torch.arange(like_tensor.shape[0])
            return torch.tensor_split(ixs, ixs.shape[0] // batch_size)

        def predict(self, X, n_samples=2000):
            """ Draw samples from the mixture for each input row.

                Returns
                -------
                torch.Tensor, shape (batch, n_samples, output_size)
                    Samples in the scaled output space.
            """
            self.eval()
            with torch.no_grad():
                preds = self(X.to(self.device))
                return self.mdn_mv_util.sample(preds, n=n_samples)

        def train_and_validate(self, X_train, y_train, X_val, y_val, optimizer='adam', max_epoch=5, batch_size=64, lr=1e-3, weight_decay=1e-6, max_patience=2, verbose=False):
            """ Train with early stopping on the validation NLL.

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

            best_val, best_state, bad_epochs = float("inf"), deepcopy(self.state_dict()), 0
            history = {"epoch": [], "train_loss": [], "val_loss": []}

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    optimizer.zero_grad()
                    outputs = self(X_train[ixs])
                    loss = self.loss_fn(outputs, y_train[ixs])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    optimizer.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(self(X_val), y_val).item()

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