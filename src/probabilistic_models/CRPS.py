from .ProbabilisticModel import ProbabilisticModel
import warnings
import torch
import torch.nn as nn
from torchnaut import crps
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import solve_triangular
from ..utils.gpu_utils import use_device, rng_devices
from ..utils.reference_mapping import (sorted_samples_and_copula, empirical_score, empirical_quantile_from_score,
                                COPULA_FACTORS, TAIL_MODES)


class CRPS(ProbabilisticModel):
    """ CRPS (energy score) model for probabilistic prediction.

        An implicit generative network: random noise is injected after the first
        hidden layer (torchnaut EpsilonSampler), so every forward pass returns a set
        of samples from the predictive distribution. The network is trained with the
        multivariate CRPS (energy score). There is no explicit density, so all
        statistics (mean, median, variance) are estimated from samples.

        Forward mode  (inverse=False): input = parameters q,  output = QoIs.
        Inverse mode  (inverse=True):  input = QoIs,           output = parameters q.

        Devices: everything runs on the CPU unless a device such as 'cuda:1' is given
        explicitly. Only the network's forward and backward passes (which include the
        noise sampling) run on that device; data, scaling and the statistics of the
        samples stay on the CPU. Between calls the network is kept on the CPU and the
        GPU memory is released. With a seed, results on CPU and GPU differ, because
        the two devices have different random number generators.

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
        predict_device : str
            Default device of the prediction methods (default 'cpu').
        predict_gpu_memory_fraction : float or None
            Default GPU memory limit of the prediction methods (default None).
    """

    POINT_ESTIMATES = ('mean', 'median')
    _SAVED_ATTRS = ('point_estimate', 'n_eval_samples', 'eval_seed', 'history', 'train_config',
                    'cv_train_losses', 'cv_val_losses')

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
        self.predict_device = 'cpu'
        self.predict_gpu_memory_fraction = None

        if inverse:
            self.input_names, self.output_names = list(QoI_names), list(Q.variable_names())
        else:
            self.input_names, self.output_names = list(Q.variable_names()), list(QoI_names)
        self.input_size, self.output_size = len(self.input_names), len(self.output_names)

        self.model = self._build_inner()

    def _config(self):
        """ Constructor arguments (except Q), used by save_model / load_model. """
        return {'QoI_names': list(self.QoI_names), 'layers': self.layers,
                'n_epsilon': self.n_epsilon, 'inverse': self.inverse}

    def _build_inner(self):
        """ Create a freshly initialized inner network (on the CPU). """
        return self.CRPSInner(input_size=self.input_size, output_size=self.output_size,
                              layers=self.layers, n_epsilon=self.n_epsilon)

    # ------------------------------------------------------------------ #
    #  Sampling and prediction                                             #
    # ------------------------------------------------------------------ #
    def _map_sample_chunks(self, features, reduce, n_samples, seed, chunk_size, device, gpu_memory_fraction):
        """ Draw samples chunk by chunk and apply `reduce` to each chunk on the CPU.

            Inputs are processed in chunks because the activations grow as
            n_rows * n_samples * width. Each chunk of samples, shape
            (chunk, n_samples, n_outputs), is moved to the CPU right away, so the
            GPU only ever holds one chunk. With a seed, the global RNG state is
            restored afterwards, so other random code is not affected.

            Returns
            -------
            list
                reduce(chunk) for every chunk, in order.
        """
        if device is None:
            device = getattr(self, 'predict_device', 'cpu')
            if gpu_memory_fraction is None:
                gpu_memory_fraction = getattr(self, 'predict_gpu_memory_fraction', None)
        X = torch.tensor(self.q_scaler.transform(features), dtype=torch.float32)
        if X.shape[0] == 0:
            raise ValueError("features has no rows")
        results = []
        with use_device(self.model, device, gpu_memory_fraction) as dev:
            self.model.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=rng_devices(dev)):
                if seed is not None:
                    torch.manual_seed(seed)
                for start in range(0, X.shape[0], chunk_size):
                    chunk = self.model(X[start:start + chunk_size].to(dev), n_samples=n_samples).cpu()
                    results.append(reduce(chunk))
        return results

    def predict(self, features, point=None, n_samples=None, seed=None, chunk_size=256, device=None,
                gpu_memory_fraction=None):
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
            device : str, optional
                Device of the network's forward pass: 'cpu' or e.g. 'cuda:1'
                (defaults to self.predict_device, i.e. 'cpu').
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device.

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

        z = torch.cat(self._map_sample_chunks(features, reduce, n_samples, seed, chunk_size, device,
                                              gpu_memory_fraction))
        # StandardScaler is a per-dimension affine map: sample mean and median commute with it.
        return self.y_scaler.inverse_transform(z.numpy())

    def predict_mean_and_var(self, features, n_samples=None, seed=None, chunk_size=256, device=None,
                             gpu_memory_fraction=None):
        """ Sample estimates of the predictive mean and variance per input row.

            Returns
            -------
            mean : np.ndarray, shape (n, n_outputs)
            var : np.ndarray, shape (n, n_outputs)
                Predictive (marginal) variance in physical units.
        """
        n_samples = n_samples or getattr(self, 'n_eval_samples', 2000)
        seed = getattr(self, 'eval_seed', 0) if seed is None else seed
        stats = self._map_sample_chunks(features, lambda s: (s.mean(dim=1), s.var(dim=1)), n_samples, seed,
                                        chunk_size, device, gpu_memory_fraction)
        mean = self.y_scaler.inverse_transform(torch.cat([m for m, _ in stats]).numpy())
        var = torch.cat([v for _, v in stats]).numpy() * self.y_scaler.scale_ ** 2
        return mean, var

    def sample(self, features, n_samples=2000, seed=None, chunk_size=256, device=None, gpu_memory_fraction=None):
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
            device, gpu_memory_fraction :
                See predict.

            Returns
            -------
            np.ndarray, shape (n, n_samples, n_outputs)
                Samples in physical units.
        """
        z = torch.cat(self._map_sample_chunks(features, lambda s: s, n_samples, seed, chunk_size, device,
                                              gpu_memory_fraction)).numpy()
        return z * self.y_scaler.scale_ + self.y_scaler.mean_

    def inference(self, features, n_samples=2000, seed=None, chunk_size=256, device=None, gpu_memory_fraction=None):
        """ Draw samples from the predictive distribution.

            Returns
            -------
            np.ndarray
                Shape (n_samples, n_outputs) for a single input row (as before),
                shape (n, n_samples, n_outputs) for several rows.
        """
        out = self.sample(features, n_samples=n_samples, seed=seed, chunk_size=chunk_size, device=device,
                          gpu_memory_fraction=gpu_memory_fraction)
        return out[0] if out.shape[0] == 1 else out

    # ------------------------------------------------------------------ #
    #  Effect removal (see ProbabilisticModel.map_to_reference)            #
    # ------------------------------------------------------------------ #
    def _reference_map(self, q_obs, z_obs, q_ref, n_samples=2000, n_ref_samples=10000, seed=None,
                       copula='cholesky', tails='extrapolate', chunk_size=256, device=None, gpu_memory_fraction=None):
        """ Empirical marginals + Gaussian copula mapping (Algorithm 2 of the method).

            For every measurement, N samples at q_obs give its marginal percentiles and
            the normal-score correlation R_obs; M samples at q_ref give the reference
            marginals and R_ref. With R = A A^T:
                z = Phi^-1(F_obs(f_obs)),  w = A_obs^-1 z,  z_ref = A_ref w,
                f_ref,j = F_ref,j^-1(Phi(z_ref,j)).
            Rows with the same reference condition share one set of reference samples
            (with all inputs subtracted: a single set for all rows).

            Returns the mapped (scaled) outputs, the joint (copula) scores w and the
            marginal scores z of the measurements.

            Options (pass to map_to_reference / subtract_effects)
            -------
            n_samples : int
                Samples N per measurement (default 2000). Percentiles are resolved
                only within [1/(2N), 1 - 1/(2N)].
            n_ref_samples : int
                Samples M per reference condition (default 10000).
            seed : int, optional
                Random seed (default self.eval_seed): reproducible results.
            copula : {'cholesky', 'symmetric'}
                Factor of R. 'cholesky': Rosenblatt transform of the copula, independent
                percentiles, depends on the output order. 'symmetric': R^(1/2),
                order-independent, percentiles not exactly independent.
            tails : {'extrapolate', 'clip'}
                Measurements beyond the most extreme sample. 'extrapolate' (default):
                the quantile functions are continued linearly in normal-score space
                with their tail slopes, so large deviations (e.g. damage) keep their
                size. 'clip' (Section 6.1 of the method): percentile of the extreme
                sample, so the mapped value saturates near the extreme reference sample.
            chunk_size : int
                Measurements per network pass (default 256); the reference pass uses
                proportionally fewer rows, so both hold about the same number of samples.
            device, gpu_memory_fraction :
                See predict.

            Notes
            -----
            The copula is meaningful only for a model trained with the multivariate
            energy score (as here). It is assumed Gaussian at a fixed condition; this
            fails mainly for clearly multi-modal (clustered) joint distributions.
        """
        if tails not in TAIL_MODES:
            raise ValueError(f"tails must be one of {TAIL_MODES}, got '{tails}'")
        if copula not in COPULA_FACTORS:
            raise ValueError(f"copula must be one of {COPULA_FACTORS}, got '{copula}'")
        seed = getattr(self, 'eval_seed', 0) if seed is None else seed
        n, D = z_obs.shape
        w = np.empty((n, D))
        marginal = np.empty((n, D))

        # 1) measurements: marginal percentiles and decorrelated normal scores
        offset = [0]

        def reduce_obs(chunk):
            a, c = offset[0], chunk.shape[0]
            offset[0] += c
            sorted_s, A = sorted_samples_and_copula(chunk.double().numpy(), copula)
            z = empirical_score(sorted_s, z_obs[a:a + c], tails=tails)
            marginal[a:a + c] = z
            if copula == 'cholesky':
                w[a:a + c] = np.stack([solve_triangular(A[i], z[i], lower=True) for i in range(c)])
            else:
                w[a:a + c] = np.linalg.solve(A, z[..., None])[..., 0]

        self._map_sample_chunks(q_obs, reduce_obs, n_samples, seed, chunk_size, device, gpu_memory_fraction)

        # 2) reference conditions: re-correlate and invert the reference marginals
        q_ref_unique, inverse = np.unique(np.asarray(q_ref, dtype=float), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        q_ref_unique = pd.DataFrame(q_ref_unique, columns=q_ref.columns)
        z_ref = np.empty((n, D))
        offset = [0]

        def reduce_ref(chunk):
            a, c = offset[0], chunk.shape[0]
            offset[0] += c
            sorted_s, A = sorted_samples_and_copula(chunk.double().numpy(), copula)
            rows = np.flatnonzero((inverse >= a) & (inverse < a + c))
            idx = inverse[rows] - a
            zr = np.einsum('rij,rj->ri', A[idx], w[rows])
            z_ref[rows] = empirical_quantile_from_score(sorted_s, zr, index=idx, tails=tails)

        ref_chunk = max(1, chunk_size * n_samples // n_ref_samples)
        self._map_sample_chunks(q_ref_unique, reduce_ref, n_ref_samples, seed, ref_chunk, device,
                                gpu_memory_fraction)
        return z_ref, w, marginal

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #
    def evaluate_model(self, y_train, X_test, y_test, verbose=False, point='mean', n_samples=2000, seed=0,
                       levels=(0.5, 0.9, 0.95), forward_model=None, device='cpu', gpu_memory_fraction=None):
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
            device : str, optional
                Device of the network's forward pass (default 'cpu').
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device.

            Returns
            -------
            list
                Forward: [per-output DataFrame, aggregated dict].
                Inverse: [per-parameter DataFrame, summary dict, forward DataFrame or None].
        """
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        names = ('point_estimate', 'n_eval_samples', 'eval_seed', 'predict_device', 'predict_gpu_memory_fraction')
        defaults = ('mean', 2000, 0, 'cpu', None)
        saved = tuple(getattr(self, n, d) for n, d in zip(names, defaults))
        for n, v in zip(names, (point, n_samples, seed, device, gpu_memory_fraction)):
            setattr(self, n, v)
        try:
            # if self.inverse:
            #     return self._evaluate_inverse_model(y_train, X_test, y_test, verbose=verbose, n_samples=n_samples,
            #                                         seed=seed, levels=levels, forward_model=forward_model)
            return self._evaluate_model(y_train, X_test, y_test, verbose=verbose)
        finally:
            for n, v in zip(names, saved):
                setattr(self, n, v)

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, q, y, k_fold=None, device='cpu', split_type='shuffle', random_seed=1997, train_test_ratio=0.8,
              **params):
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
            device : str, optional
                'cpu' (default) or an explicit GPU such as 'cuda:1'. Only the forward
                and backward passes run there; the network returns to the CPU and the
                GPU memory is released after training.
            split_type : {'shuffle', 'no_shuffle', 'reverse'}, optional
                How the validation set for early stopping is chosen
                (default 'shuffle'; 'no_shuffle' takes the last rows).
            random_seed : int, optional
                Seed of the train/validation split and the cross-validation folds.
            train_test_ratio : float, optional
                Fraction of the rows used for training; the rest is the validation
                set for early stopping (default 0.8).
            **params : dict
                Passed to CRPSInner.train_and_validate, e.g. gpu_memory_fraction,
                max_epoch, max_patience, batch_size, lr, val_seed, verbose.
        """
        self._check_finite(q, y)
        q_train, q_val, y_train, y_val = self.train_test_split(q, y, train_test_ratio=train_test_ratio, random_seed=random_seed,
                                                               split_type=split_type)
        # Keep only the training part as data of the model (used e.g. as prior in plot_posterior);
        # the validation rows are only needed for early stopping during this call.
        self.samples, self.QoI = q_train, y_train

        self.train_config = dict(params, device=str(device), split_type=split_type, random_seed=random_seed,
                                 train_test_ratio=train_test_ratio)

        self.q_scaler = StandardScaler().fit(q_train)
        self.y_scaler = StandardScaler().fit(y_train)

        q_train_tensor = torch.tensor(self.q_scaler.transform(q_train), dtype=torch.float32)
        q_val_tensor = torch.tensor(self.q_scaler.transform(q_val), dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_scaler.transform(y_train), dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_scaler.transform(y_val), dtype=torch.float32)

        print(f'----- Training started for \'CRPSModel\' model -----')

        if k_fold is not None:
            self.cv_train_losses, self.cv_val_losses = self.cross_validation(
                q_train_tensor, y_train_tensor, k_fold, random_seed=random_seed, device=device, **params)
            self.model = self._build_inner()      # fresh network for the final fit

        _, self.history = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor,
                                                        device=device, **params)
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
        return history["train_loss"][-1], float(np.nanmin(history["val_loss"]))

    class CRPSInner(nn.Module):
        """ Inner network: MLP with noise injected after the first hidden layer.

            Parameters
            ----------
            input_size : int
                Number of input features.
            output_size : int
                Number of output features.
            layers : list of dict
                Hidden layer configuration. Each dictionary may contain the keys
                'neurons', 'activation' and 'dropout'; missing keys or None use the
                defaults (32, 'relu', 0.0).
            n_epsilon : int
                Dimension of the injected noise vector.
        """

        def __init__(self, input_size, output_size, layers, n_epsilon):
            super().__init__()

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
                neurons = layer.get('neurons')
                activation = layer.get('activation')
                dropout = layer.get('dropout')
                neurons = default_neurons if neurons is None else neurons
                activation = default_activation if activation is None else activation
                dropout = default_dropout if dropout is None else dropout

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

        @property
        def device(self):
            """ Device the network currently lives on (the CPU between calls). """
            return next(self.parameters()).device

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
            """ Draw samples for each input row, on the network's current device.

                Returns
                -------
                torch.Tensor, shape (batch, n_samples, output_size)
                    Samples in the scaled output space.
            """
            self.eval()
            with torch.no_grad():
                return self(X, n_samples=n_samples)

        def _validation_loss(self, X_val, y_val, device, batch_size):
            """ Mean validation energy score, computed in batches (the data stay on the CPU). """
            total = 0.0
            for start in range(0, X_val.shape[0], batch_size):
                xb = X_val[start:start + batch_size].to(device)
                yb = y_val[start:start + batch_size].to(device)
                total += self.loss_fn(xb, yb).item() * xb.shape[0]
            return total / X_val.shape[0]

        def train_and_validate(self, X_train, y_train, X_val, y_val, device='cpu', gpu_memory_fraction=None,
                               optimizer='adam', max_epoch=500, batch_size=64, val_batch_size=256, lr=1e-3,
                               weight_decay=1e-6, max_patience=20, min_delta=1e-6, val_seed=0, verbose=False, log_every=10):
            """ Train with early stopping on the validation energy score.

                Parameters
                ----------
                device : str, optional
                    'cpu' (default) or an explicit GPU such as 'cuda:1'. The data stay
                    on the CPU; only the current batch is moved to the device.
                gpu_memory_fraction : float, optional
                    GPU memory limit, see utils.gpu_utils.resolve_device.
                val_batch_size : int, optional
                    Batch size of the validation pass (default 256; each row expands
                    to 100 samples inside the network).
                val_seed : int or None, optional
                    Seed for the noise used in the validation loss. A fixed seed
                    removes sampling noise from the early-stopping criterion, so
                    epochs are compared on the same random draws (default 0).
                max_patience : int, optional
                    Stop after this many epochs without improvement (default 20).
                min_delta : float, optional
                    Minimum decrease of the validation loss that counts as an
                    improvement (default 1e-6).
                log_every : int, optional
                    With verbose, print every log_every epochs (default 10).

                Returns
                -------
                self
                    The trained network (best validation state), on the CPU.
                dict
                    History with keys 'epoch', 'train_loss', 'val_loss'.
            """
            self.batch_size = batch_size

            optimizers = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
            if optimizer.lower() not in optimizers:
                raise ValueError(f"Unknown optimizer '{optimizer}'")

            history = {"epoch": [], "train_loss": [], "val_loss": []}
            # The loop runs in its own method, so all its GPU tensors (batches, optimizer
            # state) are freed before use_device releases the GPU memory.
            with use_device(self, device, gpu_memory_fraction) as dev:
                best_state = self._fit(dev, X_train, y_train, X_val, y_val, history,
                                       optimizers[optimizer.lower()], max_epoch, batch_size, val_batch_size,
                                       lr, weight_decay, max_patience, min_delta, val_seed, verbose, log_every)

            self.load_state_dict(best_state)
            self.eval()
            return self, history

        def _fit(self, dev, X_train, y_train, X_val, y_val, history, optimizer_cls, max_epoch, batch_size,
                 val_batch_size, lr, weight_decay, max_patience, min_delta, val_seed, verbose, log_every):
            """ Training loop; returns the best state (on the CPU). """
            opt = optimizer_cls(self.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
            # Snapshots of the best weights are kept on the CPU, not on the GPU.
            best_val, best_state, bad_epochs = float("inf"), self._cpu_state(), 0

            for epoch in range(max_epoch):
                self.train()
                losses = []
                for ixs in self.get_batch_ixs(X_train, batch_size, permute=True):
                    xb, yb = X_train[ixs].to(dev), y_train[ixs].to(dev)
                    opt.zero_grad()
                    loss = self.loss_fn(xb, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    opt.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=rng_devices(dev)):
                    if val_seed is not None:
                        torch.manual_seed(val_seed)
                    val_loss = self._validation_loss(X_val, y_val, dev, val_batch_size)
                train_loss = float(np.mean(losses))

                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                if verbose and (epoch % log_every == 0 or epoch == max_epoch - 1):
                    print(f"epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f}")

                if not (np.isfinite(val_loss) and np.isfinite(train_loss)):
                    if not np.isfinite(best_val):
                        raise FloatingPointError(
                            f"Loss is not finite at epoch {epoch} (train {train_loss}, val {val_loss}) and no "
                            f"earlier epoch was valid; the network did not train. Try a lower lr, other "
                            f"layers, or check the data for NaN/inf.")
                    warnings.warn(f"Loss became non-finite at epoch {epoch}; stopping and keeping the best "
                                  f"state (val loss {best_val:.4f}).")
                    break

                if val_loss < best_val - min_delta:
                    best_val, best_state, bad_epochs = val_loss, self._cpu_state(), 0
                else:
                    bad_epochs += 1
                if bad_epochs > max_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val loss {best_val:.4f})")
                    break
            return best_state

        def _cpu_state(self):
            return {k: v.detach().to("cpu", copy=True) for k, v in self.state_dict().items()}
