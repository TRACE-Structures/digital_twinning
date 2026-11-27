from .ProbabilisticModel import ProbabilisticModel
import warnings
import torch
import torch.nn as nn
from torchnaut.mdn import MDNMV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ..utils.gpu_utils import use_device
from ..utils.reference_mapping import mixture_rosenblatt_map, mixture_marginal_scores


class MDN(ProbabilisticModel):
    """ Mixture Density Network (MDN) model for probabilistic modeling.

        Models the full conditional density p(output | input) as a Gaussian mixture
        whose parameters are produced by a neural network (Bishop, 1994).

        Forward mode  (inverse=False): input = parameters q,  output = QoIs.
        Inverse mode  (inverse=True):  input = QoIs,           output = parameters q.

        Devices: everything runs on the CPU unless a device such as 'cuda:1' is given
        explicitly. Only the network's forward and backward passes run on that device;
        data, scaling, the mixture statistics and sampling stay on the CPU. Between
        calls the network is kept on the CPU and the GPU memory is released.

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
        predict_device : str
            Default device of the prediction methods (default 'cpu').
        predict_gpu_memory_fraction : float or None
            Default GPU memory limit of the prediction methods (default None).
    """

    POINT_ESTIMATES = ('mean', 'median', 'branch')
    _SAVED_ATTRS = ('point_estimate', 'history', 'train_config', 'cv_train_losses', 'cv_val_losses')

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
        self.predict_device = 'cpu'
        self.predict_gpu_memory_fraction = None

        if inverse:
            self.input_names, self.output_names = list(QoI_names), list(Q.variable_names())
        else:
            self.input_names, self.output_names = list(Q.variable_names()), list(QoI_names)

        self.model = self._build_inner()

    def _config(self):
        """ Constructor arguments (except Q), used by save_model / load_model. """
        return {'QoI_names': list(self.QoI_names), 'layers': self.layers,
                'n_components': self.n_components, 'inverse': self.inverse}

    def _build_inner(self):
        """ Create a freshly initialized inner network (on the CPU). """
        return self.MDNInner(input_size=len(self.input_names), output_size=len(self.output_names),
                             layers=self.layers, n_components=self.n_components)

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def _network_outputs(self, features, device=None, gpu_memory_fraction=None, chunk_size=1024):
        """ Raw network outputs for each input row, returned on the CPU.

            Only the forward pass runs on `device`, in chunks of `chunk_size` rows.
        """
        if device is None:
            device = getattr(self, 'predict_device', 'cpu')
            if gpu_memory_fraction is None:
                gpu_memory_fraction = getattr(self, 'predict_gpu_memory_fraction', None)
        X = torch.tensor(self.q_scaler.transform(features), dtype=torch.float32)
        if X.shape[0] == 0:
            raise ValueError("features has no rows")
        outputs = []
        with use_device(self.model, device, gpu_memory_fraction) as dev:
            self.model.eval()
            with torch.no_grad():
                for start in range(0, X.shape[0], chunk_size):
                    outputs.append(self.model(X[start:start + chunk_size].to(dev)).cpu())
        return torch.cat(outputs)

    def predict_distribution(self, features, device=None, gpu_memory_fraction=None, chunk_size=1024):
        """ Conditional mixture distribution for each input row.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
            device : str, optional
                Device of the network's forward pass: 'cpu' or e.g. 'cuda:1'
                (defaults to self.predict_device, i.e. 'cpu').
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device.
            chunk_size : int, optional
                Input rows per forward pass (default 1024).

            Returns
            -------
            torch.distributions.MixtureSameFamily
                Batch of n mixtures on the CPU, defined in the SCALED output space.
        """
        outputs = self._network_outputs(features, device, gpu_memory_fraction, chunk_size)
        return self.model.mdn_mv_util.get_dist(outputs)

    def predict(self, features, point=None, device=None, gpu_memory_fraction=None, chunk_size=1024):
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
            device, gpu_memory_fraction, chunk_size :
                See predict_distribution.

            Returns
            -------
            np.ndarray, shape (n, n_outputs)
                Point predictions in physical units.
        """
        point = point or getattr(self, 'point_estimate', 'mean')
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")
        dist = self.predict_distribution(features, device, gpu_memory_fraction, chunk_size)

        if point == 'mean':
            z = dist.mean
        elif point == 'median':
            z = self._marginal_median(dist)
        else:  # 'branch'
            k = dist.mixture_distribution.probs.argmax(dim=-1)
            z = dist.component_distribution.mean[torch.arange(k.shape[0]), k]

        # StandardScaler is a per-dimension affine map: mean, median and component
        # centres all commute with it, so transform the scaled estimate back.
        return self.y_scaler.inverse_transform(z.numpy())

    def predict_mean_and_var(self, features, device=None, gpu_memory_fraction=None, chunk_size=1024):
        """ Conditional mean and (marginal) variance of the outputs (Bishop eq. 45, 47).

            Returns
            -------
            mean : np.ndarray, shape (n, n_outputs)
            var : np.ndarray, shape (n, n_outputs)
                Predictive variance = average component variance + spread of the
                component means, in physical units.
        """
        dist = self.predict_distribution(features, device, gpu_memory_fraction, chunk_size)
        mean = self.y_scaler.inverse_transform(dist.mean.numpy())
        var = dist.variance.numpy() * self.y_scaler.scale_ ** 2
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

    def inference(self, features, n_samples=2000, seed=None, device=None, gpu_memory_fraction=None,
                  chunk_size=1024):
        """ Draw samples from the predicted conditional distribution.

            The network runs on `device`; the sampling itself runs on the CPU.

            Parameters
            ----------
            features : array-like, shape (n, n_inputs)
            n_samples : int, optional
                Number of samples per input row (default is 2000).
            seed : int, optional
                Random seed (default None: not reproducible). The global RNG state is
                restored afterwards.
            device, gpu_memory_fraction, chunk_size :
                See predict_distribution.

            Returns
            -------
            np.ndarray
                Shape (n_samples, n_outputs) for a single input row (as before),
                shape (n, n_samples, n_outputs) for several rows.
        """
        dist = self.predict_distribution(features, device, gpu_memory_fraction, chunk_size)
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.manual_seed(seed)
            samples = dist.sample((n_samples,)).permute(1, 0, 2).numpy()
        out = samples * self.y_scaler.scale_ + self.y_scaler.mean_
        return out[0] if out.shape[0] == 1 else out


    # ------------------------------------------------------------------ #
    #  Effect removal (see ProbabilisticModel.map_to_reference)            #
    # ------------------------------------------------------------------ #
    def mixture_parameters(self, features, device=None, gpu_memory_fraction=None, chunk_size=1024):
        """ Mixture parameters for each input row, in double precision, scaled output space.

            The raw network outputs are converted to float64 before the softmax and the
            softplus of the Cholesky diagonal, so the tail percentiles keep their accuracy.

            Returns
            -------
            log_w : np.ndarray, shape (n, K)
                Normalized log mixture weights.
            mu : np.ndarray, shape (n, K, D)
                Component means.
            L : np.ndarray, shape (n, K, D, D)
                Lower-triangular scale factors, Sigma_k = L_k L_k^T.
        """
        outputs = self._network_outputs(features, device, gpu_memory_fraction, chunk_size).double()
        dist = self.model.mdn_mv_util.get_dist(outputs)
        comp = dist.component_distribution
        log_w = dist.mixture_distribution.logits
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        if isinstance(comp, torch.distributions.MultivariateNormal):
            mu, L = comp.loc, comp.scale_tril
        else:   # independent normal components
            mu, L = comp.mean, torch.diag_embed(comp.stddev)
        return log_w.numpy(), mu.numpy(), L.numpy()

    def _reference_map(self, q_obs, z_obs, q_ref, eps=None, n_iter=80, device=None, gpu_memory_fraction=None,
                       chunk_size=1024):
        """ Rosenblatt transform through the predicted mixtures (Algorithm 1 of the method).

            Exact for the model's mixture: no sampling, so the result is deterministic.
            The outputs are processed in the training column order (rows of L_k).

            Returns the mapped (scaled) outputs, the joint (Rosenblatt) scores and the
            marginal scores of the measurements.

            Options (pass to map_to_reference / subtract_effects)
            -------
            eps : float, optional
                Clip the percentiles to [eps, 1 - eps] before inversion, as in Section 3.4
                of the method (e.g. 1e-9, about 6 sigma). Default None: no clipping; the
                CDFs are evaluated in log space, so measurements far in the tails (e.g.
                from damage) are mapped exactly instead of being compressed.
            n_iter : int
                Bisection steps of the inverse CDF (default 80).
            device, gpu_memory_fraction, chunk_size :
                See predict_distribution.
        """
        kw = dict(device=device, gpu_memory_fraction=gpu_memory_fraction, chunk_size=chunk_size)
        mix_obs = self.mixture_parameters(q_obs, **kw)
        # Rows sharing the same reference condition need the network only once.
        q_ref_unique, inverse = np.unique(np.asarray(q_ref, dtype=float), axis=0, return_inverse=True)
        q_ref_unique = pd.DataFrame(q_ref_unique, columns=q_ref.columns)
        mix_ref = tuple(a[inverse.ravel()] for a in self.mixture_parameters(q_ref_unique, **kw))
        z_ref, joint = mixture_rosenblatt_map(z_obs, mix_obs, mix_ref, eps=eps, n_iter=n_iter)
        return z_ref, joint, mixture_marginal_scores(z_obs, mix_obs)

    # ------------------------------------------------------------------ #
    #  Evaluation                                                        #
    # ------------------------------------------------------------------ #
    def evaluate_model(self, y_train, X_test, y_test, verbose=False, point='mean', device='cpu',
                       gpu_memory_fraction=None):
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
            device : str, optional
                Device of the network's forward pass (default 'cpu').
            gpu_memory_fraction : float, optional
                GPU memory limit, see utils.gpu_utils.resolve_device.

            Returns
            -------
            list
                [DataFrame of per-output metrics, dict of aggregated metrics]
        """
        if self.inverse:
            raise NotImplementedError("evaluate_model is only available for forward MDNs (inverse=False).")
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")

        saved = (getattr(self, 'point_estimate', 'mean'), getattr(self, 'predict_device', 'cpu'),
                 getattr(self, 'predict_gpu_memory_fraction', None))
        self.point_estimate, self.predict_device, self.predict_gpu_memory_fraction = point, device, gpu_memory_fraction
        try:
            return self._evaluate_model(y_train, X_test, y_test, verbose=verbose)
        finally:
            self.point_estimate, self.predict_device, self.predict_gpu_memory_fraction = saved


    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, q, y, device='cpu', k_fold=None, split_type='shuffle', random_seed=1997, train_test_ratio=0.8,
              **params):
        """ Train the MDN model.

            Parameters
            ----------
            q : array-like
                Model inputs (parameters in forward mode, QoIs in inverse mode).
            y : array-like
                Model outputs (QoIs in forward mode, parameters in inverse mode).
            device : str, optional
                'cpu' (default) or an explicit GPU such as 'cuda:1'. Only the forward
                and backward passes run there; the network returns to the CPU and the
                GPU memory is released after training.
            k_fold : int, optional
                If given, run k-fold cross-validation on the training part first
                (losses stored in self.cv_train_losses / self.cv_val_losses), then
                fit the final model.
            split_type : {'shuffle', 'no_shuffle', 'reverse'}, optional
                How the validation set for early stopping is chosen
                (default 'shuffle'; 'no_shuffle' takes the last rows).
            random_seed : int, optional
                Seed of the train/validation split and the cross-validation folds.
            train_test_ratio : float, optional
                Fraction of the rows used for training; the rest is the validation
                set for early stopping (default 0.8).
            **params : dict
                Passed to MDNInner.train_and_validate, e.g. gpu_memory_fraction,
                max_epoch, max_patience, batch_size, lr, verbose.
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

        print(f'----- Training started for \'MDN\' model -----')

        if k_fold is not None:
            self.cv_train_losses, self.cv_val_losses = self.cross_validation(
                q_train_tensor, y_train_tensor, k_fold, random_seed=random_seed, device=device, **params)
            self.model = self._build_inner()      # fresh network for the final fit

        _, self.history = self.model.train_and_validate(q_train_tensor, y_train_tensor, q_val_tensor, y_val_tensor,
                                                        device=device, **params)
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
        return history["train_loss"][-1], float(np.nanmin(history["val_loss"]))

    class MDNInner(nn.Module):
        """ Inner class defining the architecture of the MDN model.

            Parameters
            ----------
            input_size : int
                The number of input features.
            output_size : int
                The number of output features.
            layers : list of dict
                A list specifying the configuration of each hidden layer. Each dictionary may contain
                the keys 'neurons', 'activation' and 'dropout'; missing keys or None use the defaults
                (32, 'relu', 0.0).
            n_components : int
                The number of mixture components in the MDN.
        """
        def __init__(self, input_size, output_size, layers, n_components):
            super().__init__()

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

            self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, self.mdn_mv_util.network_output_dim))

        @property
        def device(self):
            """ Device the network currently lives on (the CPU between calls). """
            return next(self.parameters()).device

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
            """ Draw samples from the mixture for each input row, on the network's current device.

                Returns
                -------
                torch.Tensor, shape (batch, n_samples, output_size)
                    Samples in the scaled output space.
            """
            self.eval()
            with torch.no_grad():
                preds = self(X.to(self.device))
                return self.mdn_mv_util.sample(preds, n=n_samples)

        def _validation_loss(self, X_val, y_val, device, batch_size):
            """ Mean validation NLL, computed in batches (the data stay on the CPU). """
            total = 0.0
            for start in range(0, X_val.shape[0], batch_size):
                xb = X_val[start:start + batch_size].to(device)
                yb = y_val[start:start + batch_size].to(device)
                total += self.loss_fn(self(xb), yb).item() * xb.shape[0]
            return total / X_val.shape[0]

        def train_and_validate(self, X_train, y_train, X_val, y_val, device='cpu', gpu_memory_fraction=None,
                               optimizer='adam', max_epoch=500, batch_size=64, val_batch_size=1024, lr=1e-3,
                               weight_decay=1e-6, max_patience=20, min_delta=1e-6, verbose=False, log_every=10):
            """ Train with early stopping on the validation NLL.

                Parameters
                ----------
                device : str, optional
                    'cpu' (default) or an explicit GPU such as 'cuda:1'. The data stay
                    on the CPU; only the current batch is moved to the device.
                gpu_memory_fraction : float, optional
                    GPU memory limit, see utils.gpu_utils.resolve_device.
                val_batch_size : int, optional
                    Batch size of the validation pass (default 1024).
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
                                       lr, weight_decay, max_patience, min_delta, verbose, log_every)

            self.load_state_dict(best_state)
            self.eval()
            return self, history

        def _fit(self, dev, X_train, y_train, X_val, y_val, history, optimizer_cls, max_epoch, batch_size,
                 val_batch_size, lr, weight_decay, max_patience, min_delta, verbose, log_every):
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
                    loss = self.loss_fn(self(xb), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
                    opt.step()
                    losses.append(loss.item())
                scheduler.step()

                self.eval()
                with torch.no_grad():
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
