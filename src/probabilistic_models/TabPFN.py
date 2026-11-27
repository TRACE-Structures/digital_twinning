from .ProbabilisticModel import ProbabilisticModel
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr, ndtri
from scipy.linalg import solve_triangular
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def bar_icdf(criterion, logits, p):
    """ Per-row inverse CDF of a TabPFN FullSupportBarDistribution.

        Same as criterion.icdf (half-normal tails included), but p holds one
        probability per row, shape (n,).
    """
    p = torch.as_tensor(p, dtype=logits.dtype, device=logits.device).clamp(1e-12, 1 - 1e-12)
    probs = logits.softmax(-1)
    cum = torch.cumsum(probs, -1)
    idx = torch.searchsorted(cum, p[:, None]).squeeze(-1).clamp(0, criterion.num_bars - 1)
    before = torch.cat((torch.zeros_like(cum[:, :1]), cum[:, :-1]), -1).gather(-1, idx[:, None]).squeeze(-1)
    cond = ((p - before) / probs.gather(-1, idx[:, None]).squeeze(-1)).clamp(0.0, 1.0)
    widths, borders = criterion.bucket_widths, criterion.borders
    values = borders[idx] + widths[idx] * cond
    left = criterion.halfnormal_with_p_weight_before(widths[0])
    right = criterion.halfnormal_with_p_weight_before(widths[-1])
    values = torch.where(idx == 0, borders[1] - left.icdf(1.0 - cond), values)
    return torch.where(idx == criterion.num_bars - 1, borders[-2] + right.icdf(cond), values)


class TabPFN(ProbabilisticModel):
    """ TabPFN (v2) regressors as a probabilistic model.

        One TabPFNRegressor per output gives the full conditional marginal
        distribution p(f_j | x) of every output (a bar distribution with half-normal
        tails), in context: "training" only stores the data. The dependence between
        the outputs is a Gaussian copula with a constant correlation R, estimated
        from out-of-fold normal scores of the training data.

        Effect removal: since R does not depend on x, the copula cancels in the
        mapping, which becomes an exact per-output quantile mapping
            f_ref,j = F_j^-1( F_j(f_j | x_obs) | x_ref ).
        R is used for the joint scores w = A^-1 z (R = A A^T), i.e. for the alarm
        distance d2 = z^T R^-1 z.

        Forward mode (inverse=False): input = parameters q, output = QoIs.
        Inverse mode (inverse=True):  input = QoIs, output = parameters q.

        Everything runs on the CPU unless a device such as 'cuda:1' is given. TabPFN
        is slow on the CPU and, for v2, refuses more than 1000 training rows there
        unless the environment variable TABPFN_ALLOW_CPU_LARGE_DATASET=true is set.
    """

    POINT_ESTIMATES = ('mean', 'median')
    _SAVED_ATTRS = ()

    def __init__(self, Q, QoI_names, inverse=False, n_estimators=8, random_state=0, version='v2',
                 ignore_pretraining_limits=False):
        self.Q = Q
        self.QoI_names = QoI_names
        self.inverse = inverse
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.version = version
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.point_estimate = 'mean'
        self.predict_device = 'cpu'
        if inverse:
            self.input_names, self.output_names = list(QoI_names), list(Q.variable_names())
        else:
            self.input_names, self.output_names = list(Q.variable_names()), list(QoI_names)

    # ------------------------------------------------------------------ #
    #  Regressors                                                          #
    # ------------------------------------------------------------------ #
    def _new_regressor(self, device='cpu'):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        return TabPFNRegressor.create_default_for_version(
            ModelVersion(self.version), device=device, n_estimators=self.n_estimators,
            random_state=self.random_state, ignore_pretraining_limits=self.ignore_pretraining_limits)

    def _fit_regressors(self, X, Y, device):
        regressors = []
        for j in range(Y.shape[1]):
            reg = self._new_regressor(device)
            reg.fit(X, Y[:, j])
            reg.to('cpu')                         # kept on the CPU between calls
            regressors.append(reg)
        return regressors

    def _full(self, regressors, j, X, device=None):
        """ (logits, criterion) of output j in the scaled output space, on the CPU. """
        device = device or getattr(self, 'predict_device', 'cpu')
        reg = regressors[j]
        try:
            reg.to(device)
            out = reg.predict(X, output_type='full')
        finally:
            reg.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        criterion = out['criterion'].to('cpu')
        logits = torch.as_tensor(out['logits']).to('cpu', criterion.borders.dtype)
        if logits.shape[-1] != criterion.num_bars:
            raise ValueError(f"Output '{self.output_names[j]}' is constant in the training data; remove it.")
        return logits, criterion

    def _marginal_scores(self, regressors, X, Z, device=None, eps=1e-12):
        """ Normal scores Phi^-1(F_j(z_j | x)) of every output (Z in the scaled space). """
        scores = np.empty(Z.shape)
        for j in range(Z.shape[1]):
            logits, crit = self._full(regressors, j, X, device)
            y = torch.as_tensor(Z[:, j:j + 1], dtype=logits.dtype)
            u = crit.cdf(logits, y)[:, 0].numpy()
            scores[:, j] = ndtri(np.clip(u, eps, 1 - eps))
        return scores

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, q, y, device='cpu', k_fold=None, split_type=None, random_seed=1997, train_test_ratio=None,
              **params):
        """ Store the data as TabPFN context and estimate the output copula.

            Parameters
            ----------
            q, y : array-like
                Model inputs and outputs (all rows are used as context).
            device : str, optional
                'cpu' (default) or an explicit GPU such as 'cuda:1'.
            k_fold : int, optional
                Folds for the out-of-fold normal scores that give the copula
                correlation R (default None: 5 folds). In-sample scores would be too
                narrow, because TabPFN sees the training rows in its context.
            split_type, train_test_ratio : optional
                Accepted for the same call as MDN / CRPS, but not used: TabPFN has no
                early stopping, so no validation split is needed and all rows are
                used as context.
            random_seed : int, optional
                Seed of the folds.
        """
        k_fold = 5 if k_fold is None else int(k_fold)
        if k_fold < 2:
            raise ValueError("k_fold must be at least 2 (out-of-fold scores for the copula).")
        self._check_finite(q, y)
        self.samples, self.QoI = q, y
        self.q_scaler = StandardScaler().fit(q)
        self.y_scaler = StandardScaler().fit(y)
        X = self.q_scaler.transform(q)
        Y = self.y_scaler.transform(y)
        self.train_config = dict(params, device=str(device), k_fold=k_fold, random_seed=random_seed)

        print(f"----- Training started for 'TabPFN' model ({len(Y)} context rows, {Y.shape[1]} outputs) -----")
        oof = np.empty(Y.shape)
        for fold, (tr, va) in enumerate(KFold(k_fold, shuffle=True, random_state=random_seed).split(X)):
            print(f"Copula fold {fold + 1}/{k_fold}")
            regs = self._fit_regressors(X[tr], Y[tr], device)
            oof[va] = self._marginal_scores(regs, X[va], Y[va], device)
        self.oof_scores = oof
        self.copula_corr = np.corrcoef(oof.T) if Y.shape[1] > 1 else np.ones((1, 1))
        self.regressors = self._fit_regressors(X, Y, device)
        print("----- Training ended for 'TabPFN' model -----")

    def train_and_validate(self, xi_train, yt_train, xi_val, yt_val, **params):
        raise NotImplementedError("TabPFN is not trained with a loss; use train().")

    def _config(self):
        raise NotImplementedError     # no '.pt' format: save_model(format='pickle') only

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #
    def predict(self, features, point=None, device=None):
        point = point or self.point_estimate
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")
        X = self.q_scaler.transform(features)
        z = np.empty((X.shape[0], len(self.output_names)))
        for j in range(z.shape[1]):
            logits, crit = self._full(self.regressors, j, X, device)
            z[:, j] = (crit.mean(logits) if point == 'mean' else crit.median(logits)).numpy()
        return self.y_scaler.inverse_transform(z)

    def predict_mean_and_var(self, features, device=None):
        X = self.q_scaler.transform(features)
        D = len(self.output_names)
        mean, var = np.empty((X.shape[0], D)), np.empty((X.shape[0], D))
        for j in range(D):
            logits, crit = self._full(self.regressors, j, X, device)
            mean[:, j], var[:, j] = crit.mean(logits).numpy(), crit.variance(logits).numpy()
        return self.y_scaler.inverse_transform(mean), var * self.y_scaler.scale_ ** 2

    def sample(self, features, n_samples=2000, seed=None, device=None):
        """ Joint samples (marginals from TabPFN, dependence from the copula), shape (n, n_samples, D). """
        X = self.q_scaler.transform(features)
        n, D = X.shape[0], len(self.output_names)
        rng = np.random.default_rng(seed)
        A = np.linalg.cholesky(self.copula_corr + 1e-10 * np.eye(D))
        u = ndtr(rng.standard_normal((n, n_samples, D)) @ A.T)
        out = np.empty((n, n_samples, D))
        for j in range(D):
            logits, crit = self._full(self.regressors, j, X, device)
            rep = logits.repeat_interleave(n_samples, dim=0)
            out[:, :, j] = bar_icdf(crit, rep, u[:, :, j].ravel()).numpy().reshape(n, n_samples)
        return out * self.y_scaler.scale_ + self.y_scaler.mean_

    def inference(self, features, n_samples=2000, seed=None, device=None):
        out = self.sample(features, n_samples=n_samples, seed=seed, device=device)
        return out[0] if out.shape[0] == 1 else out

    def evaluate_model(self, y_train, X_test, y_test, verbose=False, point='mean', device='cpu'):
        """ Evaluate a forward model with the same metrics as MDN / CRPS (see _evaluate_model). """
        if self.inverse:
            raise NotImplementedError("evaluate_model is only available for forward models (inverse=False).")
        if point not in self.POINT_ESTIMATES:
            raise ValueError(f"point must be one of {self.POINT_ESTIMATES}, got '{point}'")
        saved = (self.point_estimate, getattr(self, 'predict_device', 'cpu'))
        self.point_estimate, self.predict_device = point, device
        try:
            return self._evaluate_model(y_train, X_test, y_test, verbose=verbose)
        finally:
            self.point_estimate, self.predict_device = saved

    # ------------------------------------------------------------------ #
    #  Effect removal (see ProbabilisticModel.map_to_reference)            #
    # ------------------------------------------------------------------ #
    def _reference_map(self, q_obs, z_obs, q_ref, device=None, eps=1e-12):
        """ Per-output quantile mapping through the TabPFN marginals.

            Returns the mapped (scaled) outputs, the joint copula scores
            w = A^-1 z and the marginal scores z of the measurements.
        """
        X_obs = self.q_scaler.transform(q_obs)
        X_ref = self.q_scaler.transform(q_ref)
        D = z_obs.shape[1]
        marginal = np.empty((len(X_obs), D))
        z_ref = np.empty((len(X_obs), D))
        for j in range(D):
            logits_o, crit = self._full(self.regressors, j, X_obs, device)
            y = torch.as_tensor(z_obs[:, j:j + 1], dtype=logits_o.dtype)
            u = np.clip(crit.cdf(logits_o, y)[:, 0].numpy(), eps, 1 - eps)
            marginal[:, j] = ndtri(u)
            logits_r, crit_r = self._full(self.regressors, j, X_ref, device)
            z_ref[:, j] = bar_icdf(crit_r, logits_r, u).numpy()
        A = np.linalg.cholesky(self.copula_corr + 1e-10 * np.eye(D))
        joint = solve_triangular(A, marginal.T, lower=True).T
        return z_ref, joint, marginal

    # ------------------------------------------------------------------ #
    #  Pickling: the regressors are refitted from the stored context      #
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        state = super().__getstate__()
        state.pop('regressors', None)            # TabPFN weights are not stored in the file
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'q_scaler'):
            self.regressors = self._fit_regressors(self.q_scaler.transform(self.samples),
                                                   self.y_scaler.transform(self.QoI), 'cpu')
