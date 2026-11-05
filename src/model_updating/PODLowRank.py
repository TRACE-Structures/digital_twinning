"""
Low-rank (POD) representation of the measurable output.

Implements Section 3.2 of
    A. Stanić, M. Nikolić, H. G. Matthies, N. Friedman,
    "Probabilistic Identification of Parameters in Dynamic Fracture Propagation",
    Int. J. Numer. Methods Eng. 127 (2026) e70282,
including footnote 3 (scaled projection, Eq. 33) and footnote 5 (covariance
estimated from response samples rather than from gPCE coefficients).

Convention
----------
Samples are ROWS: ``Y`` has shape (N, m) with N realisations of the
m-dimensional measurable vector (same as scikit-learn). The paper writes
column vectors, so its  Y_r = P (Y - mu_Y)  becomes  Yr = (Y - mu) @ P.T  here.

Mapping to the paper
--------------------
    mu_          mu_Y                         Eq. (19)
    eigvals_     lambda_i = sigma_i^2         Eq. (20)-(23)
    V_           V  (eigenvectors as columns) Eq. (20)-(23)
    cum_ratio_   rho_L as a function of m_r   Eq. (24)
    P_           P = V_r^T   (or P_hat = S_r^-1 V_r^T if scaled=True, Eq. 33)
    transform    Y_r = P (Y - mu_Y)           Eq. (25)
    project_measurement   z_mr = P (z_m - mu_Y)
    project_error_cov     C_Er = P C_E P^T
    ReducedGaussianLikelihood   L_r(q) = pi_Er(z_mr - P(M(S(q)) - mu_Y))   Eq. (26)
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class PODLowRank:
    """POD / Karhunen-Loeve low-rank representation of a random output vector Y.

    Parameters
    ----------
    energy : float
        Target variance ratio rho_L (Eq. 24) used to pick m_r, e.g. 0.98.
        Ignored when ``n_modes`` is given.
    n_modes : int, optional
        Fix m_r explicitly.
    scaled : bool
        If True use P_hat = S_r^{-1} V_r^T (footnote 3, Eq. 33), giving unit-variance
        reduced coordinates. The paper found the unscaled version more stable for
        MCMC, so the default is False.
    method : {"svd", "eig"}
        "eig" builds C_Y explicitly and solves C_Y v_i = lambda_i v_i (Eq. 21-23),
        exactly as written in the paper. "svd" gives identical results via the SVD
        of the centred snapshot matrix; it is more accurate and much cheaper when
        m is large (it never forms the m x m covariance).
    """

    def __init__(self, energy=0.98, n_modes=None, scaled=False, method="svd"):
        if method not in ("svd", "eig"):
            raise ValueError("method must be 'svd' or 'eig'")
        self.energy = energy
        self.n_modes = n_modes
        self.scaled = scaled
        self.method = method

    # ------------------------------------------------------------------ fit
    def fit(self, Y):
        """Compute mu_Y, the POD basis and the projection matrix P from samples.

        ``Y`` (N, m) should be the TRAINING samples only (footnote 5: the
        covariance is estimated from samples of the response).
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y must have shape (n_samples, n_outputs)")
        N, m = Y.shape
        if N < 2:
            raise ValueError("Need at least two samples to estimate a covariance")

        self.mu_ = Y.mean(axis=0)                              # Eq. (19)
        Yt = Y - self.mu_                                      # fluctuating part

        if self.method == "eig":
            C = Yt.T @ Yt / (N - 1)                            # Eq. (20)
            lam, V = np.linalg.eigh(C)                         # Eq. (21)-(23)
            order = np.argsort(lam)[::-1]
            lam, V = np.clip(lam[order], 0.0, None), V[:, order]
        else:
            _, s, Vt = np.linalg.svd(Yt, full_matrices=False)
            lam, V = s**2 / (N - 1), Vt.T                      # same eigenpairs

        V = self._fix_signs(V)

        total = lam.sum()                                      # = trace(C_Y)
        if total <= 0:
            raise ValueError("Samples have zero variance; nothing to decompose")
        self.eigvals_ = lam
        self.sigma_ = np.sqrt(lam)
        self.V_ = V
        self.cum_ratio_ = np.cumsum(lam) / total               # Eq. (24)

        if self.n_modes is not None:
            m_r = int(self.n_modes)
        else:
            m_r = int(np.searchsorted(self.cum_ratio_, self.energy - 1e-12) + 1)
        m_r = max(1, min(m_r, int(np.sum(lam > lam[0] * 1e-14))))
        self.m_r_ = m_r
        self.rho_L_ = float(self.cum_ratio_[m_r - 1])

        self._build_projection(V[:, :m_r], self.sigma_[:m_r])
        return self

    def _build_projection(self, Vr, sr):
        if self.scaled:
            self.P_ = Vr.T / sr[:, None]                       # Eq. (33): S_r^-1 V_r^T
            self.P_lift_ = Vr * sr[None, :]                    # V_r S_r (inverse map)
        else:
            self.P_ = Vr.T                                     # Eq. (25): P = V_r^T
            self.P_lift_ = Vr

    @staticmethod
    def _fix_signs(V):
        """Make each eigenvector's largest-magnitude entry positive (reproducibility)."""
        idx = np.argmax(np.abs(V), axis=0)
        signs = np.sign(V[idx, np.arange(V.shape[1])])
        signs[signs == 0] = 1.0
        return V * signs

    # ------------------------------------------------------------ mappings
    def transform(self, Y):
        """Y_r = P (Y - mu_Y)   (Eq. 25). Accepts (m,) or (N, m)."""
        Y = np.asarray(Y, dtype=float)
        return (Y - self.mu_) @ self.P_.T

    def inverse_transform(self, Yr):
        """Lift reduced coordinates back to the full space: mu_Y + V_r (S_r) Y_r."""
        Yr = np.asarray(Yr, dtype=float)
        return self.mu_ + Yr @ self.P_lift_.T

    def project_measurement(self, z_m):
        """z_mr = P (z_m - mu_Y)."""
        return self.transform(z_m)

    def project_error_cov(self, C_E):
        """C_Er = P C_E P^T.

        ``C_E`` may be a scalar variance, a 1-D array of variances (diagonal C_E,
        length m), or a full (m, m) covariance matrix.
        """
        P = self.P_
        C_E = np.asarray(C_E, dtype=float)
        if C_E.ndim == 0:
            return float(C_E) * (P @ P.T)
        if C_E.ndim == 1:
            return (P * C_E[None, :]) @ P.T
        return P @ C_E @ P.T

    # ----------------------------------------------------------- diagnostics
    def reconstruction_mae(self, Y):
        """Mean absolute error of the low-rank reconstruction (cf. Figure 10a)."""
        Y = np.asarray(Y, dtype=float)
        return float(np.mean(np.abs(self.inverse_transform(self.transform(Y)) - Y)))

    def n_modes_for(self, energy):
        """Smallest m_r with rho_L >= energy (Eq. 24)."""
        return int(np.searchsorted(self.cum_ratio_, energy - 1e-12) + 1)


class BlockPOD(PODLowRank):
    """Separate POD for groups of output columns (e.g. u_x and u_y as in Figs. 9-10).

    The overall projection P is block-diagonal, so ``transform``,
    ``project_measurement``, ``project_error_cov`` and the likelihood work exactly
    as for a single POD. Reduced coordinates are concatenated block by block.

    Parameters
    ----------
    blocks : list of index arrays / slices into the m output columns.
    **pod_kwargs : passed to each per-block PODLowRank (energy, n_modes, scaled, method).
    """

    def __init__(self, blocks, **pod_kwargs):
        super().__init__(**pod_kwargs)
        self.blocks = blocks
        self.pod_kwargs = pod_kwargs

    def fit(self, Y):
        Y = np.asarray(Y, dtype=float)
        m = Y.shape[1]
        cols = [np.arange(m)[b] for b in self.blocks]
        self.pods_ = [PODLowRank(**self.pod_kwargs).fit(Y[:, c]) for c in cols]

        self.mu_ = Y.mean(axis=0)
        m_r_total = sum(p.m_r_ for p in self.pods_)
        self.P_ = np.zeros((m_r_total, m))
        self.P_lift_ = np.zeros((m, m_r_total))
        r0 = 0
        for c, p in zip(cols, self.pods_):
            r1 = r0 + p.m_r_
            self.P_[r0:r1, c] = p.P_
            self.P_lift_[c, r0:r1] = p.P_lift_
            r0 = r1
        self.m_r_ = m_r_total
        return self


class ReducedGaussianLikelihood:
    """Gaussian likelihood in the reduced space, Eq. (26).

    With unbiased Gaussian error E ~ N(0, C_E), the projected error is
    E_r ~ N(0, P C_E P^T) and

        log L_r(q) = log N(z_mr - y_r(q); 0, C_Er),

    where y_r(q) = P(M(S(q)) - mu_Y) is the reduced model output, i.e. what the
    surrogate (gPCE, or any other proxy) predicts.

    Parameters
    ----------
    pod : fitted PODLowRank or BlockPOD
    z_m : (m,) full-space measurement
    C_E : error covariance in the full space (scalar, (m,) variances, or (m, m))
    jitter : relative diagonal regularisation of C_Er (it can be ill-conditioned
             when C_E is small compared with the retained POD variance).
    """

    def __init__(self, pod, z_m, C_E, jitter=1e-10):
        self.pod = pod
        self.z_r = pod.project_measurement(z_m)
        C_Er = pod.project_error_cov(C_E)
        C_Er = 0.5 * (C_Er + C_Er.T)
        C_Er = C_Er + jitter * np.mean(np.diag(C_Er)) * np.eye(C_Er.shape[0])
        self.C_Er = C_Er
        self._cho = cho_factor(C_Er, lower=True)
        logdet = 2.0 * np.sum(np.log(np.diag(self._cho[0])))
        self._const = -0.5 * (logdet + C_Er.shape[0] * np.log(2.0 * np.pi))

    def log_likelihood(self, y_r):
        """log L_r for reduced predictions y_r of shape (m_r,) or (B, m_r)."""
        y_r = np.asarray(y_r, dtype=float)
        res = self.z_r - y_r                                   # eps_r = z_mr - y_r
        res2 = np.atleast_2d(res)
        quad = np.einsum("bi,ib->b", res2, cho_solve(self._cho, res2.T))
        out = self._const - 0.5 * quad
        return out[0] if res.ndim == 1 else out

    def log_likelihood_full(self, y):
        """Same, but from full-space model outputs M(S(q)) of shape (m,) or (B, m)."""
        return self.log_likelihood(self.pod.transform(y))


def error_std(sigma_meas, Y_true_test, Y_pred_test):
    """Error-model standard deviation as in Eqs. (30)-(32):

        sigma_E = sigma_meas + sigma_proxy,

    with sigma_proxy the per-component std of the surrogate's discrepancy on test
    data (surrogate prediction lifted back to the full space vs. the true model
    output). Returns a length-m vector; square it to get the diagonal of C_E.
    """
    disc = np.asarray(Y_true_test, float) - np.asarray(Y_pred_test, float)
    return sigma_meas + disc.std(axis=0, ddof=1)
