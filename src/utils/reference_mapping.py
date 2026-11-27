""" Mapping measured outputs to a reference condition (removal of environmental effects).

    NumPy core of the two pipelines of 'Removing Environmental Effects from Modal
    Frequencies' (distribution_effect_removing.pdf). A measured output vector f_obs at
    the actual inputs x_obs is moved to the same position (percentile) of the
    distribution predicted for the reference inputs x_ref:

        f_ref = F^-1( F(f_obs | x_obs) | x_ref )

    - Mixture density networks (Sections 3-5): exact Rosenblatt transform of the
      Cholesky-parametrised Gaussian mixture, mixture_rosenblatt_map().
    - Sample-based (CRPS / energy score) models (Section 6): empirical marginals and
      a Gaussian copula, see sorted_samples_and_copula(), empirical_score(),
      empirical_quantile_from_score().

    Everything here is model-independent; the MDN and CRPS classes provide the
    mixture parameters or samples.
"""

import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr, logsumexp
from scipy.linalg import solve_triangular
from scipy.stats import chi2


_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


# ---------------------------------------------------------------------- #
#  Mixture density networks: Rosenblatt transform (Algorithm 1)           #
# ---------------------------------------------------------------------- #
def _mixture_quantile(log_tail, upper, log_w, m, s, n_iter=80):
    """ Vectorized bisection for the 1-D mixture quantile, eq. (8), in log space.

        Solves log F(f) = log_tail (lower half) or log(1 - F(f)) = log_tail
        (upper half, upper=True) for every row. Working with log CDF / log survival
        keeps the mapping exact far in the tails (no clipping at about 6 sigma), so
        large deviations, e.g. caused by damage, are not compressed.

        log_tail, upper : (n,);  log_w, m, s : (n, K)
    """
    # bracket, eq. (11), widened so that it contains the target tail probability
    c = np.maximum(10.0, np.sqrt(-2.0 * log_tail) + 3.0)[:, None]
    lo = (m - c * s).min(axis=1)
    hi = (m + c * s).max(axis=1)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        z = (mid[:, None] - m) / s
        below = np.where(upper,
                         logsumexp(log_w + log_ndtr(-z), axis=1) > log_tail,   # survival still too large
                         logsumexp(log_w + log_ndtr(z), axis=1) < log_tail)    # CDF still too small
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def mixture_rosenblatt_map(f_obs, mix_obs, mix_ref, eps=None, n_iter=80):
    """ Map measurements to the reference condition through Gaussian mixtures.

        Implements Algorithm 1 for mixtures with lower-triangular (Cholesky) scale
        factors, Sigma_k = L_k L_k^T. The outputs are processed in their column
        order (the rows of L_k). For D = 1 this is the univariate mapping of
        Section 3; for K = 1 it reduces to eq. (10) / (21).

        Parameters
        ----------
        f_obs : np.ndarray, shape (n, D)
            Measured outputs.
        mix_obs, mix_ref : tuple (log_w, mu, L)
            Mixture parameters predicted for x_obs and x_ref, one mixture per row:
            log_w (n, K) log mixture weights, mu (n, K, D) component means,
            L (n, K, D, D) lower-triangular scale factors.
        eps : float, optional
            If given, percentiles are clipped to [eps, 1 - eps] as in Section 3.4
            (e.g. 1e-9, i.e. about 6 sigma). Default None: no clipping; the CDF and
            its inverse are evaluated in log space, which stays exact far in the tails.
        n_iter : int, optional
            Bisection steps (default 80).

        Returns
        -------
        f_ref : np.ndarray, shape (n, D)
            Outputs mapped to the reference condition.
        z : np.ndarray, shape (n, D)
            Normal scores of the Rosenblatt percentiles, u = Phi(z). For the healthy
            structure and a calibrated model they are independent N(0, 1). Computed
            from log CDF / log survival, so they stay exact where u rounds to 0 or 1.
    """
    f_obs = np.asarray(f_obs, dtype=float)
    l_o, mu_o, L_o = (np.asarray(a, dtype=float) for a in mix_obs)
    l_r, mu_r, L_r = (np.asarray(a, dtype=float) for a in mix_ref)
    n, K, D = mu_o.shape
    if f_obs.shape != (n, D):
        raise ValueError(f"f_obs has shape {f_obs.shape}, expected {(n, D)}")
    l_o, l_r = l_o.copy(), l_r.copy()
    log_floor = -np.inf if eps is None else np.log(eps)

    e_o = np.zeros((n, K, D))         # standardized residuals of every component
    e_r = np.zeros((n, K, D))
    f_ref = np.empty((n, D))
    z = np.empty((n, D))

    for j in range(D):
        # conditional components, eq. (17), and normalized log-weights, eq. (19)
        m_o = mu_o[:, :, j] + np.einsum('nki,nki->nk', L_o[:, :, j, :j], e_o[:, :, :j])
        m_r = mu_r[:, :, j] + np.einsum('nki,nki->nk', L_r[:, :, j, :j], e_r[:, :, :j])
        s_o, s_r = L_o[:, :, j, j], L_r[:, :, j, j]
        lw_o = l_o - logsumexp(l_o, axis=1, keepdims=True)
        lw_r = l_r - logsumexp(l_r, axis=1, keepdims=True)

        # forward step at x_obs, eq. (13): log u and log(1 - u)
        z_o = (f_obs[:, j, None] - m_o) / s_o
        log_u = logsumexp(lw_o + log_ndtr(z_o), axis=1)
        log_s = logsumexp(lw_o + log_ndtr(-z_o), axis=1)
        # inverse step at x_ref, eq. (14), matched on the smaller tail
        upper = log_s < log_u
        z[:, j] = _scores_from_log_tails(log_u, log_s)
        log_tail = np.maximum(np.where(upper, log_s, log_u), log_floor)
        f_ref[:, j] = _mixture_quantile(log_tail, upper, lw_r, m_r, s_r, n_iter=n_iter)

        # condition on f_j: residuals, eq. (18), and log-weights, eq. (19)
        z_r = (f_ref[:, j, None] - m_r) / s_r
        e_o[:, :, j], e_r[:, :, j] = z_o, z_r
        l_o += -0.5 * z_o ** 2 - _LOG_SQRT_2PI - np.log(s_o)
        l_r += -0.5 * z_r ** 2 - _LOG_SQRT_2PI - np.log(s_r)

    return f_ref, z


def _scores_from_log_tails(log_u, log_s):
    """ Normal score from log CDF and log survival, accurate in both tails. """
    upper = log_s < log_u
    return np.where(upper, -ndtri(np.exp(np.maximum(log_s, -700.0))), ndtri(np.exp(np.maximum(log_u, -700.0))))


def mixture_marginal_scores(f, mix):
    """ Normal scores of every output in its own marginal distribution.

        The marginal of output j of the mixture is the 1-D mixture with means
        mu_kj and standard deviations sqrt(sum_i L_kji^2). Unlike the Rosenblatt
        scores, the marginal score of an output does not depend on the other
        outputs, so it tells which output deviates (the Rosenblatt scores of the
        later outputs also react to a change of an earlier one).

        f : (n, D);  mix : (log_w, mu, L) as in mixture_rosenblatt_map.
        Returns z : (n, D).
    """
    log_w, mu, L = (np.asarray(a, dtype=float) for a in mix)
    log_w = log_w - logsumexp(log_w, axis=1, keepdims=True)
    sd = np.sqrt((L ** 2).sum(axis=-1))                           # (n, K, D)
    zc = (np.asarray(f, dtype=float)[:, None, :] - mu) / sd
    log_u = logsumexp(log_w[:, :, None] + log_ndtr(zc), axis=1)
    log_s = logsumexp(log_w[:, :, None] + log_ndtr(-zc), axis=1)
    return _scores_from_log_tails(log_u, log_s)


# ---------------------------------------------------------------------- #
#  Sample-based models: empirical marginals + Gaussian copula (Alg. 2)    #
# ---------------------------------------------------------------------- #
COPULA_FACTORS = ('cholesky', 'symmetric')


def sorted_samples_and_copula(samples, factor='cholesky', jitter=1e-10):
    """ Sorted marginal samples and the normal-score correlation factor.

        Parameters
        ----------
        samples : np.ndarray, shape (m, N, D)
            N samples of the D outputs for each of m inputs.
        factor : {'cholesky', 'symmetric'}
            Factor A of the normal-score correlation R = A A^T. 'cholesky' gives the
            Rosenblatt transform of the Gaussian copula (depends on the output order,
            independent percentiles); 'symmetric' uses R^(1/2) (order-independent).
        jitter : float
            Eigenvalue floor; keeps A invertible for (nearly) collinear outputs.

        Returns
        -------
        sorted_s : np.ndarray, shape (m, N, D)
            Samples sorted along the sample axis.
        A : np.ndarray, shape (m, D, D)
            Factor of the normal-score correlation matrix.
    """
    if factor not in COPULA_FACTORS:
        raise ValueError(f"factor must be one of {COPULA_FACTORS}, got '{factor}'")
    samples = np.asarray(samples, dtype=float)
    m, N, D = samples.shape
    order = np.argsort(samples, axis=1, kind='stable')
    sorted_s = np.take_along_axis(samples, order, axis=1)

    # normal scores Phi^-1((rank + 1/2) / N); every column holds the same set of
    # scores (mean 0, same variance), so R = Z^T Z / sum(scores^2)
    grid = ndtri((np.arange(N) + 0.5) / N)
    scores = np.empty_like(samples)
    np.put_along_axis(scores, order, np.broadcast_to(grid[None, :, None], samples.shape), axis=1)
    R = np.einsum('mni,mnj->mij', scores, scores) / np.sum(grid ** 2)

    lam, V = np.linalg.eigh(R)
    lam = np.maximum(lam, jitter)
    if factor == 'symmetric':
        A = np.einsum('mij,mj,mkj->mik', V, np.sqrt(lam), V)
    else:
        A = np.linalg.cholesky(np.einsum('mij,mj,mkj->mik', V, lam, V))
    return sorted_s, A


TAIL_MODES = ('clip', 'extrapolate')


def _tail_slopes(sorted_s, k):
    """ Slope df/dz of the quantile function at both ends, from the k outermost gaps. """
    N = sorted_s.shape[1]
    zk = ndtri((k + 0.5) / N) - ndtri(0.5 / N)                     # > 0
    lower = (sorted_s[:, k] - sorted_s[:, 0]) / zk
    upper = (sorted_s[:, -1] - sorted_s[:, -1 - k]) / zk
    return lower, upper                                          # (m, D) each


def empirical_score(sorted_s, f, tails='clip', tail_k=None):
    """ Normal score Phi^-1(F(f)) of the piecewise-linear empirical CDF, eq. (28).

        Inside the sample range: Hazen plotting positions (i - 1/2)/N, linear
        interpolation. Beyond the most extreme samples:
          'clip'        : the score of the extreme sample, i.e. u in [1/(2N), 1 - 1/(2N)]
                          (Section 6.1);
          'extrapolate' : the quantile function is continued linearly in normal-score
                          space with the slope of the outermost tail_k samples
                          (Gaussian-like tail), so large deviations are not compressed.

        sorted_s : (m, N, D) sorted samples, or (1, N, D): one sample for all rows.
        f : (m, D) values (any number of rows if sorted_s has one row).
        Returns z : (m, D) normal scores (use ndtr(z) for the percentiles).
    """
    m, N, D = sorted_s.shape
    n = f.shape[0]
    cols = np.arange(D)[None, :]
    if m == 1 and n != 1:
        rows = np.zeros((n, 1), dtype=np.int64)
        k = np.stack([np.searchsorted(sorted_s[0, :, j], f[:, j], side='left') for j in range(D)], axis=1)
    else:
        rows = np.arange(n)[:, None]
        k = (sorted_s < f[:, None, :]).sum(axis=1)             # samples below f, 0..N
    i_lo, i_hi = np.clip(k - 1, 0, N - 1), np.clip(k, 0, N - 1)
    s_lo, s_hi = sorted_s[rows, i_lo, cols], sorted_s[rows, i_hi, cols]
    gap = s_hi - s_lo
    t = np.clip(np.divide(f - s_lo, gap, out=np.zeros_like(f), where=gap > 0), 0.0, 1.0)
    z = ndtri((i_lo + t + 0.5) / N)
    if tails == 'extrapolate':
        b_lo, b_hi = _tail_slopes(sorted_s, tail_k or max(1, N // 100))
        z_edge = ndtri(0.5 / N)
        r = rows[:, 0]
        lo_s, hi_s, b_lo, b_hi = sorted_s[r, 0], sorted_s[r, -1], b_lo[r], b_hi[r]
        below, above = f < lo_s, f > hi_s
        z = np.where(below & (b_lo > 0), z_edge + np.divide(f - lo_s, b_lo, where=b_lo > 0,
                                                            out=np.zeros_like(f)), z)
        z = np.where(above & (b_hi > 0), -z_edge + np.divide(f - hi_s, b_hi, where=b_hi > 0,
                                                             out=np.zeros_like(f)), z)
    elif tails != 'clip':
        raise ValueError(f"tails must be one of {TAIL_MODES}, got '{tails}'")
    return z


def empirical_quantile_from_score(sorted_s, z, index=None, tails='clip', tail_k=None):
    """ Inverse of empirical_score: value at normal score z.

        sorted_s : (m, N, D) sorted samples;  z : (r, D) normal scores.
        index : (r,) int, optional
            Row of sorted_s used for every row of z (default: row i for row i).
            Avoids copying the sample array when many rows share a distribution.
        Returns f : (r, D).
    """
    m, N, D = sorted_s.shape
    r = z.shape[0]
    index = np.arange(r) if index is None else np.asarray(index)
    pos = np.clip(ndtr(z) * N - 0.5, 0.0, N - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, N - 1)
    cols = np.arange(D)[None, :]
    s0 = sorted_s[index[:, None], i0, cols]
    s1 = sorted_s[index[:, None], i1, cols]
    f = s0 + (pos - i0) * (s1 - s0)
    if tails == 'extrapolate':
        b_lo, b_hi = _tail_slopes(sorted_s, tail_k or max(1, N // 100))
        z_edge = ndtri(0.5 / N)
        f = np.where(z < z_edge, sorted_s[index, 0] + (z - z_edge) * b_lo[index], f)
        f = np.where(z > -z_edge, sorted_s[index, -1] + (z + z_edge) * b_hi[index], f)
    elif tails != 'clip':
        raise ValueError(f"tails must be one of {TAIL_MODES}, got '{tails}'")
    return f


# ---------------------------------------------------------------------- #
#  Scores relative to a reference sample (deterministic / SHAP models)    #
# ---------------------------------------------------------------------- #
def reference_scores(reference_values, values, tails='extrapolate'):
    """ Marginal and joint normal scores of values relative to a reference sample.

        Used when the model gives no distribution (PredictiveModel, SHAP
        subtraction): the reference distribution is the empirical distribution of the
        reference remained effects (pooled over all conditions). Marginals: empirical
        CDF as in eq. (28); dependence: Gaussian copula of the reference, as in
        Section 6.2 with the same distribution on both sides.

        Parameters
        ----------
        reference_values : (N, D) reference sample (e.g. remained effects of healthy data).
        values : (n, D) values to score.
        tails : {'extrapolate', 'clip'}, see empirical_score.

        Returns
        -------
        marg_ref, joint_ref : (N, D) scores of the reference sample itself.
        marg, joint : (n, D) scores of values. The joint scores w = A^-1 z
            (R = A A^T, Cholesky) are independent N(0, 1) under the Gaussian copula,
            so sum_j w_j^2 = z^T R^-1 z ~ chi2(D), independent of the output order.
    """
    ref = np.asarray(reference_values, dtype=float)[None]
    sorted_s, A = sorted_samples_and_copula(ref, 'cholesky')
    marg_ref = empirical_score(sorted_s, ref[0], tails=tails)
    marg = empirical_score(sorted_s, np.asarray(values, dtype=float), tails=tails)
    A = A[0]
    joint_ref = solve_triangular(A, marg_ref.T, lower=True).T
    joint = solve_triangular(A, marg.T, lower=True).T
    return marg_ref, joint_ref, marg, joint


# ---------------------------------------------------------------------- #
#  Alarm summary shared by PredictiveModel and ProbabilisticModel         #
# ---------------------------------------------------------------------- #
def outlier_alert(joint_ref, joint_mon, marg_ref, marg_mon, remained_ref, remained_mon, alpha=0.01,
                  threshold_ratio=0.1):
    """ Outlier statistics of the monitoring data and the alert decision.

        A measurement is an outlier if d2 = sum_j w_j^2 of its joint scores exceeds
        the chi2(D) quantile 1 - alpha. The alert is raised if more than
        threshold_ratio of the monitoring measurements are outliers. Per output,
        a marginal score |z_j| > Phi^-1(1 - alpha/2) counts as an outlier.

        Parameters
        ----------
        joint_ref, joint_mon, marg_ref, marg_mon : pd.DataFrame
            Joint and marginal normal scores of the reference and monitoring data.
        remained_ref, remained_mon : pd.DataFrame
            Remained (mapped) effects in physical units.

        Returns
        -------
        dict with the keys limit, z_single, d2_ref, d2_mon (pd.Series), outliers
        (bool array), error_ratio, reference_ratio, alert, problematic_rows,
        reference_statistics, monitoring_statistics.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    D = joint_mon.shape[1]
    limit = float(chi2.ppf(1 - alpha, D))
    z_single = float(ndtri(1 - alpha / 2))
    d2_ref = (joint_ref ** 2).sum(axis=1)
    d2_mon = (joint_mon ** 2).sum(axis=1)
    outliers = (d2_mon > limit).to_numpy()
    error_ratio = float(outliers.mean())

    problematic_rows = marg_mon[outliers].copy()
    problematic_rows['d2'] = d2_mon[outliers]

    def stats(remained, marg):
        table = round(remained.describe().T, 4).reindex(columns=['25%', '50%', '75%', 'count', 'max', 'mean',
                                                                 'min', 'std'])
        table['outlier_ratio'] = (marg.abs() > z_single).mean().round(4).to_numpy()
        return table

    return {'limit': limit, 'z_single': z_single, 'd2_ref': d2_ref, 'd2_mon': d2_mon, 'outliers': outliers,
            'error_ratio': error_ratio, 'reference_ratio': float((d2_ref > limit).mean()),
            'alert': bool(error_ratio > threshold_ratio), 'problematic_rows': problematic_rows,
            'reference_statistics': stats(remained_ref, marg_ref),
            'monitoring_statistics': stats(remained_mon, marg_mon)}
