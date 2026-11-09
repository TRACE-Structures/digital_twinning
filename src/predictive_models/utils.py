from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import re

FREQ_RE = re.compile(r"^f\.(\d+)$")
EIGVEC_RE = re.compile(r"^S\.(\d+)\.(\d+)\.([xyzXYZ])$")
_DIR_PRIORITY = {"x": 0, "y": 1, "z": 2}  # preferred display order when inferring


def identity_func(x):
    '''
    Identity function that returns the input as a NumPy array.

    Parameters:
    -----------
    x : pd.DataFrame or np.ndarray
        Input data to be converted.

    Returns:
    -------
    np.ndarray
        Input data as a NumPy array.    
    '''

    if isinstance(x, np.ndarray):
        return x
    return x.to_numpy()

def identity_inverse_func(x):
    '''
    Inverse identity function that returns the input as is.

    Parameters:
    -----------
    x : np.ndarray
        Input data.
        
    Returns:
    -------
    np.ndarray
        Input data as is.
    '''

    return x

def get_scaler(scale_method, y=None):
    """ Returns a scaler object based on the specified scaling method.

        Parameters
        ----------
        scale_method : str
            Scaling method to use ('identity' or 'minmax').
        y : array-like, optional
            Target data for fitting the scaler (required for 'minmax' scaling).

        Returns
        -------
        scaler : object
            Scaler object corresponding to the specified scaling method.
    """

    if scale_method == 'identity':
        scaler = FunctionTransformer(func=identity_func, inverse_func=identity_inverse_func)
    elif scale_method == 'minmax':
        scaler = MinMaxScaler((0.05, 0.95))
        scaler.fit(y)
    return scaler

def reorder_header(columns, m=None, n=None, directions=None):
    """
        Validate that `columns` contains exactly the expected set of
        'f.<i>' and 'S.<i>.<j>.<d>' columns for m modes, n nodes, and the
        given directions -- in ANY order -- and return them reordered into
        canonical order. Raises ValueError on any unrecognized, missing,
        or out-of-range column.

        Parameters
        ----------
        columns : list of str
            List of column names to validate and reorder.
        m : int, optional
            Number of frequency modes. If None, inferred from the columns.
        n : int, optional
            Number of nodes. If None, inferred from the columns.
        directions : list of str, optional
            List of directions. If None, inferred from the columns.

        Returns
        -------
        list of str
            Reordered list of column names in canonical order.
    """
    columns = list(columns)

    freq_found, eigvec_found, unrecognized = {}, {}, []
    for col in columns:
        fm = FREQ_RE.match(col)
        if fm:
            freq_found[int(fm.group(1))] = col
            continue
        em = EIGVEC_RE.match(col)
        if em:
            mode_id, node_id, dir_id = int(em.group(1)), int(em.group(2)), em.group(3).lower()
            eigvec_found[(mode_id, node_id, dir_id)] = col
            continue
        unrecognized.append(col)

    if unrecognized:
        raise ValueError(
            f"Header contains columns that don't match 'f.<mode>' or "
            f"'S.<mode>.<node>.<dir>': {unrecognized}"
        )

    if m is None:
        if not freq_found:
            raise ValueError("No 'f.<mode>' columns found; cannot infer number of modes.")
        m = max(freq_found)

    if directions is None:
        found_dirs = {d for (_, _, d) in eigvec_found}
        if not found_dirs:
            raise ValueError("No 'S.<mode>.<node>.<dir>' columns found; cannot infer directions.")
        directions = sorted(found_dirs, key=lambda d: (_DIR_PRIORITY.get(d, 99), d))
    else:
        directions = [d.lower() for d in directions]

    if n is None:
        found_nodes = {node for (_, node, _) in eigvec_found}
        if not found_nodes:
            raise ValueError("No 'S.<mode>.<node>.<dir>' columns found; cannot infer number of nodes.")
        n = max(found_nodes)

    expected_freq = [f"f.{i}" for i in range(1, m + 1)]
    expected_eigvec = [(i, j, d) for i in range(1, m + 1) for j in range(1, n + 1) for d in directions]

    missing_freq = [i for i in range(1, m + 1) if i not in freq_found]
    missing_eigvec = [key for key in expected_eigvec if key not in eigvec_found]
    if missing_freq or missing_eigvec:
        parts = []
        if missing_freq:
            parts.append("missing frequency columns: " + ", ".join(f"f.{i}" for i in missing_freq))
        if missing_eigvec:
            parts.append("missing eigenvector columns: " + ", ".join(f"S.{i}.{j}.{d}" for i, j, d in missing_eigvec))
        raise ValueError("Header is incomplete -- " + "; ".join(parts))

    extra_freq = [freq_found[i] for i in freq_found if i not in range(1, m + 1)]
    extra_eigvec = [eigvec_found[key] for key in eigvec_found if key not in expected_eigvec]
    if extra_freq or extra_eigvec:
        raise ValueError(
            f"Header contains unexpected entries outside the expected "
            f"m={m}, n={n}, directions={directions}: {extra_freq + extra_eigvec}"
        )

    return expected_freq + [f"S.{i}.{j}.{d}" for (i, j, d) in expected_eigvec]

def reorder_dataframe(df, m=None, n=None, directions=None):
    """ Returns df with columns reordered canonically. Raises ValueError if invalid.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame to reorder.
        m : int, optional
            Number of frequency modes. If None, inferred from the DataFrame.
        n : int, optional
            Number of nodes. If None, inferred from the DataFrame.
        directions : list of str, optional
            List of directions. If None, inferred from the DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns reordered canonically.
    """
    order = reorder_header(df.columns, m=m, n=n, directions=directions)
    return df[order]

def _infer_dims(columns, directions=None):
    """ Infer (m, n, directions) from column names, without validation.
    Assumes the columns already form a complete, correctly named set
    (e.g. as returned by reorder_dataframe).

        Parameters
        ----------
        columns : list of str
            List of column names to infer dimensions from.
        directions : list of str, optional
            List of directions. If None, inferred from the columns.

        Returns
        -------
        m : int
            Number of frequency modes.
        n : int
            Number of nodes.
        directions : list of str
            List of directions in canonical order.
    """
    columns = list(columns)
    m = 0
    for col in columns:
        fm = FREQ_RE.match(col)
        if fm:
            m = max(m, int(fm.group(1)))

    found_nodes, found_dirs = set(), set()
    for col in columns:
        em = EIGVEC_RE.match(col)
        if em:
            found_nodes.add(int(em.group(2)))
            found_dirs.add(em.group(3).lower())
    n = max(found_nodes) if found_nodes else 0

    if directions is None:
        directions = sorted(found_dirs, key=lambda d: (_DIR_PRIORITY.get(d, 99), d))
    else:
        directions = [d.lower() for d in directions]

    return m, n, directions

def extract_arrays(df, m=None, n=None, directions=None):
    """
        Extract frequencies and eigenvectors from a DataFrame whose header is
        already known-valid and in canonical order (e.g. the output of
        reorder_dataframe). Does NOT re-validate the header.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the frequency and eigenvector data.
        m : int, optional
            Number of frequency modes. If None, inferred from the DataFrame columns.
        n : int, optional
            Number of nodes. If None, inferred from the DataFrame columns.
        directions : list of str, optional
            List of directions. If None, inferred from the DataFrame columns.

        Returns
        -------
        frequencies : ndarray, shape (n_samples, n_modes)
        eigenvectors: ndarray, shape (n_samples, n_modes, n_nodes * n_directions)
            Last axis ordered as: all nodes for direction[0], then all nodes for
            direction[1], ... i.e. [d0_node1, ..., d0_nodeN, d1_node1, ..., dD_nodeN]
    """
    m, n, directions = _infer_dims(df.columns, directions=directions) if (m is None or n is None or directions is None) \
        else (m, n, [d.lower() for d in directions])

    freq_cols = [f"f.{i}" for i in range(1, m + 1)]
    frequencies = df[freq_cols].to_numpy(dtype=np.float64)

    n_samples = len(df)
    eigenvectors = np.empty((n_samples, m, n * len(directions)), dtype=np.float64)
    for mode_idx, i in enumerate(range(1, m + 1)):
        cols = [f"S.{i}.{j}.{d}" for d in directions for j in range(1, n + 1)]
        eigenvectors[:, mode_idx, :] = df[cols].to_numpy(dtype=np.float64)

    return frequencies, eigenvectors

def get_modes_and_nodes(df):
    """ Infer the number of frequency modes and nodes from a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the frequency and eigenvector data.

        Returns
        -------
        m : int
            Number of frequency modes.
        n : int
            Number of nodes. """
    
    m, n, d = _infer_dims(df.columns)
    return m, n*len(d)