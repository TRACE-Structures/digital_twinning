from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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
