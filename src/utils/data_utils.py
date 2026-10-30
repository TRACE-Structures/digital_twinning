import numpy as np
import uncertain_variables as uv
import pyuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertain_variables import Variable, UniformDistribution, VariableSet
import seaborn as sns

def gaussian(x, mu, sig):
    ''' Gaussian function pdf value at x.
    
        Parameters
        ----------
        x : float or array_like
            Points at which to evaluate the Gaussian.
        mu : float
            Mean of the Gaussian.
        sig : float
            Standard deviation.
        
        Returns
        -------
        float or ndarray
            Gaussian pdf value(s) at x. '''
    
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def get_distribution(X, parameter_idx, verbose=False):
    ''' Determine if data for a parameter is better fit by Uniform or Normal distribution.
    
        Parameters
        ----------
        X : pandas.DataFrame
            Input data containing parameters as columns.
        parameter_idx : int
            Column index of parameter in X.
        verbose : bool, optional
            If True, print error metrics (default False).
        
        Returns
        -------
        str
            'Uniform' or 'Normal' distribution label. '''
    
    hist, bin_edges = np.histogram((X.values[:, parameter_idx]), density=True, bins=50)
    mu = np.mean(X.values[:, parameter_idx])
    sigma = np.std(X.values[:, parameter_idx])
    gaussian_curve = np.zeros(len(bin_edges))
    uniform_line = np.zeros(len(bin_edges))
    for i in range(len(bin_edges)):
        gaussian_curve[i] = gaussian(bin_edges[i], mu, sigma)
        uniform_line[i] = np.mean(hist)
    gaussian_error = np.mean(np.sqrt((hist - gaussian_curve[:-1])**2))
    uniform_error = np.mean(np.sqrt((hist - uniform_line[:-1])**2))
    if verbose:
        print('Gaussian error: {}'.format(gaussian_error))
        print('Uniform error: {}'.format(uniform_error))
    if gaussian_error > uniform_error:
        return 'Uniform'
    else:
        return 'Normal'
    
def get_variableset_from_data(data, verbose=False):
    ''' Create a VariableSet from data by choosing Uniform or Normal distribution per parameter.
    
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame of samples where columns correspond to parameters.
        verbose : bool, optional
            Whether to print chosen distribution info (default False).
        
        Returns
        -------
        VariableSet
            VariableSet composed of Variables with fitted distributions. '''
    
    Q = uv.VariableSet()
    for i in range(data.values.shape[1]):
        dist = get_distribution(data, i)    
        if dist == 'Uniform':
            l_bound = np.min(data.values, axis=0)[i]
            u_bound = np.max(data.values, axis=0)[i]
            P = uv.Variable(data.columns[i], uv.UniformDistribution(l_bound, u_bound))
            if verbose:
                print('{} - UniformDistribution({}, {})'.format(data.columns[i], l_bound, u_bound))
        elif dist == 'Normal':
            mu = np.mean(data.values[:, i])
            sigma = np.std(data.values[:, i])
            P = uv.Variable(data.columns[i], uv.NormalDistribution(mu, sigma))
            if verbose:
                print('{} - NormalDistribution({}, {})'.format(data.columns[i], mu, sigma))
        Q.add(P)
    return Q