""" Digital Twin class for model updating using MCMC """

import emcee
import time
import pandas as pd
import numpy as np
from digital_twinning.utils import utils

class DigitalTwin:
    """ Digital Twin class for model updating using MCMC

        Attributes
        ----------
        model : PredictiveModel object
            Predictive model used as digital twin

        E : Error model object
            Error model representing the discrepancy between model predictions and measurements

        Q : VariableSet object
            Variable set defining the probabilistic variables of the model

        Methods
        -------
        __init__(model, E)
            Initialize the Digital Twin with given model and error model
        
        get_logprob(q)
            Compute the log-probability of the parameters q given the measurements

        update(y_m, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False, name_pairs=None)
            Update the digital twin using MCMC with given measurements y_m

        likelihood(q, y_m)
            Compute the likelihood of the measurements y_m given parameters q

        loglikelihood(q, y_m)
            Compute the log-likelihood of the measurements y_m given parameters q

        prior(q)
            Compute the prior probability of the parameters q

        logprior(q)
            Compute the log-prior probability of the parameters q

        get_mean_and_var_of_posterior()
            Get the mean and variance of the posterior parameter distribution

        get_posterior_samples()
            Get the samples from the posterior parameter distribution

        get_MAP()
            Get the maximum a posteriori estimate of the parameters
        get_indices_from_mx(H)
            Get the indices of parameters and measurements from the mapping matrix H """
       
    def __init__(self, model, E):
        """ Initialize the Digital Twin with given model and error model

            Parameters
            ----------
            model : PredictiveModel object
                Predictive model used as digital twin

            E : VariableSet object
                Error model representing the discrepancy between model predictions and measurements"""
        
        self.model = model
        self.E = E
        self.Q = model.Q
        
        
    def get_logprob(self, q):
        """ Compute the log-probability of the parameters q given the measurements

            Parameters
            ----------
            q : array_like
                Parameter values at which to compute the log-probability

            Returns
            -------
            logprob : float
                Log-probability of the parameters q given the measurements """
        
        if self.Q_ == 'default':
            logprob = self.loglikelihood(q, self.y_m) + self.logprior(q)
            return logprob
        else:
            logprob = self.loglikelihood(q, self.y_m) + self.Q_.logpdf(q)
            return logprob 
        
    def update(self, y_m, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False, name_pairs=None):
        """ Update the digital twin using MCMC with given measurements y_m

            Parameters
            ----------
            y_m : array_like
                Measured quantities of interest

            nwalkers : int, optional
                Number of MCMC walkers, by default 150

            nburn : int, optional
                Number of burn-in steps, by default 100

            niter : int, optional
                Number of MCMC iterations, by default 350

            Q_ : str or VariableSet, optional
                VariableSet object defining the prior distribution of parameters, by default 'default'

            plot_samples : bool, optional
                If True, plot the MCMC samples, by default False

            name_pairs : _type_, optional
                List of tuples defining mapping between model outputs and measurements, by default None """
        
        self.Q_ = Q_
        if name_pairs is not None:
            H = self.model.create_mx_from_tuple(name_pairs, list(y_m.keys()))
            self.q_indices, y_indices = self.get_indices_from_mx(H)
            y_m = y_m.to_numpy().reshape(-1, 1)[y_indices]
            self.E = self.E.diminished_paramset(y_indices)
        else:
            self.q_indices = None
            y_m = y_m.to_numpy()
        self.y_m = y_m
        num_param = self.Q.num_params()
        
        if Q_ == 'default':
            p0 = self.Q.sample(nwalkers)
        else:
            p0 = Q_.sample(nwalkers)
        self.p0 = p0

        
        print('MCMC creating')
        sampler = emcee.EnsembleSampler(nwalkers, num_param, self.get_logprob)#, pool=pool)
        start_time = time.time()

        print('Burning period')
        state = sampler.run_mcmc(p0, nburn, progress = True)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = True)
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.sampler = sampler
            
        if plot_samples:
            # TODO sns pairplot
            pass
        
    def likelihood(self, q, y_m):
        """ Compute the likelihood of the measurements y_m given parameters q

            Parameters
            ----------
            q : array_like
                Parameter values

            y_m : array_like
                Measured quantities of interest

            Returns
            -------
            p : float
                Likelihood of the measurements y_m given parameters q"""
        
        # TODO inverse_transform to predicted data
        #q = self.model.get_scaled_q(q.reshape(1,-1))
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        d = y_m - self.model.predict(q_df)
        d = d.transpose()
        p = self.E.pdf(d)
        return p
    
    def loglikelihood(self, q, y_m):
        """ Compute the log-likelihood of the measurements y_m given parameters q

            Parameters
            ----------
            q : array_like
                Parameter values

            y_m : array_like
                Measured quantities of interest

            Returns
            -------
            logp : float
                Log-likelihood of the measurements y_m given parameters q"""
        
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        if self.q_indices is None:
            d = y_m - self.model.predict(q_df)
            d = d.transpose()
        else:
            d = y_m - self.model.predict(q_df).reshape(-1, 1)[self.q_indices]
        logp = self.E.logpdf(d)
        return logp
    
    def prior(self, q):
        """ Compute the prior probability of the parameters q

            Parameters
            ----------
            q : array_like
                Parameter values

            Returns
            -------
            pr : float
                Prior probability of the parameters q """
        
        pr = self.Q.pdf(q.reshape(1,-1))
        return pr
    
    def logprior(self, q):
        """ Compute the log-prior probability of the parameters q

            Parameters
            ----------
            q : array_like
                Parameter values

            Returns
            -------
            logpr : float
                Log-prior probability of the parameters q """
        
        logpr = self.Q.logpdf(q.reshape(-1,1))
        return logpr
    
    def get_mean_and_var_of_posterior(self):
        """ Get the mean and variance of the posterior parameter distribution

            Returns
            -------
            means_df : pandas.DataFrame
                DataFrame containing the mean of the posterior parameter distribution
            
            variances_df : pandas.DataFrame
                DataFrame containing the variance of the posterior parameter distribution """
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        means = np.mean(post_samples, axis=0)
        variances = np.var(post_samples, axis=0)
        means_df = pd.DataFrame(means.reshape(1,-1), columns=self.Q.param_names())
        variances_df = pd.DataFrame(variances.reshape(1,-1), columns=self.Q.param_names())
        return means_df, variances_df
    
    def get_posterior_samples(self):
        """ Get the samples from the posterior parameter distribution

            Returns
            -------
            post_samples_df : pandas.DataFrame
                DataFrame containing the samples from the posterior parameter distribution"""
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        post_samples_df = pd.DataFrame(post_samples, columns=self.Q.param_names())
        return post_samples_df
    
    def get_MAP(self): # maximum a posterior estimate
        """ Get the maximum a posteriori estimate of the parameters

            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the maximum a posteriori estimate of the parameters """
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        
        map_estimate = np.zeros((1, post_samples.shape[1]))
        for p in range(post_samples.shape[1]):
            p_estimate = utils.estimate_maxima(post_samples[:,p])
            map_estimate[0,p] = p_estimate

        map_df = pd.DataFrame(map_estimate, columns=self.Q.param_names())
        return map_df
    
    def get_indices_from_mx(self, H):
        """ Get the indices of parameters and measurements from the mapping matrix H

            Parameters
            ----------
            H : array_like
                Mapping matrix between model outputs and measurements

            Returns
            -------
            q_indices : list
                List of indices of parameters corresponding to measurements
            
            y_indices : list
                List of indices of measurements corresponding to parameters """
        
        q_indices = []
        y_indices = []
        for i in range(H.shape[1]):
            if 1 in H[:, i]:
                q_indices.append(np.where(H[:, i]==1)[0][0])

        for i in range(H.shape[0]):
            if 1 in H[i, :]:
                y_indices.append(np.where(H[i, :]==1)[0][0])

        return q_indices, y_indices
                