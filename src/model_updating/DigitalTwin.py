""" Digital Twin class for model updating using MCMC """

import emcee
import time
import pandas as pd
import numpy as np
import uncertain_variables as uv
from .plotting_functions import MCMC_plot
from scipy.optimize import minimize


class DigitalTwin:
    """ Digital Twin class for model updating using MCMC

        Attributes
        ----------
        model : PredictiveModel object
            Predictive model used as digital twin

        E : Error model object
            Error model representing the discrepancy between model predictions and measurements

        Q : VariableSet object
            Variable set defining the probabilistic variables of the model """
       
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
        self.Q = self.get_Q_germ()

    def get_Q_germ(self):
        
        Q_germ = uv.VariableSet()
        variable_names = self.model.Q.variable_names()
        for i in range(len(variable_names)):
            name = variable_names[i]
            dist = self.model.Q.get_variables()[name].get_base_dist()            
            P = uv.Variable(name, dist)
            Q_germ.add(P)
        return Q_germ
        
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
        
    def update(self, y_m, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False, name_pairs=None, progress=True):
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
                List of tuples defining mapping between model outputs and measurements, by default None
            progress : bool, optional
                If True, show progress bar during MCMC sampling, by default True"""
        
        self.Q_ = Q_
        if name_pairs is not None:
            H = self.model.create_mx_from_tuple(name_pairs, list(y_m.keys()))
            self.q_indices, y_indices = self.get_indices_from_mx(H)
            y_m = y_m.to_numpy().reshape(-1, 1)[y_indices]
            self.E = self.E.diminished_variable_set(y_indices)
        else:
            self.q_indices = None
            y_m = y_m.to_numpy()
        self.y_m = y_m
        num_param = self.Q.num_variables()
        
        if Q_ == 'default':
            p0 = self.Q.sample(nwalkers)
        else:
            p0 = Q_.sample(nwalkers)
        self.p0 = p0

        
        print('MCMC creating')
        sampler = emcee.EnsembleSampler(nwalkers, num_param, self.get_logprob)#, pool=pool)
        start_time = time.time()

        print('Burning period')
        state = sampler.run_mcmc(p0, nburn, progress = progress)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = progress)
    
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
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.variable_names())
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
        
        q = q.reshape(1, -1)
        q_df = pd.DataFrame(q, columns=self.Q.variable_names())
        if self.q_indices is None:
            q_model = self.model.Q.germ2variable(q)
            if np.any(~np.isfinite(q_model)):
                return -np.inf
            q_df = pd.DataFrame(q_model, columns=self.model.Q.variable_names())
            pred = self.model.predict(q_df)
            if np.any(~np.isfinite(pred)):
                return -np.inf
            d = y_m - pred
            d = d.transpose()
        else:
            pred = self.model.predict(q_df).reshape(-1, 1)[self.q_indices]
            if np.any(~np.isfinite(pred)):
                return -np.inf
            d = y_m - pred
        logp = self.E.logpdf(d)
        logp = np.asarray(logp)
        if not np.all(np.isfinite(logp)):
            return -np.inf
        return float(logp)
    
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
    
    def get_mean_and_var_of_posterior(self, scaled=True):
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
        if scaled:
            means = self.model.Q.germ2variable(means.reshape(1,-1))
            variances = self.model.Q.germ2variable(variances.reshape(1,-1))
        means_df = pd.DataFrame(means, columns=self.Q.variable_names())
        variances_df = pd.DataFrame(variances, columns=self.Q.variable_names())
        return means_df, variances_df
    
    def get_posterior_samples(self, scaled=True):
        """ Get the samples from the posterior parameter distribution

            Returns
            -------
            post_samples_df : pandas.DataFrame
                DataFrame containing the samples from the posterior parameter distribution"""
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        if scaled:
            post_samples = self.model.Q.germ2variable(post_samples)

        post_samples_df = pd.DataFrame(post_samples, columns=self.Q.variable_names())
        return post_samples_df
    
    # def get_MAP_old(self, scaled = True): # maximum a posterior estimate
    #     """ Get the maximum a posteriori estimate of the parameters

    #         Returns
    #         -------
    #         map_df : pandas.DataFrame
    #             DataFrame containing the maximum a posteriori estimate of the parameters """
        
    #     sampler = self.sampler
    #     post_samples = sampler.get_chain(flat=True)
        
    #     map_estimate = np.zeros((1, post_samples.shape[1]))
    #     for p in range(post_samples.shape[1]):
    #         p_estimate = utils.estimate_maxima(post_samples[:,p])
    #         map_estimate[0,p] = p_estimate

    #     if scaled:
    #         map_estimate = self.model.Q.germ2variable(map_estimate)

    #     map_df = pd.DataFrame(map_estimate, columns=self.Q.variable_names())
    #     return map_df


    def get_MAP(self, scaled=True, refine=True, n_starts=5, space='germ', return_logprob=False):
        """ Joint maximum a posteriori estimate of the parameters

            Takes the MCMC sample with the highest log-posterior (always a point on
            the posterior) and, optionally, refines it with a multistart Nelder-Mead
            search on the exact log-posterior. Unlike combining the modes of the 1-D
            marginals, this gives the mode of the JOINT posterior.

            Parameters
            ----------
            scaled : bool, optional
                Return the estimate in physical parameters (germ2variable), by default True

            refine : bool, optional
                Refine the best samples with a local optimizer, by default True

            n_starts : int, optional
                Number of best distinct samples used as optimizer starts, by default 5

            space : {'germ', 'physical'}, optional
                Space in which the density is maximized. 'germ' (default) maximizes the
                posterior density of the germ variables the sampler works in. 'physical'
                maximizes the density of the physical parameters q, which includes the
                Jacobian of the germ -> q map: log pi_q = log pi_theta - log|det dq/dtheta|.
                The two modes differ for nonlinear maps (e.g. lognormal parameters).

            return_logprob : bool, optional
                Also return the maximized log density, by default False

            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the joint MAP estimate of the parameters

            Notes
            -----
            If the posterior is a ridge or has several equally probable modes (a
            non-identifiable combination of parameters), the MAP is not unique and
            should not be reported as a single point estimate. """
        
        if space not in ('germ', 'physical'):
            raise ValueError("space must be 'germ' or 'physical'")

        chain = self.sampler.get_chain(flat=True)
        logp = self.sampler.get_log_prob(flat=True)

        def objective_logp(theta):
            lp = np.asarray(self.get_logprob(np.asarray(theta, dtype=float)), dtype=float)
            if lp.size != 1:
                raise ValueError(f"get_logprob returned {lp.size} values instead of one; "
                                 f"check the shape expected by Q.logpdf in logprior()")
            lp = lp.item()
            if not np.isfinite(lp):
                return -np.inf
            if space == 'physical':
                lp -= self._log_abs_det_jacobian(theta)
            return lp

        # starting points: best distinct samples (by the objective actually maximized)
        order = np.argsort(logp)[::-1]
        cand = []
        for i in order:
            if not any(np.allclose(chain[i], c) for c in cand):
                cand.append(chain[i])
            if len(cand) >= max(n_starts, 1) * (4 if space == 'physical' else 1):
                break
        cand_lp = [objective_logp(c) for c in cand]
        best_idx = np.argsort(cand_lp)[::-1][:max(n_starts, 1)]
        best_theta, best_lp = cand[best_idx[0]], cand_lp[best_idx[0]]

        if refine:
            for j in best_idx:
                res = minimize(lambda th: -objective_logp(th), cand[j], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000 * chain.shape[1]})
                lp = -res.fun
                if np.isfinite(lp) and lp > best_lp:
                    best_theta, best_lp = res.x, lp

        map_estimate = np.asarray(best_theta, dtype=float).reshape(1, -1)
        if scaled:
            map_estimate = self.model.Q.germ2variable(map_estimate)

        map_df = pd.DataFrame(map_estimate, columns=self.Q.variable_names())
        return (map_df, best_lp) if return_logprob else map_df

    def _log_abs_det_jacobian(self, theta, h=1e-6):
        """ log|det d q / d theta| of the germ -> physical map (central differences) """
        
        theta = np.asarray(theta, dtype=float)
        n = theta.size
        J = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            qp = np.asarray(self.model.Q.germ2variable((theta + e).reshape(1, -1)), dtype=float).ravel()
            qm = np.asarray(self.model.Q.germ2variable((theta - e).reshape(1, -1)), dtype=float).ravel()
            J[:, j] = (qp - qm) / (2 * h)
        sign, logdet = np.linalg.slogdet(J)
        return logdet if sign != 0 else np.inf
    
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


    def plot_MCMC(self, X_train, extra_point=None, extra_point_name='Extra point', map=True, custom_xlim=None, custom_ylim=None):
        fig = MCMC_plot(self, X_train, extra_point=extra_point, extra_point_name=extra_point_name, map=map, custom_xlim=custom_xlim, custom_ylim=custom_ylim)
        return fig