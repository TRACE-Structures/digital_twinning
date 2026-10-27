""" JointManager for multi-model updating using MCMC"""

import uncertain_variables as uv
import pandas as pd
import emcee
import time
import numpy as np
from scipy.optimize import minimize
from .plotting_functions import multimodel_MCMC_plot


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class JointManager:
    """ Class to manage joint model updating for multiple models through one or more shared parameters
    
        Attributes
        ----------
        models : list of SurrogateModel
            List of surrogate models for each model
            
        joint_parameters : dict
            Dictionary defining joint parameters across the models
            
        Q : VariableSet
            Joint VariableSet for all models
            
        indices : list of list of int
            Indices mapping model parameters to joint parameters
            
        scale : list of list of float
            Scaling factors for each model's parameters
            
        shift : list of list of float
            Shift values for each models's parameters """
    
    def __init__(self, models, joint_parameters):
        """ Initialize the JointManager with given models and joint parameters

            Parameters
            ----------
            models : list of SurrogateModel
                List of surrogate models for each model

            joint_parameters : dict
                Dictionary defining joint parameters across models """
        
        self.models = models
        self.joint_parameters = joint_parameters
        self.Q, self.indices = self.get_joint_paramset_and_indices(models, joint_parameters)
        self._map_cache = None
    
    def get_joint_paramset_and_indices(self, models, joint_parameters):
        """ Create joint VariableSet and mapping indices

            Parameters
            ----------
            models : list of SurrogateModel
                List of surrogate models of the models

            joint_parameters : dict
                Dictionary defining joint parameters across models

            Returns
            -------
            Q_joint : VariableSet
                Joint VariableSet for all models

            indices : list of list of int
                Indices mapping model parameters to joint parameters

            scale : list of list of float
                Scaling factors for each model's parameters

            shift : list of list of float
                Shift values for each model's parameters """
        
        model_parameters = []
        for b in range(len(models)):
            model_parameters.append(models[b].Q.variable_names())
        joint_names = list(joint_parameters.keys())
        for i in range(len(joint_parameters)):
            parameter_list = joint_parameters[joint_names[i]]
            assert len(parameter_list) == len(model_parameters), "The jointparameters should contain every model. Put a '-' when you don't want to use the model for parameter join"
            for j in range(len(model_parameters)):
                if parameter_list[j] != '-':
                    assert parameter_list[j] in model_parameters[j], "Parameter '{}' is not in parameter list {}".format(parameter_list[j], model_parameters[j])

        indices = []
        for i in range(len(model_parameters)):
            indices.append([None]*len(model_parameters[i]))

        Q_joint = uv.VariableSet()

        for i in range(len(joint_parameters)):
            joint_name = joint_names[i]
            parameter_list = joint_parameters[joint_name]
            joint_dist = None
            for j in range(len(models)):
                variable_name = parameter_list[j]
                if variable_name != '-':
                    dist = models[j].Q.get_variables()[variable_name].get_base_dist()
                    if joint_dist == None:
                        joint_dist = dist
                    assert dist == joint_dist, "Joining variables with different distribution type is not supported"
                    idx = models[j].Q.variable_names().index(variable_name)
                
                    indices[j][idx] = i
                    model_parameters[j].remove(variable_name)
            J = uv.Variable(joint_name, joint_dist)
            Q_joint.add(J)
            
        for i in range(len(model_parameters)):
            for j in range(len(model_parameters[i])):
                variable_name = model_parameters[i][j]
                dist = models[i].Q.variables[variable_name].get_base_dist()
                V = uv.Variable(variable_name, dist)    
                Q_joint.add(V)
                
                idx = Q_joint.variable_names().index(variable_name)
                original_index = models[i].Q.variable_names().index(variable_name)
                indices[i][original_index]  = idx

        return Q_joint, indices


    def get_logprob(self, q):
        """ Compute log-probability for MCMC

            Parameters
            ----------
            q : array_like
                Parameter values

            Returns
            -------
            logprob : float
                Log-probability of the parameter values """
        
        if self.Q_ == 'default':
            logprob = self.loglikelihood(q, self.y_m) + self.logprior(q)
            return logprob
        else:
            logprob = self.loglikelihood(q, self.y_m) + self.Q_.logpdf(q)
            return logprob
        
    def logprior(self, q):
        """ Compute log-prior probability

            Parameters
            ----------
            q : array_like
                Parameter values

            Returns
            -------
            logpr : float
                Log-prior probability of the parameter values """
        
        logpr = self.Q.logpdf(q.reshape(-1,1))
        return logpr
    
    def loglikelihood(self, q, y_m):
        """ Compute log-likelihood of observed data

            Parameters
            ----------
            q : array_like
                Parameter values

            y_m : array_like
                Measured data

            Returns
            -------
            logp : float
                Log-likelihood of the observed data given the parameter values """
        
        q = q.reshape(1, -1)
        q_df = pd.DataFrame(q, columns=self.Q.variable_names())
        predictions = np.zeros((1, 0))
        for i in range(len(self.models)):
            q_model = self.models[i].Q.germ2variable(q[:, self.indices[i]])
            q_df = pd.DataFrame(q_model, columns=self.models[i].Q.variable_names())
            y_model = self.models[i].predict(q_df)
            predictions = np.concatenate((predictions, y_model), axis=1)
        d = y_m - predictions
        d = d.transpose()
        logp = self.E.logpdf(d)
        return logp
    
    def update(self, y_list, sigmas, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False, progress=True):
        """ Perform MCMC update with observed data

            Parameters
            ----------
            y_list : list of pandas.DataFrame
                List of observed data DataFrames for each model

            sigmas : list of pandas.DataFrame
                List of standard deviation DataFrames for each model

            nwalkers : int, optionnal
                Number of MCMC walkers, by default 150

            nburn : int, optional
                Number of burn-in steps, by default 100

            niter : int, optional
                Number of MCMC iterations, by default 350

            Q_ : str or VariableSet, optional
                VariableSet object defining the prior distribution of parameters, by default 'default'

            plot_samples : bool, optional
                If True, plot MCMC samples, by default False """
        
        self.Q_ = Q_
        self._map_cache = None
        y_m = np.zeros((1, 0))
        for i in range(len(y_list)):
            y_model = y_list[i].to_numpy()
            y_m = np.concatenate((y_m, y_model), axis=1)
        self.y_m = y_m
        num_param = self.Q.num_variables()

        joint_sigma = np.zeros((1, 0))
        for i in range(len(sigmas)):
            model_sigma = sigmas[i].values
            joint_sigma = np.concatenate((joint_sigma, model_sigma), axis=1)
        joint_sigma = joint_sigma.reshape(-1, 1)
        self.E = uv.generate_stndrn_variable_set(joint_sigma)


        if Q_ == 'default':
            #logprob = lambda q: self.loglikelihood(q, y_m) + self.logprior(q)
            p0 = self.Q.sample(nwalkers)
        else:
            #logprob = lambda q: self.loglikelihood(q, y_m) + Q_.logpdf(q)
            p0 = Q_.sample(nwalkers)
        self.p0 = p0

        
        print('MCMC creating')
        sampler = emcee.EnsembleSampler(nwalkers, num_param, self.get_logprob) #pool=pool
        start_time = time.time()

        print('Burning period')
        state = sampler.run_mcmc(p0, nburn, progress = progress)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = progress)
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.sampler = sampler

    # ------------------------------------------------------------------ helpers
    def _germ_to_model(self, theta, i):
        """ Physical parameters of model i for joint germ vector(s) theta (n, n_joint) """
        
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.asarray(self.models[i].Q.germ2variable(theta[:, self.indices[i]]), dtype=float)
 
    def _variable_owner(self):
        """ For each joint variable j: (model index i, local index k) of the first model using it """
        
        owner = [None] * self.Q.num_variables()
        for i, idx in enumerate(self.indices):
            for k, j in enumerate(idx):
                if owner[j] is None:
                    owner[j] = (i, k)
        return owner
 
    def _joint_logprob(self, theta):
        """ Joint log-posterior as a finite Python float (-inf outside the support) """
        
        lp = np.asarray(self.get_logprob(np.asarray(theta, dtype=float)), dtype=float)
        if lp.size != 1:
            raise ValueError(f"get_logprob returned {lp.size} values instead of one; "
                             f"check the shape expected by Q.logpdf in logprior()")
        lp = lp.item()
        return lp if np.isfinite(lp) else -np.inf
 
    def _log_abs_det_jacobian(self, theta, h=1e-6):
        """ log|det dq/dtheta| of the germ -> physical map of the joint variables.
 
            Assumes independent variables (each germ2variable acts component-wise), so
            the Jacobian is diagonal. A joint variable shared by several models is
            mapped with the first model that uses it (see _variable_owner). """
        
        theta = np.asarray(theta, dtype=float).ravel()
        total = 0.0
        for j, (i, k) in enumerate(self._variable_owner()):
            e = np.zeros_like(theta)
            e[j] = h
            dq = (self._germ_to_model(theta + e, i)[0, k] - self._germ_to_model(theta - e, i)[0, k]) / (2 * h)
            if dq == 0 or not np.isfinite(dq):
                return np.inf
            total += np.log(abs(dq))
        return total
 
    # ---------------------------------------------------------------------- MAP
    def get_MAP(self, scaled=False, refine=True, n_starts=5, space='germ', return_logprob=False):
        """ Joint maximum a posteriori estimate of the joint variable set
 
            Mode of the JOINT posterior: the sample with the highest stored
            log-posterior, optionally refined by a multistart Nelder-Mead search on
            the exact joint log-posterior. The result is cached until the next
            update() (or a call with different settings).
 
            Parameters
            ----------
            scaled : bool, optional
                Return physical values instead of germs, by default False. A joint
                variable shared by several models is mapped with the first model that
                uses it; use get_MAP_by_model for each model's own physical values.
 
            refine : bool, optional
                Refine the best samples with a local optimizer, by default True
 
            n_starts : int, optional
                Number of best distinct samples used as optimizer starts, by default 5
 
            space : {'germ', 'physical'}, optional
                Density that is maximized. 'germ' (default): the posterior density of
                the joint germs the sampler works in. 'physical': the density of the
                physical parameters (adds -log|det dq/dtheta|, see
                _log_abs_det_jacobian). The modes differ for nonlinear maps.
 
            return_logprob : bool, optional
                Also return the maximized log density, by default False
 
            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the joint MAP estimate (columns: joint variables)
 
            Notes
            -----
            If the posterior is a ridge or has several equally probable modes, the
            MAP is not unique and should not be reported as a single point estimate. """
        
        theta_map, best_lp = self._compute_joint_MAP(refine, n_starts, space)
        map_estimate = theta_map.reshape(1, -1)
        if scaled:
            owner = self._variable_owner()
            map_estimate = np.array([[self._germ_to_model(theta_map, i)[0, k] for (i, k) in owner]])
        map_df = pd.DataFrame(map_estimate, columns=self.Q.variable_names())
        return (map_df, best_lp) if return_logprob else map_df
 
    def _compute_joint_MAP(self, refine=True, n_starts=5, space='germ'):
        """ Joint MAP in germ space (cached); returns (theta_map, log density) """
        
        if space not in ('germ', 'physical'):
            raise ValueError("space must be 'germ' or 'physical'")
        key = (refine, n_starts, space)
        if self._map_cache is not None and self._map_cache[0] == key:
            return self._map_cache[1], self._map_cache[2]
 
        chain = self.sampler.get_chain(flat=True)
        logp = self.sampler.get_log_prob(flat=True)
 
        def objective(theta):
            lp = self._joint_logprob(theta)
            if np.isfinite(lp) and space == 'physical':
                lp -= self._log_abs_det_jacobian(theta)
            return lp if np.isfinite(lp) else -np.inf
 
        # candidate starts: best distinct samples by stored log-posterior
        n_starts = max(int(n_starts), 1)
        n_cand = n_starts * (4 if space == 'physical' else 1)
        cand = []
        for i in np.argsort(logp)[::-1]:
            if not any(np.allclose(chain[i], c) for c in cand):
                cand.append(chain[i])
            if len(cand) >= n_cand:
                break
        cand_lp = np.array([objective(c) for c in cand])
        order = np.argsort(cand_lp)[::-1][:n_starts]
        best_theta, best_lp = cand[order[0]].copy(), float(cand_lp[order[0]])
 
        if refine:
            for j in order:
                res = minimize(lambda th: -objective(th), cand[j], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-10,
                                        'maxiter': 2000 * chain.shape[1]})
                if np.isfinite(res.fun) and -res.fun > best_lp:
                    best_theta, best_lp = np.asarray(res.x, dtype=float), float(-res.fun)
 
        self._map_cache = (key, best_theta, best_lp)
        return best_theta, best_lp
 
    def get_MAP_by_model(self, scaled=True, refine=True, n_starts=5, space='germ'):
        """ Joint MAP estimate expressed in the parameters of each model
 
            Each model receives its slice of the JOINT MAP vector (so shared
            parameters are consistent across models), mapped to its own physical
            parameters if scaled=True. This is not the mode of each model's
            marginal posterior.
 
            Parameters
            ----------
            scaled : bool, optional
                If True, return physical values, by default True
 
            refine, n_starts, space :
                See get_MAP.
 
            Returns
            -------
            map_dfs : list of pandas.DataFrame
                List of DataFrames containing the MAP estimate for each model """
        
        theta_map, _ = self._compute_joint_MAP(refine, n_starts, space)
        map_dfs = []
        for i in range(len(self.models)):
            if scaled:
                values = self._germ_to_model(theta_map, i)
            else:
                values = theta_map[self.indices[i]].reshape(1, -1)
            map_dfs.append(pd.DataFrame(values, columns=self.models[i].Q.variable_names()))
        return map_dfs
 
    # -------------------------------------------------------------- statistics
    def get_mean_and_var_by_model(self, scaled=True):
        """ Get mean and variance of posterior for each model
        
            Parameters
            ----------
            scaled : bool, optional
                If True, statistics of the physical parameters, otherwise of the
                germs, by default True
                
            Returns
            -------
            mean_dfs : list of pandas.DataFrame
                List of DataFrames containing the mean of the posterior for each model
                
            varance_dfs : list of pandas.DataFrame
                List of DataFrames containing the variance of the posterior for each model """
        
        mean_dfs = []
        varance_dfs = []
        for sample_df in self.get_posterior_samples_by_model(scaled):
            mean_dfs.append(sample_df.mean().to_frame().T)
            varance_dfs.append(sample_df.var(ddof=0).to_frame().T)
        return mean_dfs, varance_dfs
    
    def get_data_of_model(self, model_index, scaled=True):
        """ Get MAP, mean, variance, and samples for a specific model
 
            Parameters
            ----------
            model_index : int
                Index of the model model
 
            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the MAP estimate for the model
                
            mean_df : pandas.DataFrame
                DataFrame containing the mean of the posterior for the model
                
            var_df : pandas.DataFrame
                DataFrame containing the variance of the posterior for the model 
            
            sample_df : pandas.DataFrame
                DataFrame containing the samples from the posterior for the model"""
        
        maps = self.get_MAP_by_model(scaled)
        means, vars = self.get_mean_and_var_by_model(scaled)
        samples = self.get_posterior_samples_by_model(scaled)
        map_df = maps[model_index]
        mean_df = means[model_index]
        var_df = vars[model_index]
        sample_df = samples[model_index]
        return map_df, mean_df, var_df, sample_df
    
    def get_posterior_samples_by_model(self, scaled=True):
        """ Get posterior samples for each model
 
            Parameters
            ----------
            scaled : bool, optional
                If True, return scaled samples, by default True
 
            Returns
            -------
            sample_dfs : list of pandas.DataFrame
                List of DataFrames containing the samples from the posterior for each model"""
        
        post_samples = self.sampler.get_chain(flat=True)
        sample_dfs = []
        for i in range(len(self.models)):
            if scaled:
                model_samples = self._germ_to_model(post_samples, i)
            else:
                model_samples = post_samples[:, self.indices[i]]
            sample_dfs.append(pd.DataFrame(model_samples, columns=self.models[i].Q.variable_names()))
        return sample_dfs
    
    def get_posterior_samples(self):
        """ Get posterior samples for joint parameters (germs)
 
            Returns
            -------
            post_samples_df : pandas.DataFrame
                DataFrame containing the samples from the posterior parameter distribution """
        
        post_samples = self.sampler.get_chain(flat=True)
        post_samples_df = pd.DataFrame(post_samples, columns=self.Q.variable_names())
        return post_samples_df

    def plot_multimodel_MCMC(self, extra_point=None, extra_point_name='Extra point', map_point=True, model_names=None, formatted_param_names=None, map_kwargs=None):
        figs = multimodel_MCMC_plot(self, extra_point=extra_point, extra_point_name=extra_point_name, map_point=map_point, model_names=model_names, formatted_param_names=formatted_param_names, map_kwargs=map_kwargs)
        return figs