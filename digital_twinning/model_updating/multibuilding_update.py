""" JointManager for multi-building model updating using MCMC"""

import uncertain_variables as uv
import pandas as pd
import emcee
import time
from digital_twinning.utils import utils
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class JointManager:
    """ Class to manage joint model updating for multiple buildings through one or more shared parameters
    
        Attributes
        ----------
        models : list of SurrogateModel
            List of surrogate models for each building
            
        joint_parameters : dict
            Dictionary defining joint parameters across buildings
            
        Q : VariableSet
            Joint VariableSet for all buildings
            
        indices : list of list of int
            Indices mapping building parameters to joint parameters
            
        scale : list of list of float
            Scaling factors for each building's parameters
            
        shift : list of list of float
            Shift values for each building's parameters
            
        Methods
        -------
        __init__(models, joint_parameters)
            Initialize the JointManager with given models and joint parameters
            
        generate_joint_stdrn_simparamset(sigmas)
            Generate joint standard random simulation variable set
            
        choose_distribution(buildings, joint_names, mode='auto')
            Choose distribution for joint parameters
            
        get_joint_paramset_and_indices(buildings, joint_parameters)
            Create joint VariableSet and mapping indices
            
        get_logprob(q)
            Compute log-probability for MCMC
            
        logprior(q)
            Compute log-prior probability
            
        loglikelihood(q, y_m)
            Compute log-likelihood of observed data
            
        update(y_list, sigmas, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False)
            Perform MCMC update with observed data
            
        get_mean_and_var_by_model(scaled=True)
            Get mean and variance of posterior for each building
            
        get_MAP_by_model(scaled=True)
            Get maximum a posteriori estimate for each building
            
        get_MAP()
            Get maximum a posteriori estimate for joint parameters
            
        get_data_of_model(model_index)
            Get MAP, mean, variance, and samples for a specific building
            
        get_posterior_samples_by_model(scaled=True)
            Get posterior samples for each building
            
        get_posterior_samples()
            Get posterior samples for joint parameters """
    
    def __init__(self, models, joint_parameters):
        """ Initialize the JointManager with given models and joint parameters

            Parameters
            ----------
            models : list of SurrogateModel
                List of surrogate models for each building

            joint_parameters : dict
                Dictionary defining joint parameters across buildings """
        
        self.models = models
        self.joint_parameters = joint_parameters
        self.Q, self.indices, self.scale, self.shift = self.get_joint_paramset_and_indices(models, joint_parameters)

    def generate_joint_stdrn_simparamset(self, sigmas):
        """ Generate joint standard random simulation variable set

            Parameters
            ----------
            sigmas : list of pandas.DataFrame
                List of standard deviation DataFrames for each building
            
            Returns
            -------
            E_joint : VariableSet
                Joint standard random simulation variable set """
        
        joint_sigma = np.zeros((1, 0))
        for i in range(len(sigmas)):
            building_sigma = sigmas[i].values
            joint_sigma = np.concatenate((joint_sigma, building_sigma), axis=1)
        joint_sigma = joint_sigma.reshape(-1, 1)
        E_joint = utils.generate_stdrn_simparamset(joint_sigma)
        return E_joint
    
    def choose_distribution(self, buildings, joint_names, mode='auto'):
        """ Choose distribution for joint parameters

            Parameters
            ----------
            buildings : list of SurrogateModel
                List of surrogate models of the buildings

            joint_names : list of str
                List of joint parameter names for each building

            mode : str, optional
                Mode for choosing distribution ('first_dist', 'uniform', 'normal', 'auto'), by default 'auto'

            Returns
            -------
            distribution : Distribution
                Chosen distribution for the joint parameter"""
        
        if mode == 'first_dist':
            for i in range(len(joint_names)):
                if joint_names[i] != '-':
                    break
            distribution = buildings[i].Q.params[joint_names[i]]
            return distribution
        
        elif mode == 'uniform':
            distribution = uv.UniformDistribution(0, 1)
            return distribution
        
        elif mode == 'normal':
            distribution = uv.NormalDistribution(0, 1)
            return distribution
        
        elif mode == 'auto':
            for i in range(len(joint_names)):
                if joint_names[i] != '-':
                    break
            if buildings[i].Q.params[joint_names[i]].get_dist_type() == 'unif':
                distribution = uv.UniformDistribution(0, 1)
                return distribution
            elif buildings[i].Q.params[joint_names[i]].get_dist_type() == 'norm':
                distribution = uv.NormalDistribution(0, 1)
                return distribution                
    
    def get_joint_paramset_and_indices(self, buildings, joint_parameters):
        """ Create joint VariableSet and mapping indices

            Parameters
            ----------
            buildings : list of SurrogateModel
                List of surrogate models of the buildings

            joint_parameters : dict
                Dictionary defining joint parameters across buildings

            Returns
            -------
            Q_joint : VariableSet
                Joint VariableSet for all buildings

            indices : list of list of int
                Indices mapping building parameters to joint parameters

            scale : list of list of float
                Scaling factors for each building's parameters

            shift : list of list of float
                Shift values for each building's parameters """
        
        building_parameters = []
        for b in range(len(buildings)):
            building_parameters.append(buildings[b].Q.param_names())
        joint_names = list(joint_parameters.keys())
        for i in range(len(joint_parameters)):
            parameter_list = joint_parameters[joint_names[i]]
            assert len(parameter_list) == len(building_parameters), "The jointparameters should contain every building. Put a '-' when you don't want to use the building for parameter join"
            for j in range(len(building_parameters)):
                if parameter_list[j] != '-':
                    assert parameter_list[j] in building_parameters[j], "Parameter '{}' is not in parameter list {}".format(parameter_list[j], building_parameters[j])

        Q_joint = uv.VariableSet()

        indices = []
        scale = []
        shift = []
        for i in range(len(building_parameters)):
            indices.append([None]*len(building_parameters[i]))
            scale.append([1]*len(building_parameters[i]))
            shift.append([0]*len(building_parameters[i]))
        
        for i in range(len(joint_parameters)):
            distribution = self.choose_distribution(buildings, joint_parameters[joint_names[i]], mode='auto')
            JointParameter = uv.Variable(joint_names[i], distribution)
            Q_joint.add(JointParameter)
            scale0 = distribution.get_bounds()[1] - distribution.get_bounds()[0]
            shift0 = distribution.get_bounds()[0]
            for j in range(len(building_parameters)):
                param_name = joint_parameters[joint_names[i]][j]
                if param_name != '-':
                    idx = buildings[j].Q.param_names().index(param_name)
                    indices[j][idx] = i
                    
                    lower_bound = buildings[j].Q.get_bounds()[idx][0]
                    higher_bound = buildings[j].Q.get_bounds()[idx][1]
                    scale[j][idx] = (higher_bound - lower_bound) / scale0
                    shift[j][idx] = lower_bound - scale0*shift0
                    building_parameters[j].remove(param_name)

        for i in range(len(building_parameters)):
            for j in range(len(building_parameters[i])):
                Parameter = uv.Variable(building_parameters[i][j], buildings[i].Q.params[building_parameters[i][j]])    
                Q_joint.add(Parameter)

        for i in range(len(building_parameters)):
            for j in range(len(building_parameters[i])):
                parameter = building_parameters[i][j]
                idx = Q_joint.param_names().index(parameter)
                original_index = buildings[i].Q.param_names().index(parameter)
                indices[i][original_index]  = idx   
        
        return Q_joint, indices, scale, shift
    
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
        
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        predictions = np.zeros((1, 0))
        for i in range(len(self.models)):
            q_model = q[self.indices[i]] * self.scale[i] + self.shift[i]
            q_df = pd.DataFrame(q_model.reshape(1,-1), columns=self.models[i].Q.param_names())
            y_model = self.models[i].predict(q_df)
            predictions = np.concatenate((predictions, y_model), axis=1)
        d = y_m - predictions
        d = d.transpose()
        logp = self.E.logpdf(d)
        return logp
    
    def update(self, y_list, sigmas, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False):
        """ Perform MCMC update with observed data

            Parameters
            ----------
            y_list : list of pandas.DataFrame
                List of observed data DataFrames for each building

            sigmas : list of pandas.DataFrame
                List of standard deviation DataFrames for each building

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
        y_m = np.zeros((1, 0))
        for i in range(len(y_list)):
            y_building = y_list[i].to_numpy()
            y_m = np.concatenate((y_m, y_building), axis=1)
        self.y_m = y_m
        num_param = self.Q.num_params()
        self.E = self.generate_joint_stdrn_simparamset(sigmas)


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
        state = sampler.run_mcmc(p0, nburn, progress = True)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = True)
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.sampler = sampler

    def get_mean_and_var_by_model(self, scaled=True):
        """ Get mean and variance of posterior for each building
        
            Parameters
            ----------
            scaled : bool, optional
                If True, return scaled mean and variance, by default True
                
            Returns
            -------
            mean_dfs : list of pandas.DataFrame
                List of DataFrames containing the mean of the posterior for each building
                
            varance_dfs : list of pandas.DataFrame
                List of DataFrames containing the variance of the posterior for each building """
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        mean_dfs = []
        varance_dfs = []
        for i in range(len(self.models)):
            model_samples = post_samples[:, self.indices[i]]
            means = np.mean(model_samples, axis=0)
            variances = np.var(model_samples, axis=0)

            if scaled:
                means = means * self.scale[i] + self.shift[i]
                variances = variances * self.scale[i]

            means_df = pd.DataFrame(means.reshape(1,-1), columns=self.models[i].Q.param_names())
            variances_df = pd.DataFrame(variances.reshape(1,-1), columns=self.models[i].Q.param_names())
            mean_dfs.append(means_df)
            varance_dfs.append(variances_df)
        return mean_dfs, varance_dfs

    def get_MAP_by_model(self, scaled=True):
        """ Get maximum a posteriori estimate for each building
            
            Parameters
            ----------
            scaled : bool, optional
                If True, return scaled MAP estimate, by default True
                
            Returns
            -------
            map_dfs : list of pandas.DataFrame
                List of DataFrames containing the MAP estimate for each building """
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)

        map_dfs = []
        for i in range(len(self.models)):
            map_estimate = np.zeros((1, len(self.indices[i])))
            model_samples = post_samples[:, self.indices[i]]
            for p in range(len(self.indices[i])):
                p_estimate = utils.estimate_maxima(model_samples[:,p])
                map_estimate[0,p] = p_estimate
            if scaled:
                map_estimate = map_estimate * self.scale[i] + self.shift[i]
            map_df = pd.DataFrame(map_estimate, columns=self.models[i].Q.param_names())
            map_dfs.append(map_df)
        return map_dfs        
        
    def get_MAP(self): # maximum a posterior estimate
        """ Get maximum a posteriori estimate of the joint variable set

            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the maximum a posteriori estimate of the joint variable set """
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        
        map_estimate = np.zeros((1, post_samples.shape[1]))
        for p in range(post_samples.shape[1]):
            p_estimate = utils.estimate_maxima(post_samples[:,p])
            map_estimate[0,p] = p_estimate

        map_df = pd.DataFrame(map_estimate, columns=self.Q.param_names())
        return map_df
    
    def get_data_of_model(self, model_index):
        """ Get MAP, mean, variance, and samples for a specific building

            Parameters
            ----------
            model_index : int
                Index of the building model

            Returns
            -------
            map_df : pandas.DataFrame
                DataFrame containing the MAP estimate for the building
                
            mean_df : pandas.DataFrame
                DataFrame containing the mean of the posterior for the building
                
            var_df : pandas.DataFrame
                DataFrame containing the variance of the posterior for the building 
            
            sample_df : pandas.DataFrame
                DataFrame containing the samples from the posterior for the building"""
        
        maps = self.get_MAP_by_model()
        means, vars = self.get_mean_and_var_by_model()
        samples = self.get_posterior_samples_by_model()
        map_df = maps[model_index]
        mean_df = means[model_index]
        var_df = vars[model_index]
        sample_df = samples[model_index]
        return map_df, mean_df, var_df, sample_df
    
    def get_posterior_samples_by_model(self, scaled=True):
        """ Get posterior samples for each building

            Parameters
            ----------
            scaled : bool, optional
                If True, return scaled samples, by default True

            Returns
            -------
            sample_dfs : list of pandas.DataFrame
                List of DataFrames containing the samples from the posterior for each building"""
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        sample_dfs = []
        for i in range(len(self.models)):
            model_samples = post_samples[:, self.indices[i]]
            if scaled:
                model_samples = model_samples * self.scale[i] + self.shift[i]
            sample_dfs.append(pd.DataFrame(model_samples, columns=self.models[i].Q.param_names()))
        return sample_dfs
    
    def get_posterior_samples(self):
        """ Get posterior samples for joint parameters

            Returns
            -------
            post_samples_df : pandas.DataFrame
                DataFrame containing the samples from the posterior parameter distribution """
        
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        post_samples_df = pd.DataFrame(post_samples, columns=self.Q.param_names())
        return post_samples_df