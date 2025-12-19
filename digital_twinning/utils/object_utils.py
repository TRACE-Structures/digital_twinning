import numpy as np
import uncertain_variables as uv
import matplotlib.pyplot as plt
import emcee
import time
import seaborn as sns
import pandas as pd
import shap

def plot_sobol_sensitivity(sobol_index, y, param_name=None):
    ''' Plot Sobol sensitivity indices as stacked area plot or pie chart.

        Parameters
        ----------
        sobol_index : pandas.DataFrame
            Sobol sensitivity indices (rows: timesteps/modes, columns: parameters).
        y : pandas.DataFrame
            Quantity of interest values for variance calculation.
        param_name : str or None, optional
            Specific parameter/mode to plot as pie chart. If None, plots stacked area.

        Returns
        -------
        matplotlib.figure.Figure
            Sensitivity plot figure. '''
    
    fig, ax = plt.subplots(figsize=(8, 6))
 
    y_var = np.broadcast_to(y.var(axis=0).to_numpy().reshape(-1, 1), sobol_index.shape)
    y_var_df = pd.DataFrame(y_var, columns=sobol_index.columns, index=sobol_index.index)
    partial_variance = np.multiply(sobol_index, y_var)

    color_map = plt.cm.viridis(np.linspace(0, 1, partial_variance.shape[1]+1))
    colors = {partial_variance.columns[i]: color_map[i] for i in range(len(partial_variance.columns))}

    if param_name is not None:
        # Handle specific parameter case
        threshold = 0.1
        loc_par_var = partial_variance.loc[param_name]
        df = pd.DataFrame(loc_par_var)
        df[df < 0] = 0
        df['percentage'] = df[param_name] / df[param_name].sum()
        under_threshold = df.loc[df['percentage'] < threshold].sum()
        remaining = [y_var_df.loc[param_name][0], 1] - df.sum()
        others = under_threshold + remaining
        colors['others'] = color_map[-1]
        df = df[df['percentage'] >= threshold]
        df.loc['others'] = others
        pie_colors = [colors[x] for x in df.index]

        ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
        ax.pie(df[param_name], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
    else:
        # General case: Stacked area plot
        timesteps = sobol_index.shape[0]
        time = np.arange(0, timesteps, 1)

        # Total variance plot
        ax.plot(time, y_var[:, 0], color='black', label='Total variance', linewidth=0.8)

        # Stacked area plot
        stackplot_colors = [colors[col] for col in y_var_df.columns]
        ax.stackplot(time, partial_variance.to_numpy().transpose(), alpha=0.5, labels=sobol_index.columns, colors=stackplot_colors)

        # Format the plot
        ax.set_xlim((0, timesteps - 1))
        ax.set_xlabel('Timestep', fontsize=14)
        ax.set_ylabel('Variance', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

    return fig


def plot_shap_values(model, q=None, param_name=None):
    ''' Generate SHAP plots (waterfall or summary) for model explanations.

        Parameters
        ----------
        model : object
            Model object with .model.explainer, .Q.param_names(), .QoI_names, and optionally .X_train.
        q : pandas.DataFrame or None, optional
            Input data for SHAP computation. If None, uses model.X_train.
        param_name : str or None, optional
            Specific QoI to plot. If None, plots bar summary for first QoI.

        Returns
        -------
        matplotlib.figure.Figure
            SHAP visualization figure. '''
    
    if q is None:
        q = model.X_train
    else:
        q = pd.DataFrame(q, columns=model.Q.param_names())
    
    xi = q
    explainer = model.model.explainer
    shap_values = explainer(xi)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if param_name is not None:
        param_index = model.QoI_names.index(param_name)
        QoI_shap = shap_values[:, :, param_index]

        if QoI_shap.shape[0] == 1:
            ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
            shap.plots.waterfall(QoI_shap[0, :], show=False)
        else:
            ax.set_title(f"Used Quantity of Interest parameter: {param_name}", fontsize=16)
            shap.summary_plot(QoI_shap, xi, show=False)
    else:
        ax.set_title(f"Average Influence of {shap_values.shape[0]} sample of Parameters on  Quantity of Interest", fontsize=16)
        shap.summary_plot(shap_values[:, :, 0], xi, plot_type="bar", show=False)

    return fig

def plot_MCMC_results(Q, sampler, num_param, nwalkers, scale):
    ''' Create pairplot of MCMC posterior samples with parameter bounds.

        Parameters
        ----------
        Q : VariableSet
            Parameter set with bounds and names.
        sampler : emcee.EnsembleSampler
            Fitted MCMC sampler.
        num_param : int
            Number of parameters.
        nwalkers : int
            Number of MCMC walkers (unused in current implementation).
        scale : float or array_like
            Scaling factor for printing means.

        Returns
        -------
        None
            Displays pairplot and prints scaled means. '''
    
    param_bounds = Q.get_bounds()

    post_samples = sampler.get_chain(flat = True)
    h = sns.PairGrid(pd.DataFrame(post_samples[:, :], columns = Q.param_names()))
    h.map_diag(plt.hist, color="#2f779dff", bins = 15, linewidth = 0.3)
    h.map_upper(sns.scatterplot, color = "#2f779dff", s = 10, linewidth=0.3)
    h.map_lower(sns.scatterplot, color = "#2f779dff", s = 10, linewidth=0.3)
    for i in range(num_param):
        for j in range(num_param):
            h.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                h.axes[i, j].set_ylim(param_bounds[i])

    means = np.zeros((num_param, 1))
    for i in range(num_param):
        means[i] = np.mean(post_samples[:, i])

    print('Means from MCMC:')
    print(means*scale)
    plt.show()

def MCMC_run_and_plot(nwalkers, niter, nburn, scale, manager):
    ''' Run MCMC inference and generate posterior plots.

        Parameters
        ----------
        nwalkers : int
            Number of MCMC walkers.
        niter : int
            Number of MCMC iterations (post-burn).
        nburn : int
            Number of burn-in iterations.
        scale : float or array_like
            Scaling factor for results.
        manager : object
            Object with .Q (VariableSet) and .pdf_func (log-probability function).

        Returns
        -------
        None
            Runs MCMC and displays posterior plots. '''
    
    Q = manager.Q
    num_param = Q.num_params()
    pdf_func = manager.pdf_func

    p0 = Q.sample(nwalkers)
    
    print('MCMC creating')
    sampler = emcee.EnsembleSampler(nwalkers, num_param, pdf_func)
    start_time = time.time()

    print('Burning period')
    state = sampler.run_mcmc(p0, nburn, progress = True)
    sampler.reset()

    print('MCMC running')
    sampler.run_mcmc(state, niter, progress = True)

    print("--- %s seconds ---" % (time.time() - start_time))

    plot_MCMC_results(Q, sampler, num_param, nwalkers, scale)


def generate_stdrn_simparamset(sigma):
    ''' Create VariableSet of zero-mean normal variables with given standard deviations.

        Parameters
        ----------
        sigma : array_like
            Standard deviations for each variable.

        Returns
        -------
        VariableSet
            Set of Normal(0, sigma[i]) variables named 'pn_1', 'pn_2', etc. '''
    
    Q = uv.VariableSet()
    for i in range(len(sigma)):
        s = uv.Variable('pn_' + str(i+1), uv.NormalDistribution(0, sigma[i]))
        Q.add(s)
    return Q

def set_prior_paramset(name, a, b):
    ''' Create uniform prior VariableSet from parameter names and bounds.

        Parameters
        ----------
        name : list of str
            Parameter names.
        a : array_like
            Lower bounds for each parameter.
        b : array_like
            Upper bounds for each parameter.

        Returns
        -------
        VariableSet
            VariableSet with Uniform(a[i], b[i]) distributions.

        Raises
        ------
        AssertionError
            If lengths of name, a, and b don't match. '''
    
    Q = uv.VariableSet()
    assert(len(name)==len(a)==len(b))
    for i in range(len(name)):
        Q.add(uv.Variable(name[i], uv.UniformDistribution(a[i], b[i])))
    return Q
