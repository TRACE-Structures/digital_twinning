from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def MCMC_plot(DigitalTwin, X_train, extra_point=None, extra_point_name='Extra point', map=True, custom_xlim=None, custom_ylim=None):
    ''' Create overlaid pairplot of prior and posterior samples from MCMC with highlighted points.

        Parameters
        ----------
        model_obj : object
            Model object with parameter info.
        X_train : array_like or DataFrame
            Training samples representing prior.
        DigitalTwin : object or None
            Object containing sampler for posterior samples.
        nwalkers : int or None, optional
            Number of walkers; unused here.
        extra_point : DataFrame or array_like, optional
            Additional point to plot.
        extra_point_name : str, optional
            Label for extra point in legend.
        map_point : DataFrame or None, optional
            MAP point to highlight.
        custom_xlim : tuple or None, optional
            Custom x-axis limits.
        custom_ylim : tuple or None, optional
            Custom y-axis limits.

        Returns
        -------
        matplotlib.figure.Figure
            Pairplot figure with priors, posteriors, and highlighted points. '''
    
    prior_samples = pd.DataFrame(X_train, columns=DigitalTwin.model.Q.variable_names())

    posterior_samples = (
        pd.DataFrame(DigitalTwin.model.Q.germ2variable(DigitalTwin.sampler.get_chain(flat=True)), columns=DigitalTwin.model.Q.variable_names())
        if DigitalTwin else None
    )
    if extra_point is not None and not isinstance(extra_point, pd.DataFrame):
        extra_point = pd.DataFrame(extra_point, columns=DigitalTwin.model.Q.variable_names())


    if map:
        map_point = DigitalTwin.get_MAP()
    else:
        map_point = None
        
    mean_value = posterior_samples.mean()
    param_names = DigitalTwin.model.Q.variable_names()
    n_params = len(param_names)
    prior_color = '#b8b8b8'
    posterior_color = '#1a80bb'
    MAP_color = '#ea801c'
    
    prior_samples['Dataset'] = 'Prior'
    if posterior_samples is not None:
        posterior_samples['Dataset'] = 'Posterior'
        combined = pd.concat([prior_samples, posterior_samples])
    else:
        combined = prior_samples

    fig = sns.pairplot(
        combined,
        hue='Dataset',
        palette={'Prior': prior_color, 'Posterior': posterior_color},
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 10},
        diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
    )

    for i in range(n_params):
        for j in range(n_params):
            ax = fig.axes[i, j]
            if i != j:
                ax.scatter(mean_value[j], mean_value[i], 
                            color='red', s=10, zorder=1, marker='o')
                if map_point is not None:
                    ax.scatter(map_point.iloc[0, j], map_point.iloc[0, i], 
                                color=MAP_color, s=10, zorder=3, marker='o')
                if extra_point is not None:
                    ax.scatter(extra_point.iloc[0, j], extra_point.iloc[0, i], 
                                color='purple', s=10, zorder=3, marker='o')
            else:
                ax.vlines(mean_value[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                            color='red', zorder=1)
                if map_point is not None:    
                    ax.vlines(map_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color=MAP_color, zorder=1)
                if extra_point is not None:
                    ax.vlines(extra_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color='purple', zorder=1)

    param_bounds = DigitalTwin.model.Q.get_bounds()
    for i in range(n_params):
        for j in range(n_params):
            fig.axes[i, j].set_xlim(param_bounds[j])
            if i != j:
                fig.axes[i, j].set_ylim(param_bounds[i])
    legend_elements = []
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=prior_color, markersize=8, label='Prior'))
    if posterior_samples is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=posterior_color, markersize=8, label='Posterior'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', markersize=8, label='Mean Value'))
    
    if map_point is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=MAP_color, markersize=8, label='MAP Point'))
    if extra_point is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='purple', markersize=8, label=extra_point_name))

    if fig._legend is not None:
        fig._legend.remove()
    fig.fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95))

    plt.suptitle('Prior and Posterior Distributions', y=1.02)
    return fig


def multimodel_MCMC_plot(jointManager, extra_point=None, extra_point_name='Extra point', map_point=True, model_names=None, formatted_param_names=None, map_kwargs=None):
    ''' Plot prior and posterior distributions for multiple models with optional highlights.
 
        Parameters
        ----------
        jointManager : JointManager
            Manager containing the models, the sampler and the index mapping.
        extra_point : DataFrame or None, optional
            Additional point to plot.
        extra_point_name : str, optional
            Label for extra point in legend.
        map_point : bool, optional
            Whether to plot the MAP point (joint MAP, sliced for each model).
        model_names : list of str or None, optional
            Optional model names for titles.
        formatted_param_names : list of list of str or None, optional
            Optional formatted parameter names for axes labels.
        map_kwargs : dict or None, optional
            Passed to jointManager.get_MAP_by_model (refine, n_starts, space).
 
        Returns
        -------
        list of matplotlib.figure.Figure
            List of pairplot figures for each model. '''
    
    posterior_by_model = jointManager.get_posterior_samples_by_model(scaled=True)
    if map_point:
        # joint MAP computed once (cached in the JointManager), then sliced per model
        map_by_model = jointManager.get_MAP_by_model(scaled=True, **(map_kwargs or {}))
 
    prior_color = '#b8b8b8'
    posterior_color = '#1a80bb'
    MAP_color = '#ea801c'
 
    figures = []
    for m in range(len(jointManager.models)):
        Q_m = jointManager.models[m].Q
        param_names = Q_m.variable_names()
        n_params = len(param_names)
 
        prior_samples = pd.DataFrame(Q_m.sample(1000), columns=param_names)
        posterior_samples = posterior_by_model[m].copy()
        mean_value = posterior_samples.mean().to_numpy()
        if map_point:
            map_p = map_by_model[m].iloc[0].to_numpy()
 
        prior_samples['Dataset'] = 'Prior'
        posterior_samples['Dataset'] = 'Posterior'
        combined = pd.concat([prior_samples, posterior_samples])
 
        fig = sns.pairplot(
            combined,
            hue='Dataset',
            palette={'Prior': prior_color, 'Posterior': posterior_color},
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 10},
            diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
        )
 
        for i in range(n_params):
            for j in range(n_params):
                ax = fig.axes[i, j]
                if formatted_param_names is not None and j == 0:
                    ax.set_ylabel(formatted_param_names[m][i])
 
                if formatted_param_names is not None and i == n_params-1:
                    ax.set_xlabel(formatted_param_names[m][j])
                    ax.tick_params(axis='x', labelrotation=90)
 
                if i != j:
                    ax.scatter(mean_value[j], mean_value[i], 
                            color='red', s=10, zorder=4, marker='o')
                    if map_point:
                        ax.scatter(map_p[j], map_p[i], 
                                    color=MAP_color, s=10, zorder=5, marker='o')
                    if extra_point is not None:
                        ax.scatter(extra_point.iloc[0, j], extra_point.iloc[0, i], 
                                    color='purple', s=10, zorder=5, marker='o')
                else:
                    ax.vlines(mean_value[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                color='red', zorder=1)
                    if map_point:
                        ax.vlines(map_p[i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                    color=MAP_color, zorder=1)
                    if extra_point is not None:
                        ax.vlines(extra_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1], 
                                    color='purple', zorder=1)
 
        param_bounds = Q_m.get_bounds()
        for i in range(n_params):
            for j in range(n_params):
                fig.axes[i, j].set_xlim(param_bounds[j])
                if i != j:
                    fig.axes[i, j].set_ylim(param_bounds[i])
 
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=prior_color, markersize=8, label='Prior'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=posterior_color, markersize=8, label='Posterior'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Mean Value'),
        ]
        if map_point:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=MAP_color, markersize=8, label='MAP Point'))
        if extra_point is not None:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='purple', markersize=8, label=extra_point_name))
        if fig._legend is not None:
            fig._legend.remove()
        fig.fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95))
 
        if model_names is None:
            fig.fig.suptitle('Prior and Posterior Distributions of model {}.'.format(m+1), y=1.02)
        else:
            fig.fig.suptitle('Prior and Posterior Distributions of the {} model'.format(model_names[m]), y=1.02)
        figures.append(fig)
    return figures
