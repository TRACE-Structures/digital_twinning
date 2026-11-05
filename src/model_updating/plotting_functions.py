import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def MCMC_plot(DigitalTwin, X_train, extra_point=None, extra_point_name='Extra point', map=True,
              custom_xlim=None, custom_ylim=None, max_plot_samples=4000, map_point=None,
              map_kwargs=None, corner=False, random_seed=0):
    ''' Create overlaid pairplot of prior and posterior samples from MCMC with highlighted points.

        Only a random subsample of the samples is drawn (max_plot_samples per
        dataset), which is what makes the plot fast; the posterior mean is computed
        from the full chain.

        Parameters
        ----------
        DigitalTwin : DigitalTwin
            Updated DigitalTwin (with sampler).
        X_train : array_like or DataFrame
            Samples representing the prior (physical parameters).
        extra_point : DataFrame or array_like, optional
            Additional point to plot.
        extra_point_name : str, optional
            Label for extra point in legend.
        map : bool, optional
            Whether to plot the MAP point, by default True.
        custom_xlim, custom_ylim : list or dict, optional
            Axis limits per parameter, as a list of (low, high) in parameter order or
            a dict {name: (low, high)}. Override Q.get_bounds() for the x axes
            (custom_xlim) and y axes (custom_ylim) of that parameter.
        max_plot_samples : int or None, optional
            Maximum number of prior and of posterior points drawn, by default 4000.
            None draws all samples (slow for long chains).
        map_point : DataFrame, optional
            Precomputed MAP point (physical values); skips DigitalTwin.get_MAP().
        map_kwargs : dict, optional
            Passed to DigitalTwin.get_MAP (e.g. refine, n_starts, space).
        corner : bool, optional
            Draw only the lower triangle (about half the panels), by default False.
        random_seed : int, optional
            Seed for the subsampling, by default 0.

        Returns
        -------
        seaborn.PairGrid
            Pairplot with priors, posteriors, and highlighted points. '''
    
    param_names = DigitalTwin.model.Q.variable_names()
    n_params = len(param_names)
    rng = np.random.default_rng(random_seed)

    def subsample(df):
        if max_plot_samples is None or len(df) <= max_plot_samples:
            return df
        return df.iloc[rng.choice(len(df), max_plot_samples, replace=False)]

    # full chain -> statistics; subsample -> plotting
    posterior_full = pd.DataFrame(
        DigitalTwin.model.Q.germ2variable(DigitalTwin.sampler.get_chain(flat=True)), columns=param_names)
    mean_value = posterior_full.mean().to_numpy()
    posterior_samples = subsample(posterior_full).copy()
    prior_samples = subsample(pd.DataFrame(np.asarray(X_train), columns=param_names)).copy()

    if extra_point is not None and not isinstance(extra_point, pd.DataFrame):
        extra_point = pd.DataFrame(np.atleast_2d(extra_point), columns=param_names)
    extra_p = extra_point.iloc[0].to_numpy() if extra_point is not None else None

    if map_point is None and map:
        map_point = DigitalTwin.get_MAP(**(map_kwargs or {}))      # cached in DigitalTwin
    map_p = map_point.iloc[0].to_numpy() if (map and map_point is not None) else None

    prior_color = '#b8b8b8'
    posterior_color = '#1a80bb'
    MAP_color = '#ea801c'

    prior_samples['Dataset'] = 'Prior'
    posterior_samples['Dataset'] = 'Posterior'
    combined = pd.concat([prior_samples, posterior_samples], ignore_index=True)

    fig = sns.pairplot(
        combined,
        hue='Dataset',
        palette={'Prior': prior_color, 'Posterior': posterior_color},
        diag_kind='kde',
        corner=corner,
        plot_kws={'alpha': 0.5, 's': 10, 'linewidth': 0, 'rasterized': True},
        diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
    )

    def limits(custom, j, default):
        if custom is None:
            return default
        value = custom.get(param_names[j]) if isinstance(custom, dict) else custom[j]
        return default if value is None else value

    param_bounds = DigitalTwin.model.Q.get_bounds()
    for i in range(n_params):
        for j in range(n_params):
            ax = fig.axes[i, j]
            if ax is None:                       # upper triangle when corner=True
                continue
            ax.set_xlim(limits(custom_xlim, j, param_bounds[j]))
            if i != j:
                ax.set_ylim(limits(custom_ylim, i, param_bounds[i]))
                ax.scatter(mean_value[j], mean_value[i], color='red', s=10, zorder=4, marker='o')
                if map_p is not None:
                    ax.scatter(map_p[j], map_p[i], color=MAP_color, s=10, zorder=5, marker='o')
                if extra_p is not None:
                    ax.scatter(extra_p[j], extra_p[i], color='purple', s=10, zorder=5, marker='o')
            else:
                ax.axvline(mean_value[i], color='red', zorder=1)
                if map_p is not None:
                    ax.axvline(map_p[i], color=MAP_color, zorder=1)
                if extra_p is not None:
                    ax.axvline(extra_p[i], color='purple', zorder=1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=prior_color, markersize=8, label='Prior'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=posterior_color, markersize=8, label='Posterior'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Mean Value'),
    ]
    if map_p is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=MAP_color, markersize=8, label='MAP Point'))
    if extra_p is not None:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='purple', markersize=8, label=extra_point_name))

    if fig._legend is not None:
        fig._legend.remove()
    fig.fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.95))

    fig.fig.suptitle('Prior and Posterior Distributions', y=1.02)
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
