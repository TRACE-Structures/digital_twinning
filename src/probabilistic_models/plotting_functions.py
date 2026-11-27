import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

def posterior_plot(model_obj, mdn_samples, extra_point=None, extra_point_name='Extra point', map_point=None,
                   X_train=None):
    ''' Create overlaid pairplot of prior and posterior samples with highlighted points.

        Parameters
        ----------
        model_obj : ProbabilisticModel
            Model object with parameter info (model_obj.Q).
        mdn_samples : array_like or DataFrame
            Posterior samples, columns ordered as model_obj.Q.variable_names().
        extra_point : DataFrame or array_like, optional
            Additional point to plot.
        extra_point_name : str, optional
            Label for extra point in legend.
        map_point : DataFrame or None, optional
            MAP point to highlight.
        X_train : array_like or DataFrame, optional
            Prior samples. Defaults to the parameter values the model was trained
            on, model_obj.training_parameters().

        Returns
        -------
        seaborn.PairGrid
            Pairplot figure with priors, posteriors, and highlighted points. '''

    param_names = list(model_obj.Q.variable_names())
    n_params = len(param_names)

    if X_train is None:
        prior_samples = model_obj.training_parameters()
    else:
        prior_samples = pd.DataFrame(X_train, columns=param_names).copy()

    posterior_samples = pd.DataFrame(mdn_samples, columns=param_names).copy()
    if extra_point is not None and not isinstance(extra_point, pd.DataFrame):
        extra_point = pd.DataFrame(extra_point, columns=param_names)

    mean_value = posterior_samples.mean()
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
        plot_kws={'alpha': 0.5, 's': 10},
        diag_kws={'alpha': 0.2, 'common_norm': False, 'zorder': 3}
    )

    for i in range(n_params):
        for j in range(n_params):
            ax = fig.axes[i, j]
            if i != j:
                ax.scatter(mean_value.iloc[j], mean_value.iloc[i],
                            color='red', s=10, zorder=1, marker='o')
                if map_point is not None:
                    ax.scatter(map_point.iloc[0, j], map_point.iloc[0, i],
                                color=MAP_color, s=10, zorder=3, marker='o')
                if extra_point is not None:
                    ax.scatter(extra_point.iloc[0, j], extra_point.iloc[0, i],
                                color='purple', s=10, zorder=3, marker='o')
            else:
                ax.vlines(mean_value.iloc[i], ax.get_ylim()[0], ax.get_ylim()[1],
                            color='red', zorder=1)
                if map_point is not None:
                    ax.vlines(map_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1],
                                color=MAP_color, zorder=1)
                if extra_point is not None:
                    ax.vlines(extra_point.iloc[0, i], ax.get_ylim()[0], ax.get_ylim()[1],
                                color='purple', zorder=1)

    param_bounds = model_obj.Q.get_bounds()
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
