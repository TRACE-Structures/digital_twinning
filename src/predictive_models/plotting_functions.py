import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

def plot_remained_effects(higher_bound, lower_bound, margin, higher_measured_bound, lower_measured_bound, QoI_names, xticks=True):
    ''' Plot remaining effect bounds with measured min-max and acceptable margins.

        Parameters
        ----------
        higher_bound : array_like
            Upper bounds of reference effect.
        lower_bound : array_like
            Lower bounds of reference effect.
        margin : float
            Margin threshold defining acceptable regions.
        higher_measured_bound : array_like
            Upper bounds of measured data.
        lower_measured_bound : array_like
            Lower bounds of measured data.
        QoI_names : list of str
            Names of quantities of interest.
        xticks : bool, optional
            Whether to display x-axis tick labels.

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing effect bounds and monitoring data. '''
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(0, len(higher_bound))
    ax.fill_between(x, higher_bound + margin, lower_bound - margin, color = '#1a80bb', alpha=0.3, label='Acceptable region')

    ax.fill_between(x, higher_measured_bound, lower_measured_bound, color = 'gray', alpha=0.3)


    ax.plot(higher_bound, color='#1a80bb', label='Reference min-max')
    ax.plot(lower_bound, color='#1a80bb')

    ax.plot(higher_measured_bound, color ='black', label='Monitoring min-max')
    ax.plot(lower_measured_bound, color ='black')

    up_up_indexes = np.where(higher_measured_bound > higher_bound + margin)[0]
    up_low_indexes = np.where(higher_measured_bound < lower_bound - margin)[0]
    low_up_indexes = np.where(lower_measured_bound > higher_bound + margin)[0]
    low_low_indexes = np.where(lower_measured_bound < lower_bound - margin)[0]
    
    upper_indexes = np.sort(np.concatenate((up_up_indexes, up_low_indexes)))
    lower_indexes = np.sort(np.concatenate((low_up_indexes, low_low_indexes)))

    if (len(upper_indexes) > 0) and (len(lower_indexes) > 0):
        ax.plot(upper_indexes, higher_measured_bound[upper_indexes], 'ro', label='Point outside acceptable region')
        ax.plot(lower_indexes, lower_measured_bound[lower_indexes], 'ro')

    elif len(upper_indexes) > 0:
        ax.plot(upper_indexes, higher_measured_bound[upper_indexes], 'ro', label='Point outside acceptable region')
    elif len(lower_indexes) > 0:
        ax.plot(lower_indexes, lower_measured_bound[lower_indexes], 'ro', label='Point outside acceptable region')

    if xticks:
        ax.set_xticks(x, QoI_names)
    else:
        ax.set_xticklabels([])
    ax.set_xlabel('Predicted features', fontsize=12)
    fig.legend()
    return fig

def plot_shap_single_waterfall(model, silent, q, param_name):
    ''' SHAP waterfall plot for an individual prediction.

        Parameters
        ----------
        model : object
            Fitted model object with SHAP explainer.
        q : pandas.DataFrame or array_like
            Input features for prediction.
        param_name : str
            Target parameter name.

        Returns
        -------
        matplotlib.figure.Figure
            Waterfall plot figure. '''
    
    if type(q) != pd.DataFrame:
        q = pd.DataFrame([q], columns=model.Q.variable_names())
            
    explainer = model.explainer
    shap_values = explainer(q, silent=silent)
    labels = model.Q.variable_names()
    param_index = model.QoI_names.index(param_name)
    values = shap_values[0, :, param_index].data
    bar_lengths = shap_values[0, :, param_index].values
    expected_value = shap_values[0, :, param_index].base_values
    sorted_indices = np.argsort(np.abs(bar_lengths))
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_bar_lengths = [bar_lengths[i] for i in sorted_indices]
    colors = ['#E53935' if val >= 0 else '#1E88E5' for val in sorted_bar_lengths]

    fig, ax = plt.subplots()
    bar_height = 0.915
    x_min = min(min(expected_value + np.cumsum(sorted_bar_lengths)), expected_value)
    x_max = max(max(expected_value + np.cumsum(sorted_bar_lengths)), expected_value)
    x_range = x_max - x_min
    x_range_ratio = x_range * 0.12
    ax.set_xlim(x_min - x_range_ratio, x_max + x_range_ratio)
    y_pos = list(range(len(sorted_labels)))
    y_min = min(y_pos) - (3 / 4) * bar_height
    y_max = max(y_pos) + (2 / 3) * bar_height
    ax.set_ylim(y_min, y_max)
    prediction = expected_value
    for i in range(len(sorted_labels)):
        bar_length = sorted_bar_lengths[i]
        bar = ax.barh(y_pos[i], bar_length, color=colors[i], height=bar_height, left=prediction, align='center')
        text_x = prediction + bar_length
        text_x_correction = x_range*0.02
        sign=''
        x = text_x
        if sorted_bar_lengths[i] >= 0:
            text_x += text_x_correction
            alignment = 'left'
            color = '#E53935'
            sign = '+'
        else:
            text_x -= text_x_correction
            alignment = 'right'
            color = '#1E88E5'
        ax.text(text_x, y_pos[i], f'{sign}{bar_length:.2f}', 
                va='center', ha=alignment, color=color, fontsize=10)
        
        # Update the left position for the next bar
        prediction += bar_length
        
    # Add triangles at the end of bars
        y = y_pos[i]
        half_bar_height = bar_height/2*0.9
        
        if bar_length >= 0:
            triangle = np.array([[x, y - half_bar_height], [x, y + half_bar_height], [x + x_range*0.01, y]])
        else:
            triangle = np.array([[x, y - half_bar_height], [x, y + half_bar_height], [x - x_range*0.01, y]])
        
        ax.add_patch(plt.Polygon(triangle, color=color, label="_nolegend_"))

    ax.axvline(expected_value, color='gray', linestyle='dashed', label=f'Expected Value = {expected_value:.3f}')
    ax.axvline(prediction, color='gray', linestyle='dashed', label=f'Prediction = {prediction:.3f}')
    
    x_label_pos = ax.xaxis.get_ticklabels()[0].get_position()[1]
    ax.text(expected_value, (y_min - half_bar_height)/2, f'Expected Value = {expected_value:.3f}', va='center', fontsize=10, color='black', ha='center')
    ax.text(prediction, y_max + 0.04, f'Prediction = {prediction:.3f}', fontsize=10, color='black', ha='center')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{l} = {v:.3f}' for v, l in zip(sorted_values, sorted_labels)])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.set_title("SHAP Explanation for Individual Prediction", fontsize=16, pad=20)

    return fig


def plot_shap_multiple_waterfalls(model, silent, q, param_name=None, show_param_values=False):
    ''' Plot SHAP waterfall plots for multiple predictions or quantities of interest.

        Parameters
        ----------
        model : object
            Fitted model object with SHAP explainer.
        q : pandas.DataFrame or array_like
            Input feature sets.
        param_name : str or None, optional
            Filter QoI names to plot (default is None, plot all).
        show_param_values : bool, optional
            Whether to show parameter values in legend (default False).

        Returns
        -------
        matplotlib.figure.Figure
            Figure with SHAP waterfall bar plots for multiple QoIs. '''
    
    if type(q) != pd.DataFrame:
        q = pd.DataFrame([q], columns=model.Q.variable_names())

    explainer = model.explainer
    shap_values = explainer(q, silent=silent)
    
    labels = model.Q.variable_names()
    QoI_names = model.QoI_names
    num_params = len(QoI_names)
    num_features = len(labels)
    
    all_expected_value = shap_values[0, :, :].base_values
    all_bar_lengths = shap_values[0, :, :].values
    values = shap_values[0, :, 0].data
    all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
    
    title = "Prediction of QoI and Effects of Varying Parameters"
    
    if param_name is not None:
        column_names = model.QoI_names
        if param_name == 'freq':     
            freqs = [freq for freq in column_names if freq.startswith('f')]
            QoI_names = freqs
            num_params = len(freqs)
            freqs_indices = [i for i, e in enumerate(column_names) if e in freqs]
            all_expected_value = all_expected_value[freqs_indices]
            all_bar_lengths = all_bar_lengths[:, freqs_indices]
            all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
            
            title = "Prediction of Frequencies and Effects of Varying Parameters"
        elif param_name not in column_names:
            prefix = param_name
            modes = [mode for mode in column_names if mode.startswith(prefix)]
            QoI_names = modes
            num_params = len(modes)
            if num_params == 0:
                raise ValueError(f"There is no mode starting with {prefix}")
            modes_indices = [i for i, e in enumerate(column_names) if e in modes]
            all_expected_value = all_expected_value[modes_indices]
            all_bar_lengths = all_bar_lengths[:, modes_indices]
            all_labels = np.array([[labels[i]]*num_params for i in range(num_features)])
            
            title = f"Prediction of Mode {param_name} and Effects of Varying Parameters"
    
    sorted_indices = np.argsort(np.abs(all_bar_lengths), axis=0)#[::-1]  # Indices for descending order
    sorted_all_labels = np.take_along_axis(all_labels, sorted_indices, axis=0)
    sorted_bar_lengths = np.take_along_axis(all_bar_lengths, sorted_indices, axis=0)
    
    fig, ax = plt.subplots(figsize=(num_params*1.8, 5))
    
    bar_width = (1-0.2)/num_features
    x_pos = np.arange(num_features * num_params)
    
    y_min = np.min(all_expected_value + np.cumsum(sorted_bar_lengths, axis=0))
    y_max = np.max(all_expected_value + np.cumsum(sorted_bar_lengths, axis=0))
    y_range = y_max - y_min
    y_range_ratio = (y_max - y_min)*0.22
    ax.set_ylim(y_min-y_range_ratio, y_max+y_range_ratio)

    all_colors = []
    all_bottoms = []
    all_predictions = all_expected_value + np.sum(sorted_bar_lengths, axis=0)

    bottom = all_expected_value
    for i in range(num_features):
        all_bottoms.append(list(bottom))
        colors = []
        bar_lengths = sorted_bar_lengths[i,:]
        bottom += bar_lengths
        
        for param_index in range(num_params):
            colors.append('#E53935' if bar_lengths[param_index] >= 0 else '#1E88E5')
        all_colors.append(colors)

    br = [np.arange(num_params)]
    for i in range(1, num_features):
        br.append([x + bar_width for x in br[i - 1]])
    
    ax.plot(br[-1], all_predictions, color='black', linewidth=0.8, marker='x', label="Prediction")
        
    for param_index in range(num_features):
        compensation = [-y_range*0.01 if x >= 0 else y_range*0.01 for x in sorted_bar_lengths[param_index]]
        bars = plt.bar((br[param_index]), sorted_bar_lengths[param_index], bottom=all_bottoms[param_index], color=all_colors[param_index], width=bar_width, label="_nolegend_")
    
        for bar, height, color in zip(bars, sorted_bar_lengths[param_index], all_colors[param_index]):
            x = bar.get_x() + bar.get_width()/2
            y = bar.get_y() + bar.get_height()
            half_bar_width = bar.get_width()/2 *0.9
            
            if height >= 0:
                triangle = np.array([[x - half_bar_width, y], [x + half_bar_width, y], [x, y + y_range*0.01]])
            else:
                triangle = np.array([[x - half_bar_width, y], [x + half_bar_width, y], [x, y - y_range*0.01]])
            
            ax.add_patch(plt.Polygon(triangle, color=color, label="_nolegend_"))

        for i, (x, bottom, height, color) in enumerate(zip(br[param_index], all_bottoms[param_index], sorted_bar_lengths[param_index], all_colors[param_index])):
            sign = '+' if height >= 0 else ''
            text_x_correction = y_range*0.03 if height >= 0 else -y_range*0.03
            ax.text(x, (bottom+height) + text_x_correction, f'{sign}{height:.2f}', rotation=90, ha='center', 
                    va='bottom' if height >= 0 else 'top', color=color, fontsize=9)
            
        
        if show_param_values:
            ax.plot([], [], ' ', label= f"{labels[param_index]} = {values[param_index]:.6f}")
    
    ax.plot([], [], label='Positive effect', marker='s', markersize=8,
            markeredgecolor='#E53935', markerfacecolor='#E53935', linestyle='')
    ax.plot([], [], label='Negative effect', marker='s', markersize=8,
            markeredgecolor='#1E88E5', markerfacecolor='#1E88E5', linestyle='')

    x_pos = [item for sublist in br for item in sublist]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_all_labels.flatten(), rotation=45, fontsize=10)
    
    br = np.array(br)
    group_labels = QoI_names
    group_positions = np.mean(br, axis=0)

    for pos, label in zip(group_positions, group_labels):
        ax.text(pos, y_min - y_range_ratio*2, label, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("Prediction with SHAP Value Impact")
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlim(min(x_pos) - 2*bar_width, max(x_pos) + 2*bar_width)

    return fig



def plot_shap_beeswarm(model, silent, param_name, q=None):
    ''' Generate SHAP beeswarm plot showing feature effects on a QoI.

        Parameters
        ----------
        model : object
            Model object with stored SHAP values and explainer.
        param_name : str
            Quantity of interest to plot.
        q : pandas.DataFrame or None, optional
            Feature values corresponding to SHAP values. If None, uses stored SHAP values.

        Returns
        -------
        matplotlib.figure.Figure
            Beeswarm plot figure. '''
    
    if q is not None:
        explainer = model.explainer
        shap_values = explainer(q, silent=silent)
    else:
        shap_values = model.shap_values

    fig, ax = plt.subplots(figsize=(8, 6))
    
    if param_name is not None:
        param_index = model.QoI_names.index(param_name)
        QoI_shap = shap_values[:, :, param_index]

        ax.set_title(f"SHAP Explanation for {param_name} Quantity of Interest", fontsize=16)
        shap.summary_plot(QoI_shap, q, show=False)
    return fig


def plot_effects(effects, xticks=True):
    ''' Plot predicted values with subtractive effects removed.

        Parameters
        ----------
        effects : array_like
            Effect values to plot.
        xticks : bool, optional
            Whether to show x-axis tick labels (default is True).

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing plotted effects. '''
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(effects.transpose())
    ax.set_title('Predictions with Subtraced Effects', fontsize=16)
    if not xticks:
        ax.set_xticklabels([])
    ax.set_xlabel('Predicted features', fontsize=12)
    return fig


def plot_sobol_sensitivity(sobol_index, y, param_name=None):
    ''' Plot Sobol sensitivity indices as stacked area or pie charts.

        Parameters
        ----------
        sobol_index : pandas.DataFrame
            DataFrame of Sobol sensitivity indices.
        y : pandas.DataFrame
            Quantity of interest (QoI) data.
        param_name : str or None, optional
            Specific parameter or mode to plot sensitivities for.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object of the plotted sensitivities. '''
    
    y_var = np.broadcast_to(y.var(axis=0).to_numpy().reshape(-1, 1), sobol_index.shape)
    y_var_df = pd.DataFrame(y_var, columns=sobol_index.columns, index=sobol_index.index)
    partial_variance = np.multiply(sobol_index, y_var)

    color_map = plt.cm.viridis(np.linspace(0, 1, partial_variance.shape[1]+1))
    colors = {partial_variance.columns[i]: color_map[i] for i in range(len(partial_variance.columns))}
    modes = partial_variance.index
    y_label = 'Labels'

    if param_name == 'freq':
        f_modes = [mode for mode in partial_variance.index if mode.startswith('f')]
        num_plots = len(f_modes)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        if num_plots == 1:
            axes = [axes]

        for i, mode in enumerate(f_modes):
            threshold = 0.01
            loc_par_var = partial_variance.loc[mode]
            df = pd.DataFrame(loc_par_var)
            df[df < 0] = 0
            df['percentage'] = df[mode] / df[mode].sum()
            under_threshold = df.loc[df['percentage'] < threshold].sum()
            remaining = [y_var_df.loc[mode][0], 1] - df.sum()
            others = under_threshold + remaining
            colors['others'] = color_map[-1]
            df = df[df['percentage'] >= threshold]
            df.loc['others'] = others
            if df.loc['others']['percentage'] < 1e-06:
                df.drop(index='others', inplace=True)

            pie_colors = [colors[x] for x in df.index]
            axes[i].pie(df['percentage'], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
            axes[i].set_title(f"Sobol sensitivity for: {mode}", fontsize=12)
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    if param_name in partial_variance.index:
        threshold = 0.01
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

        ax.set_title(f"Sobol sensitivity for: {param_name}", fontsize=16)
        ax.pie(df['percentage'], labels=df.index, colors=pie_colors, wedgeprops={"alpha": 0.5})
        return fig

    elif param_name is not None and param_name not in partial_variance.columns:
        prefix = param_name
        index_names = partial_variance.index
        modes = [item for item in index_names if item.startswith(prefix)]
        if len(modes) == 0:
            raise ValueError(f"There is no mode starting with {prefix}")
        partial_variance = partial_variance.loc[modes]
        y_var_df = y_var_df.loc[modes]
        y_label = param_name

    timesteps = partial_variance.shape[0]
    ax.plot(modes, y_var_df.iloc[:, 0], color='black', label='Total variance', linewidth=0.8)
    stackplot_colors = [colors[col] for col in y_var_df.columns]
    ax.stackplot(modes, partial_variance.to_numpy().transpose(), alpha=0.5, labels=partial_variance.columns, colors=stackplot_colors)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlim((0, timesteps - 1))
    ax.set_xlabel(y_label, fontsize=14)
    ax.set_ylabel('Variance', fontsize=14)
    ax.set_title('Sobol sensitivity', fontsize=16)
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)

    return fig
