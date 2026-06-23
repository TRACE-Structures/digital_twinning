'''Preprocessing functions for mode shape data handling in digital twinning.'''

import pyuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertain_variables import Variable, UniformDistribution, VariableSet
import seaborn as sns

def compute_dot_products(input_eigenvectors : np.ndarray, reference_eigenvectors: np.ndarray) -> np.ndarray:
    '''
    Calculates dot products between eigenvectors and reference eigenvectors.
    
    Parameters
    ----------
    input_eigenvectors : ndarray of shape (n_samples, n_nodes)
        Array of eigenvectors.
        
    reference_eigenvectors : ndarray of shape (n_clusters, n_nodes)
        Array of reference_eigenvectors.
        
    Returns
    -------
    dot_product : ndarray of shape (n_samples, 1, n_clusters)
        Matrix of dot products between input eigenvectors and typical eigenvectors.
    '''
    
    num_of_samples = input_eigenvectors.shape[0]
    num_of_modes = input_eigenvectors.shape[1]
    num_of_components = input_eigenvectors.shape[2]
    input_eigenvectors_reshaped = input_eigenvectors.reshape(-1, num_of_components)
    dot_products = np.dot(input_eigenvectors_reshaped, reference_eigenvectors.T)
    dot_products = dot_products.reshape(num_of_samples, num_of_modes, -1)
    return dot_products

def flip_eigenvectors(input_eigenvectors: np.ndarray, reference_eigenvectors: np.ndarray | None = None) -> np.ndarray:
    '''
    Flips eigenvectors by clusters to the direction of the typical eigenvectors.
    
    Parameters
    ----------
    input_eigenvectors : ndarray of shape (n_samples, n_nodes)
        Array of eigenvectors.
        
    reference_eigenvectors : ndarray of shape (n_clusters, n_nodes), default=None
        Array of reference_eigenvectors.
    
    Returns
    -------
    flipped_eigenvectors : ndarray
        Flipped versions of the input eigenvectors.
    '''
    
    input_eigenvectors = np.expand_dims(input_eigenvectors, axis=1)
    reference_eigenvectors = input_eigenvectors[0].copy() if reference_eigenvectors is None else reference_eigenvectors
    dot_products = compute_dot_products(input_eigenvectors, reference_eigenvectors)
    max_abs_indices = np.argmax(np.abs(dot_products), axis=-1)
    max_abs_signs = np.sign(dot_products[np.arange(dot_products.shape[0])[:, None], np.arange(dot_products.shape[1]), max_abs_indices])
    flipped_eigenvectors = input_eigenvectors*max_abs_signs[:,:,np.newaxis]
    flipped_eigenvectors = flipped_eigenvectors.reshape(flipped_eigenvectors.shape[0], flipped_eigenvectors.shape[2])
    return flipped_eigenvectors

sensor_cols = ['phi_sensor_1_real', 'phi_sensor_1_imag', 'phi_sensor_2_real',
       'phi_sensor_2_imag', 'phi_sensor_3_real', 'phi_sensor_3_imag',
       'phi_sensor_4_real', 'phi_sensor_4_imag', 'phi_sensor_5_real',
       'phi_sensor_5_imag', 'phi_sensor_6_real', 'phi_sensor_6_imag']
modeshape_cols = ['frequency'] + sensor_cols

def df_from_unv(unv_filename):
    """
    Reads a UNV file and returns a Pandas DataFrame with the data.
    """
    extracted_data = []
    try:
        read_uff = pyuff.UFF(unv_filename)
        all_datasets = read_uff.read_sets()

        for dataset in all_datasets:
            if isinstance(dataset, dict) and dataset.get('type') == 55:
                node_nums = dataset.get('node_nums')
                r1_data = dataset.get('r1')
                r2_data = dataset.get('r2')
                frequency = dataset.get('freq', 0.0)
                xi = dataset.get('modal_damp_vis', 0.0) 
                mode_id = dataset.get('mode_n', 0)
                time = dataset.get('id1', None)

                if node_nums is not None and r1_data is not None:
                    row_data = {
                        'Datetime': time,
                        'mode_id': mode_id,
                        'frequency': frequency,
                        'xi': xi,
                    }
                    for i, node_num in enumerate(node_nums):
                        row_data[f'phi_sensor_{node_num}_real'] = r1_data[i]
                        row_data[f'phi_sensor_{node_num}_imag'] = r2_data[i]

                    extracted_data.append(row_data)

        if extracted_data:
            df = pd.DataFrame(extracted_data)
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            return df
        else:
            print("\nNo valid Dataset 55 data arrays (node_nums/r1) found.")
            return pd.DataFrame()

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def preprocess_weather_data(weather_file, time_column='date'):
    '''
    Preprocesses the weather data from a CSV file.

    Parameters:
    -----------
        weather_file: str
            path to the weather CSV file

    Returns:
    --------
        weather_df: pd.DataFrame
            preprocessed weather DataFrame
    '''

    weather_df = pd.read_csv(weather_file, index_col=None)
    if time_column in weather_df.columns:
        weather_df['Datetime'] = pd.to_datetime(weather_df[time_column], utc=True)
        weather_df = weather_df.drop(columns=[time_column])
    weather_df = weather_df.dropna()
    return weather_df

def merge_dataframes(df_modeshapes, df_weather, time_column='Datetime'):
    '''
    Merges mode shape DataFrame with weather DataFrame based on nearest timestamps.

    Parameters:
    -----------
        df_modeshapes: pd.DataFrame 
            Dataframe containing mode shape data
        df_weather: pd.DataFrame 
            Dataframe containing weather data

    Returns:
    --------
        merged_df: pd.DataFrame
            Merged DataFrame
    '''
    if time_column in df_modeshapes.columns and time_column in df_weather.columns:
        df_a = df_modeshapes.sort_values(by=time_column)
        df_b = df_weather.sort_values(by=time_column)
        merged_df = pd.merge_asof(df_a, df_b, on=time_column, direction='nearest') # merge based on the nearest time
    else:
        merged_df = df_modeshapes.join(df_weather, how="inner")
    if 'mode id' in merged_df.columns: # is this still needed?
        merged_df = merged_df.rename(columns={'mode id': 'mode_id'})

    return merged_df

def separate__and_flip_modes(merged_df):
    '''
    Separates the modes from the merged DataFrame and flips the eigenvectors.

    Parameters:
    -----------
        merged_df: pd.DataFrame
            DataFrame containing the merged data

    Returns:
    -----------
        modeshapes_dict: dictionary
            Dictionary with mode ids as keys and DataFrames as values
    '''

    modeshapes_df = merged_df.copy()
    modes = set(modeshapes_df['mode_id'])
    modeshapes_dict = {}
    for mode in modes:
        modeshapes_dict[mode] = modeshapes_df[modeshapes_df['mode_id'] == mode].drop(['mode_id'], axis=1).sort_values(by='Datetime')

        new_df = modeshapes_dict[mode]
        cols = new_df.columns
        df = new_df[sensor_cols].copy()
        new_df1 = pd.DataFrame(flip_eigenvectors(df.values, reference_eigenvectors=None), columns=sensor_cols)
        for col in cols:
            if col not in sensor_cols:
                new_df1[col] = new_df[col].values
        modeshapes_dict[mode] = new_df1.copy()

    return modeshapes_dict

def merge_modes(modeshapes_dict):
    '''
    Merges the modes from the modeshapes dictionary back into a single DataFrame.

    Parameters
    -----------
        modeshapes_dict: dictionary
            Dictionary with mode ids as keys and DataFrames as values

    Returns
    -------
        modeshapes_df: pd.DataFrame
            mMrged DataFrame containing all modes
    '''

    for i in modeshapes_dict.keys():
        df = modeshapes_dict[i]
        df['mode_id'] = pd.Series([i] * df.shape[0])
        modeshapes_dict[i] = df

    modeshapes_df = pd.concat([modeshapes_dict[i] for i in modeshapes_dict.keys()])

    return modeshapes_df

def select_modes(df, modes, X_cols):
    '''
    Selects specified modes from the DataFrame and pivots the data so that each mode's data is in separate columns.

    Parameters
    -----------
        df: pd.DataFrame
            DataFrame containing the merged data
        modes: list
            List of mode ids to select
        X_cols: list
            List of weather columns to be used

    Returns:
    -----------
        final_df: pd.DataFrame
            Dataframe with selected modes and pivoted structure
    '''

    # Pivot the DataFrame
    if 'Datetime' not in X_cols:
        X_cols = ['Datetime'] + X_cols

    modeshape_cols2 = [col for col in modeshape_cols if col in df.columns]
    pivoted_df = df.pivot_table(index=X_cols,
                                columns='mode_id',
                                values=modeshape_cols2,
                                aggfunc='first') # 'first' works assuming no duplicates for (time, mode_id)

    # Flatten the MultiIndex columns
    pivoted_df.columns = [f'{col[0]}_mode_{col[1]}' for col in pivoted_df.columns]
    final_df = pivoted_df.reset_index()

    ordered_columns = X_cols
    for mode in modes:
        for col_name in modeshape_cols2:
            ordered_columns.append(f'{col_name}_mode_{mode}')

    final_df = final_df[ordered_columns]

    return final_df

def select_cols(data, modes, X_cols, y_cols):
    '''
    Selects input features and output targets from the DataFrame based on specified modes and columns.

    Parameters:
    -----------
        data: pd.DataFrame
            Dataframe containing the merged data
        modes: list
            List of mode ids
        X_cols: list
            List of weather columns to be used
        y_cols
            Can be 'frequency', 'modeshapes', 'frequency+modeshapes', or a list of column names

    Returns:
    --------
        X: pd.DataFrame
            DataFrame of input features
        y: pd.DataFrame
            DataFrame of output targets
        Q: VariableSet
            VariableSet of input variables
        y_names: list
            List of output target names
    '''

    modes = sorted(modes)
    df = select_modes(data, modes, X_cols)

    selected_cols = []

    if y_cols == 'frequency':
        for col in df.columns:
            if 'frequency' in col:
                selected_cols.append(col)
    elif y_cols == 'modeshapes':
        for mode in modes:
            selected_cols.extend([f"{prefix}_mode_{mode}" for prefix in sensor_cols])
    elif y_cols == 'frequency+modeshapes':
        for mode in modes:
            selected_cols.extend([f"{prefix}_mode_{mode}" for prefix in modeshape_cols])

    X = df[X_cols]
    selected = [col for col in selected_cols if col in df.columns]
    y = df[selected]

    # preserve timestamps
    timestamps = df["Datetime"]
    X.index = timestamps
    y.index = timestamps

    Q = VariableSet()
    for col in X_cols:
        Q.add(Variable(col, UniformDistribution(df[col].min(), df[col].max())))
    y_names = selected

    return X, y, Q, y_names

def plot_correlation(input_df: pd.DataFrame,
                     output_df: pd.DataFrame,
                     input_features: list = None,
                     output_features: list = None,
                     annot: bool = True,
                     mode: str = 'input-output') -> plt.Figure:
    '''
    Plots a correlation heatmap between input and output features.

    Parameters:
    -----------
        input_df: pd.DataFrame
            Input features DataFrame
        output_df: pd.DataFrame
            Output targets DataFrame
        input_features: list, optional
            List of input feature names to include
        output_features: list, optional
            List of output feature names to include
        annot: bool
            Whether to annotate the heatmap with correlation values
        mode: str
            'input-output', 'input-input', or 'output-output'

    Returns:
    -------
        matplotlib Figure
    '''

    base_cell_size = 0.8
    max_figsize = (20, 15)

    # Keep only numeric columns
    input_df = input_df.select_dtypes(include='number')
    output_df = output_df.select_dtypes(include='number')

    # Filter selected features
    if input_features:
        input_df = input_df[input_features]
    if output_features:
        output_df = output_df[output_features]

    # Determine correlation matrix based on mode
    if mode == 'input-output':
        correlations = pd.DataFrame(index=input_df.columns, columns=output_df.columns)
        for input_col in input_df.columns:
            for output_col in output_df.columns:
                correlations.loc[input_col, output_col] = input_df[input_col].corr(output_df[output_col])
    elif mode == 'input-input':
        correlations = input_df.corr()
    elif mode == 'output-output':
        correlations = output_df.corr()
    else:
        raise ValueError("Invalid mode. Choose from 'input-output', 'input-input', or 'output-output'.")

    correlations = correlations.astype(float)

    # Determine dynamic figure size
    height = min(max_figsize[1], max(4, base_cell_size * correlations.shape[0]))
    width = min(max_figsize[0], max(4, base_cell_size * correlations.shape[1]))
    fig, ax = plt.subplots(figsize=(width, height))

    # Plot heatmap
    sns.heatmap(correlations, cmap='coolwarm', annot=annot, fmt=".2f",
                cbar=True, ax=ax, vmin=-1, vmax=1)

    # Set axis labels
    if mode == 'input-output':
        ax.set_xlabel('Output Features')
        ax.set_ylabel('Input Features')
        ax.set_title('Correlation Between Input and Output Features')
    elif mode == 'input-input':
        ax.set_title('Correlation Between Input Features')
    elif mode == 'output-output':
        ax.set_title('Correlation Between Output Features')

    plt.tight_layout()
    return fig

def scatter_plot(input_df: pd.DataFrame,
                 output_df: pd.DataFrame,
                 x_col: str,
                 y_col: str,
                 hue_col: str = None,
                 title: str = None) -> plt.Figure:
    '''
    Creates a scatter plot using two features from input or output dataframes.

    Parameters:
    -----------
        input_df : pd.DataFrame
            DataFrame containing input features
        output_df: pd.DataFrame
            DataFrame containing output features
        x_col: str
            Name of feature to use for x-axis
        y_col: str
            Name of feature to use for y-axis
        hue_col: str, optional
            Name of feature to color by
        title : str, optional
            Title for the plot

    Returns:
    -------
        matplotlib Figure
    '''

    # Merge input and output DataFrames for flexibility
    if 'Datetime' in input_df.columns and 'Datetime' in output_df.columns:
        df = merge_dataframes(input_df, output_df)
        df = separate__and_flip_modes(df)
        df = merge_modes(df)
        df = df.sort_values(by='Datetime')

    else:
        df = pd.concat([input_df, output_df], axis=1)

    # Check if requested columns exist
    for col in [x_col, y_col, hue_col]:
        if col and col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input or output DataFrames.")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)

    if not title:
        title = f'Scatter Plot of {x_col} vs {y_col}'

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)

    return fig

def remove_constant_columns(df: pd.DataFrame) -> (pd.DataFrame):
    '''
    Removes columns with constant values from the DataFrame.

    Parameters
    ----------
        df: pd.DataFrame
            Input DataFrame

    Returns
    -----------
        df: pd.DataFrame
            DataFrame with constant columns removed
    '''

    constant_cols = [col for col in df.columns if df[col].dropna().nunique() == 1]
    if constant_cols:
        print(f"Constant columns {constant_cols} were found and excluded.")
        df = df.drop(columns=constant_cols)
    return df

def effects_to_json(effects):
    """
    Converts a DataFrame of effects into a JSON-like structure.

    Parameters:
    -----------
        effects: pd.DataFrame
            DataFrame containing the effects data
    Returns:
    --------
        output: list
            List of dictionaries representing the effects in a JSON-like format
    """

    modes = sorted(
        {
            int(col.split("_mode_")[-1])
            for col in effects.columns
            if "_mode_" in col
        }
    )

    sensors = sorted(
        {
            col.split("_")[2]
            for col in effects.columns
            if col.startswith("phi_sensor_")
        },
        key=int
    )

    output = []

    for timestamp, row in effects.iterrows():

        oma = {
            "timestamp_start": str(timestamp),
            "timestamp_end": str(timestamp),
            "OMA_id": None,
            "flag": 0,
            "sensors": sensors,
            "modes": []
        }

        for mode_id in modes:

            mode = {
                "mode_id": mode_id,
                "frequency": row[f"frequency_mode_{mode_id}"],
                "phi": []
            }

            for sensor in sensors:

                mode["phi"].append({
                    "real": row[f"phi_sensor_{sensor}_real_mode_{mode_id}"],
                    "imag": row[f"phi_sensor_{sensor}_imag_mode_{mode_id}"]
                })

            oma["modes"].append(mode)

        output.append(oma)

    return output

def json2unv(json, file_name, sensor_coords=None, 
                    units_code=1, units_description='mm/newton', temp_mode=1, length=1.0, force=1.0, temp=1.0, temp_offset=1.0,
                    def_cs=0, disp_cs=0, color=0,
                    id1='NONE', id2='NONE', id3='NONE', id4='NONE', id5='NONE', model_type=1, analysis_type=2, data_ch=2, 
                    spec_data_type=8, data_type=2, n_data_per_node=3, load_case=1, modal_m=0, modal_damp_vis=0., modal_damp_his=0.):
    """
    Converts JSON data to UNV format.

    Parameters:
    -----------
        json: list
            List of dictionaries containing the OMA data
        file_name: str
            Name of the output UNV file
        sensor_coords: dict, optional
            Dictionary mapping sensor IDs to coordinates
        units_code: int
            Code for the units
        units_description: str
            Description of the units
        temp_mode: int
            Temperature mode
        length: float
            Length value
        force: float
            Force value
        temp: float
            Temperature value
        temp_offset: float
            Temperature offset value
        def_cs: list, optional
            List of default coordinate systems
        disp_cs: list, optional
            List of displacement coordinate systems
        color: list, optional
            List of colors
        id1, id2, id3, id4, id5: str, optional
            IDs for the UNV file
        model_type: int
            Type of the model
        analysis_type: int
            Type of the analysis
        data_ch: int
            Data channel number
        spec_data_type: int
            Specific data type

    Returns:
    --------
        None

    """

    unv_data = []
    imags = []

    if sensor_coords is None:
        sensor_coords = {
            '1': (0.0, 0.0, 0.0),
            '2': (0.0, 0.0, 0.0),
            '3': (0.0, 0.0, 0.0),
            '4': (0.0, 0.0, 0.0),
            '5': (0.0, 0.0, 0.0),
            '6': (0.0, 0.0, 0.0),
        }

    ## Prepare type 164 data
    units_data = pyuff.prepare_164(
        units_code=units_code,
        units_description=units_description,
        temp_mode=temp_mode,
        length=length,
        force=force,
        temp=temp,
        temp_offset=temp_offset
    )
    unv_data.append(units_data)

    ## Prepare type 15 data
    node_nums = []
    x, y, z = [], [], []

    for sensor_id in sorted(sensor_coords.keys(), key=int):
        node_nums.append(int(sensor_id))
        xi, yi, zi = sensor_coords[sensor_id]
        x.append(xi)
        y.append(yi)
        z.append(zi)

    if def_cs==0:
        def_cs=[0]*len(node_nums)
    if disp_cs==0:
        disp_cs=[0]*len(node_nums)
    if color==0:
        color=[0]*len(node_nums)
    node_data = pyuff.prepare_15(
        node_nums=node_nums,
        def_cs=def_cs,
        disp_cs=disp_cs,
        color=color,
        x=x,
        y=y,
        z=z
    )
    unv_data.append(node_data)

    ## Prepare type 55 data
    for oma in json:
        sensors = oma['sensors']
        node_nums = np.array([int(s) for s in sensors])
        timestamp = str(oma['timestamp_start'])

        modes = oma['modes']
        for idx, mode in enumerate(modes):
            freq = mode['frequency']
            mode_id = mode.get('mode_id', None)
            if mode_id < 0: # unv does not support negative mode IDs, skip if invalid
                continue
            phi = mode['phi']

            # Take the real and imaginary parts separately
            r1 = np.array([p['real'] for p in phi])
            r2 = np.array([p['imag'] for p in phi])
            imags.append(r2)
            r3 = np.zeros(len(r1))  # third component set to zero

            mode_data = pyuff.prepare_55(
                id1=timestamp,
                id2=id2,
                id3=id3,
                id4=id4,
                id5=id5,
                model_type=model_type,
                analysis_type=analysis_type,
                data_ch=data_ch,
                spec_data_type=spec_data_type,
                data_type=data_type,
                n_data_per_node=n_data_per_node,
                load_case=load_case,
                mode_n=mode_id,
                freq=freq,
                modal_m=modal_m,
                modal_damp_vis=modal_damp_vis,
                modal_damp_his=modal_damp_his,
                node_nums=node_nums,
                r1=r1,
                r2=r2,
                r3=r3
            )
            unv_data.append(mode_data)

    # UNV file saving
    uff = pyuff.UFF(file_name)
    uff.write_sets(unv_data)
    print(np.mean(imags), np.std(imags), imags)