# Digital Twinning

A comprehensive Python package for digital twin model updating and predictive modeling using machine learning and uncertainty quantification techniques.

## Overview

The Digital Twinning package provides tools for creating data-driven predictive models and updating them with measurement data using Bayesian inference (MCMC). It combines surrogate modeling, machine learning, and probabilistic model updating for structural health monitoring and digital twin applications.

## Features

### Predictive Models
- **Deep Neural Networks (DNN)**: Flexible neural network architectures with customizable layers and activation functions
- **Gradient Boosted Trees (GBT)**: Support for multiple implementations (XGBoost, CatBoost, LightGBM, scikit-learn)
- **Linear Regression**: Basic linear regression models for baseline comparisons
- **gPCE Models**: Generalized Polynomial Chaos Expansion for uncertainty quantification

### Model Updating
- **Bayesian Model Updating**: MCMC-based parameter estimation using emcee
- **Multi-Building Updates**: Joint parameter estimation across multiple structures
- **Prior and Posterior Analysis**: Tools for analyzing parameter distributions

### Model Interpretability
- **SHAP Analysis**: Feature importance and explanation using SHAP values
- **Sobol Sensitivity Analysis**: Global sensitivity analysis for parameter importance
- **Visualization Tools**: Comprehensive plotting utilities for model analysis

### Data Handling
- **Mode Shape Data**: Classes for handling eigenfrequencies and eigenvectors
- **Data Preprocessing**: Tools for preparing structural dynamics data
- **Value per DoF**: Structured data handling for nodal degrees of freedom

## Installation

```bash
pip install -e .
```

### Dependencies
- numpy
- scikit-learn
- scipy
- pandas
- torch
- emcee
- shap
- xgboost
- catboost
- lightgbm
- uncertain_variables

## Quick Start

### 1. Creating a Predictive Model

```python
from digital_twinning.predictive_models import PredictiveModel
from uncertain_variables import VariableSet

# Define parameter set
Q = VariableSet([...])  # Your uncertain variables

# Define quantities of interest
QoI_names = ['freq_1', 'freq_2', 'freq_3']

# Create a Gradient Boosted Trees model
model = PredictiveModel(Q, QoI_names, method='GBT')

# Train the model
model.train(q_train, y_train, n_est=150, max_d=3)

# Make predictions
predictions = model.predict(q_test)
```

### 2. Model Updating with MCMC

```python
from digital_twinning.model_updating import DigitalTwin

# Create error model
E = VariableSet([...])  # Error variables

# Initialize digital twin
dt = DigitalTwin(model, E)

# Update with measurements
y_measured = [...]  # Your measurement data
dt.update(y_measured, nwalkers=150, nburn=100, niter=350)

# Get posterior statistics
mean, var = dt.get_mean_and_var_of_posterior()
MAP = dt.get_MAP()
```

### 3. Multi-Building Model Updating

```python
from digital_twinning.model_updating import JointManager

# Define models for multiple buildings
models = [model1, model2, model3]

# Define joint parameters
joint_parameters = {
    'param1': [0, 1, 2],  # Shared across all buildings
    'param2': [0, 1]      # Shared across first two buildings
}

# Create joint manager
jm = JointManager(models, joint_parameters)

# Perform joint update
jm.update(y_measured_dict, nwalkers=150, nburn=100, niter=350)
```

### 4. Model Interpretability

```python
# SHAP analysis
shap_values = model.get_shap_values(q_test)
model.plot_shap_beeswarm(q_test)

# Sobol sensitivity analysis
sobol_indices = model.get_sobol_sensitivity(max_index=2)
model.plot_sobol_sensitivity(y_train, max_index=1)

# Subtract parameter effects
cleaned_data = model.subtract_effects(q_test, 'freq_1', ['param1', 'param2'])
```

## Module Structure

```
digital_twinning/
├── __init__.py
├── setup.py
├── data_handling/
│   ├── data_handler.py          # Classes for mode shape data
│   └── preprocess_modeshape_data.py  # Data preprocessing utilities
├── model_updating/
│   ├── digital_twin.py          # MCMC-based model updating
│   └── multibuilding_update.py  # Joint updating for multiple structures
├── predictive_models/
│   ├── predictive_model.py      # Base predictive model class
│   ├── dnn_model.py             # Deep Neural Network implementation
│   ├── gbt_model.py             # Gradient Boosted Trees implementation
│   └── linreg_model.py          # Linear Regression implementation
└── utils/
    ├── utils.py                 # General utility functions
    ├── gbt_plot_utils.py        # GBT-specific plotting utilities
    ├── object_utils.py          # Object manipulation utilities
    └── plotting_functions.py    # Plotting functions
```

## Key Classes

### PredictiveModel
The base class for all predictive models. Supports training, prediction, cross-validation, and model interpretability.

**Methods:**
- `train()`: Train the model with optional k-fold cross-validation
- `predict()`: Make predictions on new data
- `get_shap_values()`: Compute SHAP values for feature importance
- `get_sobol_sensitivity()`: Perform Sobol sensitivity analysis
- `save_model()` / `load_model()`: Serialize and deserialize models

### DigitalTwin
MCMC-based Bayesian model updating for parameter estimation.

**Methods:**
- `update()`: Update parameters using measurement data
- `get_mean_and_var_of_posterior()`: Get posterior statistics
- `get_MAP()`: Get maximum a posteriori estimate
- `loglikelihood()`: Compute log-likelihood of measurements
- `logprior()`: Compute log-prior of parameters

### JointManager
Manage joint model updating for multiple buildings with shared parameters.

**Methods:**
- `update()`: Perform joint update across all buildings
- `get_joint_paramset_and_indices()`: Create joint parameter space
- `generate_joint_stdrn_simparamset()`: Generate joint simulation parameter sets

### DNNModel
Deep Neural Network implementation with PyTorch backend.

**Features:**
- Flexible architecture with customizable layers
- Multiple activation functions (ReLU, GELU, Tanh, etc.)
- Dropout regularization
- Early stopping
- GPU support

### GBTModel
Gradient Boosted Trees with multiple backend options.

**Supported Backends:**
- XGBoost
- CatBoost
- LightGBM
- scikit-learn GradientBoostingRegressor

## Data Classes

### ValuePerDoF
Store displacement/velocity values per degree of freedom for a node.

### Mode
Store modal data including eigenfrequencies and eigenvectors (mode shapes).

**Methods:**
- `return_eigenfrequency()`: Get the eigenfrequency
- `return_eigenvector()`: Get eigenvector for specified nodes/directions
- `calculate_MAC()`: Calculate Modal Assurance Criterion

## Utilities

### Visualization Functions
- `plot_3Dscatter()`: 3D scatter plots with optional surfaces
- `plot_sobol_sensitivity()`: Visualize Sobol indices
- `plot_posterior_samples()`: Plot MCMC posterior samples
- `plot_correlation_matrix()`: Correlation heatmaps

### Analysis Functions
- `calculate_MAC()`: Modal Assurance Criterion between mode shapes
- `normalize_mode_shape()`: Normalize eigenvectors
- `compute_error_metrics()`: Calculate MAE, MSE, RMSE, etc.

## Advanced Features

### Cross-Validation
```python
# K-fold cross-validation during training
model.train(q_train, y_train, k_fold=5, n_est=150)
```

### Hyperparameter Tuning
```python
# Grid search over hyperparameters
best_params = model.tune_hyperparameters(
    q_train, y_train,
    param_grid={'n_est': [100, 150, 200], 'max_d': [3, 5, 7]}
)
```

### Model Persistence
```python
# Save trained model
model.save_model(name='my_model', path='./models/')

# Load model
from digital_twinning.predictive_models import PredictiveModel
model = PredictiveModel.load_model('./models/my_model.pkl')
```

## Examples

See the notebooks directory for detailed examples:
- Model training and validation
- MCMC-based parameter estimation
- Multi-building joint updates
- SHAP and Sobol sensitivity analysis

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Authors

- András Urbanics
- Áron Friedman
- Bence Popovics
- Emese Vastag
- Noémi Friedman

**Contact:** popbence@hun-ren.sztaki.hu

## License

This project is licensed under the GNU General Public License v3 (GPLv3).

## Citation

If you use this package in your research, please cite:

```bibtex
@software{digital_twinning,
  title = {Digital Twinning: A Python Package for Model Updating and Predictive Modeling},
  author = {Urbanics, András and Friedman, Áron and Popovics, Bence and Vastag, Emese and Friedman, Noémi},
  year = {2025},
  url = {https://github.com/TRACE-Structures/digital_twinning}
}
```

## Related Projects

- [gPCE_model](https://github.com/TRACE-Structures/gPCE_model/): Generalized Polynomial Chaos Expansion
- [uncertain_variables](https://github.com/TRACE-Structures/uncertain_variables/): Probabilistic variable management

## References

- emcee: MCMC sampling library
- SHAP: SHapley Additive exPlanations
- SALib: Sensitivity Analysis Library
- XGBoost, CatBoost, LightGBM: Gradient boosting implementations

## Development Status

This package is currently in **Alpha** stage (v0.1.0). APIs may change in future releases.
