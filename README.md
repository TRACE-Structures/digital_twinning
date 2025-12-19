# digital_twinning

[![DOI](https://zenodo.org/badge/1113759480.svg)](https://doi.org/10.5281/zenodo.17990407)

A comprehensive Python package for digital twin model updating and predictive modeling using machine learning and uncertainty quantification techniques.

## Overview

The Digital Twinning package provides tools for creating data-driven predictive models and updating them with measurement data using Bayesian inference (MCMC). It combines surrogate modeling, machine learning, and probabilistic model updating for structural health monitoring and digital twin applications.

## Installation

```bash
pip install digital-twinning
```

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
Gradient Boosted Decision Trees with multiple backend options.

**Supported Backends:**
- XGBoost
- CatBoost
- LightGBM
- scikit-learn GradientBoostingRegressor

## Authors and acknowledgment

The code is developed by András Urbanics, Áron Friedman, Bence Popovics, Emese Vastag, Elmar Zander and Noémi Friedman in the TRACE-Structures group.

This work has been funded by the European Commission Horizon Europe Innovation Action project 101092052 [BUILDCHAIN](https://buildchain-project.eu/)

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-only). See the [LICENSE](https://github.com/TRACE-Structures/digital_twinning/tree/main?tab=GPL-3.0-1-ov-file) file for details.

## Related Projects

- [gPCE_model](https://github.com/TRACE-Structures/gPCE_model/): Generalized Polynomial Chaos Expansion
- [uncertain_variables](https://github.com/TRACE-Structures/uncertain_variables/): Probabilistic variable management

## Support

For issues, questions, or contributions, please refer to the project repository or contact the authors.
