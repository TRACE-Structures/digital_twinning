![Digital Twinning](https://raw.githubusercontent.com/TRACE-Structures/digital_twinning/main/docs/digital_twinning_logo.png)

# Digital Twinning

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17990407-blue)](https://doi.org/10.5281/zenodo.17990407)
[![Read the Docs](https://img.shields.io/readthedocs/digital-twinning?style=flat&logo=readthedocs)](https://digital-twinning.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/digital-twinning?style=flat)](https://pypi.org/project/digital-twinning/)

A Python package for predictive modeling, probabilistic inference, and Bayesian model updating for digital-twin applications.

- Website: [BUILDCHAIN](https://buildchain.ilab.sztaki.hu/)
- Source code: [TRACE-Structures/digital_twinning](https://github.com/TRACE-Structures/digital_twinning)
- Documentation: [Read the Docs](https://digital-twinning.readthedocs.io/en/latest/)
- Bug reports: [GitHub Issues](https://github.com/TRACE-Structures/digital_twinning/issues)

## Overview

The package combines machine-learning surrogate models, probabilistic models, and MCMC-based parameter estimation. It is intended for workflows where model predictions are trained from simulation or measurement data and then updated using observed quantities of interest.

Inputs and outputs are represented using `pandas.DataFrame` objects in the predictive-model workflows. Model parameters are described with `VariableSet` objects from the `uncertain_variables` package.

## Features

### Predictive models

The `digital_twinning.predictive_models` package provides:

- **DNN** — feed-forward neural-network regression with configurable hidden layers, activations, dropout, and optional cross-validation.
- **GBT** — gradient-boosted regression using XGBoost.
- **LinReg** — linear regression baseline.
- **gPCE** — generalized Polynomial Chaos Expansion through `gPCE_model`.
- **MOSAIC** — surrogate modeling through the `mosaictools` package.

Predictive models share a common interface for training and prediction. The base class also provides evaluation, cross-validation, model serialization, and sensitivity-analysis functionality. SHAP computation is implemented by individual model classes where supported.

### Probabilistic models

The `digital_twinning.probabilistic_models` package includes:

- **MDN** — a mixture-density network for modeling conditional distributions.
- **CRPS** — a probabilistic neural model trained using the Continuous Ranked Probability Score.

These models provide an `inference` method for generating samples conditioned on input features.

### Bayesian model updating

The `digital_twinning.model_updating` package includes:

- **DigitalTwin** — updates one predictive model against measured quantities using MCMC sampling with `emcee`. It supports posterior samples, posterior summaries, MAP estimation, and plotting.
- **JointManager** — performs joint updates for multiple models with shared parameters and provides per-model posterior summaries and samples.

### Data and visualization utilities

The `digital_twinning.utils` package contains helpers for data preparation and visualization, including:

- Reading and preprocessing modal-shape and weather data.
- Merging, selecting, and filtering data columns and modes.
- Converting mode-shape data to and from UNV-related representations.
- Plotting model results, sensitivity analyses, and MCMC outputs.

## Installation

Install the published package with pip:

```bash
pip install digital-twinning
```

The package metadata lists dependencies including PyTorch, scikit-learn, XGBoost, `emcee`, `SALib`, `shap`, `gPCE-model`, `mosaictools`, and `uncertain-variables`. The project is distributed as `digital_twinning` version `2.0.0`.

## Source layout

```text
src/
├── model_updating/       # DigitalTwin and JointManager
├── predictive_models/    # DNN, GBT, LinReg, gPCE, MOSAIC
├── probabilistic_models/ # MDN and CRPS
└── utils/                # Data processing and plotting helpers
```

## Examples

The repository includes notebooks demonstrating package workflows under [demo/notebooks](https://github.com/TRACE-Structures/digital_twinning/tree/main/demo/notebooks).

These notebooks are the best starting point for seeing how the models are configured and used with the project’s example datasets.

## Authors and acknowledgment

Developed by András Urbanics, Áron Friedman, Bence Popovics, Emese Vastag, and Noémi Friedman in the TRACE-Structures group.

This work was funded by the European Commission through the Horizon Europe Innovation Action [BUILDCHAIN](https://buildchain-project.eu/) (grant agreement No. 101092052) up to and including v1.0.0, and by the PRODIGI project from later versions onward.

## License

This project is licensed under the GNU General Public License v3.0 only. See the [LICENSE](LICENSE) file for details.

## Related projects

- [uncertain_variables](https://github.com/TRACE-Structures/uncertain_variables/) — probabilistic variable management.
- [gPCE_model](https://github.com/TRACE-Structures/gPCE_model/) — generalized Polynomial Chaos Expansion.
- [mosaictools](https://github.com/blazkurent/mosaictools) - a Python toolkit for surrogate modeling of modal properties using the MOSAIC method.

## Support

For bug reports and questions, use the [project issue tracker](https://github.com/TRACE-Structures/digital_twinning/issues).