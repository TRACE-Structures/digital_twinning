Predictive models
=================

Predictive models map input features to one or more quantities of interest.
The package provides a shared :class:`PredictiveModel` interface, concrete
regression implementations, and plotting helpers for model interpretation and
sensitivity analysis.

.. rubric:: Model implementations

* :class:`~digital_twinning.predictive_models.PredictiveModel` defines the
  common training, prediction, evaluation, and analysis interface.
* :class:`~digital_twinning.predictive_models.DNN` provides neural-network
  regression.
* :class:`~digital_twinning.predictive_models.GBT` provides gradient-boosted
  tree regression.
* :class:`~digital_twinning.predictive_models.LinReg` provides a linear
  regression baseline.
* :class:`~digital_twinning.predictive_models.gPCE` wraps generalized
  Polynomial Chaos Expansion.
* :class:`~digital_twinning.predictive_models.MOSAIC` provides MOSAIC-based
  surrogate modeling.

.. rubric:: Typical workflow

Prepare feature and target data, create a model with the desired configuration,
train it, and call ``predict`` with new feature data. Implementations inherit
common utilities for data splitting, cross-validation, evaluation, and model
serialization from ``PredictiveModel``. Refer to each model page for its
implementation-specific parameters.

.. toctree::
   :caption: Predictive models
   :maxdepth: 1

   digital_twinning.predictive_models.predictive_model
   digital_twinning.predictive_models.dnn_model
   digital_twinning.predictive_models.gbt_model
   digital_twinning.predictive_models.linreg_model
   digital_twinning.predictive_models.gpce_model
   digital_twinning.predictive_models.mosaic_model
   digital_twinning.predictive_models.plotting_functions
