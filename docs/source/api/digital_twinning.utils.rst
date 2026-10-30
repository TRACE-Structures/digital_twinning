Utilities
=========

The utility package contains data-preparation helpers for structural
measurements, mode-shape processing, parameter distributions, and plotting.
Utilities are grouped by their primary task:

* **Data utilities** provide distribution calculations and construction of
  ``VariableSet`` objects from data.
* **Mode-shape preprocessing** reads and transforms modal and weather data,
  merges datasets, and selects modes or columns for modeling.
* **Plotting utilities** visualize mode shapes and sensor data.

.. rubric:: Suggested data-preparation sequence

For modal-analysis workflows, read or construct the mode-shape data, preprocess
weather data when needed, merge the datasets, and then select the modes and
columns used by a model. Consult the preprocessing reference for function
arguments and supported data layouts.

.. toctree::
   :caption: Utilities
   :maxdepth: 1

   digital_twinning.utils.data_utils
   digital_twinning.utils.preprocess_modeshape_data
   digital_twinning.utils.plotting_functions