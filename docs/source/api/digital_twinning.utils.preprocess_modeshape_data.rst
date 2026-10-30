Mode-shape and weather-data preprocessing
=========================================

Tools for reading and transforming modal data, preparing weather observations,
merging datasets, and selecting modeling inputs and targets. The module also
contains helpers for comparing, plotting, and exporting mode-shape data.

.. rubric:: Workflow

#. Read mode-shape data from a UNV file or construct a compatible data frame.
#. Normalize and preprocess weather data if it is part of the feature set.
#. Merge the datasets, then select modes and columns for the modeling task.
#. Use the export helpers when transformed mode-shape results need to be
   serialized or written to UNV.

.. currentmodule:: digital_twinning.utils.preprocess_modeshape_data

.. automodule:: digital_twinning.utils.preprocess_modeshape_data
   :members:
   :member-order: bysource
