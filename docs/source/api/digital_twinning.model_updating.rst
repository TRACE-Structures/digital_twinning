Bayesian model updating
=======================

The model-updating package combines predictive models with measurement data to
estimate uncertain parameters. ``DigitalTwin`` updates a single predictive
model, while ``JointManager`` coordinates updates across multiple models with
shared parameters.

.. rubric:: Updating workflow

#. Train or otherwise prepare the predictive model.
#. Create an error model describing measurement/model discrepancy.
#. Construct a ``DigitalTwin`` and call ``update`` with measurements and the
   desired MCMC settings.
#. Inspect the posterior samples and summaries, or use ``JointManager`` for a
   coordinated multi-model update.

The plotting module provides posterior visualizations for individual and joint
updates. See the class pages for supported inputs, options, and result
accessors.

.. toctree::
   :caption: Model-updating
   :maxdepth: 1

   digital_twinning.model_updating.digital_twin
   digital_twinning.model_updating.multimodel_update
   digital_twinning.model_updating.plotting_functions