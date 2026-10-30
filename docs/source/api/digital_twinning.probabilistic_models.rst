Probabilistic models
====================

Probabilistic models represent conditional output distributions rather than a
single point prediction. They expose an ``inference`` interface for sampling
outputs conditioned on input features and share a common base class.

.. rubric:: Models

* :class:`~digital_twinning.probabilistic_models.ProbabilisticModel` defines
  the shared probabilistic-model contract.
* :class:`~digital_twinning.probabilistic_models.MDN` models a conditional
  distribution with a mixture density network.
* :class:`~digital_twinning.probabilistic_models.CRPS` trains a probabilistic
  neural model using the Continuous Ranked Probability Score.

.. rubric:: Choosing a model

Use ``MDN`` when a mixture representation of the conditional distribution is
appropriate. Use ``CRPS`` when training with the Continuous Ranked Probability
Score is preferred. The model pages document constructors, training options,
and inference methods.

.. toctree::
   :caption: Probabilistic models
   :maxdepth: 1

   digital_twinning.probabilistic_models.probabilistic_model
   digital_twinning.probabilistic_models.mdn_model
   digital_twinning.probabilistic_models.crps_model
   digital_twinning.probabilistic_models.plotting_functions
