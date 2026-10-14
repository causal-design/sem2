Core API
========

Models
------

* ``sem2.model.models.base.SEM2`` — general continuous model.
* ``sem2.model.models.linear.SEM2Linear`` — linear model and analytical moments.
* ``sem2.model.models.discrete.SEM2Discrete`` — finite-state model.

Data and configuration
----------------------

* ``sem2.data.records`` — named data batches, variable metadata and scalar results.
* ``sem2.data.preparation.prepare_data`` — DataFrame preparation and alignment.
* ``sem2.default_configs`` — node and training settings.

Linear-model operators
----------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   sem2.operators.linear.LogLikelihood
   sem2.operators.linear.SEM2LinearModelOperator
   sem2.operators.linear.PathAnalysis

Training
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sem2.trainer.trainer.fit

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   sem2.trainer.trainer.SEM2Trainer
