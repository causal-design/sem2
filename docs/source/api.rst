Core API
========

Models
------

Import model classes from their defining modules:

* ``sem2.model.models.base.SEM2``: general continuous model.
* ``sem2.model.models.linear.SEM2Linear``: linear model and analytical moments.
* ``sem2.model.models.discrete.SEM2Discrete``: finite-state model.

Data and configuration
----------------------

``sem2.data.records`` defines ``DataBatch``, ``ScalarResult`` and
``VariableInfo``. ``sem2.data.preparation.prepare_data`` cleans and aligns a
DataFrame against a model, and ``SEM2Trainer`` turns its values into tensor
minibatches, so there is no separate dataloader entry point. Node and training
configurations live in ``sem2.default_configs``.

Objectives live in ``sem2.operators``; import each from its defining module.
The trainer minimizes an explicitly supplied data-taking scalar objective.

Linear-model operators
----------------------

Classes defined in ``sem2.operators.linear``. Each class has its own page,
including its public methods. ``PathAnalysis`` is not yet implemented.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   sem2.operators.linear.LogLikelihood
   sem2.operators.linear.SEM2LinearModelOperator
   sem2.operators.linear.PathAnalysis

Fitting
-------

Fitting updates the supplied model in place. Each training call creates a
fresh optimizer; it does not reset model parameters or establish convergence.
Instrumental-variable objectives may require observations at every instrument
level in each batch; ordinary shuffled minibatches do not guarantee this.

.. autofunction:: sem2.trainer.trainer.fit

.. autoclass:: sem2.trainer.trainer.SEM2Trainer
   :members: train
