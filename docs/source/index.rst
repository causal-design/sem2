SEM2
====

SEM2 is a research-stage PyTorch package for structural equation models.
Define a model with formulas, prepare named data, and fit it with an explicit
objective. Model-implied interventions do not by themselves establish causal
identification.

.. Main comment: SEM2 has been described very generically in the docs/readme
   - would perhaps be good to explain a bit more, e.g. highlight its generative nature
   that allows for modelling flexibility?
.. not sure if I understand what model-implied interventions mean?
   maybe explain more?

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   api

.. toctree::
   :maxdepth: 2
   :caption: Models and data

   models
   data
   configuration

.. toctree::
   :maxdepth: 1
   :caption: Training

   training

.. toctree::
   :maxdepth: 2
   :caption: Operators

   objectives
   queries
   operators
   operator-contracts

.. toctree::
   :maxdepth: 1
   :caption: Model construction

   formulas
   nodes
