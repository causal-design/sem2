Data and results
================

Preparation
-----------

.. autofunction:: sem2.data.preparation.prepare_data

Named values
------------

.. autoclass:: sem2.data.records.VariableInfo

.. autoclass:: sem2.data.records.DataBatch
   :members: names, select, subset, to
   :special-members: __getitem__, __len__

.. autoclass:: sem2.data.records.ScalarResult

Numerical scoring
-----------------

.. autofunction:: sem2.data.conversion.databatch_to_tensor

Synthetic fixture
-----------------

.. autoclass:: sem2.data.dummy.DummyDiscreteDataloader
