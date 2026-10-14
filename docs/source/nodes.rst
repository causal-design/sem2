Nodes and support
=================

Domains and base interfaces
---------------------------

.. autoclass:: sem2.model.components.nodes.RandomVariableSupport

.. autoclass:: sem2.model.components.nodes.RandomVariable
   :members: support

.. autoclass:: sem2.model.components.nodes.SEM2Node
   :members: forward, set_value, get_value, reset_value

.. autoclass:: sem2.model.components.nodes.NodeEndogenous

.. autoclass:: sem2.model.components.nodes.NodeNonlinear

Exogenous and continuous mechanisms
-----------------------------------

.. autoclass:: sem2.model.components.nodes.NodeExogenous

.. autoclass:: sem2.model.components.nodes.NodeLinear
   :members: effective_weight, intercept

.. autoclass:: sem2.model.components.nodes.NodeMLP

Discrete mechanisms
-------------------

.. autoclass:: sem2.model.components.nodes.NodeMLPDiscrete
   :members: support

.. autoclass:: sem2.model.components.nodes.DiscreteModes
   :members:
   :undoc-members:
