Formula parsing
===============

.. autoclass:: sem2.model.construction.formula_parser.FormulaParser
   :members: parse

.. autoclass:: sem2.model.construction.sem_types.SEMType
   :members:
   :undoc-members:

   .. :undoc-members: currently displays the attributes in an ugly way
      maybe just add the exact attribute names in the itemization in markdown?

Syntax records
--------------

.. autoclass:: sem2.model.construction.sem_ast.Distribution

.. autoclass:: sem2.model.construction.sem_ast.LinearTerm

.. autoclass:: sem2.model.construction.sem_ast.MechanismTerm

.. autoclass:: sem2.model.construction.sem_ast.ExogenousDecl

.. autoclass:: sem2.model.construction.sem_ast.EndogenousDecl

.. autoclass:: sem2.model.construction.sem_ast.NodeSpec

Distribution validation
-----------------------

.. autoclass:: sem2.model.construction.formula_parser.DistributionRegistry
   :members: validate

.. autofunction:: sem2.model.construction.formula_parser.is_list_or_float_like
