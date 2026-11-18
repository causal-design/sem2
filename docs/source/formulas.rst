Formula parsing
===============

Bidirected edges between exogenous variables
--------------------------------------------

Write ``EX <-> EY`` to share a noise source between mechanisms. For example::

    from sem2.model.models.base import SEM2

    model = SEM2("X <- n(EX); Y <- X + n(EY); EX <-> EY")
    draws, _ = model.sample(10)
    assert draws.names == ("X", "Y")

This formula gives the mechanism for ``X`` a private standard-normal draw
and a shared draw. The neural term in ``Y`` gets its own private draw and
the **same** shared draw. ``X + n(EY)`` keeps the noise contribution additive;
``n(X,EY)`` instead lets the network use both inputs jointly.

Endogenous variables are defined by ``<-``. Other variables on the right
are exogenous, with a standard-normal default. An isolated exogenous variable
can have an explicit law, such as ``E ~ Uniform(0,1)``. Ordinary samples
return only endogenous variables. For example, ``Y <- 0 + 1*E`` exposes
an exact copy of an isolated source.

In a chain ``A <-> B <-> C`` only adjacent pairs have edges. The compiler
creates one independent scalar normal source per nonempty bidirected clique:
three private and two shared sources for this chain. Adding ``A <-> C``
creates seven sources, including one shared by all three. Inspect
``model.admg`` for the original vertices and edges and ``model.clique_sources``
for the mapping to executable roots. Dense graphs can have exponentially
many cliques.

An exogenous variable with a bidirected neighbour expands to inputs of
``n(...)`` or ``d(...; levels=k)``. It cannot also have an explicit law or
appear as a linear term (including with a fixed coefficient).
Discrete mechanisms still return probabilities conditional on their inputs:
finite endogenous states are enumerated, and continuous sources are sampled.

Parser API
----------

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
