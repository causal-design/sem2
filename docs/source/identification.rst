Graphical identification
========================

``is_identified`` checks whether an intervention distribution is identified from
one observed joint distribution under the supplied causal graph. It runs the
Shpitser--Pearl ID algorithm through ``y0``; it does not fit or sample a model.
The dependency is pinned to the tested release, ``y0==0.2.11``.

Supply a model or formula, the observed variable names, and the treatment and
outcome names. All scientific variables outside ``observed`` are treated as
latent. Observation status is local to the call, not stored on the model.

For example, the front-door graph identifies the effect despite an unobserved
common cause of treatment and outcome:

.. code-block:: python

   from sem2.identification import is_identified, project_to_admg

   formula = "U,e ~ Normal(0,1); A <- U; M <- A; Y <- M + U + e"
   observed = ["A", "M", "Y"]
   assert is_identified(formula, observed=observed, treatments="A", outcomes="Y")

   # Hiding the mediator removes the information needed for identification.
   assert not is_identified(formula, observed=["A", "Y"],
                            treatments="A", outcomes="Y")

   graph = project_to_admg(formula, observed=observed)
   # graph.directed stores causal edges; graph.undirected stores bidirected edges.

Treatments and outcomes may each contain several names. They must be disjoint,
observed, user-declared variables; outcomes must be nonempty. An empty treatment
collection requests an ordinary observational marginal. Invalid inputs raise
errors rather than returning ``False``.

Graph semantics and limits
--------------------------

Compiler helpers are inlined before latent projection. The projection preserves
directed paths through latent variables and latent common causes, rather than
simply deleting unobserved vertices. The returned ``y0.graph.NxMixedGraph`` also
retains isolated observed variables.

The check uses declared dependencies only, ignoring coefficients (including fixed
zeros), noise distributions, fitted weights, and deterministic restrictions.
A ``True`` result assumes the causal graph and the usual support conditions are
valid; it does not establish positivity in a dataset. A ``False`` result concerns
the full intervention distribution in the nonparametric graph model. It does not
rule out identification of a particular mean or contrast, or identification
under additional model restrictions such as linearity.

Conditional effects, latent treatment/outcome targets, data fusion, and
numerical bounds are outside this interface. No intervention values or category
scores are needed for this graph-level check.

API
---

.. autofunction:: sem2.identification.project_to_admg

.. autofunction:: sem2.identification.is_identified

References
----------

* Shpitser and Pearl (2006), *Identification of Joint Interventional
  Distributions in Recursive Semi-Markovian Causal Models*.
* `y0 identification API <https://y0.readthedocs.io/en/latest/identification.html>`_.
