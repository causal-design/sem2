"""Graphical identification against known causal graphs, without fitting."""
import pytest

from sem2.identification import is_identified, project_to_admg
from sem2.model.models.base import SEM2


def test_front_door_identification_depends_on_observing_the_mediator():
    formula = 'U,e ~ Normal(0,1); A <- U; M <- A; Y <- M + U + e'
    assert is_identified(formula, observed=['A', 'M', 'Y'], treatments='A', outcomes='Y')
    assert not is_identified(formula, observed=['A', 'Y'], treatments='A', outcomes='Y')
    # An unsupported latent target is an error, not a nonidentification result.
    with pytest.raises(ValueError, match='observed variables'):
        is_identified(formula, observed=['A', 'Y'], treatments='U', outcomes='Y')


def test_projection_preserves_paths_without_crossing_observed_nodes_or_colliders():
    graph = project_to_admg(
        'U,E,F,I ~ Normal(0,1); A <- U; H <- A; B <- H; '
        'L <- U; C <- L; K <- E + F; Z <- K',
        observed=['A', 'B', 'C', 'E', 'F', 'Z', 'I'],
    )
    assert {node.name for node in graph.nodes()} == {'A', 'B', 'C', 'E', 'F', 'Z', 'I'}
    assert {(a.name, b.name) for a, b in graph.directed.edges()} == {
        ('A', 'B'), ('E', 'Z'), ('F', 'Z'),
    }
    # U confounds A and C, but the observed A blocks projection through to B.
    # Marginalizing the collider K must not induce an E <-> F edge.
    assert {frozenset((a.name, b.name)) for a, b in graph.undirected.edges()} == {
        frozenset(('A', 'C')),
    }


def test_helpers_preserve_scientific_graph():
    formula = 'u,v,w ~ Normal(0,1); x <- n(u,n(v,w)); y <- n(x,w)'
    model = SEM2(formula)
    observed = ['u', 'x', 'y']

    projected = project_to_admg(model, observed=observed)
    flat = project_to_admg('u,v,w ~ Normal(0,1); x <- n(u,v,w); y <- n(x,w)',
                           observed=observed)
    assert projected == flat
    assert not is_identified(model, observed=observed, treatments='x', outcomes='y')


def test_joint_intervention_can_be_identified_when_a_subintervention_is_not():
    # Shpitser (2023), "When does the ID algorithm fail?", Figure 1(b).
    formula = ('H1,H2 ~ Normal(0,1); W <- H1 + H2; A1 <- W; '
               'A2 <- H2; Y <- A1 + A2 + H1')
    observed = ['W', 'A1', 'A2', 'Y']
    assert is_identified(formula, observed=observed, treatments=['A1', 'A2'], outcomes='Y')
    assert not is_identified(formula, observed=observed, treatments='A2', outcomes='Y')
