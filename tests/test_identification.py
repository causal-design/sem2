"""Graphical identification against known causal graphs, without fitting."""
import pytest
import torch

from sem2.identification import is_identified, project_to_admg
from sem2.model.models.base import SEM2


@pytest.mark.parametrize('formula, observed, expected', [
    ('A,e ~ Normal(0,1); Y <- A + e', ['A', 'Y'], True),
    ('C,e ~ Normal(0,1); A <- C; Y <- A + C + e', ['C', 'A', 'Y'], True),
    ('U,e ~ Normal(0,1); A <- U; Y <- A + U + e', ['A', 'Y'], False),
    ('U,e ~ Normal(0,1); A <- U; M <- A; Y <- M + U + e', ['A', 'M', 'Y'], True),
], ids=['randomized', 'back-door', 'confounded', 'front-door'])
def test_id_matches_reference_graphs(formula, observed, expected):
    # These are graphical claims: the numerical/linear restrictions are ignored.
    assert is_identified(formula, observed=observed, treatments='A', outcomes='Y') is expected


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


def test_helpers_preserve_scientific_graph_and_do_not_change_model_or_rng():
    formula = 'u,v,w ~ Normal(0,1); x <- n(u,n(v,w)); y <- n(x,w)'
    model = SEM2(formula)
    observed = ['u', 'x', 'y']
    state = torch.random.get_rng_state()
    projected = project_to_admg(model, observed=observed)
    flat = project_to_admg('u,v,w ~ Normal(0,1); x <- n(u,v,w); y <- n(x,w)',
                           observed=observed)
    assert projected == flat
    assert not is_identified(model, observed=observed, treatments='x', outcomes='y')
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.model_formula == formula
    assert any(spec.is_helper for spec in model.node_specs.values())


def test_joint_intervention_can_be_identified_when_a_subintervention_is_not():
    # Shpitser (2023), "When does the ID algorithm fail?", Figure 1(b).
    formula = ('H1,H2 ~ Normal(0,1); W <- H1 + H2; A1 <- W; '
               'A2 <- H2; Y <- A1 + A2 + H1')
    observed = ['W', 'A1', 'A2', 'Y']
    assert is_identified(formula, observed=observed, treatments=['A1', 'A2'], outcomes='Y')
    assert not is_identified(formula, observed=observed, treatments='A2', outcomes='Y')
    assert is_identified(formula, observed=observed, treatments=[], outcomes=['W', 'Y'])


def test_invalid_identification_queries_raise_instead_of_reporting_nonidentification():
    formula = 'u ~ Normal(0,1); exposure <- n(n(u)); outcome <- exposure + u'
    with pytest.raises(ValueError, match='Unknown or internal'):
        project_to_admg(formula, observed=['exposure:0', 'outcome'])
    with pytest.raises(ValueError, match='observed variables'):
        is_identified(formula, observed=['exposure', 'outcome'], treatments='u', outcomes='outcome')
    with pytest.raises(ValueError, match='disjoint'):
        is_identified(formula, observed=['exposure'], treatments='exposure', outcomes='exposure')
    with pytest.raises(ValueError, match='nonempty'):
        is_identified(formula, observed=['exposure'], treatments='exposure', outcomes=[])
    with pytest.raises(TypeError, match='strings'):
        project_to_admg(formula, observed=[1])
