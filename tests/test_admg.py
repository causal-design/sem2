import pytest
import torch

from sem2.data.records import DataBatch
from sem2.identification import is_identified, project_to_admg
from sem2.model.models.base import SEM2
from sem2.model.models.discrete import SEM2Discrete


def test_bow_agrees_with_explicit_dag_before_and_after_intervention():
    formula = "X <- n(EX); Y <- X + n(EY); EX <-> EY"
    explicit = "X <- n(a,c); Y <- X + n(b,c); a,b,c ~ Normal(0,1)"
    models = []
    for text in [formula, explicit]:
        torch.manual_seed(11)
        models.append(SEM2(text).eval())
    for intervention in [None, DataBatch(torch.full((8, 1), 2.), ["X"])]:
        draws = []
        for model in models:
            torch.manual_seed(42)
            batch, _ = model.sample(8, do_intervention=intervention)
            assert batch.names == ("X", "Y")
            draws.append(batch.values)
        torch.testing.assert_close(*draws)
    graph = project_to_admg(models[0], observed=["X", "Y"])
    assert [(u.name, v.name) for u, v in graph.directed.edges()] == [("X", "Y")]
    assert [{u.name, v.name} for u, v in graph.undirected.edges()] == [{"X", "Y"}]
    assert not is_identified(formula, observed=["X", "Y"], treatments="X", outcomes="Y")


def test_chain_vs_triangle_and_shared_inputs_in_one_mechanism():
    text = "A <-> B <-> C; Z <- n(A,B,C)"
    dag = SEM2(text)
    expected = {("A",), ("B",), ("C",), ("A", "B"), ("B", "C")}
    assert set(dag.clique_sources) == expected
    triangle = SEM2(text + "; C <-> A; B <-> A")
    assert set(triangle.clique_sources) == expected | {("A", "C"), ("A", "B", "C")}
    assert len(triangle.node_specs["Z"].predecessors) == len(set(triangle.node_specs["Z"].predecessors)) == 7


def test_discrete_sources_are_integrated_conditionally():
    model = SEM2Discrete("X <- d(EX;levels=2); Y <- d(X,EY;levels=3); EX <-> EY")
    dist, _ = model.sample(5)
    assert dist.states.names == ("X", "Y")
    assert dist.probabilities.shape == (5, 6)
    torch.testing.assert_close(dist.probabilities.sum(1), torch.ones(5))


def test_isolated_inferred_noise_matches_declared_standard_normal():
    model = SEM2("Y <- n(E)")
    assert model.admg.endogenous == {"Y"} and model.admg.exogenous == {"E"}
    assert model.node_specs["E"].dist_params == ([0., 1.], {})


@pytest.mark.parametrize(("formula", "error"), [
    ("X <- n(a); Y <- n(b); X <-> b", "exogenous endpoints"),
    ("X <- n(a); Y <- n(b); a <-> b; a ~ Normal(0,1)", "declare a distribution"),
    ("X <- n(a); Y <- 2*b; a <-> b", "linear term"),
    ("X <- n(a); Y <- b; a <-> b", "linear term"),
])
def test_unsupported_noise_declarations(formula, error):
    with pytest.raises(ValueError, match=error):
        SEM2(formula)
