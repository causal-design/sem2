"""Model-structure construction: the SEM2 formula language frontend.

Pipeline:  formula string --(FormulaParser.parse)--> AST --(lower)--> NodeSpec
           NodeSpec dict --(_DAG)--> graph + topological order + inferred SEM type
"""
