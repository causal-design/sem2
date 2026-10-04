"""Sphinx configuration for the package documentation."""
import importlib.metadata

project = "SEM2"
release = importlib.metadata.version("sem2")
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
autodoc_typehints = "none"
autoclass_content = "both"
html_theme = "alabaster"
exclude_patterns = []
