"""Sphinx configuration for the package documentation."""
import importlib.metadata

project = "SEM2"
release = importlib.metadata.version("sem2")
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]
autosummary_generate = True
templates_path = ["_templates"]
autodoc_typehints = "none"
autoclass_content = "both"
html_theme = "pydata_sphinx_theme"
html_sidebars = {"**": ["grouped-navigation.html"]}
html_theme_options = {"navbar_center": []}
exclude_patterns = []
