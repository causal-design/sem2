# Package documentation

The Sphinx site lives in `source/`. Historical meetings, plans and reports are outside the Sphinx source tree and are not included in the site.

From the repository root:

```sh
python -m pip install -e '.[docs]'
python -m sphinx -M html docs/source docs/_build -W --keep-going
```

Open `docs/_build/html/index.html`. The initial site is deliberately small; extend the core API and guides alongside implementation. Autodoc imports the installed package without mocking its dependencies.

## Local preview

To serve the built site on your own computer:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs/_build/html
```

Visit http://127.0.0.1:8000. After editing `docs/source/`, rebuild and refresh the browser. Stop the server with Ctrl+C. GitHub Pages and deployment automation are deferred.

## GitHub Wiki: maintainer setup

Enable **Settings → General → Features → Wikis**, then create the wiki's **Home** page using `wiki/Home.md`. The Wiki is stored in GitHub's separate wiki repository; this file is only the initial page content, not an automatic synchronization mechanism. Keep package documentation in Sphinx rather than maintaining two copies.
