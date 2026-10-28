# Package documentation

The public Sphinx site lives in `source/`. Internal meeting records, plans, and research reports are maintained in the private [sem2-docs repository](https://github.com/causal-design/sem2-docs) and are not included in this repository or the site.

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

Visit http://127.0.0.1:8000. After editing `docs/source/`, rebuild and refresh the browser. Stop the server with Ctrl+C.

## GitHub Pages

The [Documentation workflow](../.github/workflows/docs.yml) builds the Sphinx
site on pull requests targeting `main` and on pushes to `main`. Build warnings
fail the check. It installs the real package with its documentation dependencies
and uses CPU PyTorch to avoid downloading CUDA libraries on the runner.

Pull requests only build the documentation. After a successful build on `main`,
the workflow uploads `docs/_build/html` and deploys it to GitHub Pages. Generated
HTML stays out of Git; a separate `gh-pages` branch is not needed.

### One-time repository administrator setup

1. Open **Settings → Pages → Build and deployment**.
2. Set **Source** to **GitHub Actions**.
3. If the `github-pages` environment restricts deployment branches, allow `main`.
4. Merge the Pages setup pull request. The push to `main` starts the first build
   and deployment. To retry after changing settings, open **Actions →
   Documentation → Run workflow** and select `main`.

The expected public URL is <https://causal-design.github.io/sem2/>. It becomes
available after Pages is enabled and the first deployment succeeds. Contributors
with write access can propose workflow changes, but a repository administrator
must enable Pages. See [GitHub's custom workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## GitHub Wiki: maintainer setup

Enable **Settings → General → Features → Wikis**, then create the wiki's **Home** page using `wiki/Home.md`. The Wiki is stored in GitHub's separate wiki repository; this file is only the initial page content, not an automatic synchronization mechanism. Keep package documentation in Sphinx rather than maintaining two copies.
