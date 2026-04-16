# marimo_gradient_descent_regularization

Interactive marimo app for Lecture 4 of aDAV: fitting coefficients with gradient descent and controlling model complexity with regularization.

## Run locally

```bash
uv sync
sh ./run.sh
```

## Files

- `app.py`: the single-file marimo app
- `pyproject.toml`: local dependencies for running and exporting the app
- `docs/`: GitHub Pages / Pyodide export generated with marimo

## Export to WASM

```bash
uv run python -m marimo export html-wasm app.py -o docs --mode run -f
```

## Live app

After GitHub Pages is enabled for this repository, the app should be available at:

`https://personalwebsite.github.io/marimo_gradient_descent_regularization/`
