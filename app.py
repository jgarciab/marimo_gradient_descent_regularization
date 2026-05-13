import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="full",
    app_title="Fitting Models and Controlling Complexity: Gradient Descent and Regularization",
)


@app.cell
def imports():
    import marimo as mo

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#667788",
            "axes.labelcolor": "#2F3441",
            "xtick.color": "#445160",
            "ytick.color": "#445160",
            "text.color": "#2F3441",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#D3DCE4",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Trebuchet MS", "DejaVu Sans", "Arial"],
            "font.size": 10.1,
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return mo, np, plt


@app.cell
def theme():
    train_color = "#2A6F97"
    valid_color = "#C9791B"
    chosen_color = "#3F7D52"
    lasso_color = "#B75A3C"
    data_color = "#6B7280"
    path_color = "#A65E46"
    mse_color = "#4C6F91"
    penalty_color = "#B98932"
    cost_color = "#4B5563"
    split_a_color = "#C9791B"
    split_b_color = "#6C8BA4"
    cv_color = chosen_color
    coef_palette = ["#3F7D52", "#5F86A8", "#B75A3C", "#8A7E45", "#8B94A3"]

    style = """
    <style>
    html, body, div, span, p, li, label, button, input, select, textarea, h1, h2, h3, h4 {
      font-family: "Segoe UI", "Trebuchet MS", "DejaVu Sans", Arial, sans-serif !important;
      text-align: left !important;
    }

    .markdown h1,
    .markdown h2,
    .markdown h3,
    .markdown h4 {
      text-align: left !important;
      margin-top: 0.04rem !important;
      margin-bottom: 0.04rem !important;
      line-height: 1.2 !important;
    }

    .markdown h1 {
      font-size: 2.05rem !important;
      font-weight: 650 !important;
    }

    .markdown p,
    .markdown ul,
    .markdown ol {
      text-align: left !important;
      margin-top: 0.04rem !important;
      margin-bottom: 0.12rem !important;
      line-height: 1.45 !important;
    }

    .markdown li {
      margin: 0.05rem 0 !important;
    }

    .markdown ul,
    .markdown ol {
      padding-left: 1.15rem !important;
    }

    .results-panel {
      display: grid;
      gap: 0.34rem;
      margin-top: 0.04rem;
    }

    .results-card {
      border: 1px solid #D9E1E7;
      border-radius: 10px;
      padding: 0.55rem 0.65rem;
      background: #FAFCFE;
    }

    .results-heading {
      font-size: 0.84rem;
      font-weight: 600;
      color: #5D6B78;
      margin-bottom: 0.35rem;
      letter-spacing: 0.01em;
    }

    .results-row {
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: baseline;
      margin: 0.12rem 0;
    }

    .results-row + .results-row {
      border-top: 1px solid #EDF2F6;
      padding-top: 0.22rem;
    }

    .results-label {
      color: #5D6B78;
      font-size: 0.84rem;
    }

    .results-value {
      color: #1F2A36;
      font-weight: 600;
      font-size: 0.96rem;
      text-align: right;
    }
    </style>
    """
    return (
        chosen_color,
        coef_palette,
        cost_color,
        cv_color,
        data_color,
        lasso_color,
        mse_color,
        path_color,
        penalty_color,
        split_a_color,
        split_b_color,
        style,
        train_color,
        valid_color,
    )


@app.cell
def demo_data(np):
    x_one = np.array([0.0, 1.0, 2.0, 3.0])
    y_one = np.array([0.0, 1.0, 2.0, 3.0])
    x_two = np.array([0.0, 1.0, 2.0, 3.0])
    y_two = np.array([1.0, 2.0, 3.0, 4.0])
    return x_one, x_two, y_one, y_two


@app.cell
def optimization_helpers(np, x_one, x_two, y_one, y_two):
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    def mse_1p(beta1: float) -> float:
        return mse(y_one, beta1 * x_one)

    def grad_1p(beta1: float) -> float:
        residual = y_one - beta1 * x_one
        return float(-2.0 * np.mean(x_one * residual))

    def run_gd_1p(start: float, alpha: float, steps: int, bound: float = 14.0) -> np.ndarray:
        path = [float(start)]
        for _ in range(int(steps)):
            current = path[-1]
            next_beta = current - alpha * grad_1p(current)
            if abs(next_beta) > bound:
                break
            path.append(next_beta)
        return np.asarray(path)

    def mse_2p(beta0: float, beta1: float) -> float:
        return mse(y_two, beta0 + beta1 * x_two)

    def grad_2p(beta0: float, beta1: float) -> tuple[float, float]:
        residual = y_two - beta0 - beta1 * x_two
        grad0 = float(-2.0 * np.mean(residual))
        grad1 = float(-2.0 * np.mean(x_two * residual))
        return grad0, grad1

    def run_gd_2p(
        start0: float,
        start1: float,
        alpha: float,
        steps: int,
        bound: float = 10.0,
    ) -> np.ndarray:
        path = [(float(start0), float(start1))]
        for _ in range(int(steps)):
            beta0, beta1 = path[-1]
            grad0, grad1 = grad_2p(beta0, beta1)
            next0 = beta0 - alpha * grad0
            next1 = beta1 - alpha * grad1
            if abs(next0) > bound or abs(next1) > bound:
                break
            path.append((next0, next1))
        return np.asarray(path)

    return grad_1p, mse, mse_1p, mse_2p, run_gd_1p, run_gd_2p


@app.cell
def resampling_helpers(np):
    def split_indices(
        n_samples: int,
        train_frac: float,
        valid_frac: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_samples)
        n_train = int(np.floor(n_samples * train_frac))
        n_valid = int(np.floor(n_samples * valid_frac))
        train_idx = np.sort(order[:n_train])
        valid_idx = np.sort(order[n_train:n_train + n_valid])
        test_idx = np.sort(order[n_train + n_valid:])
        return train_idx, valid_idx, test_idx

    def kfold_indices(n_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_samples)
        folds = np.array_split(order, n_splits)
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_index in range(n_splits):
            valid_idx = np.sort(folds[fold_index])
            train_idx = np.sort(np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_index]))
            rows.append((train_idx, valid_idx))
        return rows

    return kfold_indices, split_indices


@app.cell
def regularization_helpers(mse_1p, mse_2p, np):
    def penalty_raw(values: np.ndarray, penalty: str) -> float:
        values = np.asarray(values, dtype=float)
        if penalty == "ridge":
            return float(np.sum(values**2))
        return float(np.sum(np.abs(values)))

    def penalty_contribution(values: np.ndarray, lambda_value: float, penalty: str) -> float:
        return float(lambda_value * penalty_raw(values, penalty))

    def cost_1p(beta1: float, lambda_value: float, penalty: str) -> float:
        return mse_1p(beta1) + penalty_contribution(np.array([beta1]), lambda_value, penalty)

    def cost_2p(beta0: float, beta1: float, lambda_value: float, penalty: str) -> float:
        return mse_2p(beta0, beta1) + penalty_contribution(np.array([beta0, beta1]), lambda_value, penalty)

    def prepare_features(X: np.ndarray, standardize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        means = X.mean(axis=0)
        centered = X - means
        scales = X.std(axis=0, ddof=0)
        scales[scales == 0.0] = 1.0
        if standardize:
            transformed = centered / scales
        else:
            transformed = centered
            scales = np.ones(X.shape[1], dtype=float)
        return transformed, means, scales

    def soft_threshold(value: float, gamma: float) -> float:
        if value > gamma:
            return value - gamma
        if value < -gamma:
            return value + gamma
        return 0.0

    def fit_ols(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float | bool]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return {"intercept": float(coef[0]), "beta": np.asarray(coef[1:], dtype=float), "converged": True}

    def fit_ridge(
        X: np.ndarray,
        y: np.ndarray,
        lambda_value: float,
        standardize: bool = True,
    ) -> dict[str, np.ndarray | float | bool]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        transformed, means, scales = prepare_features(X, standardize=standardize)
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        n_features = transformed.shape[1]
        gram = transformed.T @ transformed / len(X)
        rhs = transformed.T @ y_centered / len(X)
        beta_scaled = np.linalg.solve(gram + lambda_value * np.eye(n_features), rhs)
        beta = beta_scaled / scales
        intercept = y_mean - means @ beta
        return {
            "intercept": float(intercept),
            "beta": beta,
            "beta_scaled": beta_scaled,
            "converged": True,
        }

    def fit_lasso(
        X: np.ndarray,
        y: np.ndarray,
        lambda_value: float,
        standardize: bool = True,
        max_iter: int = 6000,
        tol: float = 1e-6,
    ) -> dict[str, np.ndarray | float | bool]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        transformed, means, scales = prepare_features(X, standardize=standardize)
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        n_features = transformed.shape[1]
        beta = np.zeros(n_features, dtype=float)
        converged = False
        for _ in range(max_iter):
            previous = beta.copy()
            fitted = transformed @ beta
            for column in range(n_features):
                residual = y_centered - fitted + transformed[:, column] * beta[column]
                rho = float(np.mean(transformed[:, column] * residual))
                z_value = float(np.mean(transformed[:, column] ** 2))
                new_beta = soft_threshold(rho, lambda_value) / z_value
                fitted = fitted + transformed[:, column] * (new_beta - beta[column])
                beta[column] = new_beta
            if np.max(np.abs(beta - previous)) < tol:
                converged = True
                break
        beta_unscaled = beta / scales
        intercept = y_mean - means @ beta_unscaled
        return {
            "intercept": float(intercept),
            "beta": beta_unscaled,
            "beta_scaled": beta.copy(),
            "converged": converged,
        }

    def fit_model(
        X: np.ndarray,
        y: np.ndarray,
        penalty: str,
        lambda_value: float,
        standardize: bool = True,
    ) -> dict[str, np.ndarray | float | bool]:
        if penalty == "ridge":
            return fit_ridge(X, y, lambda_value=lambda_value, standardize=standardize)
        if penalty == "lasso":
            return fit_lasso(X, y, lambda_value=lambda_value, standardize=standardize)
        raise ValueError(f"Unknown penalty: {penalty}")

    def predict_linear(model: dict[str, np.ndarray | float | bool], X: np.ndarray) -> np.ndarray:
        return float(model["intercept"]) + np.asarray(X, dtype=float) @ np.asarray(model["beta"], dtype=float)

    def selected_variables(beta: np.ndarray, names: list[str], threshold: float = 1e-5) -> list[str]:
        return [name for name, value in zip(names, beta) if abs(value) > threshold]

    def coefficient_summary(beta: np.ndarray, names: list[str], digits: int = 2) -> str:
        values = np.asarray(beta, dtype=float)
        return ", ".join(f"{name}: {value:.{digits}f}" for name, value in zip(names, values))

    return (
        coefficient_summary,
        cost_1p,
        cost_2p,
        fit_lasso,
        fit_model,
        penalty_contribution,
        predict_linear,
        selected_variables,
    )


@app.cell
def simulation_helpers(np):
    def simulate_selection_data(
        n_samples: int = 140,
        seed: int = 2026,
    ) -> dict[str, np.ndarray | list[str]]:
        rng = np.random.default_rng(seed)
        x_signal = rng.normal(size=n_samples)
        x_copy = 0.88 * x_signal + 0.35 * rng.normal(size=n_samples)
        x_noise_1 = rng.normal(size=n_samples)
        x_noise_2 = rng.normal(size=n_samples)
        x_noise_3 = rng.normal(size=n_samples)
        X = np.column_stack([x_signal, x_copy, x_noise_1, x_noise_2, x_noise_3])
        beta_true = np.array([1.8, 0.0, 0.0, 0.0, 0.0])
        y = 1.0 + X @ beta_true + rng.normal(0.0, 1.8, size=n_samples)
        names = ["Signal", "Copy", "Noise 1", "Noise 2", "Noise 3"]
        return {"X": X, "y": y, "beta_true": beta_true, "names": names}

    def simulate_scaled_data(
        n_samples: int = 180,
        seed: int = 42,
    ) -> dict[str, np.ndarray | list[str]]:
        rng = np.random.default_rng(seed)
        latent = rng.normal(size=(n_samples, 4))
        latent = (latent - latent.mean(axis=0)) / latent.std(axis=0, ddof=0)
        signal_small = latent[:, 0]
        signal_large = 100.0 * latent[:, 1]
        noise_small = latent[:, 2]
        noise_large = 100.0 * latent[:, 3]
        y = 2.4 * latent[:, 0] + 2.4 * latent[:, 1] + rng.normal(0.0, 0.8, size=n_samples)
        X = np.column_stack([signal_small, signal_large, noise_small, noise_large])
        names = ["a (signal)", "b (signal)", "c (noise)", "d (noise)"]
        return {"X": X, "y": y, "names": names}

    return simulate_scaled_data, simulate_selection_data


@app.cell
def selection_helpers(
    fit_model,
    kfold_indices,
    mse,
    np,
    predict_linear,
    split_indices,
):
    def evaluate_lambda_grid(
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray,
        penalty: str,
        split_seed: int,
        standardize: bool = True,
    ) -> tuple[list[dict[str, float]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        train_idx, valid_idx, test_idx = split_indices(len(X), train_frac=0.60, valid_frac=0.20, seed=split_seed)
        rows: list[dict[str, float]] = []
        for lambda_value in lambdas:
            model = fit_model(X[train_idx], y[train_idx], penalty=penalty, lambda_value=float(lambda_value), standardize=standardize)
            rows.append(
                {
                    "lambda": float(lambda_value),
                    "train_mse": mse(y[train_idx], predict_linear(model, X[train_idx])),
                    "validation_mse": mse(y[valid_idx], predict_linear(model, X[valid_idx])),
                    "test_mse": mse(y[test_idx], predict_linear(model, X[test_idx])),
                }
            )
        return rows, (train_idx, valid_idx, test_idx)

    def evaluate_cv_grid(
        X: np.ndarray,
        y: np.ndarray,
        lambdas: np.ndarray,
        penalty: str,
        n_splits: int,
        seed: int,
        standardize: bool = True,
    ) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for lambda_value in lambdas:
            fold_mses = []
            for train_idx, valid_idx in kfold_indices(len(X), n_splits=n_splits, seed=seed):
                model = fit_model(
                    X[train_idx],
                    y[train_idx],
                    penalty=penalty,
                    lambda_value=float(lambda_value),
                    standardize=standardize,
                )
                fold_mses.append(mse(y[valid_idx], predict_linear(model, X[valid_idx])))
            rows.append({"lambda": float(lambda_value), "cv_mse": float(np.mean(fold_mses))})
        return rows

    def coefficient_path_rows(
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        lambdas: np.ndarray,
        penalty: str,
        standardize: bool = True,
    ) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for lambda_value in lambdas:
            model = fit_model(X, y, penalty=penalty, lambda_value=float(lambda_value), standardize=standardize)
            for name, value in zip(names, np.asarray(model["beta"], dtype=float)):
                rows.append({"lambda": float(lambda_value), "name": name, "beta": float(value)})
        return rows

    return (
        coefficient_path_rows,
        evaluate_cv_grid,
        evaluate_lambda_grid,
    )


@app.cell
def ui_helpers(mo):
    def finish_figure(fig):
        fig.tight_layout(pad=0.75, w_pad=0.9, h_pad=0.9)
        return fig

    def counter_button(label: str, kind: str):
        return mo.ui.button(
            value=0,
            on_click=lambda value: value + 1,
            label=label,
            kind=kind,
            full_width=True,
        )

    def two_col(main: object, side: object):
        return mo.hstack([side, main], widths=[1.02, 1.68], gap=0.65, align="start", wrap=True)

    def section_md(title: str, kicker: str, body: str = ""):
        blocks = [mo.md(f"## {title}"), mo.md(f"*{kicker}*")]
        if body:
            blocks.append(mo.md(body))
        return mo.vstack(blocks, gap=0.03)

    def metrics_md(metrics: list[tuple[str, str]]):
        if not metrics:
            return mo.md("")
        rows = "".join(
            f"<div class='results-row'><div class='results-label'>{label}</div><div class='results-value'>{value}</div></div>"
            for label, value in metrics
        )
        return mo.Html(f"<div class='results-panel'><div class='results-card'><div class='results-heading'>Results</div>{rows}</div></div>")

    def takeaway_md(text: str):
        return mo.md(f"**Takeaway.** {text}")

    def note_md(text: str):
        return mo.md(f"_{text}_")

    def questions_md(questions: list[str]):
        items = "\n".join(f"- {question}" for question in questions)
        return mo.vstack([mo.md("**Make sure you can answer these questions**"), mo.md(items)], gap=0.02)

    def sidebar(widgets: list[object], metrics: list[tuple[str, str]]):
        blocks: list[object] = [mo.md("**Controls**"), *widgets]
        if metrics:
            blocks.append(metrics_md(metrics))
        return mo.vstack(blocks, gap=0.20, align="stretch")

    return (
        counter_button,
        finish_figure,
        note_md,
        questions_md,
        section_md,
        sidebar,
        takeaway_md,
        two_col,
    )


@app.cell
def intro(mo, style):
    _page = mo.vstack(
        [
            mo.Html(style),
            mo.md("# Fitting Models and Controlling Complexity: Gradient Descent and Regularization"),
            mo.md(
                "This app builds intuition about two linked tasks in predictive modeling:\n\n"
                "- Estimate the coefficients for a specified model.\n"
                "- Tune model complexity to find a good balance between bias and variance."
            ),
            mo.md(
                "In the first sections, the model is already specified: for example, we have decided to use a straight line with a particular predictor. The task is then to estimate the coefficient values. Later sections ask a different question: how flexible should the model be? Regularization, validation prediction error, cross-validation, and standardization help us tune that flexibility so we do not underfit or overfit."
            ),
            mo.md(
                "How to use it: move one control at a time, watch which plot or result changes, and then read the note, takeaway, and questions. Where a `Recreate data` button appears, it draws a new synthetic dataset, so small differences across clicks are part of the lesson."
            ),
        ],
        gap=0.45,
    )
    _page
    return


@app.cell
def s1_controls(mo):
    s1_lr = mo.ui.slider(0.01, 0.50, value=0.10, step=0.01, label="Learning rate")
    s1_steps = mo.ui.slider(0, 25, value=10, step=1, label="Gradient descent steps")
    return s1_lr, s1_steps


@app.cell
def s1_section(
    chosen_color,
    cost_color,
    data_color,
    finish_figure,
    grad_1p,
    mo,
    mse_1p,
    mse_color,
    note_md,
    np,
    path_color,
    plt,
    questions_md,
    run_gd_1p,
    s1_lr,
    s1_steps,
    section_md,
    sidebar,
    takeaway_md,
    two_col,
    x_one,
    y_one,
):
    _path = run_gd_1p(start=-0.5, alpha=float(s1_lr.value), steps=int(s1_steps.value))
    _current_beta = float(_path[-1])
    _diverged = len(_path) < int(s1_steps.value) + 1
    _beta_grid = np.linspace(-0.5, 2.5, 320)
    _loss_grid = np.array([mse_1p(beta) for beta in _beta_grid])

    _fig, (_ax_fit, _ax_loss) = plt.subplots(1, 2, figsize=(6.0, 2.65))
    _x_line = np.linspace(-0.2, 3.2, 120)
    _ax_fit.scatter(x_one, y_one, color=data_color, s=55, zorder=5)
    _ax_fit.plot(_x_line, _current_beta * _x_line, color=chosen_color, linewidth=2.6)
    _ax_fit.set(xlabel="x", ylabel="y", xlim=(-0.2, 3.2), ylim=(-1.8, 4.1))

    _ax_loss.plot(_beta_grid, _loss_grid, color=mse_color, linewidth=2.0)
    _ax_loss.plot(_path, [mse_1p(beta) for beta in _path], "o-", color=path_color, linewidth=1.6, markersize=3.8)
    _ax_loss.scatter([_current_beta], [mse_1p(_current_beta)], color=chosen_color, s=62, zorder=6)
    _ax_loss.axvline(1.0, color=cost_color, linestyle="--", linewidth=1.7)
    _ax_loss.set(xlabel=r"$\beta_1$", ylabel="MSE", xlim=(-0.5, 2.5), ylim=(-0.05, 5.2))
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s1_lr, s1_steps],
        metrics=[
            ("Current beta1", f"{_current_beta:.2f}"),
            ("Current MSE", f"{mse_1p(_current_beta):.2f}"),
            ("Gradient", f"{grad_1p(_current_beta):.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "1. Model fitting: Gradient Descent with One Coefficient",
                "Let's start with the simplest case: one coefficient and one loss curve.",
                (
                    "The goal is to estimate the coefficients of a pre-specified model:\n\n"
                    r"\[\hat{y} = \beta_1 x \quad (\beta_0 = 0)\]"
                    "\n\n"
                    "Gradient descent is a rule for moving a coefficient in the direction that lowers MSE.\n\n"
                    "This section shows the same information in two views: the fitted line on the left and the loss curve on the right. You will experience how gradient descent moves the coefficient towards the minimum using the update rule:\n\n"
                    r"\[\beta_1^{(t+1)} = \beta_1^{(t)} - \alpha \,\frac{\partial \mathrm{MSE}}{\partial \beta_1}\!\left(\beta_1^{(t)}\right)\]"
                    "\n\n"
                    "Try this first: increase the number of steps, then change the learning rate. Changing the learning rate changes the path, not the location of the minimum."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md(
                "A very large learning rate makes the updates jump past the useful region."
                if _diverged
                else "The path shows repeated downhill updates on one fixed objective."
            ),
            takeaway_md("Optimization is about how we reach the minimum. It does not change which coefficient gives the lowest MSE."),
            questions_md(
                [
                    "What does gradient descent change at each step: the coefficient, the data, or the loss function?",
                    "How does the learning rate control the step size, and what happens when it is too small or too large?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s2_controls(mo):
    s2_lr = mo.ui.slider(0.01, 0.50, value=0.08, step=0.01, label="Learning rate")
    s2_steps = mo.ui.slider(0, 30, value=12, step=1, label="Gradient descent steps")
    return s2_lr, s2_steps


@app.cell
def s2_section(
    chosen_color,
    cost_color,
    data_color,
    mo,
    mse_2p,
    note_md,
    np,
    path_color,
    plt,
    run_gd_2p,
    s2_lr,
    s2_steps,
    section_md,
    sidebar,
    takeaway_md,
    two_col,
    x_two,
    y_two,
):
    _path = run_gd_2p(start0=-1.0, start1=-0.5, alpha=float(s2_lr.value), steps=int(s2_steps.value))
    _current_beta0, _current_beta1 = _path[-1]
    _beta0_grid = np.linspace(-1.0, 2.0, 300)
    _beta1_grid = np.linspace(-1.0, 2.2, 300)
    _beta0_mesh, _beta1_mesh = np.meshgrid(_beta0_grid, _beta1_grid)
    _mse_surface = np.zeros_like(_beta0_mesh)
    for _index in range(len(x_two)):
        _mse_surface += (y_two[_index] - _beta0_mesh - _beta1_mesh * x_two[_index]) ** 2
    _mse_surface /= len(x_two)

    _fig, (_ax_fit, _ax_contour) = plt.subplots(1, 2, figsize=(6.25, 2.8))
    _x_line = np.linspace(-0.2, 3.2, 120)
    _ax_fit.scatter(x_two, y_two, color=data_color, s=55, zorder=5)
    _ax_fit.plot(_x_line, _current_beta0 + _current_beta1 * _x_line, color=chosen_color, linewidth=2.6)
    _ax_fit.set(xlabel="x", ylabel="y", xlim=(-0.2, 3.2), ylim=(-2.0, 5.2))

    _levels = np.linspace(0.0, 20.0, 21)
    _contour_fill = _ax_contour.contourf(_beta0_mesh, _beta1_mesh, _mse_surface, levels=_levels, cmap="Blues", alpha=0.88, extend="max")
    _ax_contour.contour(_beta0_mesh, _beta1_mesh, _mse_surface, levels=_levels, colors="#4E79A7", linewidths=0.34, alpha=0.52)
    _ax_contour.plot(_path[:, 0], _path[:, 1], "o-", color=path_color, linewidth=1.55, markersize=3.6)
    _ax_contour.scatter([1.0], [1.0], color=cost_color, marker="*", s=92, zorder=6)
    _ax_contour.scatter([_current_beta0], [_current_beta1], color=chosen_color, s=52, zorder=6)
    _ax_contour.set(xlabel=r"$\beta_0$", ylabel=r"$\beta_1$")
    _fig.tight_layout(pad=0.75, w_pad=0.75, h_pad=0.80, rect=[0.0, 0.0, 0.92, 1.0])
    _cbar_ax = _fig.add_axes([0.935, 0.21, 0.018, 0.58])
    _cbar = _fig.colorbar(_contour_fill, cax=_cbar_ax)
    _cbar.set_label("MSE", fontsize=8)
    _cbar.ax.tick_params(labelsize=7)

    _sidebar = sidebar(
        widgets=[s2_lr, s2_steps],
        metrics=[
            ("Current beta0", f"{_current_beta0:.2f}"),
            ("Current beta1", f"{_current_beta1:.2f}"),
            ("Current MSE", f"{mse_2p(_current_beta0, _current_beta1):.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "2. Model fitting: Gradient Descent with Two Coefficients",
                "The loss is now a surface, but we are still fitting coefficients.",
                (
                    "Here too the model form is fixed:\n\n"
                    r"\[\hat{y} = \beta_0 + \beta_1 x\]"
                    "\n\n"
                    "With two coefficients, the optimization picture changes from a curve to contours. That makes it easier to see why the path can bend or zig-zag."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("The star marks the lowest-MSE coefficients. The green point shows where the algorithm is after the selected number of steps."),
            takeaway_md("Going from one coefficient to two adds geometry, not a new optimization principle."),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s3_controls(mo):
    s3_penalty = mo.ui.radio(options=["Ridge (L2)", "LASSO (L1)"], value="Ridge (L2)", label="Penalty")
    s3_lambda = mo.ui.slider(0.0, 10.0, value=1.00, step=0.05, label="Lambda")
    return s3_lambda, s3_penalty


@app.cell
def s3_section(
    chosen_color,
    cost_1p,
    cost_color,
    data_color,
    finish_figure,
    mo,
    mse_1p,
    mse_color,
    note_md,
    np,
    penalty_color,
    penalty_contribution,
    plt,
    questions_md,
    s3_lambda,
    s3_penalty,
    section_md,
    sidebar,
    takeaway_md,
    two_col,
    x_one,
    y_one,
):
    _penalty = "ridge" if s3_penalty.value == "Ridge (L2)" else "lasso"
    _lambda_value = float(s3_lambda.value)
    _beta_grid = np.linspace(-1.0, 2.5, 420)
    _mse_grid = np.array([mse_1p(beta) for beta in _beta_grid])
    _penalty_grid = np.array([penalty_contribution(np.array([beta]), _lambda_value, _penalty) for beta in _beta_grid])
    _cost_grid = np.array([cost_1p(beta, _lambda_value, _penalty) for beta in _beta_grid])
    _best_beta = float(_beta_grid[np.argmin(_cost_grid)])
    _y_limit = 20.0

    _fig, _axes = plt.subplots(2, 2, figsize=(6.0, 4.5))
    _x_line = np.linspace(-0.2, 3.2, 120)
    _axes[0, 0].scatter(x_one, y_one, color=data_color, s=50, zorder=5)
    _axes[0, 0].plot(_x_line, _best_beta * _x_line, color=chosen_color, linewidth=2.5)
    _axes[0, 0].set(xlabel="x", ylabel="y", xlim=(-0.2, 3.2), ylim=(-0.9, 4.1))

    _axes[0, 1].plot(_beta_grid, _cost_grid, color=cost_color, linewidth=2.0)
    _axes[0, 1].axvline(_best_beta, color=chosen_color, linestyle="--", linewidth=1.6)
    _axes[0, 1].set(xlabel=r"$\beta_1$", ylabel="Cost", xlim=(-1.0, 2.5), ylim=(0.0, _y_limit))

    _axes[1, 0].plot(_beta_grid, _mse_grid, color=mse_color, linewidth=2.0)
    _axes[1, 0].axvline(_best_beta, color=chosen_color, linestyle="--", linewidth=1.6)
    _axes[1, 0].set(xlabel=r"$\beta_1$", ylabel="MSE", xlim=(-1.0, 2.5), ylim=(0.0, _y_limit))

    _axes[1, 1].plot(_beta_grid, _penalty_grid, color=penalty_color, linewidth=2.0)
    _axes[1, 1].axvline(_best_beta, color=chosen_color, linestyle="--", linewidth=1.6)
    _axes[1, 1].set(xlabel=r"$\beta_1$", ylabel=r"$\lambda \times$ penalty", xlim=(-1.0, 2.5), ylim=(0.0, _y_limit))
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s3_penalty, s3_lambda],
        metrics=[
            ("Optimal beta1", f"{_best_beta:.2f}"),
            ("MSE at optimum", f"{mse_1p(_best_beta):.2f}"),
            ("λ × penalty", f"{penalty_contribution(np.array([_best_beta]), _lambda_value, _penalty):.2f}"),
            ("Cost at optimum", f"{cost_1p(_best_beta, _lambda_value, _penalty):.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "3. Regularization: Adding a Penalty to the Loss",
                "Regularization adds a penalty term to the objective. It does not replace MSE.",
                (
                    "The model form is still fixed:\n\n"
                    r"\[\hat{y} = \beta_1 x \quad (\beta_0 = 0)\]"
                    "\n\n"
                    "We still want to find the best coefficient value. What changes is the loss we minimize:\n\n"
                    r"\[\text{Loss} = \mathrm{MSE} + \lambda \times \text{penalty}\]"
                    "\n\n"
                    "We can minimize this loss with gradient descent or with other optimization methods. MSE still measures how well the line matches the observed data. Regularization adds a second term that discourages large coefficients. In plain language, the model now has to balance fit (MSE) on training data against coefficient size.\n\n"
                    "Changing $\\lambda$ does not move the MSE curve. It changes how strongly coefficient size is punished."
                ),
            ),
            mo.md(r"$L_{\mathrm{ridge}}(\beta_1) = \mathrm{MSE}(\beta_1) + \lambda \beta_1^2$"),
            mo.md(r"$L_{\mathrm{lasso}}(\beta_1) = \mathrm{MSE}(\beta_1) + \lambda |\beta_1|$"),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("The three objective panels use the same y-axis. This makes the penalty visible as a real part of the cost, not just an abstract formula."),
            takeaway_md("$\\lambda$ only acts through the penalty term. That is why the cost minimum can move even though the MSE curve stays the same."),
            questions_md(
                [
                    "When $\\lambda$ gets larger, what happens to the penalty term?",
                    "Why do larger $\\lambda$ values pull coefficients toward zero?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s4_controls(mo):
    s4_penalty = mo.ui.radio(options=["Ridge (L2)", "LASSO (L1)"], value="LASSO (L1)", label="Penalty")
    s4_lambda = mo.ui.slider(0.0, 10.0, value=1.00, step=0.05, label="Lambda")
    return s4_lambda, s4_penalty


@app.cell
def s4_section(
    chosen_color,
    cost_2p,
    data_color,
    mo,
    mse_2p,
    note_md,
    np,
    plt,
    questions_md,
    s4_lambda,
    s4_penalty,
    section_md,
    sidebar,
    takeaway_md,
    two_col,
    x_two,
    y_two,
):
    _penalty = "ridge" if s4_penalty.value == "Ridge (L2)" else "lasso"
    _lambda_value = float(s4_lambda.value)

    _beta0_grid = np.linspace(-1.0, 1.55, 320)
    _beta1_grid = np.linspace(-1.0, 1.75, 320)
    _beta0_mesh, _beta1_mesh = np.meshgrid(_beta0_grid, _beta1_grid)
    _mse_surface = np.zeros_like(_beta0_mesh)
    for _index in range(len(x_two)):
        _mse_surface += (y_two[_index] - _beta0_mesh - _beta1_mesh * x_two[_index]) ** 2
    _mse_surface /= len(x_two)
    if _penalty == "ridge":
        _penalty_surface = _lambda_value * (_beta0_mesh**2 + _beta1_mesh**2)
    else:
        _penalty_surface = _lambda_value * (np.abs(_beta0_mesh) + np.abs(_beta1_mesh))
    _cost_surface = _mse_surface + _penalty_surface
    _best_index = np.unravel_index(int(np.argmin(_cost_surface)), _cost_surface.shape)
    _best_beta0 = float(_beta0_mesh[_best_index])
    _best_beta1 = float(_beta1_mesh[_best_index])

    _fig, _axes = plt.subplots(2, 2, figsize=(6.25, 4.55))
    _x_line = np.linspace(-0.2, 3.2, 120)
    _axes[0, 0].scatter(x_two, y_two, color=data_color, s=50, zorder=5)
    _axes[0, 0].plot(_x_line, _best_beta0 + _best_beta1 * _x_line, color=chosen_color, linewidth=2.5)
    _axes[0, 0].set(xlabel="x", ylabel="y", xlim=(-0.2, 3.2), ylim=(-0.1, 5.2))

    _surface_specs = [
        (_axes[0, 1], _cost_surface, "Cost"),
        (_axes[1, 0], _mse_surface, "MSE"),
        (_axes[1, 1], _penalty_surface, r"$\lambda \times$ penalty"),
    ]
    _max_ridge_penalty = 10.0 * (_beta0_mesh**2 + _beta1_mesh**2)
    _max_lasso_penalty = 10.0 * (np.abs(_beta0_mesh) + np.abs(_beta1_mesh))
    _fixed_scale_surface = np.concatenate(
        [
            (_mse_surface + _max_ridge_penalty).ravel(),
            (_mse_surface + _max_lasso_penalty).ravel(),
            _mse_surface.ravel(),
            _max_ridge_penalty.ravel(),
            _max_lasso_penalty.ravel(),
        ]
    )
    _scale_max = float(np.percentile(_fixed_scale_surface, 86.0))
    _common_levels = np.linspace(0.0, _scale_max, 21)
    for _axis, _surface, _label in _surface_specs:
        _contour_fill = _axis.contourf(
            _beta0_mesh,
            _beta1_mesh,
            _surface,
            levels=_common_levels,
            cmap="Blues",
            alpha=0.88,
            extend="max",
        )
        if float(np.max(_surface)) > float(np.min(_surface)):
            _axis.contour(
                _beta0_mesh,
                _beta1_mesh,
                _surface,
                levels=_common_levels,
                colors="#4E79A7",
                linewidths=0.34,
                alpha=0.52,
            )
        _axis.scatter([_best_beta0], [_best_beta1], color=chosen_color, marker="*", s=82, zorder=6)
        _axis.set(xlabel=r"$\beta_0$", ylabel=r"$\beta_1$")
        _axis.text(0.02, 0.96, _label, transform=_axis.transAxes, ha="left", va="top", fontsize=9, color="#2F3441")
    _fig.tight_layout(pad=0.75, w_pad=0.75, h_pad=0.90, rect=[0.0, 0.0, 0.91, 1.0])
    _cbar_ax = _fig.add_axes([0.925, 0.18, 0.018, 0.64])
    _cbar = _fig.colorbar(_contour_fill, cax=_cbar_ax)
    _cbar.set_label("Surface value", fontsize=8)
    _cbar.ax.tick_params(labelsize=7)

    _sidebar = sidebar(
        widgets=[s4_penalty, s4_lambda],
        metrics=[
            ("Optimal beta0", f"{_best_beta0:.2f}"),
            ("Optimal beta1", f"{_best_beta1:.2f}"),
            ("MSE at optimum", f"{mse_2p(_best_beta0, _best_beta1):.2f}"),
            ("Cost at optimum", f"{cost_2p(_best_beta0, _best_beta1, _lambda_value, _penalty):.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "4. L1 vs L2: Why Ridge and Lasso Behave Differently",
                "The data are the same, but the penalty changes which coefficient values look best.",
                (
                    "The model form is still fixed:\n\n"
                    r"\[\hat{y} = \beta_0 + \beta_1 x\]"
                    "\n\n"
                    "We still want to find the best values of $\\beta_0$ and $\\beta_1$. What changes is the loss we minimize:\n\n"
                    r"\[\text{Loss} = \mathrm{MSE} + \lambda \times \text{penalty}\]"
                    "\n\n"
                    "We can minimize this loss with gradient descent or with other optimization methods.\n\n"
                    "With two coefficients, the penalty becomes a surface. Ridge makes that surface smooth and round, while lasso makes it point toward the axes.\n\n"
                    "This helps explain why lasso can set coefficients exactly to zero, while ridge usually shrinks coefficients without making them exactly zero.\n\n"
                    "**Important simplification.** In real regularized regression, the intercept is usually **not** penalized. This display penalizes both coefficients only for didactic purposes: it lets you see how the whole fitted line moves as the penalty geometry changes."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("The minimum is shown on each surface. In practice, penalizing the slope is the main modeling idea; penalizing the intercept here is a visualization choice."),
            takeaway_md("The penalty shape helps explain why lasso can remove variables and ridge usually keeps them with smaller coefficients."),
            questions_md(
                [
                    "Why does lasso often produce exact zeros while ridge usually only shrinks coefficients?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s5_controls(counter_button, mo):
    s5_penalty = mo.ui.radio(options=["Ridge (L2)", "LASSO (L1)"], value="LASSO (L1)", label="Penalty")
    s5_recreate = counter_button(label="Recreate data", kind="success")
    return s5_penalty, s5_recreate


@app.cell
def s5_section(
    chosen_color,
    coef_palette,
    coefficient_path_rows,
    coefficient_summary,
    evaluate_lambda_grid,
    finish_figure,
    fit_model,
    mo,
    note_md,
    np,
    plt,
    questions_md,
    s5_penalty,
    s5_recreate,
    section_md,
    sidebar,
    simulate_selection_data,
    takeaway_md,
    train_color,
    two_col,
    valid_color,
):
    _dataset = simulate_selection_data(n_samples=50, seed=2040 + int(s5_recreate.value or 0))
    _X = np.asarray(_dataset["X"], dtype=float)
    _y = np.asarray(_dataset["y"], dtype=float)
    _names = list(_dataset["names"])
    _penalty = "ridge" if s5_penalty.value == "Ridge (L2)" else "lasso"
    _lambdas = np.unique(np.concatenate([np.geomspace(0.02, 10.0, 42), np.array([1.0])]))
    _rows, (_train_idx, _valid_idx, _test_idx) = evaluate_lambda_grid(_X, _y, _lambdas, penalty=_penalty, split_seed=19, standardize=True)
    _best_row = min(_rows, key=lambda row: row["validation_mse"])
    _best_lambda = float(_best_row["lambda"])
    _best_model = fit_model(_X[_train_idx], _y[_train_idx], penalty=_penalty, lambda_value=_best_lambda, standardize=True)
    _coef = np.asarray(_best_model["beta"], dtype=float)
    _path_rows = coefficient_path_rows(
        _X[_train_idx],
        _y[_train_idx],
        _names,
        _lambdas,
        penalty=_penalty,
        standardize=True,
    )

    _fig, (_ax_curve, _ax_coef) = plt.subplots(1, 2, figsize=(6.15, 2.85))
    _ax_curve.semilogx(_lambdas, [row["train_mse"] for row in _rows], color=train_color, linewidth=2.0, label="Training MSE")
    _ax_curve.semilogx(_lambdas, [row["validation_mse"] for row in _rows], color=valid_color, linewidth=2.2, label="Validation MSE")
    _ax_curve.axvline(_best_lambda, color=chosen_color, linestyle="--", linewidth=1.8)
    _ax_curve.set(xlabel="Lambda", ylabel="MSE")
    _ax_curve.legend(loc="upper right", fontsize=7.5)

    for _index, _name in enumerate(_names):
        _series = [float(row["beta"]) for row in _path_rows if row["name"] == _name]
        _ax_coef.semilogx(_lambdas, _series, linewidth=2.0, color=coef_palette[_index % len(coef_palette)], label=_name)
    _ax_coef.axvline(_best_lambda, color=chosen_color, linestyle="--", linewidth=1.6)
    _ax_coef.axhline(0.0, color="#8A98A8", linewidth=1.0)
    _ax_coef.set(xlabel="Lambda", ylabel="Coefficient")
    _ax_coef.legend(loc="upper right", fontsize=7.5)
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s5_penalty, s5_recreate],
        metrics=[
            ("Sample size", f"{len(_X)}"),
            ("Chosen lambda", f"{_best_lambda:.2f}"),
            ("Validation MSE", f"{_best_row['validation_mse']:.2f}"),
            ("Chosen coefficients", coefficient_summary(_coef, _names)),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "5. Choosing Lambda with Validation",
                "Now the question is not how to fit coefficients, but which regularization strength generalizes best.",
                (
                    "The dataset here has one true signal, one correlated copy of that signal, and three pure noise variables. "
                    "The true signal is:\n\n"
                    r"\[y \sim \mathrm{Signal}\qquad\text{or, with coefficients,}\qquad y = \beta_0 + \beta_{\mathrm{Signal}}\,\mathrm{Signal} + \varepsilon\]"
                    "\n\n"
                    "The true signal is deliberately not overwhelming, so choosing lambda is not trivial.\n\n"
                    "For each candidate lambda, we fit the model on the training data and choose among the candidates using **validation prediction error**. The left plot shows why training error is not enough. The right plot shows how the coefficients change as lambda grows. In plain language, lambda is a complexity dial: low lambda gives the model more freedom, high lambda restrains it."
                ),
            ),
            mo.md(r"$\lambda^* = \arg\min_{\lambda}\ \mathrm{MSE}_{\mathrm{validation}}(\lambda)$"),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("Validation MSE is useful for choosing among candidate lambdas. Once it has influenced the choice, it is no longer a clean final estimate of generalization error."),
            takeaway_md("Lambda should be chosen with unseen-data prediction error, not with training error and not with the penalized cost itself."),
            questions_md(
                [
                    "Why is lambda chosen with validation MSE rather than with validation cost?",
                    "What happens to model complexity as lambda gets larger?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s6_controls(counter_button, mo):
    s6_penalty = mo.ui.radio(options=["Ridge (L2)", "LASSO (L1)"], value="LASSO (L1)", label="Penalty")
    s6_recreate = counter_button(label="Recreate data", kind="success")
    return s6_penalty, s6_recreate


@app.cell
def s6_section(
    evaluate_lambda_grid,
    finish_figure,
    fit_model,
    mo,
    note_md,
    np,
    plt,
    questions_md,
    s6_penalty,
    s6_recreate,
    section_md,
    sidebar,
    simulate_selection_data,
    split_a_color,
    split_b_color,
    takeaway_md,
    two_col,
):
    _dataset = simulate_selection_data(n_samples=50, seed=2110 + int(s6_recreate.value or 0))
    _X = np.asarray(_dataset["X"], dtype=float)
    _y = np.asarray(_dataset["y"], dtype=float)
    _names = list(_dataset["names"])
    _penalty = "ridge" if s6_penalty.value == "Ridge (L2)" else "lasso"
    _lambdas = np.geomspace(0.02, 2.50, 34)
    _split_a, (_train_a, _valid_a, _test_a) = evaluate_lambda_grid(_X, _y, _lambdas, penalty=_penalty, split_seed=19, standardize=True)
    _split_b, (_train_b, _valid_b, _test_b) = evaluate_lambda_grid(_X, _y, _lambdas, penalty=_penalty, split_seed=41, standardize=True)
    _best_a = min(_split_a, key=lambda row: row["validation_mse"])
    _best_b = min(_split_b, key=lambda row: row["validation_mse"])
    _model_a = fit_model(_X[_train_a], _y[_train_a], penalty=_penalty, lambda_value=float(_best_a["lambda"]), standardize=True)
    _model_b = fit_model(_X[_train_b], _y[_train_b], penalty=_penalty, lambda_value=float(_best_b["lambda"]), standardize=True)

    _coef_a = np.asarray(_model_a["beta"], dtype=float)
    _coef_b = np.asarray(_model_b["beta"], dtype=float)

    _fig, (_ax_curve, _ax_coef) = plt.subplots(1, 2, figsize=(6.15, 2.85))
    _ax_curve.semilogx(_lambdas, [row["validation_mse"] for row in _split_a], color=split_a_color, linewidth=2.0, label="Split A")
    _ax_curve.semilogx(_lambdas, [row["validation_mse"] for row in _split_b], color=split_b_color, linewidth=2.0, label="Split B")
    _ax_curve.axvline(float(_best_a["lambda"]), color=split_a_color, linestyle="--", linewidth=1.4, alpha=0.85)
    _ax_curve.axvline(float(_best_b["lambda"]), color=split_b_color, linestyle="--", linewidth=1.4, alpha=0.85)
    _ax_curve.set(xlabel="Lambda", ylabel="Validation MSE")
    _ax_curve.legend(loc="upper right", fontsize=7.8)

    _positions = np.arange(len(_names))
    _bar_width = 0.36
    _ax_coef.bar(_positions - _bar_width / 2, _coef_a, color=split_a_color, width=_bar_width, label="Split A")
    _ax_coef.bar(_positions + _bar_width / 2, _coef_b, color=split_b_color, width=_bar_width, label="Split B")
    _ax_coef.axhline(0.0, color="#8A98A8", linewidth=1.0)
    _ax_coef.set_xticks(_positions)
    _ax_coef.set_xticklabels(_names, rotation=25, ha="right", fontsize=8)
    _ax_coef.set_ylabel("Coefficient")
    _ax_coef.legend(loc="upper right", fontsize=7.8)
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s6_penalty, s6_recreate],
        metrics=[
            ("Split A lambda", f"{_best_a['lambda']:.2f}"),
            ("Split B lambda", f"{_best_b['lambda']:.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "6. Why One Validation Split Can Mislead",
                "One train-validation split can make lambda selection look more certain than it really is.",
                (
                    "The data are generated from one true signal:\n\n"
                    r"\[y = 1 + 1.8\,\mathrm{Signal} + \varepsilon,\qquad \mathrm{Copy} = 0.88\,\mathrm{Signal} + \eta\]"
                    "\n\n"
                    "`Copy` is deliberately similar to `Signal`, but it is not part of the true equation for `y`. The noise variables are unrelated to `y`.\n\n"
                    "Both curves come from the same synthetic dataset. The only difference is which observations happened to land in the validation set.\n\n"
                    "When the chosen lambda changes across plausible splits, that is a warning sign that a single validation split is not stable enough to trust on its own. In plain language, the split can be a little lucky or unlucky."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("The underlying data are unchanged. Only the train-validation split moved."),
            takeaway_md("If two reasonable validation splits recommend different lambdas, the safer response is to average over more splits."),
            questions_md(
                [
                    "Why can one train-validation split give an unstable lambda choice?",
                    "What does disagreement between splits tell you about uncertainty in model selection?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s7_controls(counter_button, mo):
    s7_penalty = mo.ui.radio(options=["Ridge (L2)", "LASSO (L1)"], value="LASSO (L1)", label="Penalty")
    s7_recreate = counter_button(label="Recreate data", kind="success")
    return s7_penalty, s7_recreate


@app.cell
def s7_section(
    chosen_color,
    coefficient_summary,
    cost_color,
    cv_color,
    evaluate_cv_grid,
    finish_figure,
    fit_model,
    mo,
    note_md,
    np,
    plt,
    questions_md,
    s7_penalty,
    s7_recreate,
    section_md,
    selected_variables,
    sidebar,
    simulate_selection_data,
    takeaway_md,
    two_col,
):
    _dataset = simulate_selection_data(n_samples=50, seed=2210 + int(s7_recreate.value or 0))
    _X = np.asarray(_dataset["X"], dtype=float)
    _y = np.asarray(_dataset["y"], dtype=float)
    _names = list(_dataset["names"])
    _penalty = "ridge" if s7_penalty.value == "Ridge (L2)" else "lasso"
    _lambdas = np.geomspace(0.02, 2.50, 34)
    _cv_rows = evaluate_cv_grid(_X, _y, _lambdas, penalty=_penalty, n_splits=5, seed=13, standardize=True)
    _best_cv = min(_cv_rows, key=lambda row: row["cv_mse"])
    _best_lambda = float(_best_cv["lambda"])
    _final_model = fit_model(_X, _y, penalty=_penalty, lambda_value=_best_lambda, standardize=True)
    _coef = np.asarray(_final_model["beta"], dtype=float)

    _fig, (_ax_curve, _ax_coef) = plt.subplots(1, 2, figsize=(6.15, 2.85))
    _ax_curve.semilogx(_lambdas, [row["cv_mse"] for row in _cv_rows], color=cv_color, linewidth=2.25)
    _ax_curve.axvline(_best_lambda, color=cost_color, linestyle="--", linewidth=1.8)
    _ax_curve.set(xlabel="Lambda", ylabel="CV MSE")

    _positions = np.arange(len(_names))
    _ax_coef.bar(_positions, _coef, color=[chosen_color if abs(value) > 1e-5 else "#BFC9D4" for value in _coef], width=0.65)
    _ax_coef.axhline(0.0, color="#8A98A8", linewidth=1.0)
    _ax_coef.set_xticks(_positions)
    _ax_coef.set_xticklabels(_names)
    _ax_coef.set_ylabel("Coefficient")
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s7_penalty, s7_recreate],
        metrics=[
            ("CV lambda", f"{_best_lambda:.2f}"),
            ("Best CV MSE", f"{_best_cv['cv_mse']:.2f}"),
            ("Selected variables", ", ".join(selected_variables(_coef, _names)) or "Intercept only"),
            ("Final coefficients", coefficient_summary(_coef, _names)),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "7. Choosing Lambda with Cross-Validation",
                "Cross-validation averages several validation roles before one lambda choice is made.",
                (
                    "Cross-validation is preferred because it averages over several train-validation splits before one lambda choice is made. Instead of trusting one arbitrary validation split, it rotates which observations play the validation role and gives a more stable estimate of out-of-sample performance.\n\n"
                    "After choosing lambda, we fit one final model using all available training data."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("Cross-validation is for model selection. A separate held-out test set, if available, is still for the final audit after the choice is made."),
            takeaway_md("Cross-validation usually gives a steadier lambda choice because it does not depend on one validation split."),
            questions_md(
                [
                    "How does k-fold cross-validation reduce dependence on one arbitrary split?",
                    "After cross-validation chooses lambda, what is the held-out test set still for?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s8_controls(counter_button, mo):
    s8_lambda = mo.ui.slider(0.20, 5.0, value=0.80, step=0.05, label="Lambda (lasso)")
    s8_recreate = counter_button(label="Recreate scaled data", kind="success")
    return s8_lambda, s8_recreate


@app.cell
def s8_section(
    chosen_color,
    finish_figure,
    fit_lasso,
    lasso_color,
    mo,
    note_md,
    np,
    plt,
    questions_md,
    s8_lambda,
    s8_recreate,
    section_md,
    selected_variables,
    sidebar,
    simulate_scaled_data,
    split_indices,
    takeaway_md,
    two_col,
):
    _dataset = simulate_scaled_data(seed=45 + int(s8_recreate.value or 0))
    _X = np.asarray(_dataset["X"], dtype=float)
    _y = np.asarray(_dataset["y"], dtype=float)
    _names = list(_dataset["names"])
    _lambda_value = float(s8_lambda.value)
    _train_idx, _, _ = split_indices(len(_X), train_frac=0.65, valid_frac=0.0, seed=11)
    _no_scale = fit_lasso(_X[_train_idx], _y[_train_idx], lambda_value=_lambda_value, standardize=False)
    _with_scale = fit_lasso(_X[_train_idx], _y[_train_idx], lambda_value=_lambda_value, standardize=True)
    _raw_sd = _X[_train_idx].std(axis=0, ddof=0)
    _effect_no = np.asarray(_no_scale["beta"], dtype=float) * _raw_sd
    _effect_yes = np.asarray(_with_scale["beta"], dtype=float) * _raw_sd
    _effect_limit = 1.08 * max(float(np.max(np.abs(_effect_no))), float(np.max(np.abs(_effect_yes))), 0.5)
    _scale_ratio = float(np.max(_raw_sd) / np.min(_raw_sd))
    _kept_no = ", ".join(selected_variables(_effect_no, _names, threshold=0.05)) or "Intercept only"
    _kept_yes = ", ".join(selected_variables(_effect_yes, _names, threshold=0.05)) or "Intercept only"

    _fig, (_ax_no, _ax_yes) = plt.subplots(1, 2, figsize=(6.15, 2.85))
    _positions = np.arange(len(_names))
    _plot_labels = ["a (signal)\nsmall units", "b (signal)\nlarge units", "c (noise)\nsmall units", "d (noise)\nlarge units"]
    _colors_no = [lasso_color if abs(value) > 0.05 else "#BFC9D4" for value in _effect_no]
    _colors_yes = [chosen_color if abs(value) > 0.05 else "#BFC9D4" for value in _effect_yes]
    _ax_no.bar(_positions, _effect_no, color=_colors_no, width=0.65)
    _ax_yes.bar(_positions, _effect_yes, color=_colors_yes, width=0.65)
    for _axis in (_ax_no, _ax_yes):
        _axis.axhline(0.0, color="#8A98A8", linewidth=1.0)
        _axis.set_xticks(_positions)
        _axis.set_xticklabels(_plot_labels, fontsize=8)
        _axis.set_ylabel("One-SD prediction effect")
        _axis.set_ylim(-0.18 * _effect_limit, _effect_limit)
    _ax_no.text(0.02, 0.96, "Without standardizing", transform=_ax_no.transAxes, ha="left", va="top", fontsize=9, color="#2F3441")
    _ax_yes.text(0.02, 0.96, "With standardizing", transform=_ax_yes.transAxes, ha="left", va="top", fontsize=9, color="#2F3441")
    _fig = finish_figure(_fig)

    _sidebar = sidebar(
        widgets=[s8_lambda, s8_recreate],
        metrics=[
            ("Largest / smallest SD", f"{_scale_ratio:.0f}×"),
            ("Kept without scaling", _kept_no),
            ("Kept with scaling", _kept_yes),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "8. Standardization Makes the Penalty Fair",
                "Ridge and lasso penalize coefficient size, and coefficient size depends on measurement units.",
                (
                    "This synthetic example has two useful signals and two useless noise variables. The true signal is:\n\n"
                    r"\[y = \beta_0 + \beta_a a + \beta_b b + \varepsilon\]"
                    "\n\n"
                    "`a (signal)` and `c (noise)` are stored in small units. `b (signal)` and `d (noise)` are stored on a scale about 100 times larger.\n\n"
                    "Without standardizing, the large-unit variables can change predictions with tiny coefficients, so the lasso penalty treats them as artificially cheap. After standardization, every predictor has mean 0 and standard deviation 1 before the penalty is applied. In plain language, variables are judged on comparable units instead of on the accident of how they were measured."
                ),
            ),
            two_col(mo.as_html(_fig), _sidebar),
            note_md("The bars show the effect of a one-standard-deviation increase in each predictor. In a real workflow, standardize inside the training process and choose lambda with validation or cross-validation."),
            takeaway_md("Standardize predictors before ridge or lasso so lambda penalizes predictive contribution more fairly, not measurement units."),
            questions_md(
                [
                    "How does converting predictors to mean 0 and standard deviation 1 make the ridge or lasso penalty fairer?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


if __name__ == "__main__":
    app.run()
