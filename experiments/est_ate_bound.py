import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import statsmodels.api as sm


def estimate_ate_residual_comp(high, x_all, y_all, min_points=3, studentized_threshold=1.0):
    """
    Estimates ATE by starting with a core set of points and iteratively adding
    more points as long as the new point's studentized residual remains below the threshold.

    Args:
        high (bool): If True, focuses on high x-values (adding points with smaller x).
                     If False, focuses on low x-values (adding points with larger x).
        x_all (np.ndarray): All x values (shape (n,) or (n,1)).
        y_all (np.ndarray): All y values (shape (n,)).
        min_points (int): Minimum number of points in the initial fit (at least 2).
        studentized_threshold (float): Maximum allowed absolute studentized residual for new points.

    Returns:
        tuple: (final_model, final_x_selected, final_y_selected, final_alpha)
    """
    # Ensure x_all is 2D
    if x_all.ndim == 1:
        x_all = x_all.reshape(-1, 1)

    n = x_all.shape[0]
    if n < 2:
        # Not enough points to fit
        return LinearRegression(), x_all, y_all, np.nan

    # Sort data by x
    order = np.argsort(x_all[:, 0])
    x_sorted = x_all[order]
    y_sorted = y_all[order]

    # Initial set of indices
    initial_size = min(max(2, min_points), n)
    if high:
        selected = list(range(n - initial_size, n))
        candidates = list(range(n - initial_size - 1, -1, -1))
    else:
        selected = list(range(initial_size))
        candidates = list(range(initial_size, n))

    # Grow the selection
    for idx in candidates:
        trial = selected.copy()
        if high:
            trial.insert(0, idx)
        else:
            trial.append(idx)

        # Fit OLS via statsmodels to compute studentized residuals
        x_trial = x_sorted[trial]
        y_trial = y_sorted[trial]
        X_design = sm.add_constant(x_trial)
        ols_res = sm.OLS(y_trial, X_design).fit()
        infl = ols_res.get_influence()
        r_int = infl.resid_studentized_internal

        # Candidate position in the trial set
        pos = 0 if high else -1
        if abs(r_int[pos]) < studentized_threshold:
            # Accept new point
            selected = trial
        else:
            break

    # Final selection
    x_sel = x_sorted[selected]
    y_sel = y_sorted[selected]

    # Fit sklearn model on final selection
    final_model = LinearRegression().fit(x_sel, y_sel)
    final_alpha = final_model.intercept_

    return final_model, x_sel, y_sel, final_alpha


def main():
    # Load the CSV file into a numpy array
    from utils import ARTEFACT_DIR
    data = np.loadtxt(f'{ARTEFACT_DIR}/results_mediated_10.csv', delimiter=',', skiprows=1,
                      usecols=(0, 1, 2, 3), max_rows=8)
    print(data)
    # Sort data by the first column (x values)
    sorted_indices = np.argsort(data[:, 0])
    data = data[sorted_indices]
    x_all = data[:, 0].reshape(-1, 1)
    y_all = data[:, 1]

    # Estimate lower bound (high=True)
    model_upper, x_upper, y_upper, intercept_upper = estimate_ate_residual_comp(
        high=True, x_all=x_all, y_all=y_all, studentized_threshold=1.5
    )
    y_pred_upper = model_upper.predict(x_upper)

    # Estimate upper bound (high=False)
    model_lower, x_lower, y_lower, intercept_lower = estimate_ate_residual_comp(
        high=False, x_all=x_all, y_all=y_all, studentized_threshold=1.5
    )
    y_pred_lower = model_lower.predict(x_lower)

    # Plot both bounds
    plt.scatter(x_all, y_all, color='blue', s=10, alpha=0.5, label='Data')
    plt.plot(x_upper, y_pred_upper, color='red', linewidth=2, label=f'Lower bound (intercept={intercept_upper:.4g})')
    plt.plot(x_lower, y_pred_lower, color='green', linewidth=2, label=f'Upper bound (intercept={intercept_lower:.4g})')
    plt.legend()
    plt.title('ATE Upper and Lower Bounds')
    plt.tight_layout()
    plt.show()

    print(f'Upper bound intercept: {intercept_upper:.4g}')
    print(f'Lower bound intercept: {intercept_lower:.4g}')

if __name__ == '__main__':
    main()