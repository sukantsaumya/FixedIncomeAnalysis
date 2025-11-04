

import numpy as np
from scipy.optimize import differential_evolution, minimize

def nelson_siegel(tau, beta0, beta1, beta2, lambda_):
    """Computes the Nelson-Siegel yield curve values with numerical stability."""
    tau = np.atleast_1d(tau)
    yields = np.zeros_like(tau, dtype=float)
    zero_mask = np.abs(tau) < 1e-10
    yields[zero_mask] = beta0 + beta1

    if np.any(~zero_mask):
        t = tau[~zero_mask]
        z = t / lambda_
        exp_z = np.exp(-z)
        factor1 = (1.0 - exp_z) / z
        factor2 = factor1 - exp_z
        yields[~zero_mask] = beta0 + beta1 * factor1 + beta2 * factor2
    return yields

def nelson_siegel_svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    """Computes the Nelson-Siegel-Svensson yield curve values with numerical stability."""
    tau = np.atleast_1d(tau)
    yields = np.zeros_like(tau, dtype=float)
    zero_mask = np.abs(tau) < 1e-10
    yields[zero_mask] = beta0 + beta1

    if np.any(~zero_mask):
        t = tau[~zero_mask]
        z1 = t / lambda1
        z2 = t / lambda2
        exp_z1 = np.exp(-z1)
        exp_z2 = np.exp(-z2)

        factor1 = (1.0 - exp_z1) / z1
        factor2 = factor1 - exp_z1
        factor3 = (1.0 - exp_z2) / z2 - exp_z2

        yields[~zero_mask] = beta0 + beta1 * factor1 + beta2 * factor2 + beta3 * factor3
    return yields

def objective_sse(params, maturities, market_yields):
    """Objective function to minimize: Sum of Squared Errors (SSE)."""
    model_yields = nelson_siegel(maturities, *params)
    return np.sum((market_yields - model_yields) ** 2)

def calibrate_yield_curve(maturities, market_yields):
    """
    Performs the complete two-stage calibration and returns the optimal parameters.
    """
    bounds = [(0, 15), (-15, 15), (-20, 20), (0.01, 10)]

    print("[Stage 1] Performing global search with Differential Evolution...")
    global_result = differential_evolution(
        func=objective_sse,
        bounds=bounds,
        args=(maturities, market_yields),
        polish=False
    )

    print("[Stage 2] Refining the solution with L-BFGS-B...")
    final_result = minimize(
        fun=objective_sse,
        x0=global_result.x,
        args=(maturities, market_yields),
        method='L-BFGS-B',
        bounds=bounds
    )
    
    final_params = final_result.x
    model_yields_final = nelson_siegel(maturities, *final_params)
    errors_bp = (market_yields - model_yields_final) * 100
    rmse_bps = np.sqrt(np.mean(errors_bp ** 2))
    
    print(f"Calibration complete. Final RMSE: {rmse_bps:.4f} bps")
    return final_params, rmse_bps
