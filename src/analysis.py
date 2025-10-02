# in src/analysis.py
from . import yield_curve_model
import numpy as np
import pandas as pd
from yield_curve_model import nelson_siegel

def run_all_analysis(final_params, portfolio_df):
    """Runs all analysis functions and prints the results."""
    print("\n--- Running Relative Value Analysis ---")
    run_relative_value_analysis(final_params, portfolio_df.copy())
    
    print("\n--- Running Duration and Scenario Analysis ---")
    run_duration_analysis(final_params, portfolio_df.copy())
    
    print("\n--- Running Key Rate Duration Analysis ---")
    run_key_rate_analysis(final_params, portfolio_df.copy())

def run_relative_value_analysis(params, portfolio_df):
    """Prices a portfolio and identifies rich/cheap bonds."""
    model_prices = portfolio_df.apply(
        lambda row: _calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], params),
        axis=1
    )
    portfolio_df['Model Price'] = model_prices
    portfolio_df['Rich/Cheap ($)'] = portfolio_df['Model Price'] - portfolio_df['Market Price']
    
    print(portfolio_df.round(4).to_string(index=False))

def run_duration_analysis(params, portfolio_df):
    """Calculates portfolio duration and runs a parallel shift scenario test."""
    durations = portfolio_df.apply(
        lambda row: _calculate_modified_duration(row['Coupon Rate (%)'], row['Years to Maturity'], params),
        axis=1
    )
    portfolio_df['Modified Duration'] = durations
    total_market_value = portfolio_df['Market Price'].sum()
    
    weighted_duration = (portfolio_df['Modified Duration'] * portfolio_df['Market Price']).sum() / total_market_value
    
    print(f"Portfolio Weighted Modified Duration: {weighted_duration:.4f} years")

    # Scenario Test
    rate_shock_bps = 100
    estimated_loss_pct = -weighted_duration * (rate_shock_bps / 10000)
    
    shocked_params = params.copy()
    shocked_params[0] += (rate_shock_bps / 100) # Shift beta0 by the shock
    
    actual_new_value = portfolio_df.apply(
        lambda row: _calculate_bond_price(row['Coupon Rate (%)'], row['Years to Maturity'], shocked_params),
        axis=1
    ).sum()

    print(f"\n--- Scenario Test: +{rate_shock_bps}bps Parallel Shift ---")
    print(f"Estimated Loss (from Duration): {estimated_loss_pct: .2%}")
    print(f"Actual Loss (from Re-Pricing):  {(actual_new_value / total_market_value - 1): .2%}")

def run_key_rate_analysis(params, portfolio_df):
    """Performs a key rate duration analysis via simulation."""
    key_rates_to_shock = [2, 5, 10, 30]
    shock_size_bps = 1.0
    
    base_curve = lambda tau: nelson_siegel(tau, *params)
    base_value = portfolio_df.apply(
        lambda row: _calculate_bond_price_custom_curve(row['Coupon Rate (%)'], row['Years to Maturity'], base_curve),
        axis=1
    ).sum()
    
    print(f"Portfolio Base Value: ${base_value:.2f}")
    
    results = []
    for key_rate in key_rates_to_shock:
        shocked_curve = lambda tau: base_curve(tau) + np.maximum(0, 1 - np.abs(tau - key_rate)) * (shock_size_bps / 100.0)
        
        shocked_value = portfolio_df.apply(
            lambda row: _calculate_bond_price_custom_curve(row['Coupon Rate (%)'], row['Years to Maturity'], shocked_curve),
            axis=1
        ).sum()
        
        value_change = shocked_value - base_value
        results.append({'Key Rate Shock (+1bp)': f"{key_rate}-Year", 'Portfolio Value Change ($)': value_change})
        
    results_df = pd.DataFrame(results)
    print(results_df.round(4).to_string(index=False))

# --- Helper functions ---

def _calculate_bond_price(coupon_rate, years_to_maturity, params, face_value=100.0):
    """Helper to price a single bond."""
    curve = lambda tau: nelson_siegel(tau, *params)
    return _calculate_bond_price_custom_curve(coupon_rate, years_to_maturity, curve, face_value)

def _calculate_bond_price_custom_curve(coupon_rate, years_to_maturity, yield_curve_func, face_value=100.0):
    """Helper to price a bond with a given curve function."""
    num_coupons = int(years_to_maturity * 2)
    coupon_times = np.arange(0.5, years_to_maturity + 0.1, 0.5)
    discount_rates = yield_curve_func(coupon_times) / 100.0
    coupon_payment = (coupon_rate / 100.0) / 2.0 * face_value
    
    pv_coupons = sum([coupon_payment / ((1 + discount_rates[i]/2)**(i+1)) for i in range(num_coupons)])
    pv_face_value = face_value / ((1 + discount_rates[-1]/2)**num_coupons)
    return pv_coupons + pv_face_value

def _calculate_modified_duration(coupon_rate, years_to_maturity, params, face_value=100.0):
    """Helper to calculate a single bond's modified duration."""
    ytm = nelson_siegel(years_to_maturity, *params).item() / 100.0
    
    num_coupons = int(years_to_maturity * 2)
    coupon_times = np.arange(0.5, years_to_maturity + 0.1, 0.5)
    coupon_payment = (coupon_rate / 100.0) / 2.0 * face_value
    
    pv_flows = [coupon_payment / ((1 + ytm/2)**(i+1)) for i in range(num_coupons)]
    pv_face = face_value / ((1 + ytm/2)**num_coupons)
    bond_price = sum(pv_flows) + pv_face
    
    weighted_time = sum([pv_flows[i] * coupon_times[i] for i in range(num_coupons)])
    weighted_time_face = pv_face * years_to_maturity
    macaulay_duration = (weighted_time + weighted_time_face) / bond_price
    
    return macaulay_duration / (1 + ytm / 2)
