#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flax degradation modeling: multi-exponential model with environmental correction,
Fourier component, T and H uncertainty propagation, integration of the Chambéry fire effect (1532),
uncertainty propagation on predicted date via finite difference method, improved display,
and integration of a complete Bayesian approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, fsolve

# =======================
# 1) Base model parameters
# =======================
sigma0 = 1103         # Initial reference strength (MPa)
sigma_infty = 0       # Residual strength (MPa)
tau = 1102            # Characteristic degradation time (years)

# =======================
# 2) Environmental correction parameters for σ₀
# =======================
alpha = 1e-4
beta = 2e-4
gamma = 1e-8
delta = 4e-8
T0 = 20.0  # Reference temperature (°C)

def sigma0_env(T, H, sigma0_base):
    """Corrects initial strength based on temperature and humidity."""
    return sigma0_base * np.exp(alpha * H + beta * (T - T0) + gamma * H**2 + delta * (T - T0)**2)

# =======================
# 3) Environmental parameters for tau modification
# =======================
T_ref = 20.0    # Reference temperature
H_ref = 60.0    # Reference humidity
kappa_T_tau = 0.01   # T influence on tau
kappa_H_tau = 0.01   # H influence on tau

def effective_tau(tau_i, T, H):
    """Calculates effective characteristic time based on T and H."""
    return tau_i / (1 + kappa_T_tau * (T - T_ref) + kappa_H_tau * (H - H_ref))

# =======================
# 4) Fourier decomposition (sine and cosine components)
# =======================
fourier_coeffs_sin = [50.0, 30.0]    # Sine amplitudes
fourier_coeffs_cos = [30.0, 15.0]    # Cosine amplitudes
fourier_periods = [2000.0, 4000.0]   # Corresponding periods (years)
fourier_phases = [0.0, 0.0]          # Phases (in radians)

def fourier_series(t, coeffs_sin, coeffs_cos, periods, phases):
    """Calculates the sum of a Fourier series."""
    result = 0.0
    for A_sin, A_cos, P, phi in zip(coeffs_sin, coeffs_cos, periods, phases):
        result += A_sin * np.sin(2 * np.pi * t / P + phi) + A_cos * np.cos(2 * np.pi * t / P + phi)
    return result

# =======================
# 5) Multi-exponential model with environmental correction, Fourier component and fire effect
# =======================
N_BRANCHES = 2  # Number of exponential branches

def multi_exponential_env_fourier(t, T, H, *params):
    """
    Complete model:
      σ(t) = σ_infty + Σ_{i=1}^{N_BRANCHES} A_i * exp(-t / τ_i_eff) + Fourier(t)
    with τ_i_eff = τ_i_corr / (1 + kappa_T_tau*(T - T_ref) + kappa_H_tau*(H - H_ref))
    Fire effect is integrated via a corrective factor of 0.5 if t >= t_fire.
    t is defined as t = 2000 - Age (so t=468 corresponds to the 1532 fire).
    """
    # Try to convert t to NumPy array
    try:
        t_np = np.asarray(t)
        if t_np.ndim == 0:
            correction_fire = 0.5 if t_np >= (2000 - 1532) else 1.0
        else:
            correction_fire = np.where(t_np >= (2000 - 1532), 0.5, 1.0)
    except Exception:
        # If t is symbolic (PyMC), use pm.math.switch
        import pymc as pm
        t_fire = 2000 - 1532  # 468 years
        correction_fire = pm.math.switch(pm.math.ge(t, t_fire), 0.5, 1.0)

    A_list = params[:N_BRANCHES]
    tau_list = params[N_BRANCHES:2*N_BRANCHES]
    sigma = sigma_infty
    env_corr = 1 + kappa_T_tau * (T - T_ref) + kappa_H_tau * (H - H_ref)
    for i in range(N_BRANCHES):
        tau_eff = (tau_list[i] * correction_fire) / env_corr
        sigma += A_list[i] * np.exp(-t / tau_eff)
    sigma += fourier_series(t, fourier_coeffs_sin, fourier_coeffs_cos, fourier_periods, fourier_phases)
    return sigma

def wrapper_for_curve_fit(t, *params):
    """Wrapper for fitting with fixed T and H (20°C, 60% RH)."""
    T_const = 20.0
    H_const = 60.0
    return multi_exponential_env_fourier(t, T_const, H_const, *params)

# =============================
# 6) Fanti sample data (simplified)
# =============================
samples_data = [
    {"Sample": "B",   "Dating": "2000 A.D.",     "σr (MPa)": 1076, "T": 18.0, "H": 65.0, "Uncertainty (yrs)": 0},
    {"Sample": "DII", "Dating": "997-1147 A.D.",   "σr (MPa)": 678,  "T": 26.0, "H": 70.0, "Uncertainty (yrs)": 150},
    {"Sample": "D",   "Dating": "544-605 A.D.",    "σr (MPa)": 63.2, "T": 12.0, "H": 70.0, "Uncertainty (yrs)": 60},
    {"Sample": "NII", "Dating": "350-230 A.D.",    "σr (MPa)": 119,  "T": 10.0, "H": 80.0, "Uncertainty (yrs)": 60},
    {"Sample": "FII", "Dating": "55-74 A.D.",      "σr (MPa)": 150,  "T": 20.0, "H": 50.0, "Uncertainty (yrs)": 20},
    {"Sample": "E",   "Dating": "405-345 B.C.",    "σr (MPa)": 140,  "T": 30.0, "H": 40.0, "Uncertainty (yrs)": 60},
    {"Sample": "HII", "Dating": "1000-720 B.C.",   "σr (MPa)": 44.1, "T": 32.0, "H": 35.0, "Uncertainty (yrs)": 280},
    {"Sample": "K",   "Dating": "2826-2478 B.C.",  "σr (MPa)": 58.9, "T": 28.0, "H": 40.0, "Uncertainty (yrs)": 348},
    {"Sample": "LII", "Dating": "3500-3000 B.C.",  "σr (MPa)": 2.11, "T": 27.0, "H": 40.0, "Uncertainty (yrs)": 500}
]
df_samples = pd.DataFrame(samples_data)

def parse_age(dating_str):
    """Extracts an average age (in AD years, negative for B.C.) from the string."""
    if "A.D." in dating_str:
        parts = dating_str.replace("A.D.", "").split("-")
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2.0
        else:
            return float(parts[0])
    elif "B.C." in dating_str:
        parts = dating_str.replace("B.C.", "").split("-")
        if len(parts) == 2:
            return -((float(parts[0]) + float(parts[1])) / 2.0)
        else:
            return -float(parts[0])
    else:
        return np.nan

def parse_dating_interval(dating_str):
    """
    Extracts the dating interval from the string.
    Returns (min_age, max_age) in AD years.
    """
    if "A.D." in dating_str:
        parts = dating_str.replace("A.D.", "").split("-")
        if len(parts) == 2:
            min_age = float(parts[0])
            max_age = float(parts[1])
            return min_age, max_age
        else:
            date = float(parts[0])
            return date, date
    elif "B.C." in dating_str:
        parts = dating_str.replace("B.C.", "").split("-")
        if len(parts) == 2:
            min_age = -float(parts[0])
            max_age = -float(parts[1])
            if min_age > max_age:
                min_age, max_age = max_age, min_age
            return min_age, max_age
        else:
            date = -float(parts[0])
            return date, date
    else:
        return None, None

# =============================
# Inversion functions to find age from given resistance
# =============================
def invert_age_in_interval(target_sigma, A_min, A_max, T, H, params, age_step=1.0):
    """
    Searches for the age in interval [A_min, A_max] that minimizes |σ_model - σ_target|.
    Returns the optimal age.
    """
    if A_min is None or A_max is None or A_min > A_max:
        return None
    age_range = np.arange(A_min, A_max + age_step, age_step)
    sigma_vals = []
    for age in age_range:
        t_val = 2000 - age
        sigma_vals.append(multi_exponential_env_fourier(t_val, T, H, *params))
    sigma_vals = np.array(sigma_vals)
    ecarts = np.abs(sigma_vals - target_sigma)
    idx_min = np.argmin(ecarts)
    return age_range[idx_min]

def func_to_solve(age, target_sigma, T, H, params):
    """Function whose root (age) satisfies multi_exponential_env_fourier(2000 - age) = σ_target."""
    t_val = 2000 - age
    return multi_exponential_env_fourier(t_val, T, H, *params) - target_sigma

def invert_age(target_sigma, T, H, params, age_guess=2000.0):
    """
    Attempts to use fsolve to find the age corresponding to σ_target.
    In case of failure, uses discrete inversion.
    """
    sol, infodict, ier, mesg = fsolve(func_to_solve, age_guess, args=(target_sigma, T, H, params), full_output=True)
    if ier == 1:
        return sol[0]
    else:
        return invert_age_in_interval(target_sigma, -3500, 2100, T, H, params)

def get_model_age_for_sample(row, T=20.0, H=60.0, params=None, age_step=0.5):
    """
    For a sample (row), if a dating interval is available,
    searches for optimal age in that interval; otherwise, uses hybrid inversion.
    """
    if params is None:
        raise ValueError("Please provide 'params' parameters from the fit.")
    A_min = row["Dating_min"]
    A_max = row["Dating_max"]
    target_sigma = row["σr (MPa)"]
    if A_min is not None and A_max is not None:
        age_opt = invert_age_in_interval(target_sigma, A_min, A_max, T, H, params, age_step=age_step)
        if age_opt is not None:
            return age_opt
    return invert_age(target_sigma, T, H, params)

# =============================
# Calculate parsed ages and corresponding duration
# =============================
df_samples["Parsed Age (AD)"] = df_samples["Dating"].apply(parse_age)
df_samples["t (yrs)"] = 2000 - df_samples["Parsed Age (AD)"]
df_samples["Dating_min"], df_samples["Dating_max"] = zip(*df_samples["Dating"].apply(parse_dating_interval))

# =============================
# Fit model to resistance data (σr)
# =============================
t_data = df_samples["t (yrs)"].values
sigma_data = df_samples["σr (MPa)"].values

# Calculate σ uncertainty (15% or minimum 1 MPa)
error_floor_fraction = 0.15
minimum_sigma_uncertainty = 1.0
sigma_uncertainty = np.maximum(minimum_sigma_uncertainty, error_floor_fraction * sigma_data)
df_samples["σr Uncertainty (MPa)"] = sigma_uncertainty

# Initial parameters for [A1, A2, τ1, τ2] (2-branch model)
p0 = [820, 220, 1075, 1075]
bounds_lower = [1, 1, 10, 10]
bounds_upper = [1e5, 1e5, 1e5, 1e5]
bounds = (bounds_lower, bounds_upper)

def residuals(params, t_data, sigma_data, T, H, sigma_uncertainty):
    sigma_model = wrapper_for_curve_fit(t_data, *params)
    return (sigma_model - sigma_data) / sigma_uncertainty

result = least_squares(
    residuals, p0, args=(t_data, sigma_data, 20.0, 60.0, sigma_uncertainty),
    bounds=bounds, loss='huber'
)
popt = result.x

A_opt = popt[:N_BRANCHES]
tau_opt = popt[N_BRANCHES:2*N_BRANCHES]

print("Optimal parameters (multi-exp + env + Fourier):")
for i in range(N_BRANCHES):
    print(f" A{i+1} = {A_opt[i]:.3f} MPa")
    print(f" tau{i+1} = {tau_opt[i]:.3f} years (base, before env correction)")

# =============================
# Monte Carlo simulation to evaluate parameter uncertainty
# =============================
n_iter = 1000
params_MC = np.zeros((n_iter, len(p0)))

for i in range(n_iter):
    sigma_sim = sigma_data + np.random.uniform(-sigma_uncertainty, sigma_uncertainty)
    result_MC = least_squares(
        residuals, p0, args=(t_data, sigma_sim, 20.0, 60.0, sigma_uncertainty),
        bounds=bounds, loss='huber'
    )
    params_MC[i, :] = result_MC.x

param_means = np.mean(params_MC, axis=0)
param_std = np.std(params_MC, axis=0)

print("\nMean parameters obtained by Monte Carlo:")
print("{:<10s} {:>15s} {:>15s}".format("Parameter", "Mean", "Std dev"))
for idx in range(N_BRANCHES):
    print("{:<10s} {:15.3f} {:15.3f}".format(f"A{idx+1}", param_means[idx], param_std[idx]))
for idx in range(N_BRANCHES):
    print("{:<10s} {:15.3f} {:15.3f}".format(f"τ{idx+1}", param_means[idx+N_BRANCHES], param_std[idx+N_BRANCHES]))

# =============================
# Calculate modeled ages for each sample
# =============================
df_samples["Model Age"] = df_samples.apply(lambda row: get_model_age_for_sample(row, T=20.0, H=60.0, params=popt, age_step=0.5), axis=1)
df_samples["Model Calculated Duration (yrs)"] = 2000 - df_samples["Model Age"]

print("\n=== Comparison: Real data vs. Age calculated by model (hybrid) ===")
print(df_samples[["Sample", "Dating", "Parsed Age (AD)", "σr (MPa)", "σr Uncertainty (MPa)",
                   "Model Calculated Duration (yrs)", "Model Age"]].to_string(index=False))

# =============================
# Visualization
# =============================
plt.figure(figsize=(8,6))
plt.errorbar(
    df_samples["Parsed Age (AD)"], df_samples["σr (MPa)"],
    xerr=df_samples["Uncertainty (yrs)"],
    yerr=df_samples["σr Uncertainty (MPa)"],
    fmt='o', color='b', capsize=5, markersize=6, elinewidth=1,
    label="C14 validated samples"
)
age_plot = np.linspace(-3500, 2100, 300)
sigma_plot = np.array([multi_exponential_env_fourier(2000 - age, 20.0, 60.0, *popt) for age in age_plot])
plt.plot(age_plot, sigma_plot, 'r-', label="Multi-exp + env + Fourier model")
plt.xlabel("Age (AD)")
plt.ylabel("Strength (MPa)")
plt.title("Comparison: Age (C14) vs. Measured strength and model")
plt.grid(True)
plt.legend()

# Add annotations for each sample with offset and bounding box
for idx, row in df_samples.iterrows():
    x_val = row["Parsed Age (AD)"]
    y_val = row["σr (MPa)"]
    sample_label = row["Sample"]
    plt.annotate(
        sample_label,
        xy=(x_val, y_val),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        ha='left',
        va='bottom',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="gray"),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
    )
plt.tight_layout()
fig = plt.gcf()  # Get current figure object
fig.savefig("figure1.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()

# =============================
# Date prediction from mechanical value (Approach 1)
# =============================
sigma_input = 243.22  # Resistance value (MPa)
predicted_age = invert_age(sigma_input, 20.0, 60.0, popt, age_guess=2000.0)
predicted_duration = 2000 - predicted_age
print(f"\nApproach 1: For a strength of {sigma_input:.1f} MPa, the predicted date is: {predicted_age:.1f} AD")
print(f"This corresponds to a calculated duration of {predicted_duration:.1f} years")

# =============================
# Monte Carlo uncertainty propagation on T and H (Approach 2)
# =============================
n_MC = 1000
predicted_dates = []
for _ in range(n_MC):
    T_sim = np.random.uniform(18, 22)
    H_sim = np.random.uniform(55, 65)
    date_sim = invert_age(sigma_input, T_sim, H_sim, popt, age_guess=2000.0)
    predicted_dates.append(date_sim)
predicted_dates = np.array(predicted_dates)
mean_date = np.mean(predicted_dates)
hdi = np.percentile(predicted_dates, [2.5, 97.5])
print(f"\nApproach 2 (Monte Carlo on T and H):")
print(f"Mean predicted date = {mean_date:.1f} AD")
print(f"95% credibility interval = [{hdi[0]:.1f}, {hdi[1]:.1f}] AD")

# =============================
# Sensitivity analysis: T and H variation
# =============================
T_values = np.linspace(18, 22, 20)
H_values = np.linspace(55, 65, 20)
T_grid, H_grid = np.meshgrid(T_values, H_values)
predicted_date_grid = np.zeros(T_grid.shape)
for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        predicted_date_grid[i, j] = invert_age(sigma_input, T_grid[i, j], H_grid[i, j], popt, age_guess=2000.0)

plt.figure(figsize=(8,6))
cp = plt.contourf(T_grid, H_grid, predicted_date_grid, cmap='viridis', levels=20)
plt.colorbar(cp, label="Predicted date (AD)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.title(f"Sensitivity analysis for σ = {sigma_input} MPa")
plt.tight_layout()
fig = plt.gcf()  # Get current figure object
fig.savefig("figure2.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()

# =============================
# Uncertainty propagation on date via finite difference method (Approach 3)
# =============================
def propagate_uncertainty(sigma_input, T, H, popt, cov_params, u_sigma, delta_sigma=0.1, delta_param=1e-3):
    """
    Calculates predicted date and its uncertainty by error propagation.
    - sigma_input: measured value of σ.
    - popt: vector of optimal parameters.
    - cov_params: parameter covariance matrix (calculated from params_MC).
    - u_sigma: uncertainty on σ measurement.
    - delta_sigma, delta_param: steps for finite differences.
    """
    t0 = invert_age(sigma_input, T, H, popt, age_guess=2000.0)
    t_plus = invert_age(sigma_input + delta_sigma, T, H, popt, age_guess=2000.0)
    dtdsigma = (t_plus - t0) / delta_sigma
    grad = np.zeros(len(popt))
    for i in range(len(popt)):
        popt_pert = popt.copy()
        popt_pert[i] += delta_param
        t_pert = invert_age(sigma_input, T, H, popt_pert, age_guess=2000.0)
        grad[i] = (t_pert - t0) / delta_param
    var_sigma = (dtdsigma ** 2) * (u_sigma ** 2)
    var_params = grad @ cov_params @ grad.T
    u_t = np.sqrt(var_sigma + var_params)
    return t0, u_t, dtdsigma, grad

cov_params = np.cov(params_MC.T)
u_sigma_input = np.mean(sigma_uncertainty)
t_predicted, u_t, dtdsigma, grad = propagate_uncertainty(sigma_input, 20.0, 60.0, popt, cov_params, u_sigma_input)
print(f"\nApproach 3 (Uncertainty propagation):")
print(f"Predicted date = {t_predicted:.1f} AD ± {u_t:.1f} years")

# =============================
# Approach 4 (Bayesian): Complete inference with PyMC
# =============================
import pymc as pm
import arviz as az

with pm.Model() as bayes_model:
    # Priors
    A1_b = pm.Normal("A1", mu=217, sigma=20)
    A2_b = pm.Normal("A2", mu=814, sigma=80)
    tau1_b = pm.Normal("tau1", mu=5900, sigma=300)
    tau2_b = pm.Normal("tau2", mu=380, sigma=60)

    T_const = 20.0
    H_const = 60.0
    t_shared = pm.Data("t_shared", t_data)

    sigma_model_bayes = multi_exponential_env_fourier(t_shared, T_const, H_const, A1_b, A2_b, tau1_b, tau2_b)

    sigma_obs = pm.Normal("sigma_obs", mu=sigma_model_bayes, sigma=sigma_uncertainty, observed=sigma_data)

    trace_bayes = pm.sample(2000, tune=5000, target_accept=0.999, return_inferencedata=True, nuts={"max_treedepth": 15}, random_seed=42)

# plt.show()
az.plot_trace(trace_bayes, compact=False, rug=True, backend_kwargs={'constrained_layout': True})  # Make sure trace contains A1, A2, tau1, tau2
plt.savefig("figure6.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n=== Approach 4 (Bayesian): Parameter summary ===")
print(az.summary(trace_bayes, round_to=1))

# Bayesian inversion: for each posterior sample, calculate predicted date for σ = 243.22 MPa.
posterior = trace_bayes.posterior
n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]

A1_samples = posterior["A1"].values.flatten()
A2_samples = posterior["A2"].values.flatten()
tau1_samples = posterior["tau1"].values.flatten()
tau2_samples = posterior["tau2"].values.flatten()

def invert_age_bayes(sigma_target, A1_, A2_, tau1_, tau2_, age_guess=2000.0):
    """Inversion for a Bayesian parameter sample."""
    def func(age):
        t_val = 2000 - age
        correction_fire = 0.5 if t_val >= 468 else 1.0
        tau1_eff = (tau1_ * correction_fire)
        tau2_eff = (tau2_ * correction_fire)
        sigma_calc = A1_ * np.exp(-t_val / tau1_eff) + A2_ * np.exp(-t_val / tau2_eff)
        sigma_calc += (50.0 * np.sin(2 * np.pi * t_val / 2000.0) +
                       30.0 * np.cos(2 * np.pi * t_val / 2000.0) +
                       30.0 * np.sin(2 * np.pi * t_val / 4000.0) +
                       15.0 * np.cos(2 * np.pi * t_val / 4000.0))
        return sigma_calc - sigma_target
    sol, infodict, ier, mesg = fsolve(func, age_guess, full_output=True)
    if ier == 1:
        return sol[0]
    else:
        return np.nan

sigma_input_bayes = 243.22
bayes_dates = []
for i in range(n_samples):
    params_sample = [A1_samples[i], A2_samples[i], tau1_samples[i], tau2_samples[i]]
    age_pred = invert_age_bayes(sigma_input_bayes, *params_sample, age_guess=2000.0)
    bayes_dates.append(age_pred)
bayes_dates = np.array(bayes_dates)
median_bayes_date = np.nanmedian(bayes_dates)
hdi_bayes = az.hdi(bayes_dates[~np.isnan(bayes_dates)], hdi_prob=0.95)

print(f"\n=== Approach 4 (Bayesian): Predicted date for σ = {sigma_input_bayes} MPa ===")
print(f"  Mean = {median_bayes_date:.1f} AD")
print(f"  95% HDI interval = [{hdi_bayes[0]:.1f}, {hdi_bayes[1]:.1f}] AD")

plt.figure(figsize=(8,4))
plt.hist(bayes_dates[~np.isnan(bayes_dates)], bins=30, color='skyblue', edgecolor='gray')
plt.xlabel("Predicted date (AD)")
plt.ylabel("Frequency")
plt.title("Posterior distribution of predicted dates (Bayesian)")
plt.tight_layout()
fig = plt.gcf()  # Get current figure object
fig.savefig("figure4.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()
