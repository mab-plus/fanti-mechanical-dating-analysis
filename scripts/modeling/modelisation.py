#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modélisation de la dégradation du lin : modèle multi-exponentiel avec correction environnementale,
composante Fourier, propagation de l'incertitude sur T et H, intégration de l'effet du feu de Chambéry (1532),
propagation des incertitudes sur la date prédite via la méthode des différences finies, affichage amélioré,
et intégration d'une approche bayésienne complète.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares, fsolve

# =======================
# 1) Paramètres du modèle de base
# =======================
sigma0 = 1103         # Résistance initiale de référence (MPa)
sigma_infty = 0       # Résistance résiduelle (MPa)
tau = 1102            # Temps caractéristique de dégradation (années)

# =======================
# 2) Paramètres de correction environnementale pour σ₀
# =======================
alpha = 1e-4
beta = 2e-4
gamma = 1e-8
delta = 4e-8
T0 = 20.0  # Température de référence (°C)

def sigma0_env(T, H, sigma0_base):
    """Corrige la résistance initiale en fonction de la température et de l'humidité."""
    return sigma0_base * np.exp(alpha * H + beta * (T - T0) + gamma * H**2 + delta * (T - T0)**2)

# =======================
# 3) Paramètres environnementaux pour la modification de tau
# =======================
T_ref = 20.0    # Température de référence
H_ref = 60.0    # Humidité de référence
kappa_T_tau = 0.01   # Influence de T sur tau
kappa_H_tau = 0.01   # Influence de H sur tau

def effective_tau(tau_i, T, H):
    """Calcule le temps caractéristique effectif en fonction de T et H."""
    return tau_i / (1 + kappa_T_tau * (T - T_ref) + kappa_H_tau * (H - H_ref))

# =======================
# 4) Décomposition de Fourier (composantes sinusoïdales et cosinusoïdales)
# =======================
fourier_coeffs_sin = [50.0, 30.0]    # Amplitudes des sinusoïdes
fourier_coeffs_cos = [30.0, 15.0]    # Amplitudes des cosinusoïdales
fourier_periods = [2000.0, 4000.0]   # Périodes correspondantes (années)
fourier_phases = [0.0, 0.0]          # Phases (en radians)

def fourier_series(t, coeffs_sin, coeffs_cos, periods, phases):
    """Calcule la somme d'une série de Fourier."""
    result = 0.0
    for A_sin, A_cos, P, phi in zip(coeffs_sin, coeffs_cos, periods, phases):
        result += A_sin * np.sin(2 * np.pi * t / P + phi) + A_cos * np.cos(2 * np.pi * t / P + phi)
    return result

# =======================
# 5) Modèle multi-exponentiel avec correction environnementale, composante Fourier et effet du feu
# =======================
N_BRANCHES = 2  # Nombre de branches exponentielles

def multi_exponential_env_fourier(t, T, H, *params):
    """
    Modèle complet :
      σ(t) = σ_infty + Σ_{i=1}^{N_BRANCHES} A_i * exp(-t / τ_i_eff) + Fourier(t)
    avec τ_i_eff = τ_i_corr / (1 + kappa_T_tau*(T - T_ref) + kappa_H_tau*(H - H_ref))
    L'effet du feu est intégré via un facteur correctif de 0.5 si t >= t_fire.
    t est défini comme t = 2000 - Age (donc t=468 correspond à l'incendie de 1532).
    """
    # On essaie de convertir t en tableau NumPy
    try:
        t_np = np.asarray(t)
        if t_np.ndim == 0:
            correction_fire = 0.5 if t_np >= (2000 - 1532) else 1.0
        else:
            correction_fire = np.where(t_np >= (2000 - 1532), 0.5, 1.0)
    except Exception:
        # Si t est symbolique (PyMC), on utilise pm.math.switch
        import pymc as pm
        t_fire = 2000 - 1532  # 468 ans
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
    """Wrapper pour l'ajustement avec T et H fixes (20°C, 60% RH)."""
    T_const = 20.0
    H_const = 60.0
    return multi_exponential_env_fourier(t, T_const, H_const, *params)

# =============================
# 6) Données des échantillons de Fanti (simplifiées)
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
    """Extrait un âge moyen (en années AD, négatif pour B.C.) depuis la chaîne."""
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
    Extrait l'intervalle de datation à partir de la chaîne.
    Renvoie (min_age, max_age) en années AD.
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
# Fonctions d'inversion pour retrouver l'âge à partir d'une résistance donnée
# =============================
def invert_age_in_interval(target_sigma, A_min, A_max, T, H, params, age_step=1.0):
    """
    Recherche l'âge dans l'intervalle [A_min, A_max] qui minimise |σ_modele - σ_target|.
    Renvoie l'âge optimal.
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
    """Fonction dont la racine (âge) vérifie multi_exponential_env_fourier(2000 - age) = σ_target."""
    t_val = 2000 - age
    return multi_exponential_env_fourier(t_val, T, H, *params) - target_sigma

def invert_age(target_sigma, T, H, params, age_guess=2000.0):
    """
    Tente d'utiliser fsolve pour retrouver l'âge correspondant à σ_target.
    En cas d'échec, utilise l'inversion discrète.
    """
    sol, infodict, ier, mesg = fsolve(func_to_solve, age_guess, args=(target_sigma, T, H, params), full_output=True)
    if ier == 1:
        return sol[0]
    else:
        return invert_age_in_interval(target_sigma, -3500, 2100, T, H, params)

def get_model_age_for_sample(row, T=20.0, H=60.0, params=None, age_step=0.5):
    """
    Pour un échantillon (row), si un intervalle de datation est disponible,
    recherche l'âge optimal dans cet intervalle ; sinon, utilise l'inversion hybride.
    """
    if params is None:
        raise ValueError("Veuillez fournir les paramètres 'params' issus du fit.")
    A_min = row["Dating_min"]
    A_max = row["Dating_max"]
    target_sigma = row["σr (MPa)"]
    if A_min is not None and A_max is not None:
        age_opt = invert_age_in_interval(target_sigma, A_min, A_max, T, H, params, age_step=age_step)
        if age_opt is not None:
            return age_opt
    return invert_age(target_sigma, T, H, params)

# =============================
# Calcul des âges parsés et durée correspondante
# =============================
df_samples["Parsed Age (AD)"] = df_samples["Dating"].apply(parse_age)
df_samples["t (yrs)"] = 2000 - df_samples["Parsed Age (AD)"]
df_samples["Dating_min"], df_samples["Dating_max"] = zip(*df_samples["Dating"].apply(parse_dating_interval))

# =============================
# Ajustement du modèle aux données de résistance (σr)
# =============================
t_data = df_samples["t (yrs)"].values
sigma_data = df_samples["σr (MPa)"].values

# Calcul de l'incertitude sur σ (15% ou 1 MPa minimum)
error_floor_fraction = 0.15
minimum_sigma_uncertainty = 1.0
sigma_uncertainty = np.maximum(minimum_sigma_uncertainty, error_floor_fraction * sigma_data)
df_samples["σr Uncertainty (MPa)"] = sigma_uncertainty

# Paramètres initiaux pour [A1, A2, τ1, τ2] (modèle à 2 branches)
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

print("Paramètres optimaux (multi-exp + env + Fourier) :")
for i in range(N_BRANCHES):
    print(f" A{i+1} = {A_opt[i]:.3f} MPa")
    print(f" tau{i+1} = {tau_opt[i]:.3f} ans (base, avant correction env)")

# =============================
# Simulation Monte Carlo pour évaluer l'incertitude des paramètres
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

print("\nParamètres moyens obtenus par Monte Carlo :")
print("{:<10s} {:>15s} {:>15s}".format("Paramètre", "Moyenne", "Ecart-type"))
for idx in range(N_BRANCHES):
    print("{:<10s} {:15.3f} {:15.3f}".format(f"A{idx+1}", param_means[idx], param_std[idx]))
for idx in range(N_BRANCHES):
    print("{:<10s} {:15.3f} {:15.3f}".format(f"τ{idx+1}", param_means[idx+N_BRANCHES], param_std[idx+N_BRANCHES]))

# =============================
# Calcul des âges modélisés pour chaque échantillon
# =============================
df_samples["Model Age"] = df_samples.apply(lambda row: get_model_age_for_sample(row, T=20.0, H=60.0, params=popt, age_step=0.5), axis=1)
df_samples["Model Calculated Duration (yrs)"] = 2000 - df_samples["Model Age"]

print("\n=== Comparatif : Données réelles vs. Âge calculé par le modèle (hybride) ===")
print(df_samples[["Sample", "Dating", "Parsed Age (AD)", "σr (MPa)", "σr Uncertainty (MPa)",
                   "Model Calculated Duration (yrs)", "Model Age"]].to_string(index=False))

# =============================
# Visualisation
# =============================
plt.figure(figsize=(8,6))
plt.errorbar(
    df_samples["Parsed Age (AD)"], df_samples["σr (MPa)"],
    xerr=df_samples["Uncertainty (yrs)"],
    yerr=df_samples["σr Uncertainty (MPa)"],
    fmt='o', color='b', capsize=5, markersize=6, elinewidth=1,
    label="Échantillons C14 validés"
)
age_plot = np.linspace(-3500, 2100, 300)
sigma_plot = np.array([multi_exponential_env_fourier(2000 - age, 20.0, 60.0, *popt) for age in age_plot])
plt.plot(age_plot, sigma_plot, 'r-', label="Modèle multi-exp + env + Fourier")
plt.xlabel("Âge (AD)")
plt.ylabel("Résistance (MPa)")
plt.title("Comparaison : Âge (C14) vs. Résistance mesurée et modèle")
plt.grid(True)
plt.legend()

# Ajout des annotations pour chaque échantillon avec offset et boîte englobante
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
# Prédiction de date à partir d'une valeur mécanique (Approche 1)
# =============================
sigma_input = 243.22  # Valeur de résistance (MPa)
predicted_age = invert_age(sigma_input, 20.0, 60.0, popt, age_guess=2000.0)
predicted_duration = 2000 - predicted_age
print(f"\nApproche 1 : Pour une résistance de {sigma_input:.1f} MPa, la date prédite est : {predicted_age:.1f} AD")
print(f"Ce qui correspond à une durée calculée de {predicted_duration:.1f} ans")

# =============================
# Propagation Monte Carlo de l'incertitude sur T et H (Approche 2)
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
print(f"\nApproche 2 (Monte Carlo sur T et H) :")
print(f"Date prédite moyenne = {mean_date:.1f} AD")
print(f"Intervalle de crédibilité 95% = [{hdi[0]:.1f}, {hdi[1]:.1f}] AD")

# =============================
# Analyse de sensibilité : Variation de T et H
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
plt.colorbar(cp, label="Date prédite (AD)")
plt.xlabel("Température (°C)")
plt.ylabel("Humidité (%)")
plt.title(f"Analyse de sensibilité pour σ = {sigma_input} MPa")
plt.tight_layout()
fig = plt.gcf()  # Get current figure object
fig.savefig("figure2.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()

# =============================
# Propagation d'incertitude sur la date via la méthode des différences finies (Approche 3)
# =============================
def propagate_uncertainty(sigma_input, T, H, popt, cov_params, u_sigma, delta_sigma=0.1, delta_param=1e-3):
    """
    Calcule la date prédite et son incertitude par propagation d'erreur.
    - sigma_input : valeur mesurée de σ.
    - popt : vecteur des paramètres optimaux.
    - cov_params : matrice de covariance des paramètres (calculée à partir de params_MC).
    - u_sigma : incertitude sur la mesure de σ.
    - delta_sigma, delta_param : pas pour les différences finies.
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
print(f"\nApproche 3 (Propagation d'incertitude) :")
print(f"Date prédite = {t_predicted:.1f} AD ± {u_t:.1f} ans")

# =============================
# Approche 4 (Bayésienne) : Inférence complète avec PyMC
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
az.plot_trace(trace_bayes)  # Assure-toi que trace contient A1, A2, tau1, tau2
plt.savefig("figure6.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n=== Approche 4 (Bayésienne) : Résumé des paramètres ===")
print(az.summary(trace_bayes, round_to=1))

# Inversion bayésienne : pour chaque échantillon postérieur, calculer la date prédite pour σ = 243.22 MPa.
posterior = trace_bayes.posterior
n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]

A1_samples = posterior["A1"].values.flatten()
A2_samples = posterior["A2"].values.flatten()
tau1_samples = posterior["tau1"].values.flatten()
tau2_samples = posterior["tau2"].values.flatten()

def invert_age_bayes(sigma_target, A1_, A2_, tau1_, tau2_, age_guess=2000.0):
    """Inversion pour un échantillon bayésien de paramètres."""
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

print(f"\n=== Approche 4 (Bayésienne) : Date prédite pour σ = {sigma_input_bayes} MPa ===")
print(f"  Moyenne = {median_bayes_date:.1f} AD")
print(f"  Intervalle HDI 95% = [{hdi_bayes[0]:.1f}, {hdi_bayes[1]:.1f}] AD")

plt.figure(figsize=(8,4))
plt.hist(bayes_dates[~np.isnan(bayes_dates)], bins=30, color='skyblue', edgecolor='gray')
plt.xlabel("Date prédite (AD)")
plt.ylabel("Fréquence")
plt.title("Distribution postérieure des dates prédites (Bayésien)")
plt.tight_layout()
fig = plt.gcf()  # Get current figure object
fig.savefig("figure4.png", dpi=300, bbox_inches="tight")
plt.close()
# plt.show()
