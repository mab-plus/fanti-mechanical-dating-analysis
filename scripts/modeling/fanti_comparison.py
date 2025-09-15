#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison between Fanti's multilinear regression (2015) and our viscoelastic model.
Shows the fundamental differences between empirical and physics-based approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

# 1) PDF vectoriel avec texte "vrai" (polices embarquées)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'

def save_figure(fig, basename, kind='combo', width_mm=90, height_mm=None, min_height_in=3.0):
    w_in = width_mm / 25.4
    w0, h0 = fig.get_size_inches()
    if height_mm is not None:
        h_in = height_mm / 25.4
    else:
        aspect = (h0 / w0) if w0 else 0.75
        h_in = max(w_in * aspect, min_height_in)   # <-- évite les panneaux “écrasés”
    # fig.set_size_inches(w_in, h_in, forward=True)
    fig.tight_layout()
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")             # PDF vectoriel
    dpi = 1000 if kind == 'lineart' else 600
    fig.savefig(f"{basename}.png", dpi=dpi, bbox_inches="tight")    # PNG conforme


# Add parent directory to path to import from modelisation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.modelisation import multi_exponential_env_fourier, popt

def add_ts_c14_refs(ax, sigma_ts=243.22, c14_age=1325, c14_err=65,
                    show_point=True, annotate=True):
    # Cibles (TS mécanique + ^14C)
    ax.axhline(sigma_ts, ls="--", lw=1.4, alpha=0.9)                          # TS (horiz.)  :contentReference[oaicite:1]{index=1}
    ax.axvline(c14_age, ls=":", lw=1.4, alpha=0.9, zorder=90)                  # ^14C (vert.)  :contentReference[oaicite:2]{index=2}
    ax.axvspan(c14_age-c14_err, c14_age+c14_err, alpha=0.15, zorder=10)        # bande ±err   :contentReference[oaicite:3]{index=3}

    if show_point:
        ax.errorbar(c14_age, sigma_ts, xerr=c14_err, fmt="o", ms=8,            # point ^14C  :contentReference[oaicite:4]{index=4}
                    mfc="gold", mec="k", mew=1.3, capsize=3, zorder=100,
                    label=r"TS $^{14}$C 1325$\pm$65 AD")
    if annotate:
        ax.annotate(r"TS $^{14}$C 1325$\pm$65 AD",                             # étiquette   :contentReference[oaicite:5]{index=5}
                    xy=(c14_age, sigma_ts), xytext=(c14_age+180, sigma_ts+80),
                    arrowprops=dict(arrowstyle="->", lw=1),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7),
                    ha="left", va="bottom", zorder=101)

    # Légende sans doublons
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False)

def load_fanti_data():
    """Load Fanti et al. (2015) mechanical data"""
    df = pd.read_csv('../../data/fanti_mechanical_data.csv')
    df['t_years'] = 2000 - df['Age_AD']
    return df

def plot_model_comparison():
    """Create comparison plot between Fanti's data and our model"""
    df_fanti = load_fanti_data()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Direct comparison
    ax1.scatter(df_fanti['Age_AD'], df_fanti['sigma_r_MPa'], 
               color='red', s=100, label='Fanti data (2015)', 
               marker='s', edgecolor='black', linewidth=1.5)
    
    # Add our model prediction
    age_range = np.linspace(-3500, 2100, 500)
    t_range = 2000 - age_range
    sigma_model = [multi_exponential_env_fourier(t, 20.0, 60.0, *popt) for t in t_range]
    
    ax1.plot(age_range, sigma_model, 'b-', label='Our viscoelastic model', linewidth=2)
    ax1.set_xlabel('Age (AD)', fontsize=12)
    ax1.set_ylabel('Mechanical strength (MPa)', fontsize=12)
    ax1.set_title('Comparison: Fanti Data vs Viscoelastic Model', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    add_ts_c14_refs(ax1)
    
    # Plot 2: Log scale to show non-linearity
    ax2.scatter(df_fanti['Age_AD'], df_fanti['sigma_r_MPa'], 
               color='red', s=100, label='Fanti data (2015)', 
               marker='s', edgecolor='black', linewidth=1.5)
    ax2.plot(age_range, sigma_model, 'b-', label='Our viscoelastic model', linewidth=2)
    ax2.set_xlabel('Age (AD)', fontsize=12)
    ax2.set_ylabel('Mechanical strength (MPa)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Log Scale View: Highlighting Non-linear Behavior', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    add_ts_c14_refs(ax2)
    
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, "Fig9_model_vs_fanti_dataset", kind='lineart', width_mm=90)
    plt.close()
    
    print("Comparison figure saved as 'Fig9_model_vs_fanti_dataset.png'")
    
    # Calculate and print comparison metrics
    print("\n=== Model Comparison Metrics ===")
    # Here you could add RMSE, R² comparisons, etc.

if __name__ == "__main__":
    plot_model_comparison()
