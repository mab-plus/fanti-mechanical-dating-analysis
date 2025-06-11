#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison between Fanti's multilinear regression (2015) and our viscoelastic model.
Shows the fundamental differences between empirical and physics-based approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import from modelisation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling.modelisation import multi_exponential_env_fourier, popt

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
    
    plt.tight_layout()
    plt.savefig('figure_comparison_fanti.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison figure saved as 'figure_comparison_fanti.png'")
    
    # Calculate and print comparison metrics
    print("\n=== Model Comparison Metrics ===")
    # Here you could add RMSE, R² comparisons, etc.

if __name__ == "__main__":
    plot_model_comparison()
