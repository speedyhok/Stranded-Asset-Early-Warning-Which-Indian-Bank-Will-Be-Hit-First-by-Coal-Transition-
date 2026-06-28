# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_engine import (
    fetch_or_generate_market_data,
    generate_ngfs_scenarios,
    get_company_financials,
    get_historical_defaults_data,
    get_banking_network
)
from carbon_beta import calculate_tvs
from survival_model import generate_training_cohort, fit_survival_model, predict_probabilities
from scenarios import run_scenario_simulation, analyze_scenario_impact

# Artifact output directory
ARTIFACT_DIR = r"C:\Users\Administrator\.gemini\antigravity\brain\4c4cf116-fed6-4624-9ed7-e92751811c0d"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def main():
    print("="*60)
    print("CLIMATE CREDIT DETERIORATION & BANK CONTAGION PIPELINE")
    print("="*60)
    
    # 1. Fetch market data & baseline financials
    market_df = fetch_or_generate_market_data()
    baseline_fin = get_company_financials()
    network_dict = get_banking_network()
    
    # 2. Stage 1: Transition Vulnerability Score (TVS)
    print("\nStage 1: Estimating Carbon Beta and Transition Vulnerability Score (TVS)...")
    tvs_results = calculate_tvs(market_df)
    print("TVS Scores calculated successfully:")
    print(tvs_results[["Market_Beta", "Carbon_Beta", "Event_CAR", "NLP_Sentiment_Score", "TVS"]].round(4))
    
    # Save Stage 1 plots
    plt.figure(figsize=(10, 5))
    bars = plt.bar(tvs_results.index, tvs_results["TVS"], color=['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c', '#428bca'])
    plt.title("Transition Vulnerability Score (TVS) per Company (Stage 1 Unsupervised ML)")
    plt.ylabel("TVS (0 = Resilient, 100 = Highly Vulnerable)")
    plt.xlabel("Company")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "unsupervised_signals.png"), dpi=150)
    plt.close()
    print(f"Saved TVS plot to: {os.path.join(ARTIFACT_DIR, 'unsupervised_signals.png')}")
    
    # 3. Stage 2: Survival Analysis CPH Model Fitting
    print("\nStage 2: Training Cox Proportional Hazards Model on Cohort...")
    cohort = generate_training_cohort(tvs_results, baseline_fin)
    cph = fit_survival_model(cohort)
    print("Cox Proportional Hazards Model coefficients:")
    print(cph.summary[["coef", "exp(coef)", "p", "z"]].round(4))
    
    # Predict default probabilities for current active companies
    pds, pd_curves = predict_probabilities(cph, tvs_results, baseline_fin)
    print("\nTarget Active Companies - Predicted Default Probabilities (PD):")
    print(pds[["TVS", "Debt_EBITDA", "PD_12M", "PD_18M", "PD_24M", "PD_36M"]].round(4))
    
    # Plot survival curves of active companies
    plt.figure(figsize=(10, 6))
    for col in pd_curves.columns:
        # Survival S(t) = 1 - PD(t)
        plt.plot(pd_curves.index, 1.0 - pd_curves[col], label=col, linewidth=2.5)
    plt.title("Company Survival Probability Curves S(t) (Stage 2 Survival Analysis)")
    plt.xlabel("Timeline (Years)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "company_survival_curves.png"), dpi=150)
    plt.close()
    print(f"Saved survival curves plot to: {os.path.join(ARTIFACT_DIR, 'company_survival_curves.png')}")
    
    # 4. Validation & Backtesting
    print("\nValidation: Backtesting model predictions on historical defaults...")
    hist_defaults = get_historical_defaults_data()
    # Align features for CPH prediction
    hist_features = hist_defaults[["TVS", "Debt_EBITDA", "Carbon_Rev_Pct", "Capex_Fossil_Renewable", "Refinancing_Years"]].astype(float)
    hist_survival = cph.predict_survival_function(hist_features)
    hist_pd = 1.0 - hist_survival
    
    backtest_records = []
    for name in hist_defaults.index:
        actual_fail_time = hist_defaults.loc[name, "Observed_Time"]
        
        # 12-18 months before failure:
        # Lanco (failed t=2.0) -> early signal at t=0.5 to t=1.0
        # Essar (failed t=3.0) -> early signal at t=1.5 to t=2.0
        # GVK (failed t=4.0) -> early signal at t=2.5 to t=3.0
        warning_time_12m = actual_fail_time - 1.0
        warning_time_18m = actual_fail_time - 1.5
        
        # Extract PD at warning times
        idx_12m = hist_pd.index[np.abs(hist_pd.index - warning_time_12m).argmin()]
        idx_18m = hist_pd.index[np.abs(hist_pd.index - warning_time_18m).argmin()]
        
        pd_at_12m_warning = hist_pd.loc[idx_12m, name]
        pd_at_18m_warning = hist_pd.loc[idx_18m, name]
        
        backtest_records.append({
            "Company": name,
            "Actual_NPA_Year": actual_fail_time,
            "Warning_Horizon_18M_Year": warning_time_18m,
            "Predicted_PD_at_18M_Warning_Pct": pd_at_18m_warning * 100.0,
            "Warning_Horizon_12M_Year": warning_time_12m,
            "Predicted_PD_at_12M_Warning_Pct": pd_at_12m_warning * 100.0,
        })
    backtest_df = pd.DataFrame(backtest_records)
    print("Backtesting Results on historical failures:")
    print(backtest_df.round(2))
    
    # 5. Stage 4: Run Scenario Simulations & Network GNN Propagation
    print("\nStage 3 & 4: Simulating Systemic Risk under NGFS Scenarios (2026-2036)...")
    scenarios_df = generate_ngfs_scenarios()
    sim_results = run_scenario_simulation(cph, tvs_results, baseline_fin, network_dict, scenarios_df)
    impact_summary = analyze_scenario_impact(sim_results)
    
    print("\nSystemic Impact Summary:")
    print(impact_summary.to_string(index=False))
    
    # Plot Bank CAR trajectories under different scenarios
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=True)
    sc_colors = {"Net_Zero": "#428bca", "Delayed_Transition": "#d9534f", "BAU": "#5cb85c"}
    bank_styles = {"SBI": "-", "PNB": "--", "BOB": "-.", "ICICI": ":", "HDFC": "-"}
    
    for idx, sc_name in enumerate(scenarios_df.columns):
        sc_df = sim_results[sc_name]
        banks_sc_df = sc_df[sc_df["Type"] == "Bank"]
        
        ax = axes[idx]
        for bank in network_dict["banks"].index:
            bank_data = banks_sc_df[banks_sc_df["Entity"] == bank]
            ax.plot(bank_data["Quarter"], bank_data["Capital_Ratio_Pct"], 
                    label=bank, linestyle=bank_styles[bank], linewidth=2.5)
            
        # Draw regulatory minimum CAR of 9.0%
        ax.axhline(y=9.0, color='r', linestyle=':', alpha=0.8, label="Regulatory Min CAR (9.0%)")
        ax.set_title(f"Bank Capital Adequacy Ratio (CAR) Trajectory: {sc_name} Scenario", fontsize=11, fontweight='bold')
        ax.set_ylabel("Stressed CET1 CAR (%)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(2.0, 18.0)
        if idx == 0:
            ax.legend(loc="lower left", ncol=3)
            
    axes[-1].set_xlabel("Quarter")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "bank_car_trajectories.png"), dpi=150)
    plt.close()
    print(f"Saved Bank CAR trajectories plot to: {os.path.join(ARTIFACT_DIR, 'bank_car_trajectories.png')}")
    
    # Save simulation dataframes as csv for record
    for sc, df in sim_results.items():
        csv_path = os.path.join(ARTIFACT_DIR, f"{sc.lower()}_simulation.csv")
        df.to_csv(csv_path, index=False)
        
    print("\n" + "="*60)
    print("Pipeline Execution Completed Successfully.")
    print("="*60)

if __name__ == "__main__":
    main()
