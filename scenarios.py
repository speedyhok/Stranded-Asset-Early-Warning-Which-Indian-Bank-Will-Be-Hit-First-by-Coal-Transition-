# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import numpy as np
import pandas as pd
import torch
from data_engine import generate_ngfs_scenarios, get_company_financials, get_banking_network
from carbon_beta import calculate_tvs
from survival_model import generate_training_cohort, fit_survival_model, predict_probabilities
from network_gnn import run_stress_propagation

def run_scenario_simulation(cph_model, baseline_tvs, baseline_fin, network_dict, scenarios_df):
    """
    Simulates the quarterly credit risk and bank capital erosion from 2026 to 2036.
    Under three NGFS scenarios.
    """
    quarters = scenarios_df.index
    company_names = list(baseline_fin.index)
    bank_names = list(network_dict["banks"].index)
    
    # Store results
    # Structure: {Scenario: {Quarter: {Bank/Company: Metrics}}}
    simulation_results = {sc: [] for sc in scenarios_df.columns}
    
    for scenario_name in scenarios_df.columns:
        print(f"Running simulation for NGFS Scenario: {scenario_name}...")
        
        for q in quarters:
            carbon_price = scenarios_df.loc[q, scenario_name]
            
            # 1. Update Company Financials & TVS dynamically based on carbon price
            current_fin = baseline_fin.copy()
            current_tvs = baseline_tvs.copy()
            
            for company in company_names:
                fin = baseline_fin.loc[company]
                tvs_row = baseline_tvs.loc[company]
                
                # EBITDA impact: drops as carbon costs rise (proportional to carbon revenue share)
                # Max EBITDA reduction capped at 75% for extreme carbon prices
                ebitda_mult = np.clip(1.0 - (fin["Carbon_Rev_Pct"] * (carbon_price - 15.0) / 400.0), 0.25, 1.0)
                adj_ebitda = fin["EBITDA_INR_Cr"] * ebitda_mult
                adj_debt_ebitda = fin["Total_Debt_INR_Cr"] / adj_ebitda
                
                # Refinancing risk impact: gets worse if Capex is fossil-heavy and carbon price is high
                # High capex ratio = fossil-heavy. It cuts down remaining years to maturity
                ref_mult = np.clip(1.0 - (fin["Capex_Fossil_Renewable"] * (carbon_price - 15.0) / 800.0), 0.10, 1.5)
                adj_ref_years = np.clip(fin["Refinancing_Years"] * ref_mult, 0.2, 10.0)
                
                # Update TVS: rises with carbon price (Carbon Beta term dominates)
                tvs_mult = (carbon_price - 15.0) * (fin["Carbon_Rev_Pct"] / 2.0)
                adj_tvs = np.clip(tvs_row["TVS"] + tvs_mult, 0.0, 100.0)
                
                # Write to current dataframe
                current_fin.loc[company, "Debt_EBITDA"] = adj_debt_ebitda
                current_fin.loc[company, "Refinancing_Years"] = adj_ref_years
                current_tvs.loc[company, "TVS"] = adj_tvs
                
            # 2. Predict forward-looking 18-month default probabilities (PD_18M) using CPH
            pds_df, _ = predict_probabilities(cph_model, current_tvs, current_fin)
            
            # 3. Propagate stress through banking network via Contagion GNN
            stressed_banks = run_stress_propagation(pds_df["PD_18M"], network_dict)
            
            # Record status
            for bank in bank_names:
                res_row = stressed_banks.loc[bank]
                simulation_results[scenario_name].append({
                    "Quarter": q,
                    "Entity": bank,
                    "Type": "Bank",
                    "Capital_Ratio_Pct": res_row["Stressed_CAR_Pct"],
                    "Initial_Capital_Ratio_Pct": res_row["Initial_CAR_Pct"],
                    "Capital_Loss_INR_Cr": res_row["Initial_Capital_INR_Cr"] - res_row["Stressed_Capital_INR_Cr"],
                    "Distress_Factor_Pct": res_row["Distress_Factor_Pct"],
                    "Status": res_row["Status"]
                })
                
            for company in company_names:
                simulation_results[scenario_name].append({
                    "Quarter": q,
                    "Entity": company,
                    "Type": "Company",
                    "TVS": current_tvs.loc[company, "TVS"],
                    "PD_18M_Pct": pds_df.loc[company, "PD_18M"] * 100.0,
                    "Debt_EBITDA": current_fin.loc[company, "Debt_EBITDA"]
                })
                
    # Format all results into scenario-wise dataframes
    formatted_dfs = {}
    for sc, records in simulation_results.items():
        formatted_dfs[sc] = pd.DataFrame(records)
        
    return formatted_dfs

def analyze_scenario_impact(sim_results):
    """
    Summarizes which banks fail/breach capital adequacy, when they fail, 
    and total capital lost under each scenario.
    """
    summary = []
    
    for sc_name, df in sim_results.items():
        banks_df = df[df["Type"] == "Bank"]
        
        # Total Capital Loss at final quarter
        final_q = banks_df["Quarter"].max()
        final_banks = banks_df[banks_df["Quarter"] == final_q]
        total_loss = final_banks["Capital_Loss_INR_Cr"].sum()
        
        # Check first breach (CAR < 9.0%)
        breached_banks = banks_df[banks_df["Capital_Ratio_Pct"] < 9.0]
        
        first_breach_bank = "None"
        first_breach_quarter = "N/A"
        
        if not breached_banks.empty:
            earliest_breach = breached_banks.sort_values(by="Quarter").iloc[0]
            first_breach_bank = earliest_breach["Entity"]
            first_breach_quarter = earliest_breach["Quarter"].strftime("%Y-%m")
            
        summary.append({
            "Scenario": sc_name,
            "Total_Capital_Loss_INR_Cr": total_loss,
            "First_Bank_To_Breach": first_breach_bank,
            "First_Breach_Date": first_breach_quarter,
            "Number_Of_Breached_Banks": len(final_banks[final_banks["Capital_Ratio_Pct"] < 9.0])
        })
        
    return pd.DataFrame(summary)

if __name__ == "__main__":
    from data_engine import fetch_or_generate_market_data
    
    # Run test
    m_df = fetch_or_generate_market_data()
    fin_df = get_company_financials()
    tvs_df = calculate_tvs(m_df)
    scenarios_df = generate_ngfs_scenarios()
    network = get_banking_network()
    
    cohort = generate_training_cohort(tvs_df, fin_df)
    cph = fit_survival_model(cohort)
    
    sim_res = run_scenario_simulation(cph, tvs_df, fin_df, network, scenarios_df)
    impact = analyze_scenario_impact(sim_res)
    print("\nNGFS Scenario Impacts on Banking Network:")
    print(impact)
