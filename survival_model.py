# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

def generate_training_cohort(active_tvs, active_financials):
    """
    Generates a realistic cohort of 50 carbon-intensive peer firms (steel, power, cement, mining)
    spanning 2015-2026. Some defaulted (Event=1), some censored (Event=0), with realistic
    relationships between TVS, leverage, and refinancing risk.
    """
    np.random.seed(101)
    n_firms = 50
    
    # Base structures
    data = []
    
    # 1. Add our 3 historical defaults (ground truth failures)
    # Lanco, Essar, GVK
    data.append({
        "Company_Name": "Lanco Infratech", "TVS": 75.0, "Debt_EBITDA": 7.5,
        "Carbon_Rev_Pct": 0.90, "Capex_Fossil_Renewable": 12.0, "Refinancing_Years": 0.8,
        "Observed_Time": 2.0, "Event": 1
    })
    data.append({
        "Company_Name": "Essar Power", "TVS": 72.0, "Debt_EBITDA": 6.8,
        "Carbon_Rev_Pct": 0.95, "Capex_Fossil_Renewable": 10.0, "Refinancing_Years": 1.2,
        "Observed_Time": 3.0, "Event": 1
    })
    data.append({
        "Company_Name": "GVK Power", "TVS": 70.0, "Debt_EBITDA": 8.2,
        "Carbon_Rev_Pct": 0.88, "Capex_Fossil_Renewable": 7.0, "Refinancing_Years": 1.5,
        "Observed_Time": 4.0, "Event": 1
    })
    
    # 2. Add our active target companies (using actual calculated TVS + financial baseline)
    for name in active_tvs.index:
        tvs = active_tvs.loc[name, "TVS"]
        fin = active_financials.loc[name]
        data.append({
            "Company_Name": name, "TVS": tvs, "Debt_EBITDA": fin["Debt_EBITDA"],
            "Carbon_Rev_Pct": fin["Carbon_Rev_Pct"], "Capex_Fossil_Renewable": fin["Capex_Fossil_Renewable"],
            "Refinancing_Years": fin["Refinancing_Years"], "Observed_Time": 10.0, "Event": 0
        })
        
    # 3. Add synthetic cohort peers (42 additional companies to stabilize CPH)
    for i in range(42):
        is_default = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_default:
            tvs = np.random.uniform(55.0, 85.0)
            debt_ebitda = np.random.uniform(4.5, 9.0)
            carbon_rev = np.random.uniform(0.70, 0.99)
            capex_ratio = np.random.uniform(4.0, 15.0)
            ref_years = np.random.uniform(0.5, 2.5)
            # Default happens early
            obs_time = np.random.uniform(1.0, 5.0)
        else:
            tvs = np.random.uniform(15.0, 55.0)
            debt_ebitda = np.random.uniform(0.5, 4.0)
            carbon_rev = np.random.uniform(0.10, 0.80)
            capex_ratio = np.random.uniform(0.5, 3.5)
            ref_years = np.random.uniform(3.0, 8.0)
            # Censored at end of observation
            obs_time = 10.0
            
        data.append({
            "Company_Name": f"Synthetic_Peer_{i+1}", "TVS": tvs, "Debt_EBITDA": debt_ebitda,
            "Carbon_Rev_Pct": carbon_rev, "Capex_Fossil_Renewable": capex_ratio, "Refinancing_Years": ref_years,
            "Observed_Time": obs_time, "Event": is_default
        })
        
    return pd.DataFrame(data)

def fit_survival_model(cohort_df):
    """
    Fits a Cox Proportional Hazards Model on the training cohort.
    """
    cph = CoxPHFitter()
    
    # We drop the name column as CPH needs numeric features
    train_features = cohort_df.drop(columns=["Company_Name"])
    
    # Fit the model
    cph.fit(train_features, duration_col="Observed_Time", event_col="Event")
    return cph

def predict_probabilities(cph_model, active_tvs, active_financials):
    """
    Predicts probability of default (PD) for our active companies at 12, 18, 24, and 36 months.
    """
    target_data = []
    names = list(active_tvs.index)
    
    for name in names:
        tvs = active_tvs.loc[name, "TVS"]
        fin = active_financials.loc[name]
        target_data.append({
            "TVS": tvs,
            "Debt_EBITDA": fin["Debt_EBITDA"],
            "Carbon_Rev_Pct": fin["Carbon_Rev_Pct"],
            "Capex_Fossil_Renewable": fin["Capex_Fossil_Renewable"],
            "Refinancing_Years": fin["Refinancing_Years"]
        })
        
    target_df = pd.DataFrame(target_data, index=names)
    
    # Predict survival function S(t)
    # The output dataframe has index = times, columns = companies
    survival_fns = cph_model.predict_survival_function(target_df)
    
    # Calculate PD = 1 - S(t)
    pd_fns = 1.0 - survival_fns
    
    # Extract specific horizons (time is in years in our model, so 12m=1.0, 18m=1.5, 24m=2.0, 36m=3.0)
    horizons = {1.0: "PD_12M", 1.5: "PD_18M", 2.0: "PD_24M", 3.0: "PD_36M"}
    results = {}
    
    for t_years, label in horizons.items():
        # Find closest index in the survival function index
        closest_t = pd_fns.index[np.abs(pd_fns.index - t_years).argmin()]
        results[label] = pd_fns.loc[closest_t]
        
    res_df = pd.DataFrame(results)
    
    # Add TVS and leverage for reference
    res_df["TVS"] = active_tvs["TVS"]
    res_df["Debt_EBITDA"] = active_financials["Debt_EBITDA"]
    
    return res_df, pd_fns

if __name__ == "__main__":
    from data_engine import fetch_or_generate_market_data, get_company_financials
    from carbon_beta import calculate_tvs
    
    # 1. Load data
    m_df = fetch_or_generate_market_data()
    fin_df = get_company_financials()
    tvs_df = calculate_tvs(m_df)
    
    # 2. Generate cohort and fit CPH
    cohort = generate_training_cohort(tvs_df, fin_df)
    cph = fit_survival_model(cohort)
    
    print("CPH Model Summary:")
    cph.print_summary()
    
    # 3. Predict
    pds, pd_curves = predict_probabilities(cph, tvs_df, fin_df)
    print("\nPredicted Probabilities of Default:")
    print(pds)
