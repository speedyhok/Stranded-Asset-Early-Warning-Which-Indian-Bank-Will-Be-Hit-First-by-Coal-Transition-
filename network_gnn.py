# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class BankContagionGNN(nn.Module):
    def __init__(self, n_banks, n_companies, lgd_corp=0.60, lgd_bank=0.80):
        super(BankContagionGNN, self).__init__()
        self.n_banks = n_banks
        self.n_companies = n_companies
        self.lgd_corp = lgd_corp
        self.lgd_bank = lgd_bank
        
        # Learnable transition weights for distress scaling (optional parametrization)
        self.distress_slope = nn.Parameter(torch.tensor(30.0)) # steepness of distress threshold
        self.distress_midpoint = nn.Parameter(torch.tensor(0.09)) # regulatory minimum 9% CAR
        
    def forward(self, 
                company_pds,          # Tensor of shape (n_companies,)
                bank_capital_0,       # Tensor of shape (n_banks,)
                bank_assets_0,        # Tensor of shape (n_banks,)
                bank_car_0,           # Tensor of shape (n_banks,)
                exposure_matrix,      # Tensor of shape (n_banks, n_companies) - Bank -> Company loans
                interbank_matrix,     # Tensor of shape (n_banks, n_banks) - Lender Bank -> Borrower Bank loans
                n_iterations=5):      # Message passing iterations
        
        # Calculate Risk-Weighted Assets (RWA) from initial Capital and CAR
        # CAR = Capital / RWA  => RWA = Capital / CAR
        bank_rwa = bank_capital_0 / bank_car_0
        
        # --- Step 1: Corporate Loan Losses ---
        # Expected corporate loss for each loan: Exposure * PD * LGD
        corp_losses = exposure_matrix * company_pds.unsqueeze(0) * self.lgd_corp
        total_corp_loss = corp_losses.sum(dim=1) # Sum across companies for each bank
        
        # Update bank capital after corporate credit losses
        capital_t = bank_capital_0 - total_corp_loss
        car_t = capital_t / bank_rwa
        
        # --- Step 2: Systemic Interbank Contagion (GNN Message Passing) ---
        # We propagate stress dynamically across the interbank graph.
        # If a bank's CAR drops below 9.0%, it passes distress messages to its creditors.
        
        for iteration in range(n_iterations):
            # Compute distress factor theta for each bank (0 = healthy, 1 = fully distressed)
            # Using a differentiable sigmoid approximation of distress:
            # theta = sigmoid(slope * (midpoint - CAR))
            theta = torch.sigmoid(self.distress_slope * (self.distress_midpoint - car_t))
            
            # Mask out extremely small values to avoid noise
            theta = torch.where(car_t >= self.distress_midpoint, torch.zeros_like(theta), theta)
            
            # Loss to lender bank k from borrower bank i: interbank_matrix[k, i] * theta[i] * LGD_bank
            # interbank_matrix shape: (Lenders, Borrowers)
            # theta shape: (Borrowers,) -> we multiply column-wise and sum over borrowers
            interbank_losses = interbank_matrix * theta.unsqueeze(0) * self.lgd_bank
            total_interbank_loss = interbank_losses.sum(dim=1) # Sum across borrowing banks
            
            # Update capital and CAR with interbank losses
            capital_t = bank_capital_0 - total_corp_loss - total_interbank_loss
            car_t = torch.clamp(capital_t / bank_rwa, min=-0.05) # Cap lowest CAR to -5%
            
        # Return final bank capitals, CAR ratios, and distress factors
        return capital_t, car_t, theta

def run_stress_propagation(pd_estimates, network_dict):
    """
    Wrapper function that prepares pandas DataFrames into PyTorch Tensors,
    runs the Contagion GNN, and formats the output.
    """
    banks_df = network_dict["banks"]
    exposures_df = network_dict["exposures"]
    interbank_df = network_dict["interbank"]
    
    # 1. Align index orders
    bank_names = list(banks_df.index)
    company_names = list(pd_estimates.index)
    
    # Extract baseline arrays
    cap_0 = torch.tensor(banks_df.loc[bank_names, "Capital_CET1_INR_Cr"].values, dtype=torch.float32)
    assets_0 = torch.tensor(banks_df.loc[bank_names, "Total_Assets_INR_Cr"].values, dtype=torch.float32)
    car_0 = torch.tensor(banks_df.loc[bank_names, "CET1_Ratio"].values, dtype=torch.float32)
    
    # Aligned matrices
    exp_matrix = torch.tensor(exposures_df.loc[bank_names, company_names].values, dtype=torch.float32)
    ib_matrix = torch.tensor(interbank_df.loc[bank_names, bank_names].values, dtype=torch.float32)
    
    # PDs
    pds = torch.tensor(pd_estimates.values, dtype=torch.float32)
    
    # Run GNN model
    model = BankContagionGNN(n_banks=len(bank_names), n_companies=len(company_names))
    model.eval()
    
    with torch.no_grad():
        cap_t, car_t, theta_t = model(pds, cap_0, assets_0, car_0, exp_matrix, ib_matrix)
        
    # Format output as DataFrame
    results = pd.DataFrame({
        "Initial_Capital_INR_Cr": banks_df.loc[bank_names, "Capital_CET1_INR_Cr"],
        "Initial_CAR_Pct": banks_df.loc[bank_names, "CET1_Ratio"] * 100.0,
        "Stressed_Capital_INR_Cr": cap_t.numpy(),
        "Stressed_CAR_Pct": car_t.numpy() * 100.0,
        "Distress_Factor_Pct": theta_t.numpy() * 100.0
    }, index=bank_names)
    
    # Add status indicator
    results["Status"] = "HEALTHY"
    results.loc[results["Stressed_CAR_Pct"] < 9.0, "Status"] = "UNDER_CAPITALIZED"
    results.loc[results["Stressed_CAR_Pct"] < 5.0, "Status"] = "BREACH"
    
    return results

if __name__ == "__main__":
    from data_engine import fetch_or_generate_market_data, get_company_financials, get_banking_network
    from carbon_beta import calculate_tvs
    from survival_model import generate_training_cohort, fit_survival_model, predict_probabilities
    
    # 1. Run upstream stages
    m_df = fetch_or_generate_market_data()
    fin_df = get_company_financials()
    tvs_df = calculate_tvs(m_df)
    
    cohort = generate_training_cohort(tvs_df, fin_df)
    cph = fit_survival_model(cohort)
    pds_df, _ = predict_probabilities(cph, tvs_df, fin_df)
    
    # 2. Get banking network
    network = get_banking_network()
    
    # 3. Propagate 18-month stress
    pd_18m = pds_df["PD_18M"]
    network_stress = run_stress_propagation(pd_18m, network)
    
    print("\nBanking Network Stress Propagation (18-Month Horizon):")
    print(network_stress)
