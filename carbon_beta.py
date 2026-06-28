# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
from data_engine import COMPANIES

# Key historical climate announcement dates
CLIMATE_EVENTS = [
    "2016-11-04", # Paris Agreement enters into force
    "2018-10-08", # IPCC Special Report on 1.5°C
    "2021-11-01", # India announces Net Zero 2070 target at COP26
    "2021-11-13", # COP26 Glasgow Pact (coal phase-down agreed)
    "2023-12-13"  # COP28 UAE Consensus (transitioning away from fossil fuels)
]

# Climate risk and mitigation lexicons
CLIMATE_RISK_KEYWORDS = [
    "carbon tax", "coal phase-out", "stranded asset", "environmental regulation", 
    "carbon price", "emission penalty", "compliance cost", "cbam", "fossil fuel phase-out",
    "carbon penalty", "carbon costs", "regulatory risk"
]

CLIMATE_MITIGATION_KEYWORDS = [
    "renewable", "solar", "wind", "electric arc", "green hydrogen", "decarbonization", 
    "net zero target", "green power", "transition capex", "ev transition", "clean energy"
]

# High-fidelity simulated earnings call transcripts for sentiment analysis
MOCK_TRANSCRIPTS = {
    "Coal India": """
    In this quarter, our focus remains meeting India's growing domestic energy demand. 
    Coal production has hit record highs. While we acknowledge global discussions around coal phase-out, 
    domestic demand is robust. We are facing some regulatory risk regarding carbon tax increases and 
    environmental regulation compliance costs, which might increase our cost of production. 
    However, the transition to renewables is still in its infancy for heavy base-load power. 
    We will continue to invest in coal rail infrastructure. There are no stranded assets on our balance sheet, 
    as coal demand is expected to peak much later in India. Some solar projects are planned, 
    but coal remains our core business.
    """,
    
    "NTPC": """
    Our thermal generation fleet continues to run at optimal PLF, but we are aggressively pivoting. 
    We have set a net zero target and are rapidly deploying renewable energy capacity. 
    We are shifting our transition capex to solar and wind power. Yes, we face some carbon price sensitivity 
    and carbon penalty risks on our older coal plants, but our plan to add 60 GW of clean energy by 2032 
    mitigates this regulatory risk. Our newer thermal plants utilize carbon capture and coal-blending technologies. 
    We are writing down or repurposing old thermal plants to prevent them from becoming stranded assets. 
    Compliance cost for new environmental regulation is manageable.
    """,
    
    "Adani Power": """
    We have seen strong revenue growth driven by higher power demand. We are expanding our thermal power capacity. 
    Regarding carbon tax and global coal phase-out discussions, our domestic contracts protect our margins. 
    Carbon price fluctuations do not directly impact our domestic PPAs. However, capital cost for fossil fuel projects 
    is rising due to ESG screening. We have minimal investment in renewable capacity inside this entity, 
    as those are housed in a sister company. Environmental regulation compliance costs have gone up due to flue-gas 
    desulfurization (FGD) installations. Refinancing risk remains a focus, and we are managing our debt maturities actively.
    """,
    
    "JSW Steel": """
    Steel demand is strong across automotive and infrastructure. We are managing carbon price risks, especially 
    with the impending European CBAM regulation, which will levy carbon costs on our exports. 
    To mitigate this regulatory risk, we are investing in electric arc furnace technology and gas-based DRI. 
    We are piloting green hydrogen injection in our blast furnaces. This transition capex will help us reduce our 
    carbon penalty exposure. Traditional coal-coke blast furnaces still dominate our production, but our long-term 
    goal is green steel.
    """,
    
    "Tata Steel": """
    Our European operations are facing massive pressure from EU ETS carbon price fluctuations and carbon tax liabilities. 
    We are decommissioning blast furnaces in the UK and shifting to electric arc furnace setups. 
    In India, we are focusing on reducing our carbon footprint through scrap recycling and green power purchasing. 
    CBAM represents a medium-term export risk. Decarbonization is central to our strategy. We are allocating 
    substantial transition capex for green steel and hydrogen pilots. Compliance costs for environmental regulation 
    will increase, but this is necessary to prevent our European assets from becoming stranded assets.
    """,

    "Company X": """
    Our operations are heavily reliant on thermal coal sourcing. We are experiencing major challenges 
    due to rising carbon tax penalty structures. Our regulatory risk has increased substantially. 
    We do not have any renewable energy capacity planned, and our transition capex is zero. 
    Older thermal machinery might become a stranded asset. Compliance costs for environmental regulation 
    are severely hurting our EBITDA. We are struggling to secure new debt, and refinancing risk is extremely high.
    """,

    "Company Y": """
    This quarter has been extremely difficult. Our coal-fired boilers are facing carbon penalty charges. 
    The carbon tax increase has hit our margins. We have no transition capex, nor do we have a net zero target. 
    Environmental regulation is forcing us to shut down lines, leading to potential stranded assets. 
    Refinancing risk is critical as banks are refusing to roll over our fossil-fuel backed loans.
    """,

    "Company Z": """
    Our carbon-heavy steel mills are heavily penalized by carbon price surges. We face direct carbon costs 
    and carbon tax liabilities that we cannot pass on. We have zero investment in green technology like electric arc 
    or green hydrogen. We face severe regulatory risk and compliance costs. Refinancing risk is imminent, 
    with short-term debt maturities coming due in months and capital markets blocked for fossil expansion.
    """
}

def estimate_carbon_beta(market_df):
    """
    Regresses daily stock returns on daily market returns and daily carbon price returns.
    """
    returns_df = np.log(market_df / market_df.shift(1)).dropna()
    betas = {}
    
    for company in COMPANIES.keys():
        # Prepare inputs
        y = returns_df[company].values.reshape(-1, 1)
        X = returns_df[["Market", "Carbon_Price_Proxy"]].values
        
        # Fit regression
        reg = LinearRegression().fit(X, y)
        
        # Extract beta coefficients
        m_beta = reg.coef_[0][0]
        c_beta = reg.coef_[0][1]
        
        betas[company] = {
            "Market_Beta": m_beta,
            "Carbon_Beta": c_beta
        }
    return pd.DataFrame(betas).T

def calculate_event_abnormal_returns(market_df):
    """
    Performs event studies around major climate announcements to extract Cumulative Abnormal Returns (CAR).
    """
    returns_df = np.log(market_df / market_df.shift(1)).dropna()
    car_scores = {}
    
    # Check overlapping dates
    available_dates = returns_df.index
    
    for company in COMPANIES.keys():
        event_cars = []
        for event_str in CLIMATE_EVENTS:
            event_date = pd.to_datetime(event_str)
            
            # Find the closest trading day in our index
            if event_date not in available_dates:
                closest_idx = available_dates.get_indexer([event_date], method="nearest")[0]
                event_date = available_dates[closest_idx]
                
            # Define event window [-2, +5] days
            event_idx = returns_df.index.get_loc(event_date)
            start_idx = max(0, event_idx - 2)
            end_idx = min(len(returns_df) - 1, event_idx + 5)
            
            # Define estimation window (90 days before event, ending 10 days before)
            est_start = max(0, event_idx - 90)
            est_end = max(1, event_idx - 10)
            
            if est_end <= est_start:
                continue
                
            est_data = returns_df.iloc[est_start:est_end]
            event_data = returns_df.iloc[start_idx:end_idx+1]
            
            # Fit market model during estimation window
            y_est = est_data[company].values.reshape(-1, 1)
            X_est = est_data["Market"].values.reshape(-1, 1)
            
            model = LinearRegression().fit(X_est, y_est)
            
            # Predict returns during event window
            X_event = event_data["Market"].values.reshape(-1, 1)
            pred_returns = model.predict(X_event).flatten()
            actual_returns = event_data[company].values
            
            # Compute Cumulative Abnormal Returns (CAR)
            abnormal_returns = actual_returns - pred_returns
            car = abnormal_returns.sum()
            event_cars.append(car)
            
        car_scores[company] = np.mean(event_cars) if event_cars else 0.0
        
    return pd.Series(car_scores)

def analyze_earnings_sentiment():
    """
    Scans earnings call transcripts using a keyword-based lexicon to compute climate risk sentiment.
    """
    sentiment_scores = {}
    
    for company, transcript in MOCK_TRANSCRIPTS.items():
        text = transcript.lower()
        
        # Count risk and mitigation terms
        risk_count = sum(text.count(kw) for kw in CLIMATE_RISK_KEYWORDS)
        mitigation_count = sum(text.count(kw) for kw in CLIMATE_MITIGATION_KEYWORDS)
        
        # Calculate score (higher risk count = higher climate concern/vulnerability)
        total_words = len(text.split())
        score = (risk_count - 0.5 * mitigation_count) / (total_words / 100.0) # Normalized per 100 words
        
        # Ensure score is positive/scaled appropriately
        sentiment_scores[company] = score
        
    return pd.Series(sentiment_scores)

def calculate_tvs(market_df):
    """
    Combines Carbon Beta, Event CARs, and NLP Sentiment to form the 
    Transition Vulnerability Score (TVS) for each company.
    """
    # 1. Carbon Beta
    betas_df = estimate_carbon_beta(market_df)
    c_beta = betas_df["Carbon_Beta"]
    
    # 2. Event CAR (more negative = more vulnerable)
    event_car = calculate_event_abnormal_returns(market_df)
    
    # 3. NLP Sentiment (higher risk count = more vulnerable)
    nlp_sentiment = analyze_earnings_sentiment()
    
    # Normalize features between 0 and 1
    # Carbon Beta: negative betas mean high vulnerability, so we scale negative carbon beta
    neg_c_beta = -c_beta
    norm_c_beta = (neg_c_beta - neg_c_beta.min()) / (neg_c_beta.max() - neg_c_beta.min() + 1e-8)
    
    # Event CAR: negative CAR means high vulnerability, so we scale negative CAR
    neg_car = -event_car
    norm_car = (neg_car - neg_car.min()) / (neg_car.max() - neg_car.min() + 1e-8)
    
    # NLP Sentiment: higher means more climate concern/vulnerability
    norm_sentiment = (nlp_sentiment - nlp_sentiment.min()) / (nlp_sentiment.max() - nlp_sentiment.min() + 1e-8)
    
    # Composite score calculation (TVS)
    # Weights: 40% Carbon Beta, 30% Event CAR, 30% NLP Sentiment
    tvs = 0.4 * norm_c_beta + 0.3 * norm_car + 0.3 * norm_sentiment
    tvs = tvs * 100.0 # Scale to 0-100
    
    # Structure output
    results = pd.DataFrame({
        "Market_Beta": betas_df["Market_Beta"],
        "Carbon_Beta": c_beta,
        "Event_CAR": event_car,
        "NLP_Sentiment_Score": nlp_sentiment,
        "TVS": tvs
    })
    
    return results

if __name__ == "__main__":
    from data_engine import fetch_or_generate_market_data
    df = fetch_or_generate_market_data()
    tvs_res = calculate_tvs(df)
    print("Transition Vulnerability Scores (TVS) calculated:")
    print(tvs_res.sort_values(by="TVS", ascending=False))
