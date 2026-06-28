# Climate Credit Risk & Systemic Contagion Model
# Copyright (c) 2026 Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# License: MIT License

import numpy as np
import pandas as pd
import yfinance as yf
import os

# Set random seed for reproducibility
np.random.seed(42)

# Indian companies to analyze
COMPANIES = {
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Adani Power": "ADANIPOWER.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Company X": "X.NS",
    "Company Y": "Y.NS",
    "Company Z": "Z.NS"
}
MARKET_INDEX = "^NSEI" # Nifty 50

# Historical defaults for validation
HISTORICAL_DEFAULTS = ["Lanco Infratech", "Essar Power", "GVK Power"]

def fetch_or_generate_market_data(start_date="2016-01-01", end_date="2026-06-01"):
    """
    Attempts to download stock price data from Yahoo Finance.
    Falls back to high-fidelity synthetic data if offline or download fails.
    """
    print("Fetching equity market data...")
    data = {}
    download_success = True
    
    # Try fetching real data
    try:
        tickers = list(COMPANIES.values()) + [MARKET_INDEX]
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        if raw_data.empty or "Close" not in raw_data:
            raise ValueError("No data returned from Yahoo Finance.")
            
        close_prices = raw_data["Close"]
        
        # Check if we got enough data
        for name, ticker in COMPANIES.items():
            if ticker in close_prices.columns and close_prices[ticker].dropna().shape[0] > 100:
                data[name] = close_prices[ticker].ffill().bfill()
            else:
                raise ValueError(f"Insufficient data for {name}")
                
        if MARKET_INDEX in close_prices.columns:
            data["Market"] = close_prices[MARKET_INDEX].ffill().bfill()
        else:
            raise ValueError("Market index data missing.")
            
        print("Successfully fetched real equity data from Yahoo Finance.")
        
    except Exception as e:
        print(f"Yahoo Finance fetch failed: {e}. Generating high-fidelity synthetic market data...")
        download_success = False
        
    if not download_success:
        # Generate correlated synthetic prices
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n_days = len(dates)
        
        # Synthetic market (Nifty 50) GBM
        m_returns = np.random.normal(0.0004, 0.012, n_days)
        m_prices = 8000 * np.exp(np.cumsum(m_returns))
        
        data["Market"] = pd.Series(m_prices, index=dates)
        
        # Beta parameters and idiosyncratic volatility
        configs = {
            "NTPC": {"m_beta": 0.9, "c_beta": -0.15, "vol": 0.015, "start": 120.0},
            "Coal India": {"m_beta": 0.7, "c_beta": -0.30, "vol": 0.018, "start": 200.0},
            "Adani Power": {"m_beta": 1.2, "c_beta": -0.25, "vol": 0.025, "start": 80.0},
            "JSW Steel": {"m_beta": 1.1, "c_beta": -0.10, "vol": 0.020, "start": 300.0},
            "Tata Steel": {"m_beta": 1.15, "c_beta": -0.12, "vol": 0.021, "start": 90.0},
            "Company X": {"m_beta": 1.4, "c_beta": -0.45, "vol": 0.035, "start": 50.0},
            "Company Y": {"m_beta": 1.5, "c_beta": -0.50, "vol": 0.040, "start": 40.0},
            "Company Z": {"m_beta": 1.6, "c_beta": -0.55, "vol": 0.045, "start": 30.0}
        }
        
        # Generate synthetic carbon price index proxy
        carbon_returns = np.random.normal(0.0006, 0.02, n_days)
        carbon_prices = 15.0 * np.exp(np.cumsum(carbon_returns)) # Starts at 15 EUR
        data["Carbon_Price_Proxy"] = pd.Series(carbon_prices, index=dates)
        
        # Generate company stock prices
        for name, config in configs.items():
            noise = np.random.normal(0, config["vol"], n_days)
            ret = config["m_beta"] * m_returns + config["c_beta"] * carbon_returns + noise
            prices = config["start"] * np.exp(np.cumsum(ret))
            data[name] = pd.Series(prices, index=dates)
            
    else:
        # If real data is fetched, let's try to fetch the real carbon price index (KRBN ETF)
        # and backfill it dynamically for dates prior to its launch (July 2020)
        dates = data["Market"].index
        dates_naive = dates.tz_localize(None) if hasattr(dates, "tz_localize") else dates
        download_carbon_success = False
        
        try:
            print("Fetching real KRBN carbon index from Yahoo Finance...")
            carbon_raw = yf.download("KRBN", start=start_date, end=end_date, progress=False)
            if not carbon_raw.empty and "Close" in carbon_raw:
                carbon_series = carbon_raw["Close"].ffill().bfill()
                # Ensure index is timezone-naive to match close_prices index
                carbon_series.index = carbon_series.index.tz_localize(None) if hasattr(carbon_series.index, "tz_localize") else carbon_series.index
                
                # Align to market dates
                carbon_aligned = carbon_series.reindex(dates_naive)
                
                # KRBN launched 2020-07-30. If our start date is earlier (2016), we must backfill.
                first_valid_idx = carbon_series.first_valid_index()
                if first_valid_idx is not None and first_valid_idx > dates_naive[0]:
                    prior_dates = dates_naive[dates_naive < first_valid_idx]
                    n_prior = len(prior_dates)
                    first_price = float(carbon_series.loc[first_valid_idx].iloc[0] if isinstance(carbon_series.loc[first_valid_idx], pd.Series) else carbon_series.loc[first_valid_idx])
                    
                    # Backfill with GBM starting from KRBN's first price, going backward
                    np.random.seed(101)
                    sim_returns = np.random.normal(0.0002, 0.015, n_prior)
                    
                    sim_prices = np.zeros(n_prior)
                    curr_price = first_price
                    for idx in reversed(range(n_prior)):
                        curr_price = curr_price / (1.0 + sim_returns[idx])
                        sim_prices[idx] = curr_price
                        
                    carbon_aligned.loc[prior_dates] = sim_prices
                    
                data["Carbon_Price_Proxy"] = pd.Series(carbon_aligned.ffill().bfill().values, index=dates)
                download_carbon_success = True
                print("Successfully fetched real KRBN carbon index and backfilled history.")
            else:
                raise ValueError("No KRBN data returned.")
        except Exception as carbon_err:
            print(f"Failed to fetch KRBN carbon index: {carbon_err}. Falling back to trend model.")
            
        if not download_carbon_success:
            n_days = len(dates)
            trend = np.linspace(5, 70, n_days) + np.random.normal(0, 10, n_days).cumsum() * 0.1
            trend = np.clip(trend, 3, 110)
            data["Carbon_Price_Proxy"] = pd.Series(trend, index=dates)
        
    df = pd.DataFrame(data)
    return df

def generate_ngfs_scenarios():
    """
    Generates NGFS Carbon Price Scenarios (USD/tCO2) from 2026 to 2036.
    - Net Zero 2050 (Orderly): Smooth, rapid increase in carbon price.
    - Delayed Transition (Disorderly): Flat/low, then sudden sharp spike.
    - Current Policies (BAU): Low/flat carbon price.
    """
    quarters = pd.date_range(start="2026-06-30", end="2036-06-30", freq="Q")
    n_q = len(quarters)
    
    nz_prices = 30.0 + (180.0 - 30.0) * (np.arange(n_q) / (n_q - 1))**1.2
    
    dt_prices = np.zeros(n_q)
    dt_prices[:16] = 15.0 + np.arange(16) * 0.5
    dt_prices[16:] = dt_prices[15] + (240.0 - dt_prices[15]) * (np.arange(n_q - 16) / (n_q - 17))**1.5
    
    bau_prices = 15.0 + np.sin(np.arange(n_q) / 2.0) * 3.0
    
    scenarios = pd.DataFrame({
        "Net_Zero": nz_prices,
        "Delayed_Transition": dt_prices,
        "BAU": bau_prices
    }, index=quarters)
    
    return scenarios

def get_company_financials():
    """
    Returns current (2026) financial baseline metrics for the active companies.
    """
    financials = {
        "Coal India": {
            "Debt_EBITDA": 0.15,
            "Carbon_Rev_Pct": 0.95,
            "Capex_Fossil_Renewable": 8.0,
            "Refinancing_Years": 8.0,
            "EBITDA_INR_Cr": 35000.0,
            "Total_Debt_INR_Cr": 5000.0
        },
        "NTPC": {
            "Debt_EBITDA": 3.8,
            "Carbon_Rev_Pct": 0.85,
            "Capex_Fossil_Renewable": 1.2,
            "Refinancing_Years": 3.5,
            "EBITDA_INR_Cr": 42000.0,
            "Total_Debt_INR_Cr": 160000.0
        },
        "Adani Power": {
            "Debt_EBITDA": 2.9,
            "Carbon_Rev_Pct": 0.98,
            "Capex_Fossil_Renewable": 4.5,
            "Refinancing_Years": 2.0,
            "EBITDA_INR_Cr": 18000.0,
            "Total_Debt_INR_Cr": 52000.0
        },
        "JSW Steel": {
            "Debt_EBITDA": 2.2,
            "Carbon_Rev_Pct": 0.80,
            "Capex_Fossil_Renewable": 3.0,
            "Refinancing_Years": 4.0,
            "EBITDA_INR_Cr": 24000.0,
            "Total_Debt_INR_Cr": 53000.0
        },
        "Tata Steel": {
            "Debt_EBITDA": 2.5,
            "Carbon_Rev_Pct": 0.78,
            "Capex_Fossil_Renewable": 2.2,
            "Refinancing_Years": 3.0,
            "EBITDA_INR_Cr": 28000.0,
            "Total_Debt_INR_Cr": 70000.0
        },
        "Company X": {
            "Debt_EBITDA": 6.5,
            "Carbon_Rev_Pct": 0.92,
            "Capex_Fossil_Renewable": 10.0,
            "Refinancing_Years": 1.0,
            "EBITDA_INR_Cr": 5000.0,
            "Total_Debt_INR_Cr": 32500.0
        },
        "Company Y": {
            "Debt_EBITDA": 7.8,
            "Carbon_Rev_Pct": 0.95,
            "Capex_Fossil_Renewable": 12.0,
            "Refinancing_Years": 0.8,
            "EBITDA_INR_Cr": 3000.0,
            "Total_Debt_INR_Cr": 23400.0
        },
        "Company Z": {
            "Debt_EBITDA": 8.5,
            "Carbon_Rev_Pct": 0.98,
            "Capex_Fossil_Renewable": 15.0,
            "Refinancing_Years": 0.5,
            "EBITDA_INR_Cr": 2000.0,
            "Total_Debt_INR_Cr": 17000.0
        }
    }
    return pd.DataFrame(financials).T

def get_historical_defaults_data():
    """
    Returns simulated pre-distress (e.g. 2014-2015) financials for companies 
    that subsequently defaulted or went NPA (Lanco, Essar, GVK) between 2015-2020.
    """
    historical = {
        "Lanco Infratech": {
            "Debt_EBITDA": 7.5,
            "Carbon_Rev_Pct": 0.90,
            "Capex_Fossil_Renewable": 12.0,
            "Refinancing_Years": 0.8,
            "TVS": 75.0,
            "Observed_Time": 2.0,
            "Event": 1
        },
        "Essar Power": {
            "Debt_EBITDA": 6.8,
            "Carbon_Rev_Pct": 0.95,
            "Capex_Fossil_Renewable": 10.0,
            "Refinancing_Years": 1.2,
            "TVS": 72.0,
            "Observed_Time": 3.0,
            "Event": 1
        },
        "GVK Power": {
            "Debt_EBITDA": 8.2,
            "Carbon_Rev_Pct": 0.88,
            "Capex_Fossil_Renewable": 7.0,
            "Refinancing_Years": 1.5,
            "TVS": 70.0,
            "Observed_Time": 4.0,
            "Event": 1
        }
    }
    return pd.DataFrame(historical).T

def get_banking_network():
    """
    Generates a simulated representation of the Indian banking network.
    """
    banks = {
        "SBI": {
            "CET1_Ratio": 0.102,
            "Capital_CET1_INR_Cr": 260000.0,
            "Total_Assets_INR_Cr": 5500000.0
        },
        "PNB": {
            "CET1_Ratio": 0.091,
            "Capital_CET1_INR_Cr": 65000.0,
            "Total_Assets_INR_Cr": 1400000.0
        },
        "BOB": {
            "CET1_Ratio": 0.108,
            "Capital_CET1_INR_Cr": 75000.0,
            "Total_Assets_INR_Cr": 1500000.0
        },
        "ICICI": {
            "CET1_Ratio": 0.155,
            "Capital_CET1_INR_Cr": 160000.0,
            "Total_Assets_INR_Cr": 1800000.0
        },
        "HDFC": {
            "CET1_Ratio": 0.162,
            "Capital_CET1_INR_Cr": 240000.0,
            "Total_Assets_INR_Cr": 2800000.0
        }
    }
    
    exposures = {
        "SBI": {"Coal India": 2500.0, "NTPC": 28000.0, "Adani Power": 12000.0, "JSW Steel": 9500.0, "Tata Steel": 11000.0, "Company X": 6000.0, "Company Y": 5000.0, "Company Z": 4000.0},
        "PNB": {"Coal India": 800.0,  "NTPC": 11000.0, "Adani Power": 6500.0,  "JSW Steel": 4000.0, "Tata Steel": 4500.0, "Company X": 4500.0, "Company Y": 3500.0, "Company Z": 3000.0},
        "BOB": {"Coal India": 1200.0, "NTPC": 9500.0,  "Adani Power": 5000.0,  "JSW Steel": 3500.0, "Tata Steel": 5000.0, "Company X": 4000.0, "Company Y": 3000.0, "Company Z": 2500.0},
        "ICICI": {"Coal India": 300.0, "NTPC": 4000.0,  "Adani Power": 2200.0,  "JSW Steel": 6000.0, "Tata Steel": 8500.0, "Company X": 1500.0, "Company Y": 1000.0, "Company Z": 800.0},
        "HDFC": {"Coal India": 200.0,  "NTPC": 3500.0,  "Adani Power": 1500.0,  "JSW Steel": 7000.0, "Tata Steel": 9000.0, "Company X": 1000.0, "Company Y": 800.0,  "Company Z": 500.0}
    }
    
    interbank = {
        "SBI":   {"SBI": 0.0, "PNB": 12000.0, "BOB": 9000.0,  "ICICI": 5000.0, "HDFC": 6000.0},
        "PNB":   {"SBI": 3000.0, "PNB": 0.0, "BOB": 2500.0,  "ICICI": 1500.0, "HDFC": 2000.0},
        "BOB":   {"SBI": 4000.0, "PNB": 3000.0, "BOB": 0.0,   "ICICI": 2000.0, "HDFC": 2500.0},
        "ICICI": {"SBI": 8000.0, "PNB": 5000.0, "BOB": 4000.0,  "ICICI": 0.0,    "HDFC": 8000.0},
        "HDFC":  {"SBI": 10000.0,"PNB": 6000.0, "BOB": 5000.0,  "ICICI": 7000.0, "HDFC": 0.0}
    }
    
    return {
        "banks": pd.DataFrame(banks).T,
        "exposures": pd.DataFrame(exposures).T,
        "interbank": pd.DataFrame(interbank).T
    }
