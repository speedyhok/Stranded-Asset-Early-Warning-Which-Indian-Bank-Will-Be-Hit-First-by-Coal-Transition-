# Climate Credit Deterioration & Bank Contagion Predictor

| Attribute | Details |
| :--- | :--- |
| **Author** | Mohibul Hoque |
| **Email** | [hokworks@gmail.com](mailto:hokworks@gmail.com) |
| **LinkedIn** | [linkedin.com/in/speedymohibul](https://linkedin.com/in/speedymohibul) |
| **License** | [MIT License](file:///c:/Users/Administrator/Desktop/CC1/LICENSE) |
| **Live Demo* | [https://stranded-asset-early-warning-which.onrender.com] |

---

## Executive Summary
This project implements an advanced, multi-stage machine learning and financial network contagion simulation framework. It is specifically designed to predict the credit deterioration of carbon-intensive Indian corporates (e.g., Coal India, NTPC, Adani Power, Tata Steel, JSW Steel) **12–18 months before** it manifests as Non-Performing Assets (NPAs) in the banking sector. 

Once firm-level credit deterioration is estimated, the system models the propagation of credit losses and liquidity stress through the interbank lending network. Using a Graph Neural Network (GNN), it computes bank-level Capital Adequacy Ratio (CAR) degradation under various Network for Greening the Financial System (NGFS) climate scenarios over a 10-year quarterly simulation horizon (2026–2036).

---

## Methodology & Model Architecture

The framework consists of four sequential analytical stages:

```
+-------------------------------------------------------------+
|                     Stage 1: TVS Score                      |
|  (Carbon Beta + Climate Event Study CAR + NLP Sentiment)    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Stage 2: Cox Survival Model                   |
|  (Estimates conditional PD curves based on TVS & Leverage)  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Stage 3: Contagion GNN                      |
|  (PyTorch GCN propagating losses across bank loan network)  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               Stage 4: NGFS Scenario Simulation             |
|   (Quarterly stress paths over a 10-year horizon, 2026-36)  |
+-------------------------------------------------------------+
```

### Stage 1: Transition Vulnerability Score (TVS)
The Transition Vulnerability Score (TVS) is an unsupervised composite indicator mapping a company's financial vulnerability to transition risk. It is calculated by normalizing and weighting three risk metrics:
1. **Carbon Beta ($\beta_C$):** Derived from a multi-factor equity regression model. Daily equity returns of each company are regressed against daily market returns (`^NSEI`) and daily changes in a carbon price index proxy (representing carbon taxation/compliance costs). A more negative carbon beta indicates that stock returns fall when carbon transition pressures rise.
2. **Event Study Cumulative Abnormal Returns (CAR):** Computes market-adjusted abnormal returns around major domestic and international climate policy announcements (e.g., Paris Agreement activation, COP26 Indian Net Zero commitments, COP28 fossil-fuel phase-down milestones). Companies with significant negative CAR over a $[-2, +5]$ trading day window around these announcements exhibit higher perceived transition risk.
3. **NLP Climate Sentiment Analysis:** Scans simulated earnings call transcripts using a keyword-based lexicon of transition risks (e.g., "carbon tax", "stranded assets", "regulatory penalties") and mitigation investments (e.g., "transition capex", "decarbonization", "renewables"). The ratio of risk to mitigation terms determines the company's sentiment score.

**TVS Formula:**
$$\text{TVS} = 0.40 \times \text{Norm}(\text{Carbon Beta}) + 0.30 \times \text{Norm}(\text{Event CAR}) + 0.30 \times \text{Norm}(\text{NLP Sentiment})$$

---

### Stage 2: Cox Proportional Hazards (CPH) Model
The survival analysis is implemented using a Cox Proportional Hazards formulation. It constructs a training cohort of 50 carbon-intensive peer firms (spanning mining, steel, power, and cement sectors from 2015-2026). The model incorporates historical corporate failures (e.g., Lanco Infratech, Essar Power, GVK Power) as default events to train the hazard rates.

The covariates utilized in the model include:
- **TVS:** The composite Transition Vulnerability Score.
- **Leverage (Debt / EBITDA):** Reflecting balance sheet fragility.
- **Carbon Revenue Share (%):** Revenue proportion derived from fossil fuels.
- **Transition Capex Ratio:** Ratio of fossil fuel capital expenditure to renewable/decarbonization capex.
- **Refinancing Timeline (Years):** Remaining years until corporate debt maturities must be rolled over.

**CPH Hazard Function:**
$$\lambda(t \,|\, X) = \lambda_0(t) \exp(\sum \beta_i X_i)$$

The fitted model is used to project conditional Probability of Default (PD) curves for target active firms over horizons up to 36 months (12M, 18M, 24M, 36M).

---

### Stage 3: Systemic Contagion GNN
Stress propagation through the financial sector is modeled via a PyTorch-based Graph Neural Network (GNN) representing the bipartite credit network:
- **Nodes:** Banks and carbon-intensive corporate borrowers.
- **Edges:** Bilateral interbank loans and corporate-to-bank credit exposures.

**Distress Message Passing Mechanism:**
1. **Initial Shock:** Corporate default probabilities (PDs) from Stage 2 are converted to expected corporate loan losses ($\text{Exposure} \times \text{PD} \times \text{LGD}_{\text{corp}}$) for each bank.
2. **Capital Degradation:** Expected losses are subtracted from the bank's Common Equity Tier 1 (CET1) capital, degrading its Capital Adequacy Ratio (CAR).
3. **Bilateral Interbank Contagion:** A differentiable distress factor ($\theta \in [0, 1]$) is computed for each bank using a sigmoid threshold around the regulatory minimum CAR (9.0%):
   $$\theta_i = \sigma \left( \alpha \times (0.09 - \text{CAR}_i) \right)$$
   Where $\alpha$ controls threshold steepness. If a bank's capital drops below the regulatory threshold, it propagates distress to its creditor banks over multiple network message-passing iterations, modeling interbank lending write-downs ($\text{Interbank Exposure} \times \theta_i \times \text{LGD}_{\text{bank}}$).

---

### Stage 4: Dynamic NGFS Climate Scenario Simulation
The pipeline models the quarterly stress path from **2026 to 2036** under three NGFS scenarios:
1. **Net Zero 2050 (Orderly):** Carbon prices rise smoothly and rapidly from \$30/tCO2 to \$180/tCO2. Corporate EBITDA is squeezed proportionally to carbon revenue, but companies with renewable capex pivot quickly.
2. **Delayed Transition (Disorderly):** Policy action is delayed. Carbon prices remain low until 2030, then spike suddenly and aggressively up to \$240/tCO2, triggering severe capital write-downs and sudden defaults.
3. **Business As Usual (BAU / Current Policies):** Low, fluctuating carbon price. Traditional fossil-heavy companies maintain cash flows in the near-term but carry long-term refinancing risk as global capital markets shift away from fossil loans.

Under each scenario, corporate financial covariates, refinancing risk, and TVS are updated dynamically each quarter to project forward-looking default probabilities and bank CAR trajectories.

---

## Historical Validation & Backtesting

The model has been validated against historical Indian corporate defaults and NPA cases (2015-2020) including Lanco Infratech, Essar Power, and GVK Power. 

The backtest results indicate that the framework successfully flags credit deterioration **12–18 months prior to actual default**:
- At the 18-month pre-default warning horizon, predicted PDs surged to **11%–28%**.
- At the 12-month pre-default warning horizon, predicted PDs increased to **31%–66%**, providing ample lead time for credit risk mitigation and capital allocation adjustments.

---

## Project Structure & Core Files

- **[data_engine.py](file:///c:/Users/Administrator/Desktop/CC1/data_engine.py):** Data ingestion module. Downloads market price data (Yahoo Finance) for target companies and index (`^NSEI`), generates synthetic fallbacks, and builds baseline banking exposure profiles.
- **[carbon_beta.py](file:///c:/Users/Administrator/Desktop/CC1/carbon_beta.py):** Stage 1 ML module. Computes Carbon Beta via regressions, conducts event studies for Cumulative Abnormal Returns, performs NLP transcript keyword scoring, and compiles the TVS.
- **[survival_model.py](file:///c:/Users/Administrator/Desktop/CC1/survival_model.py):** Stage 2 module. Compiles the peer cohort, fits the Cox Proportional Hazards model using `lifelines`, and generates target PD curves.
- **[network_gnn.py](file:///c:/Users/Administrator/Desktop/CC1/network_gnn.py):** Stage 3 module. Implements the `BankContagionGNN` PyTorch model for stress propagation across interbank lending edges.
- **[scenarios.py](file:///c:/Users/Administrator/Desktop/CC1/scenarios.py):** Stage 4 module. Simulates quarterly paths for company metrics and bank CAR under NGFS carbon price paths (Net Zero, Delayed Transition, BAU).
- **[run_pipeline.py](file:///c:/Users/Administrator/Desktop/CC1/run_pipeline.py):** Main entry point. Orchestrates the pipeline, generates visual plots, runs backtests, and exports CSV reports.
- **[prepare_web_dashboard_data.py](file:///c:/Users/Administrator/Desktop/CC1/prepare_web_dashboard_data.py):** Exports compiled simulation parameters and histories into JSON format (`modelData.js`) for local web dashboard rendering.

---

## Interactive Stress Testing Dashboard (Designed by Mohibul Hoque)

The dashboard has been built as a high-performance React application utilizing customized, responsive CSS with glassmorphism components to display credit risk transmission visually:
1. **Interactive Parameters Panel:** Real-time modification of parameters (LGD for corporates, LGD for interbank, GNN iterations, and regulatory min CAR) which immediately re-runs the Graph Neural Network simulation locally in Javascript and redraws the network.
2. **Interactive Bipartite Network Map:** A high-fidelity SVG network rendering showing the flow of risk between corporate debtors and creditor banks. It features hover fading (isolating selected nodes and their transaction peers) and click-to-view detail overlays.
3. **Stressed Loss Attribution Details:** Click on any bank node to view a micro-portfolio report, listing exact initial and stressed CET1 CAR, corporate credit expected losses, interbank contagion write-downs, and loan matrices with individual debtor firms.
4. **NGFS Climate Scenarios Timeline Chart:** An interactive quarterly SVG trajectory plot showing bank CAR values and carbon price index curves from 2026 to 2036. Moving the mouse across the plot displays a dotted vertical indicator and detailed hover tooltip card.
5. **Model Backtesting & Validation Viewer:** Evaluates historical defaults of large Indian firms, highlighting the early-warning horizon (12M and 18M) and probability metrics.

---

## Installation & Getting Started

### Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)

### Setup
1. Clone the project and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end analytical pipeline:
   ```bash
   python run_pipeline.py
   ```
   *This saves visualization plots (e.g. `unsupervised_signals.png`, `company_survival_curves.png`, `bank_car_trajectories.png`) and quarterly scenario reports to the configured artifacts directory.*

3. Export data for the Web Dashboard:
   ```bash
   python prepare_web_dashboard_data.py
   ```

4. Start the interactive React dashboard locally:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

### Render Deployment (Python Web Service)
This project is configured to run on Render as a **Python Web Service** utilizing Flask to serve the compiled React frontend static assets.

**Configuration on Render:**
1.  **Service Type:** Web Service
2.  **Environment:** Python
3.  **Build Command:** `pip install -r requirements.txt` (the compiled React dist folder is committed to git)
4.  **Start Command:** `gunicorn app:app`
