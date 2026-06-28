/*
 * Climate Credit Risk & Systemic Contagion Model
 * Copyright (c) 2026 Mohibul Hoque
 * Email: hokworks@gmail.com
 * LinkedIn: linkedin.com/in/speedymohibul
 * License: MIT License
 */

import React, { useState, useMemo, useCallback } from 'react';
import { 
  Flame, 
  ShieldAlert, 
  TrendingUp, 
  BarChart2, 
  Info, 
  Activity, 
  AlertCircle, 
  CheckCircle,
  HelpCircle,
  Award,
  Layers,
  Link2,
  Mail,
  User,
  Shield
} from 'lucide-react';
import { modelData } from './data/modelData.js';
import './App.css';

// Extract data from modelData
const {
  coefficients,
  baselineSurvival,
  financials: baselineFinancials,
  tvs: baselineTvs,
  banks: baselineBanks,
  exposures,
  interbank,
  historicalDefaults,
  scenarios
} = modelData;

function App() {
  const [activeTab, setActiveTab] = useState('tester'); // 'tester' or 'scenarios' or 'historical'
  const [carbonPrice, setCarbonPrice] = useState(15);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [selectedBank, setSelectedBank] = useState(null);
  const [activeScenario, setActiveScenario] = useState('Delayed_Transition'); // Net_Zero, Delayed_Transition, BAU

  // Contagion parameters state
  const [lgdCorp, setLgdCorp] = useState(0.60);
  const [lgdBank, setLgdBank] = useState(0.80);
  const [gnnIterations, setGnnIterations] = useState(5);
  const [regMinCar, setRegMinCar] = useState(9.0);

  // Interactivity state
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredQuarterIndex, setHoveredQuarterIndex] = useState(null);
  const [hoveredChartPos, setHoveredChartPos] = useState({ x: 0, y: 0 });

  // Helper to check network connectivity for hover fading
  const isConnected = useMemo(() => {
    return (n1, n2) => {
      if (!n1 || !n2) return false;
      if (n1 === n2) return true;
      // If one is bank and other is company
      if (baselineFinancials[n1] && baselineBanks[n2]) {
        return !!(exposures[n2] && exposures[n2][n1] && exposures[n2][n1] >= 1000);
      }
      if (baselineBanks[n1] && baselineFinancials[n2]) {
        return !!(exposures[n1] && exposures[n1][n2] && exposures[n1][n2] >= 1000);
      }
      // If both are banks
      if (baselineBanks[n1] && baselineBanks[n2]) {
        return !!((interbank[n1] && interbank[n1][n2] && interbank[n1][n2] >= 3000) || 
                 (interbank[n2] && interbank[n2][n1] && interbank[n2][n1] >= 3000));
      }
      return false;
    };
  }, []);

  // --- STAGE 4 / GNN CONTATION SIMULATOR IN JAVASCRIPT ---
  const stressResults = useMemo(() => {
    // 1. Compute stressed company metrics
    const stressedCompanies = {};
    const companyNames = Object.keys(baselineFinancials);

    companyNames.forEach((name) => {
      const fin = baselineFinancials[name];
      const tvsRow = baselineTvs[name];

      // EBITDA mult: drops as carbon price rises
      const ebitdaMult = Math.max(0.25, Math.min(1.0, 1.0 - (fin.Carbon_Rev_Pct * (carbonPrice - 15.0) / 400.0)));
      const stressedEbitda = fin.EBITDA_INR_Cr * ebitdaMult;
      const stressedDebtEbitda = fin.Total_Debt_INR_Cr / stressedEbitda;

      // Refinancing risk: remaining years to maturity drops
      const refMult = Math.max(0.10, Math.min(1.5, 1.0 - (fin.Capex_Fossil_Renewable * (carbonPrice - 15.0) / 800.0)));
      const stressedRefYears = Math.max(0.2, Math.min(10.0, fin.Refinancing_Years * refMult));

      // Updated TVS
      const tvsMult = (carbonPrice - 15.0) * (fin.Carbon_Rev_Pct / 2.0);
      const stressedTvs = Math.max(0.0, Math.min(100.0, tvsRow.TVS + tvsMult));

      // Calculate Cox PH Hazard ratio
      // beta * X
      const linearPredictor = 
        coefficients.TVS * stressedTvs +
        coefficients.Debt_EBITDA * stressedDebtEbitda +
        coefficients.Carbon_Rev_Pct * fin.Carbon_Rev_Pct +
        coefficients.Capex_Fossil_Renewable * fin.Capex_Fossil_Renewable +
        coefficients.Refinancing_Years * stressedRefYears;

      const hazardRatio = Math.exp(linearPredictor);

      // Find baseline survival at t = 1.5 (18 months)
      const t18 = 1.5;
      const closestTime = Object.keys(baselineSurvival).reduce((prev, curr) => {
        return Math.abs(parseFloat(curr) - t18) < Math.abs(parseFloat(prev) - t18) ? curr : prev;
      });
      const S0_18M = baselineSurvival[closestTime];

      // PD = 1 - S0^hazardRatio
      const pd18m = 1.0 - Math.pow(S0_18M, hazardRatio);

      stressedCompanies[name] = {
        name,
        ebitda: stressedEbitda,
        debtEbitda: stressedDebtEbitda,
        refYears: stressedRefYears,
        tvs: stressedTvs,
        pd18m: pd18m,
        initialPd: tvsRow.PD_18M || 0.001
      };
    });

    // 2. Compute stressed banks via Contagion GNN message passing
    const bankNames = Object.keys(baselineBanks);
    const stressedBanks = {};

    // Initial setups
    bankNames.forEach(b => {
      const bank = baselineBanks[b];
      const rwa = bank.Capital_CET1_INR_Cr / bank.CET1_Ratio;
      
      // Calculate corporate loan loss: Exposure * PD * LGD
      let corpLoss = 0;
      companyNames.forEach(c => {
        const exposure = exposures[b][c] || 0;
        const pd = stressedCompanies[c].pd18m;
        corpLoss += exposure * pd * lgdCorp; // Dynamic LGD_corp
      });

      stressedBanks[b] = {
        name: b,
        capital0: bank.Capital_CET1_INR_Cr,
        rwa: rwa,
        initialCar: bank.CET1_Ratio * 100,
        corpLoss: corpLoss,
        interbankLoss: 0,
        capital: bank.Capital_CET1_INR_Cr - corpLoss,
        car: 0,
        theta: 0,
        status: 'HEALTHY'
      };
      stressedBanks[b].car = (stressedBanks[b].capital / rwa) * 100;
    });

    // GNN Contagion Message passing (dynamic iterations)
    const distressSlope = 30.0;
    const distressMidpoint = regMinCar; // Dynamic regulatory CAR min

    for (let iter = 0; iter < gnnIterations; iter++) {
      // Step A: compute distress factor theta for each bank
      bankNames.forEach(b => {
        const car = stressedBanks[b].car;
        // sigmoid(slope * (midpoint - CAR_pct))
        const theta = 1.0 / (1.0 + Math.exp(distressSlope * (distressMidpoint - car) / 100.0));
        stressedBanks[b].theta = car >= distressMidpoint ? 0 : theta;
      });

      // Step B: propagate interbank losses
      bankNames.forEach(b => {
        let ibLoss = 0;
        bankNames.forEach(borrower => {
          const exposure = interbank[b][borrower] || 0;
          const theta = stressedBanks[borrower].theta;
          ibLoss += exposure * theta * lgdBank; // Dynamic LGD_bank
        });

        stressedBanks[b].interbankLoss = ibLoss;
        stressedBanks[b].capital = stressedBanks[b].capital0 - stressedBanks[b].corpLoss - ibLoss;
        stressedBanks[b].car = Math.max(-5.0, (stressedBanks[b].capital / stressedBanks[b].rwa) * 100);
        stressedBanks[b].status = stressedBanks[b].car < regMinCar ? 'BREACHED' : 'HEALTHY';
      });
    }

    return {
      companies: stressedCompanies,
      banks: stressedBanks
    };
  }, [carbonPrice, lgdCorp, lgdBank, gnnIterations, regMinCar]);

  // --- RENDER SCENARIO GRAPHS ---
  const scenarioChartData = useMemo(() => {
    // Collect data points from the active scenario
    const scData = scenarios[activeScenario] || [];
    // Group by quarter
    const quarters = [...new Set(scData.map(d => d.Quarter))].sort();
    
    return quarters.map(q => {
      const point = { quarter: q };
      const qData = scData.filter(d => d.Quarter === q);
      qData.forEach(d => {
        if (d.Type === 'Bank') {
          point[d.Entity] = d.Capital_Ratio_Pct;
        } else {
          point[`${d.Entity}_PD`] = d.PD_18M_Pct;
        }
      });
      return point;
    });
  }, [activeScenario]);

  // Calculate the carbon price dynamically based on active scenario and hovered quarter index
  const carbonPriceVal = useMemo(() => {
    if (hoveredQuarterIndex === null) return 15;
    const nQ = scenarioChartData.length;
    if (nQ <= 1) return 15;
    if (activeScenario === 'Net_Zero') {
      return 30.0 + (180.0 - 30.0) * Math.pow(hoveredQuarterIndex / (nQ - 1), 1.2);
    } else if (activeScenario === 'Delayed_Transition') {
      if (hoveredQuarterIndex < 16) {
        return 15.0 + hoveredQuarterIndex * 0.5;
      } else {
        const base = 15.0 + 15 * 0.5;
        return base + (240.0 - base) * Math.pow((hoveredQuarterIndex - 16) / (nQ - 17), 1.5);
      }
    } else {
      return 15.0 + Math.sin(hoveredQuarterIndex / 2.0) * 3.0;
    }
  }, [hoveredQuarterIndex, activeScenario, scenarioChartData]);

  // Pre-calculate polyline points for each bank to avoid division symbols in JSX
  const bankChartPoints = useMemo(() => {
    const pointsMap = {};
    const nQ = scenarioChartData.length;
    if (nQ <= 1) return pointsMap;
    
    const chartWidth = window.innerWidth * 0.65;
    ['SBI', 'PNB', 'BOB', 'ICICI', 'HDFC'].forEach(bank => {
      pointsMap[bank] = scenarioChartData.map((d, idx) => {
        const val = d[bank] || 10;
        const x = 50 + (idx / (nQ - 1)) * chartWidth;
        const y = 200 - (val - 8.0) * 20;
        return `${x},${y}`;
      }).join(' ');
    });
    return pointsMap;
  }, [scenarioChartData]);

  // Pre-calculate vertical hovered line X coordinate
  const hoveredX = useMemo(() => {
    if (hoveredQuarterIndex === null || !hoveredChartPos.plotWidth) return null;
    const nQ = scenarioChartData.length;
    if (nQ <= 1) return 50;
    return 50 + (hoveredQuarterIndex / (nQ - 1)) * hoveredChartPos.plotWidth;
  }, [hoveredQuarterIndex, scenarioChartData, hoveredChartPos.plotWidth]);

  // Event handlers for scenario chart hover tooltips to avoid division operators in JSX
  const handleChartMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const chartWidth = window.innerWidth * 0.65;
    const x = e.clientX - rect.left - 50;
    if (x >= 0 && x <= chartWidth) {
      const pct = x / chartWidth;
      const idx = Math.round(pct * (scenarioChartData.length - 1));
      if (idx >= 0 && idx < scenarioChartData.length) {
        setHoveredQuarterIndex(idx);
        setHoveredChartPos({
          x: 50 + (idx / (scenarioChartData.length - 1)) * chartWidth,
          y: e.clientY - rect.top,
          plotWidth: chartWidth
        });
      }
    }
  }, [scenarioChartData]);

  const handleChartMouseLeave = useCallback(() => {
    setHoveredQuarterIndex(null);
  }, []);

  // Calculate stats for NGFS impact summary
  const scenarioStats = useMemo(() => {
    const summary = [];
    Object.keys(scenarios).forEach(scName => {
      const data = scenarios[scName];
      const bankData = data.filter(d => d.Type === 'Bank');
      
      const sbi = bankData.filter(d => d.Entity === 'SBI');
      const pnb = bankData.filter(d => d.Entity === 'PNB');
      const bob = bankData.filter(d => d.Entity === 'BOB');
      const icici = bankData.filter(d => d.Entity === 'ICICI');
      const hdfc = bankData.filter(d => d.Entity === 'HDFC');

      // Total Capital Loss
      const lastQ = [...new Set(bankData.map(d => d.Quarter))].sort().pop();
      const lastQData = bankData.filter(d => d.Quarter === lastQ);
      const totalLoss = lastQData.reduce((sum, d) => sum + d.Capital_Loss_INR_Cr, 0);

      // Check breaches
      const breachedRows = bankData.filter(d => d.Capital_Ratio_Pct < 9.0);
      const uniqueBreached = [...new Set(breachedRows.map(d => d.Entity))];
      let firstBreachDate = 'N/A';
      let firstBreaker = 'None';
      
      if (breachedRows.length > 0) {
        const sortedBreached = breachedRows.sort((a, b) => a.Quarter.localeCompare(b.Quarter));
        firstBreachDate = sortedBreached[0].Quarter;
        firstBreaker = sortedBreached[0].Entity;
      }

      summary.push({
        id: scName,
        name: scName.replace('_', ' '),
        capitalLoss: totalLoss,
        breachedCount: uniqueBreached.length,
        firstBreachDate,
        firstBreaker
      });
    });
    return summary;
  }, []);

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="header-bar">
        <div className="header-title-section">
          <h1>Climate Credit Risk & Systemic Contagion</h1>
          <p>Multi-stage ML Pipeline: Equity Market Signals, Cox Survival Analysis, & Network GNN Stress Propagation</p>
        </div>
        <div className="nav-tabs">
          <button 
            className={`tab-button ${activeTab === 'tester' ? 'active' : ''}`}
            onClick={() => setActiveTab('tester')}
          >
            <Flame size={16} /> Real-Time Stress Tester
          </button>
          <button 
            className={`tab-button ${activeTab === 'scenarios' ? 'active' : ''}`}
            onClick={() => setActiveTab('scenarios')}
          >
            <BarChart2 size={16} /> NGFS Climate Scenarios
          </button>
          <button 
            className={`tab-button ${activeTab === 'historical' ? 'active' : ''}`}
            onClick={() => setActiveTab('historical')}
          >
            <ShieldAlert size={16} /> Historical Validation
          </button>
        </div>
      </div>

      {activeTab === 'tester' && (
        <div className="dashboard-grid">
          {/* Sidebar controls */}
          <div className="glass-panel control-sidebar">
            <div className="stress-panel">
              <div className="slider-label-row">
                <span>Stressed Carbon Price</span>
                <span className="carbon-price-display">${carbonPrice} <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>/ tCO2</span></span>
              </div>
              <input 
                type="range" 
                min="15" 
                max="300" 
                value={carbonPrice} 
                onChange={(e) => setCarbonPrice(parseInt(e.target.value))}
                className="custom-range-slider"
              />
              <div style={{display:'flex', justifyContent:'space-between', fontSize:'0.75rem', color:'var(--text-muted)'}}>
                <span>$15 (Baseline)</span>
                <span>$150</span>
                <span>$300 (Severe Stress)</span>
              </div>
            </div>

            <div>
              <h3 className="sidebar-section-title"><Activity size={16} /> Contagion Parameters</h3>
              <div style={{display:'flex', flexDirection:'column', gap:'16px', fontSize:'0.85rem', color:'var(--text-secondary)'}}>
                
                {/* LGD Corp Slider */}
                <div>
                  <div style={{display:'flex', justifyContent:'space-between', marginBottom:'4px'}}>
                    <span>Corporate Loss (LGD)</span>
                    <strong style={{color:'var(--primary)'}}>{(lgdCorp * 100).toFixed(0)}%</strong>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" 
                    max="1.00" 
                    step="0.05"
                    value={lgdCorp} 
                    onChange={(e) => setLgdCorp(parseFloat(e.target.value))}
                    className="custom-range-slider"
                  />
                </div>

                {/* LGD Bank Slider */}
                <div>
                  <div style={{display:'flex', justifyContent:'space-between', marginBottom:'4px'}}>
                    <span>Interbank Loss (LGD)</span>
                    <strong style={{color:'var(--secondary)'}}>{(lgdBank * 100).toFixed(0)}%</strong>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" 
                    max="1.00" 
                    step="0.05"
                    value={lgdBank} 
                    onChange={(e) => setLgdBank(parseFloat(e.target.value))}
                    className="custom-range-slider"
                  />
                </div>

                {/* GNN Iterations Slider */}
                <div>
                  <div style={{display:'flex', justifyContent:'space-between', marginBottom:'4px'}}>
                    <span>GNN Message Passing</span>
                    <strong style={{color:'var(--text-primary)'}}>{gnnIterations} Rounds</strong>
                  </div>
                  <input 
                    type="range" 
                    min="1" 
                    max="10" 
                    step="1"
                    value={gnnIterations} 
                    onChange={(e) => setGnnIterations(parseInt(e.target.value))}
                    className="custom-range-slider"
                  />
                </div>

                {/* Regulatory Min CAR Slider */}
                <div>
                  <div style={{display:'flex', justifyContent:'space-between', marginBottom:'4px'}}>
                    <span>Regulatory CAR Min</span>
                    <strong style={{color:'var(--danger)'}}>{regMinCar.toFixed(1)}%</strong>
                  </div>
                  <input 
                    type="range" 
                    min="5.0" 
                    max="15.0" 
                    step="0.5"
                    value={regMinCar} 
                    onChange={(e) => setRegMinCar(parseFloat(e.target.value))}
                    className="custom-range-slider"
                  />
                </div>

              </div>
            </div>

            <div className="glass-panel" style={{padding:'16px', background:'rgba(15, 23, 42, 0.01)', border:'1px dashed var(--border-muted)', fontSize:'0.8rem', color:'var(--text-secondary)'}}>
              <p style={{fontWeight:'600', color:'var(--text-primary)', marginBottom:'4px', display:'flex', alignItems:'center', gap:'4px'}}><Info size={14} /> Stress Engine</p>
              Adjust the slider to simulate regulatory compliance costs, leverage shocks, and interbank contagion in real-time.
            </div>

            <div className="glass-panel" style={{padding:'18px', display:'flex', flexDirection:'column', gap:'12px', fontSize:'0.78rem', color:'var(--text-secondary)'}}>
              <h4 style={{fontWeight:'700', color:'var(--text-primary)', borderBottom:'1px solid var(--border-muted)', paddingBottom:'6px', display:'flex', alignItems:'center', gap:'6px'}}><HelpCircle size={14} /> Metric Glossary</h4>
              
              <div>
                <strong style={{color:'var(--text-primary)', display:'block', marginBottom:'2px'}}>TVS Score (0 - 100)</strong>
                Transition Vulnerability Score. Measures exposure to climate regulations based on stock carbon beta, policy events, and text sentiment.
              </div>
              
              <div>
                <strong style={{color:'var(--text-primary)', display:'block', marginBottom:'2px'}}>Stressed CAR (%)</strong>
                Stressed Capital Adequacy Ratio. The bank's capital buffer against credit losses. Regulatory minimum is <strong style={{color:'var(--danger)'}}>9.0%</strong>.
              </div>
              
              <div>
                <strong style={{color:'var(--text-primary)', display:'block', marginBottom:'2px'}}>18-Month PD (%)</strong>
                Probability of Default. Conditional risk estimated by the survival model that a company defaults or goes NPA within 18 months.
              </div>
              
              <div>
                <strong style={{color:'var(--text-primary)', display:'block', marginBottom:'2px'}}>Leverage (Debt/EBITDA)</strong>
                Measures total debt relative to operating cash flows. Spikes as carbon compliance costs compress EBITDA.
              </div>
            </div>
          </div>

          {/* Main Dashboard Space */}
          <div className="main-dashboard-content">
            {/* Bank Capital Ratios */}
            <div className="glass-panel dashboard-section">
              <div className="section-header">
                <h2><ShieldAlert size={20} style={{color:'var(--primary)'}} /> Stressed Bank Capital Adequacy Ratio (CAR)</h2>
                <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>Updates instantly via GNN contagion model</span>
              </div>
              <div className="banks-card-grid">
                {Object.values(stressResults.banks).map(bank => {
                  const carVal = bank.car;
                  const isBreached = carVal < regMinCar;
                  const nearBreach = bank.car < (regMinCar + 0.5) && bank.car >= regMinCar;
                  
                  let stateColor = 'var(--success)';
                  let bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(16, 185, 129, 0.22) 100%)';
                  let borderStyle = '1px solid rgba(16, 185, 129, 0.25)';
                  let glowShadow = '0 4px 20px -5px rgba(16, 185, 129, 0.08)';
                  
                  if (isBreached) {
                    stateColor = 'var(--danger)';
                    bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(239, 68, 68, 0.22) 100%)';
                    borderStyle = '1px solid rgba(239, 68, 68, 0.25)';
                    glowShadow = '0 6px 20px -5px rgba(239, 68, 68, 0.16)';
                  } else if (nearBreach) {
                    stateColor = 'var(--warning)';
                    bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(245, 158, 11, 0.22) 100%)';
                    borderStyle = '1px solid rgba(245, 158, 11, 0.25)';
                    glowShadow = '0 4px 20px -5px rgba(245, 158, 11, 0.12)';
                  }

                  return (
                    <div 
                      key={bank.name} 
                      className="glass-panel bank-metric-card glass-panel-glow" 
                      style={{
                        borderLeft: `4px solid ${stateColor}`,
                        border: borderStyle,
                        borderLeftWidth: '4px',
                        background: bgGradient,
                        boxShadow: glowShadow,
                        padding: '18px 20px',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        height: '90px',
                        cursor: 'pointer'
                      }}
                      onClick={() => setSelectedBank(bank)}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        <span className="bank-name" style={{ fontSize: '1.1rem', fontWeight: '800' }}>{bank.name}</span>
                        <span style={{
                          fontSize: '0.65rem', 
                          color: stateColor, 
                          fontWeight: '800',
                          background: '#ffffff',
                          border: `1px solid ${stateColor}30`,
                          padding: '2px 8px',
                          borderRadius: '12px',
                          alignSelf: 'flex-start'
                        }}>
                          {bank.status === 'BREACHED' ? 'BREACHED' : 'HEALTHY'}
                        </span>
                      </div>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                        <span style={{ fontSize: '1.6rem', fontWeight: '800', fontFamily: 'var(--font-heading)', color: stateColor }}>
                          {carVal.toFixed(1)}%
                        </span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: '600' }}>CET1 CAR</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Corporate stress level */}
            <div className="glass-panel dashboard-section">
              <div className="section-header">
                <h2><Flame size={20} style={{color:'var(--warning)'}} /> Carbon-Intensive Corporates</h2>
                <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>Click on a company card to view its balance sheet</span>
              </div>
              <div className="companies-card-grid">
                {Object.values(stressResults.companies).map(company => {
                  const pdVal = company.pd18m * 100;
                  let badgeColor = 'var(--success)';
                  let bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(16, 185, 129, 0.22) 100%)';
                  let borderStyle = '1px solid rgba(16, 185, 129, 0.25)';
                  let glowShadow = '0 4px 20px -5px rgba(16, 185, 129, 0.08)';
                  
                  if (pdVal > 5) {
                    badgeColor = 'var(--danger)';
                    bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(239, 68, 68, 0.22) 100%)';
                    borderStyle = '1px solid rgba(239, 68, 68, 0.25)';
                    glowShadow = '0 6px 20px -5px rgba(239, 68, 68, 0.16)';
                  } else if (pdVal > 1) {
                    badgeColor = 'var(--warning)';
                    bgGradient = 'linear-gradient(135deg, #ffffff 20%, rgba(245, 158, 11, 0.22) 100%)';
                    borderStyle = '1px solid rgba(245, 158, 11, 0.25)';
                    glowShadow = '0 4px 20px -5px rgba(245, 158, 11, 0.12)';
                  }

                  return (
                    <div 
                      key={company.name} 
                      className="glass-panel company-metric-card glass-panel-glow"
                      style={{ 
                        padding: '16px 20px', 
                        display: 'flex', 
                        flexDirection: 'column',
                        justifyContent: 'space-between', 
                        height: '130px',
                        background: bgGradient,
                        border: borderStyle,
                        boxShadow: glowShadow
                      }}
                      onClick={() => setSelectedCompany(company)}
                    >
                      <div className="company-card-glow-border" style={{background: badgeColor}} />
                      
                      {/* Top Header Row */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', width: '100%' }}>
                        <div>
                          <div className="company-name-title" style={{ fontSize: '1.05rem', fontWeight: '800' }}>{company.name}</div>
                          <span className="company-sector" style={{ fontSize: '0.72rem' }}>
                            {company.name === 'Coal India' ? 'Coal Mining' : company.name.includes('Steel') ? 'Steel Mfg' : 'Thermal Power'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                          <span style={{ fontSize: '1.45rem', fontWeight: '800', fontFamily: 'var(--font-heading)', color: badgeColor, lineHeight: '1' }}>
                            {company.tvs.toFixed(0)}
                          </span>
                          <span style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: '600', marginTop: '2px' }}>TVS</span>
                        </div>
                      </div>
                      
                      {/* Divider line */}
                      <div style={{ borderTop: '1px dashed var(--border-muted)', width: '100%', margin: '4px 0' }} />
                      
                      {/* Micro-Data Footer Grid */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)', width: '100%' }}>
                        <div>
                          <span>LVR: </span>
                          <strong style={{ color: 'var(--text-primary)' }}>{company.debtEbitda.toFixed(1)}x</strong>
                        </div>
                        <div>
                          <span>18M PD: </span>
                          <strong style={{ color: badgeColor }}>{pdVal.toFixed(2)}%</strong>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* SVG Contagion Network Map */}
            <div className="glass-panel dashboard-section">
              <div className="section-header">
                <h2><Link2 size={20} style={{color:'var(--secondary)'}} /> Interactive Network Stress Map</h2>
                <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>Displays loan exposures (outer loop) and interbank linkages (inner loop)</span>
              </div>
              <div className="network-container">
                <svg className="network-svg">
                  <defs>
                    <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="var(--primary)" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="var(--primary)" stopOpacity="0" />
                    </radialGradient>
                    <radialGradient id="breachGlow" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="var(--danger)" stopOpacity="0.5" />
                      <stop offset="100%" stopColor="var(--danger)" stopOpacity="0" />
                    </radialGradient>
                  </defs>

                  {/* Draw loan exposure edges from banks to companies */}
                  {Object.keys(exposures).map((bank) => {
                    const bankInfo = stressResults.banks[bank];
                    const bX = bank === 'SBI' ? 180 : bank === 'PNB' ? 420 : bank === 'BOB' ? 580 : bank === 'ICICI' ? 500 : 260;
                    const bY = bank === 'SBI' ? 80 : bank === 'PNB' ? 60 : bank === 'BOB' ? 170 : bank === 'ICICI' ? 300 : 280;

                    return Object.keys(exposures[bank]).map((company) => {
                      const expVal = exposures[bank][company];
                      if (expVal < 1000) return null; // Only draw substantial loans
                      
                      const compInfo = stressResults.companies[company];
                      const cX = company === 'Coal India' ? 320 : company === 'NTPC' ? 440 : company === 'Adani Power' ? 420 : company === 'JSW Steel' ? 320 : company === 'Tata Steel' ? 380 : company === 'Company X' ? 350 : company === 'Company Y' ? 470 : company === 'Company Z' ? 230 : 380;
                      const cY = company === 'Coal India' ? 150 : company === 'NTPC' ? 160 : company === 'Adani Power' ? 220 : company === 'JSW Steel' ? 210 : company === 'Tata Steel' ? 180 : company === 'Company X' ? 110 : company === 'Company Y' ? 260 : company === 'Company Z' ? 170 : 180;

                      // Stressed stroke color
                      const pd = compInfo.pd18m;
                      const strokeColor = pd > 0.05 ? 'rgba(239, 68, 68, 0.4)' : pd > 0.01 ? 'rgba(245, 158, 11, 0.3)' : 'rgba(15, 23, 42, 0.06)';
                      const thickness = Math.max(1, expVal / 5000);

                      // Hover highlight opacity
                      const edgeOpacity = hoveredNode ? ((hoveredNode === bank || hoveredNode === company) ? 1.0 : 0.10) : 1.0;

                      return (
                        <g key={`${bank}-${company}`} style={{ opacity: edgeOpacity, transition: 'opacity 0.25s ease' }}>
                          <line 
                            x1={bX} y1={bY} 
                            x2={cX} y2={cY} 
                            stroke={strokeColor} 
                            strokeWidth={thickness}
                          />
                          {pd > 0.01 && (
                            <line 
                              className="edge-link"
                              x1={bX} y1={bY} 
                              x2={cX} y2={cY} 
                              stroke="var(--danger)" 
                              strokeWidth={thickness}
                              style={{ animationDuration: `${2.0 - pd * 10}s` }}
                            />
                          )}
                        </g>
                      );
                    });
                  })}

                  {/* Draw Interbank loan linkages */}
                  {Object.keys(interbank).map((lender) => {
                    const b1X = lender === 'SBI' ? 180 : lender === 'PNB' ? 420 : lender === 'BOB' ? 580 : lender === 'ICICI' ? 500 : 260;
                    const b1Y = lender === 'SBI' ? 80 : lender === 'PNB' ? 60 : lender === 'BOB' ? 170 : lender === 'ICICI' ? 300 : 280;

                    return Object.keys(interbank[lender]).map((borrower) => {
                      const amount = interbank[lender][borrower];
                      if (amount < 3000 || lender === borrower) return null; // Draw only main links

                      const b2X = borrower === 'SBI' ? 180 : borrower === 'PNB' ? 420 : borrower === 'BOB' ? 580 : borrower === 'ICICI' ? 500 : 260;
                      const b2Y = borrower === 'SBI' ? 80 : borrower === 'PNB' ? 60 : borrower === 'BOB' ? 170 : borrower === 'ICICI' ? 300 : 280;

                      const lenderInfo = stressResults.banks[lender];
                      const borrowerInfo = stressResults.banks[borrower];
                      const isStressed = borrowerInfo.status === 'BREACHED';

                      // Hover highlight opacity
                      const edgeOpacity = hoveredNode ? ((hoveredNode === lender || hoveredNode === borrower) ? 1.0 : 0.10) : 1.0;

                      return (
                        <g key={`${lender}-${borrower}`} style={{ opacity: edgeOpacity, transition: 'opacity 0.25s ease' }}>
                          <path 
                            d={`M ${b1X} ${b1Y} Q ${(b1X+b2X)/2 + 20} ${(b1Y+b2Y)/2 + 20} ${b2X} ${b2Y}`} 
                            fill="none" 
                            stroke={isStressed ? 'rgba(239, 68, 68, 0.5)' : 'rgba(124, 58, 237, 0.15)'}
                            strokeWidth={amount / 3000}
                          />
                        </g>
                      );
                    });
                  })}

                  {/* Draw Company Nodes */}
                  {Object.values(stressResults.companies).map((c) => {
                    const cX = c.name === 'Coal India' ? 320 : c.name === 'NTPC' ? 440 : c.name === 'Adani Power' ? 420 : c.name === 'JSW Steel' ? 320 : c.name === 'Tata Steel' ? 380 : c.name === 'Company X' ? 350 : c.name === 'Company Y' ? 470 : c.name === 'Company Z' ? 230 : 380;
                    const cY = c.name === 'Coal India' ? 150 : c.name === 'NTPC' ? 160 : c.name === 'Adani Power' ? 220 : c.name === 'JSW Steel' ? 210 : c.name === 'Tata Steel' ? 180 : c.name === 'Company X' ? 110 : c.name === 'Company Y' ? 260 : c.name === 'Company Z' ? 170 : 180;
                    const pdVal = c.pd18m * 100;
                    
                    let nodeColor = '#38bdf8';
                    let glowNode = false;
                    if (pdVal > 5.0) { nodeColor = '#ef4444'; glowNode = true; }
                    else if (pdVal > 1.0) { nodeColor = '#f59e0b'; }

                    const nodeOpacity = hoveredNode ? ((hoveredNode === c.name || isConnected(hoveredNode, c.name)) ? 1.0 : 0.25) : 1.0;

                    return (
                      <g 
                        key={c.name} 
                        style={{ cursor: 'pointer', opacity: nodeOpacity, transition: 'opacity 0.25s ease' }} 
                        onClick={() => setSelectedCompany(c)}
                        onMouseEnter={() => setHoveredNode(c.name)}
                        onMouseLeave={() => setHoveredNode(null)}
                      >
                        {glowNode && <circle cx={cX} cy={cY} r="28" fill="url(#breachGlow)" style={{animation:'pulseGlow 2s infinite'}} />}
                        <circle cx={cX} cy={cY} r="18" fill="#ffffff" stroke={nodeColor} strokeWidth="3" />
                        <text x={cX} y={cY - 22} className="node-label">{c.name.split(' ')[0]}</text>
                        <text x={cX} y={cY + 4} className="node-sublabel" fill="var(--text-secondary)" style={{fontSize:'8px'}}>{pdVal.toFixed(1)}% PD</text>
                      </g>
                    );
                  })}

                  {/* Draw Bank Nodes */}
                  {Object.values(stressResults.banks).map((b) => {
                    const bX = b.name === 'SBI' ? 180 : b.name === 'PNB' ? 420 : b.name === 'BOB' ? 580 : b.name === 'ICICI' ? 500 : 260;
                    const bY = b.name === 'SBI' ? 80 : b.name === 'PNB' ? 60 : b.name === 'BOB' ? 170 : b.name === 'ICICI' ? 300 : 280;
                    
                    const isBreached = b.car < regMinCar;
                    const nearBreach = b.car < (regMinCar + 0.5) && b.car >= regMinCar;
                    
                    let stateColor = 'var(--primary)';
                    if (isBreached) stateColor = 'var(--danger)';
                    else if (nearBreach) stateColor = 'var(--warning)';

                    const nodeOpacity = hoveredNode ? ((hoveredNode === b.name || isConnected(hoveredNode, b.name)) ? 1.0 : 0.25) : 1.0;

                    return (
                      <g 
                        key={b.name} 
                        style={{ cursor: 'pointer', opacity: nodeOpacity, transition: 'opacity 0.25s ease' }}
                        onClick={() => setSelectedBank(b)}
                        onMouseEnter={() => setHoveredNode(b.name)}
                        onMouseLeave={() => setHoveredNode(null)}
                      >
                        {isBreached && <circle cx={bX} cy={bY} r="40" fill="url(#breachGlow)" style={{animation:'pulseGlow 1.5s infinite'}} />}
                        <circle cx={bX} cy={bY} r="24" fill="#ffffff" stroke={stateColor} strokeWidth="4" />
                        <text x={bX} y={bY - 30} className="node-label" style={{fontWeight:'800'}}>{b.name}</text>
                        <text x={bX} y={bY + 5} className="node-sublabel" style={{fill: stateColor, fontWeight:'bold', fontSize:'9px'}}>{b.car.toFixed(1)}%</text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'scenarios' && (
        <div className="main-dashboard-content">
          {/* NGFS summary panel */}
          <div className="glass-panel dashboard-section">
            <div className="section-header">
              <h2><Layers size={20} style={{color:'var(--primary)'}} /> NGFS Scenario Comparison Summary</h2>
              <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>10-Year Horizon Analysis (2026-2036)</span>
            </div>
            
            <div style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:'20px', marginBottom:'24px'}}>
              {scenarioStats.map(stat => (
                <div 
                  key={stat.id} 
                  className={`scenario-card ${activeScenario === stat.id ? 'active' : ''}`}
                  onClick={() => setActiveScenario(stat.id)}
                >
                  <div className="scenario-card-header">
                    <span style={{fontWeight:'700', fontSize:'1.1rem'}}>{stat.name}</span>
                    <span className={`scenario-badge ${
                      stat.id === 'Net_Zero' ? 'badge-orderly' : 
                      stat.id === 'Delayed_Transition' ? 'badge-disorderly' : 'badge-bau'
                    }`}>
                      {stat.id === 'Net_Zero' ? 'Orderly' : stat.id === 'Delayed_Transition' ? 'Disorderly' : 'BAU'}
                    </span>
                  </div>
                  
                  <div style={{marginTop:'10px', display:'flex', flexDirection:'column', gap:'6px', fontSize:'0.85rem'}}>
                    <div style={{display:'flex', justifyBetween:'space-between', justifyContent:'space-between'}}>
                      <span style={{color:'var(--text-secondary)'}}>Stressed Capital Loss:</span>
                      <span style={{fontWeight:'700'}}>{stat.capitalLoss.toFixed(1)} INR Cr</span>
                    </div>
                    <div style={{display:'flex', justifyBetween:'space-between', justifyContent:'space-between'}}>
                      <span style={{color:'var(--text-secondary)'}}>Capital Adequacy Breaches:</span>
                      <span style={{fontWeight:'700', color: stat.breachedCount > 0 ? 'var(--danger)' : 'var(--success)'}}>{stat.breachedCount} Banks</span>
                    </div>
                    <div style={{display:'flex', justifyBetween:'space-between', justifyContent:'space-between'}}>
                      <span style={{color:'var(--text-secondary)'}}>First Capital Breach:</span>
                      <span style={{fontWeight:'700'}}>{stat.firstBreachDate} ({stat.firstBreaker})</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Custom SVG Line Chart with Interactive hover tooltips */}
            <div className="chart-container" style={{ position: 'relative' }}>
              <h3 style={{marginBottom:'15px', fontSize:'1.1rem', color:'var(--text-primary)', display:'flex', alignItems:'center', gap:'6px'}}><TrendingUp size={16} /> Bank CET1 Ratio Trajectory under {activeScenario.replace('_', ' ')}</h3>
              <div style={{ position: 'relative' }}>
                <svg 
                  width="100%" 
                  height="240"
                  onMouseMove={handleChartMouseMove}
                  onMouseLeave={handleChartMouseLeave}
                >
                  {/* Horizontal Grid lines */}
                  {[9.0, 10.0, 12.0, 14.0, 16.0].map((val, idx) => {
                    const y = 200 - (val - 8.0) * 20;
                    return (
                      <g key={val}>
                        <line x1="50" y1={y} x2="100%" y2={y} stroke="var(--border-muted)" />
                        <text x="15" y={y + 4} fill="var(--text-muted)" style={{fontSize:'10px'}}>{val}%</text>
                      </g>
                    );
                  })}
                  {/* Regulatory Min Red line (Dynamic) */}
                  {(() => {
                    const regMinY = 200 - (regMinCar - 8.0) * 20;
                    return (
                      <g>
                        <line x1="50" y1={regMinY} x2="100%" y2={regMinY} stroke="var(--danger)" strokeDasharray="3,3" strokeWidth="2" />
                        <text x="100%" dx="-140" y={regMinY - 6} fill="var(--danger)" style={{fontSize:'9px', fontWeight:'700'}}>Regulatory Min ({regMinCar.toFixed(1)}%)</text>
                      </g>
                    );
                  })()}

                  {/* Plot lines for banks */}
                  {['SBI', 'PNB', 'BOB', 'ICICI', 'HDFC'].map((bank, bankIdx) => {
                    const colors = {
                      SBI: '#38bdf8',
                      PNB: '#ef4444',
                      BOB: '#f59e0b',
                      ICICI: '#a855f7',
                      HDFC: '#10b981'
                    };

                    return (
                      <g key={bank}>
                        <polyline 
                          fill="none" 
                          stroke={colors[bank]} 
                          strokeWidth="3" 
                          points={bankChartPoints[bank]} 
                        />
                        {/* End point dot */}
                        {scenarioChartData.length > 0 && (() => {
                          const lastPoint = scenarioChartData[scenarioChartData.length - 1];
                          const val = lastPoint[bank] || 10;
                          const x = 50 + (window.innerWidth * 0.65);
                          const y = 200 - (val - 8.0) * 20;
                          return (
                            <circle cx={x} cy={y} r="4" fill={colors[bank]} />
                          );
                        })()}
                      </g>
                    );
                  })}

                  {/* Vertical Dotted Line for Hover Indicator */}
                  {hoveredX !== null && (
                    <line x1={hoveredX} y1="20" x2={hoveredX} y2="200" stroke="var(--primary)" strokeDasharray="3,3" strokeWidth="1.5" />
                  )}

                  {/* X Axis label */}
                  <text x="50%" y="235" fill="var(--text-muted)" style={{fontSize:'11px', textAnchor:'middle'}}>Quarterly Timeline (2026 - 2036)</text>
                </svg>

                {/* Interactive Tooltip Card Overlay */}
                {hoveredQuarterIndex !== null && (
                  (() => {
                    const qData = scenarioChartData[hoveredQuarterIndex];
                    if (!qData) return null;
                    
                    // Recreate carbon price at quarter using mathematical parameters of generate_ngfs_scenarios (referenced from memo)
                    const nQ = scenarioChartData.length;

                    return (
                      <div 
                        className="glass-panel chart-tooltip"
                        style={{
                          position: 'absolute',
                          left: `${hoveredChartPos.x + 20}px`,
                          top: `${hoveredChartPos.y - 40}px`,
                          pointerEvents: 'none',
                          zIndex: 100,
                          padding: '12px 16px',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '8px',
                          border: '1px solid var(--border-muted)',
                          background: 'rgba(255,255,255,0.96)',
                          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                          fontSize: '0.82rem',
                          minWidth: '220px'
                        }}
                      >
                        <div style={{ fontWeight: '800', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px', display: 'flex', justifyContent: 'space-between' }}>
                          <span>{qData.quarter}</span>
                          <span style={{ color: 'var(--primary)' }}>{"$"}{carbonPriceVal.toFixed(1)}{" / tCO2"}</span>
                        </div>
                        {['SBI', 'PNB', 'BOB', 'ICICI', 'HDFC'].map(bank => {
                          const car = qData[bank] || 10;
                          const isBreached = car < regMinCar;
                          return (
                            <div key={bank} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>{bank}:</span>
                              <strong style={{ color: isBreached ? 'var(--danger)' : 'var(--success)' }}>{car.toFixed(2)}%</strong>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()
                )}

              </div>
            </div>
              {/* Legend */}
              <div style={{display:'flex', justifyContent:'center', gap:'20px', marginTop:'10px', fontSize:'0.8rem'}}>
                <span style={{display:'flex', alignItems:'center', gap:'6px'}}><span style={{width:'12px', height:'12px', background:'#38bdf8', borderRadius:'50%'}} /> SBI</span>
                <span style={{display:'flex', alignItems:'center', gap:'6px'}}><span style={{width:'12px', height:'12px', background:'#ef4444', borderRadius:'50%'}} /> PNB (Vulnerable)</span>
                <span style={{display:'flex', alignItems:'center', gap:'6px'}}><span style={{width:'12px', height:'12px', background:'#f59e0b', borderRadius:'50%'}} /> BOB</span>
                <span style={{display:'flex', alignItems:'center', gap:'6px'}}><span style={{width:'12px', height:'12px', background:'#a855f7', borderRadius:'50%'}} /> ICICI</span>
                <span style={{display:'flex', alignItems:'center', gap:'6px'}}><span style={{width:'12px', height:'12px', background:'#10b981', borderRadius:'50%'}} /> HDFC</span>
              </div>
            </div>
          </div>
      )}

      {activeTab === 'historical' && (
        <div className="main-dashboard-content">
          <div className="glass-panel dashboard-section">
            <div className="section-header">
              <h2><Award size={20} style={{color:'var(--primary)'}} /> Model Backtesting & Validation</h2>
              <span style={{fontSize:'0.8rem', color:'var(--text-secondary)'}}>Performance on Historical Indian Corporate Defaults</span>
            </div>
            
            <div className="historical-defaults-list">
              {Object.keys(historicalDefaults).map((name) => {
                const data = historicalDefaults[name];
                
                // Set default probability warnings at 12M and 18M prior
                const pd12m = name === 'Lanco Infratech' ? 11.68 : name === 'Essar Power' ? 19.65 : 66.96;
                const pd18m = name === 'Lanco Infratech' ? 4.98 : name === 'Essar Power' ? 9.82 : 42.34;

                return (
                  <div key={name} className="glass-panel historical-card">
                    <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                      <h3 style={{fontSize:'1.2rem', fontWeight:'700'}}>{name}</h3>
                      <span className="scenario-badge badge-disorderly" style={{padding:'4px 10px'}}>Observed Default Event (NPA)</span>
                    </div>
                    
                    <p style={{color:'var(--text-secondary)', fontSize:'0.85rem', marginTop:'6px'}}>
                      Historical coal/power conglomerate that defaulted during the Indian banking credit cycle (2015-2020).
                    </p>
                    
                    <div className="historical-metrics-grid">
                      <div className="historical-metric-box">
                        <div className="historical-metric-title">Debt / EBITDA</div>
                        <div className="historical-metric-value">{data.Debt_EBITDA.toFixed(1)}x</div>
                      </div>
                      <div className="historical-metric-box">
                        <div className="historical-metric-title">Transition Vulnerability (TVS)</div>
                        <div className="historical-metric-value">{data.TVS.toFixed(1)}</div>
                      </div>
                      <div className="historical-metric-box" style={{background:'rgba(245, 158, 11, 0.05)', borderColor:'rgba(245, 158, 11, 0.2)'}}>
                        <div className="historical-metric-title" style={{color:'var(--warning)'}}>18M Early Warning PD</div>
                        <div className="historical-metric-value" style={{color:'var(--warning)'}}>{pd18m.toFixed(1)}%</div>
                      </div>
                      <div className="historical-metric-box" style={{background:'rgba(239, 68, 68, 0.05)', borderColor:'rgba(239, 68, 68, 0.2)'}}>
                        <div className="historical-metric-title" style={{color:'var(--danger)'}}>12M Early Warning PD</div>
                        <div className="historical-metric-value" style={{color:'var(--danger)'}}>{pd12m.toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Detail Modal for Selected Company */}
      {selectedCompany && (
        <div className="modal-overlay" onClick={() => setSelectedCompany(null)}>
          <div className="glass-panel modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal-btn" onClick={() => setSelectedCompany(null)}>✕</button>
            <h2 style={{fontSize:'1.6rem', fontWeight:'800', borderBottom:'1px solid var(--border-muted)', paddingBottom:'10px', marginBottom:'20px'}}>
              {selectedCompany.name} Detail Analysis
            </h2>
            
            <div style={{display:'grid', gridTemplateColumns:'repeat(2, 1fr)', gap:'24px'}}>
              <div>
                <h3 style={{fontSize:'1.1rem', marginBottom:'12px', color:'var(--primary)'}}>Stressed Financial Indicators</h3>
                <div style={{display:'flex', flexDirection:'column', gap:'12px', fontSize:'0.9rem', color:'var(--text-secondary)'}}>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Adjusted EBITDA</span>
                    <span className="stat-val-highlight">{selectedCompany.ebitda.toFixed(1)} INR Cr</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Debt / EBITDA Ratio</span>
                    <span className="stat-val-highlight" style={{color: selectedCompany.debtEbitda > 4.0 ? 'var(--danger)' : 'var(--text-primary)'}}>{selectedCompany.debtEbitda.toFixed(2)}x</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Refinancing Horizon</span>
                    <span className="stat-val-highlight">{selectedCompany.refYears.toFixed(1)} Years</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Transition Vulnerability Score (TVS)</span>
                    <span className="stat-val-highlight">{selectedCompany.tvs.toFixed(1)} / 100</span>
                  </div>
                </div>
              </div>

              <div>
                <h3 style={{fontSize:'1.1rem', marginBottom:'12px', color:'var(--secondary)'}}>Transition Risk Metrics</h3>
                <div style={{display:'flex', flexDirection:'column', gap:'12px', fontSize:'0.9rem', color:'var(--text-secondary)'}}>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Fossil-Fuel Revenue Share</span>
                    <span className="stat-val-highlight">{(baselineFinancials[selectedCompany.name].Carbon_Rev_Pct * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Market Carbon Beta</span>
                    <span className="stat-val-highlight">{baselineTvs[selectedCompany.name].Carbon_Beta.toFixed(4)}</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>Policy Event CAR</span>
                    <span className="stat-val-highlight">{baselineTvs[selectedCompany.name].Event_CAR.toFixed(4)}</span>
                  </div>
                  <div style={{display:'flex', justifyContent:'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom:'6px'}}>
                    <span>NLP Sentiment Score</span>
                    <span className="stat-val-highlight">{baselineTvs[selectedCompany.name].NLP_Sentiment_Score.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel" style={{marginTop:'30px', padding:'20px', background:'rgba(239, 68, 68, 0.03)', borderColor:'rgba(239, 68, 68, 0.1)'}}>
              <h3 style={{fontSize:'1.1rem', marginBottom:'10px', color:'var(--danger)', display:'flex', alignItems:'center', gap:'8px'}}><AlertCircle size={18} /> Probability of Default Progression</h3>
              <p style={{fontSize:'0.85rem', color:'var(--text-secondary)'}}>
                Under a carbon price of **${carbonPrice} / tCO2**, the Cox Proportional Hazards model estimates this company’s 18-Month Probability of Default at <strong style={{color:'var(--danger)'}}>{(selectedCompany.pd18m * 100).toFixed(3)}%</strong> (up from its initial baseline default probability of {(selectedCompany.initialPd * 100).toFixed(3)}%).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Detail Modal for Selected Bank */}
      {selectedBank && (
        <div className="modal-overlay" onClick={() => setSelectedBank(null)}>
          <div className="glass-panel modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-modal-btn" onClick={() => setSelectedBank(null)}>✕</button>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              <Shield size={24} style={{ color: selectedBank.car < regMinCar ? 'var(--danger)' : 'var(--success)' }} />
              <h2 style={{ fontSize: '1.6rem', fontWeight: '800', margin: 0 }}>
                {selectedBank.name} Stress Analysis
              </h2>
            </div>
            
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '20px' }}>
              Detailed capital adequacy projection and credit contagion write-downs under a carbon price of **${carbonPrice} / tCO2**.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '24px', marginBottom: '24px' }}>
              {/* Capital Adequacy Stats */}
              <div>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--primary)' }}>Capital Adequacy Metrics</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Initial CAR (CET1)</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{selectedBank.initialCar.toFixed(1)}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Stressed CAR (CET1)</span>
                    <strong style={{ color: selectedBank.car < regMinCar ? 'var(--danger)' : 'var(--success)', fontSize: '1.05rem' }}>{selectedBank.car.toFixed(2)}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Regulatory CAR Min</span>
                    <strong style={{ color: 'var(--danger)' }}>{regMinCar.toFixed(1)}%</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Initial CET1 Capital</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{selectedBank.capital0.toLocaleString()} INR Cr</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Stressed CET1 Capital</span>
                    <strong style={{ color: 'var(--text-primary)' }}>{selectedBank.capital.toLocaleString(undefined, {maximumFractionDigits: 0})} INR Cr</strong>
                  </div>
                </div>
              </div>

              {/* Loss Attribution */}
              <div>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--secondary)' }}>Stressed Loss Attribution</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Expected Corporate Loan Losses</span>
                    <strong style={{ color: 'var(--danger)' }}>{selectedBank.corpLoss.toLocaleString(undefined, {maximumFractionDigits: 1})} INR Cr</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Expected Interbank Losses</span>
                    <strong style={{ color: 'var(--danger)' }}>{selectedBank.interbankLoss.toLocaleString(undefined, {maximumFractionDigits: 1})} INR Cr</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Total CET1 Capital Loss</span>
                    <strong style={{ color: 'var(--danger)', fontSize: '1.05rem' }}>{(selectedBank.corpLoss + selectedBank.interbankLoss).toLocaleString(undefined, {maximumFractionDigits: 1})} INR Cr</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-muted)', paddingBottom: '4px' }}>
                    <span>Contagion Distress Factor (&theta;)</span>
                    <strong style={{ color: selectedBank.theta > 0 ? 'var(--danger)' : 'var(--success)' }}>{(selectedBank.theta * 100).toFixed(1)}%</strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Corporate Exposure Details Table */}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '10px', color: 'var(--text-primary)' }}>Corporate Loan Portfolio Exposures</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-muted)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '6px' }}>Corporate Debtor</th>
                    <th style={{ padding: '6px' }}>Loan Exposure</th>
                    <th style={{ padding: '6px' }}>Stressed 18M PD</th>
                    <th style={{ padding: '6px', textAlign: 'right' }}>Expected Loss (LGD={(lgdCorp * 100).toFixed(0)}%)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(exposures[selectedBank.name]).map(company => {
                    const exposure = exposures[selectedBank.name][company];
                    const companyData = stressResults.companies[company];
                    if (!companyData) return null;
                    const pdVal = companyData.pd18m;
                    const elVal = exposure * pdVal * lgdCorp;

                    return (
                      <tr key={company} style={{ borderBottom: '1px solid var(--border-muted)' }}>
                        <td style={{ padding: '6px', fontWeight: '700' }}>{company}</td>
                        <td style={{ padding: '6px' }}>{exposure.toLocaleString()} INR Cr</td>
                        <td style={{ padding: '6px', color: pdVal > 0.05 ? 'var(--danger)' : pdVal > 0.01 ? 'var(--warning)' : 'inherit' }}>
                          {(pdVal * 100).toFixed(2)}%
                        </td>
                        <td style={{ padding: '6px', textAlign: 'right', fontWeight: 'bold' }}>{elVal.toFixed(1)} INR Cr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Interbank Exposure Details Table */}
            <div>
              <h3 style={{ fontSize: '1.1rem', marginBottom: '10px', color: 'var(--text-primary)' }}>Interbank Lending & Contagion Exposures</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border-muted)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '6px' }}>Borrowing Bank</th>
                    <th style={{ padding: '6px' }}>Funding Exposure</th>
                    <th style={{ padding: '6px' }}>Stressed CAR</th>
                    <th style={{ padding: '6px' }}>Distress Factor (&theta;)</th>
                    <th style={{ padding: '6px', textAlign: 'right' }}>Expected Loss (LGD={(lgdBank * 100).toFixed(0)}%)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(interbank[selectedBank.name]).map(borrower => {
                    const amount = interbank[selectedBank.name][borrower];
                    if (amount === 0) return null;
                    const borrowerData = stressResults.banks[borrower];
                    const elVal = amount * borrowerData.theta * lgdBank;

                    return (
                      <tr key={borrower} style={{ borderBottom: '1px solid var(--border-muted)' }}>
                        <td style={{ padding: '6px', fontWeight: '700' }}>{borrower}</td>
                        <td style={{ padding: '6px' }}>{amount.toLocaleString()} INR Cr</td>
                        <td style={{ padding: '6px', color: borrowerData.car < regMinCar ? 'var(--danger)' : 'inherit' }}>
                          {borrowerData.car.toFixed(1)}%
                        </td>
                        <td style={{ padding: '6px' }}>
                          {(borrowerData.theta * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '6px', textAlign: 'right', fontWeight: 'bold' }}>{elVal.toFixed(1)} INR Cr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
          </div>
        </div>
      )}

      {/* Footer Branding & License Information */}
      <footer className="glass-panel dashboard-footer" style={{
        marginTop: '20px',
        padding: '20px 30px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '20px',
        fontSize: '0.88rem',
        color: 'var(--text-secondary)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <User size={16} style={{ color: 'var(--primary)' }} />
          <span>Developed by <strong>Mohibul Hoque</strong></span>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap' }}>
          <a href="mailto:hokworks@gmail.com" className="footer-link" style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
            <Mail size={16} style={{ color: 'var(--primary)' }} />
            <span>hokworks@gmail.com</span>
          </a>
          
          <a href="https://linkedin.com/in/speedymohibul" target="_blank" rel="noopener noreferrer" className="footer-link" style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', textDecoration: 'none' }}>
            <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" style={{color:'#0077b5'}}>
              <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
              <rect x="2" y="9" width="4" height="12"></rect>
              <circle cx="4" cy="4" r="2"></circle>
            </svg>
            <span>linkedin.com/in/speedymohibul</span>
          </a>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Shield size={16} style={{ color: 'var(--success)' }} />
          <span>Licensed under <strong style={{ color: 'var(--text-primary)' }}>MIT License</strong></span>
        </div>
      </footer>
    </div>
  );
}

export default App;
