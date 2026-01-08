# AI-Driven Decision Audit System

## Overview
This project is an AI-powered Decision Intelligence system designed to analyze, audit, and improve business decisions before they are executed. Instead of only predicting outcomes, the system explains why a decision may succeed or fail and suggests better alternatives using explainable AI.

## Problem Statement
Enterprises often incur losses due to incorrect business decisions such as pricing changes made without understanding demand, competition, or risk. This system helps identify risky decisions early and provides transparent explanations.

## Key Features
- Predicts expected revenue impact of business decisions
- Audits past decisions to identify loss-causing actions
- Explains decisions using SHAP-based explainable AI
- Performs what-if simulations to recommend better alternatives
- Interactive Streamlit dashboard for real-time decision analysis

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (Random Forest)
- SHAP (Explainable AI)
- Streamlit
- Matplotlib

## System Workflow
1. Business decision data is collected and processed
2. ML models predict revenue impact and risk
3. Decision audit logic compares predictions with actual outcomes
4. SHAP explains feature-level impact
5. Users interact with the system through a Streamlit dashboard

## Use Case Example
A pricing manager can test a proposed price change and instantly see:
- Expected profit or loss
- Risk level
- Key factors influencing the outcome
- Alternative pricing suggestions

## Future Enhancements
- Support for credit and inventory decisions
- Real-time data integration
- Cloud deployment and API integration

## Author
Riya Soni
