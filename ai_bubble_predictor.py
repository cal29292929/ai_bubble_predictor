# ai_bubble_predictor.py
# Version 1.0
# A conceptual model to assess the risk of an AI market bubble burst.
#
# DISCLAIMER: This script is for educational purposes only and does not constitute financial advice.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    """
    Placeholder function to load market data.
    In a real-world scenario, you would replace this with an API call to a financial data provider
    or by loading a local CSV file.

    The DataFrame should be time-indexed and contain columns for the indicators.
    """
    # --- Replace with your actual data loading mechanism ---
    # Example: df = pd.read_csv('your_market_data.csv', index_col='Date', parse_dates=True)
    
    # Creating sample dummy data for demonstration purposes
    print("---
    Loading sample data. Replace with your actual financial data source.
    ---")
    dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=100, freq='D'))
    data = {
        'nasdaq_composite': np.linspace(15000, 25000, 100) + np.random.randn(100) * 500,
        'ai_etf_price': np.linspace(50, 150, 100) + np.random.randn(100) * 10,
        'vix_index': np.linspace(25, 15, 100) + np.random.randn(100) * 2,
        'fed_interest_rate': [2.5] * 50 + [2.75] * 50, # Simulating a rate hike
        'pe_ratio_tech': np.linspace(30, 70, 100) + np.random.randn(100) * 5,
    }
    df = pd.DataFrame(data, index=dates)
    return df

def calculate_indicators(df):
    """
    Calculate metrics and indicators from the raw data that signal bubble-like behavior.
    """
    # 1. Price Acceleration: How fast are prices rising compared to historical trends?
    df['ai_etf_roc_90d'] = df['ai_etf_price'].pct_change(90) * 100 # 90-day Rate of Change

    # 2. Volatility: Is the market becoming complacent? (Low VIX can signal euphoria)
    # We will use the raw VIX index for this. Lower is riskier in a euphoric market.

    # 3. Valuation: Are price-to-earnings ratios historically high?
    # We will use the raw P/E ratio for this.

    # Remove any NaN values created by pct_change
    df.dropna(inplace=True)
    return df

def normalize_data(df):
    """
    Scale all relevant indicator values to a 0-1 range for consistent weighting.
    """
    scaler = MinMaxScaler()
    
    # Note: For VIX, lower values indicate higher risk, so we invert it by (1 - value).
    df['vix_normalized'] = 1 - scaler.fit_transform(df[['vix_index']])
    
    # For other indicators, higher values mean higher risk.
    df['roc_normalized'] = scaler.fit_transform(df[['ai_etf_roc_90d']])
    df['pe_ratio_normalized'] = scaler.fit_transform(df[['pe_ratio_tech']])
    
    return df

def calculate_bubble_risk_score(df, weights):
    """
    Calculate the final risk score based on weighted indicators.
    The score is scaled to be out of 100.
    """
    # Select the most recent data point for the current risk score
    latest_data = df.iloc[-1]

    # Calculate the weighted average of the normalized indicators
    risk_score = (
        latest_data['roc_normalized'] * weights['price_acceleration'] +
        latest_data['vix_normalized'] * weights['market_complacency'] +
        latest_data['pe_ratio_normalized'] * weights['valuation_extremes']
    )
    
    # Normalize the score to be out of 100
    normalized_score = (risk_score / sum(weights.values())) * 100
    
    return normalized_score

def main():
    """
    Main execution function.
    """
    print("Starting AI Bubble Risk Analysis...")

    # Define the weights for each indicator. These can be adjusted based on market analysis.
    # Total weight should ideally sum to 1.0 for clarity.
    indicator_weights = {
        'price_acceleration': 0.4,   # Rapid price increase is a strong signal
        'valuation_extremes': 0.4,   # High P/E ratios are a classic bubble sign
        'market_complacency': 0.2    # Low volatility (VIX) indicates euphoria
    }

    # 1. Load Data
    market_data = load_data()
    if market_data.empty:
        print("No data loaded. Exiting.")
        return

    # 2. Calculate Indicators
    indicator_df = calculate_indicators(market_data)

    # 3. Normalize Data for Scoring
    normalized_df = normalize_data(indicator_df)

    # 4. Calculate Final Risk Score
    risk_score = calculate_bubble_risk_score(normalized_df, indicator_weights)

    # 5. Output Results
    print("\n--- AI Bubble Risk Score ---")
    print(f"Current Score: {risk_score:.2f} / 100")
    
    if risk_score > 60:
        print("Risk Level: HIGH. Market exhibits strong characteristics of a speculative bubble.")
    elif risk_score > 30:
        print("Risk Level: MODERATE. Market is showing signs of overheating. Caution is advised.")
    else:
        print("Risk Level: LOW. Market appears to be operating within normal parameters.")
        
    print("\nThis model is a conceptual tool and not financial advice.")


if __name__ == "__main__":
    main()
