"""
Generate PM2.5 Forecasts for 2026-2030 using PyTorch Best Models (XGBoost)
Creates year-by-year plots showing seasonal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENERATING PM2.5 FORECASTS FOR 2026-2030 (DPI SET TO 500)")
print("="*80)

def create_temporal_features(df):
    """Add cyclical temporal features"""
    df['month_sin'] = np.sin(2 * np.pi * df['DOY'] / 365.25)
    df['month_cos'] = np.cos(2 * np.pi * df['DOY'] / 365.25)
    df['day_sin'] = np.sin(2 * np.pi * df['DOY'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['DOY'] / 365.25)
    return df

def create_lag_features(df, target_col='pm25', lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for XGBoost"""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='pm25', windows=[7, 14, 30]):
    """Create rolling statistics"""
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window=window).max()
    return df

def train_xgboost_model(city):
    """Train XGBoost on historical data"""
    print(f"\nTraining XGBoost for {city}...")
    
    # Load historical data
    train_data = pd.read_csv(f'Data/{city}_Ready.csv')
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='mixed', errors='coerce')
    train_data = train_data.sort_values('Date').reset_index(drop=True)
    
    # Use 2017-2023 for training
    train_data = train_data[train_data['Date'] < '2024-01-01'].copy()
    
    # Feature engineering
    train_data = create_temporal_features(train_data)
    train_data = create_lag_features(train_data)
    train_data = create_rolling_features(train_data)
    
    # Drop NaN rows created by lag/rolling features
    train_data = train_data.dropna()
    
    # Feature columns
    feature_cols = ['Temp', 'Wind', 'Humidity', 'Precip', 
                    'month_sin', 'month_cos', 'day_sin', 'day_cos']
    
    # Add lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        feature_cols.append(f'pm25_lag_{lag}')
    
    # Add rolling features
    for window in [7, 14, 30]:
        feature_cols.extend([f'pm25_roll_mean_{window}', f'pm25_roll_std_{window}',
                            f'pm25_roll_min_{window}', f'pm25_roll_max_{window}'])
    
    X_train = train_data[feature_cols]
    y_train = train_data['pm25']
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    print(f"{city} model trained!")
    
    return model, feature_cols, train_data

def generate_future_predictions(city, model, feature_cols, historical_data):
    """Generate predictions for 2026-2030"""
    print(f"\nGenerating forecasts for {city} (2026-2030)...")
    
    # Load future weather data
    future_data = pd.read_csv(f'Data/Prediction_files/{city}_Future_Weather_2026_2030.csv')
    
    # Standardize column names
    if 'date' in future_data.columns:
        future_data.rename(columns={'date': 'Date'}, inplace=True)
    if 'Date' not in future_data.columns and 'YEAR' in future_data.columns and 'DOY' in future_data.columns:
        future_data['Date'] = pd.to_datetime(future_data['YEAR'].astype(str) + '-' + future_data['DOY'].astype(str), format='%Y-%j')
    else:
        future_data['Date'] = pd.to_datetime(future_data['Date'], format='mixed', errors='coerce')
    
    future_data = future_data.sort_values('Date').reset_index(drop=True)
    
    # Add required columns if missing
    if 'YEAR' not in future_data.columns:
        future_data['YEAR'] = future_data['Date'].dt.year
    if 'DOY' not in future_data.columns:
        future_data['DOY'] = future_data['Date'].dt.dayofyear
    
    # Create temporal features
    future_data = create_temporal_features(future_data)
    
    # Initialize pm25 column with historical mean
    future_data['pm25'] = historical_data['pm25'].mean()
    
    # Iterative forecasting to generate lag and rolling features
    predictions = []
    
    for idx in range(len(future_data)):
        # Get the recent history for lag/rolling features
        if idx == 0:
            # Use last 30 days from historical data
            recent_history = historical_data[['pm25']].tail(30).copy()
            recent_history = pd.concat([recent_history, 
                                      pd.DataFrame({'pm25': predictions})], 
                                      ignore_index=True)
        else:
            # Use previous predictions
            recent_history = pd.DataFrame({'pm25': predictions[-30:]})
        
        # Create lag features
        row_features = {}
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(recent_history) >= lag:
                row_features[f'pm25_lag_{lag}'] = recent_history['pm25'].iloc[-lag]
            else:
                row_features[f'pm25_lag_{lag}'] = historical_data['pm25'].mean()
        
        # Create rolling features
        for window in [7, 14, 30]:
            if len(recent_history) >= window:
                row_features[f'pm25_roll_mean_{window}'] = recent_history['pm25'].tail(window).mean()
                row_features[f'pm25_roll_std_{window}'] = recent_history['pm25'].tail(window).std()
                row_features[f'pm25_roll_min_{window}'] = recent_history['pm25'].tail(window).min()
                row_features[f'pm25_roll_max_{window}'] = recent_history['pm25'].tail(window).max()
            else:
                row_features[f'pm25_roll_mean_{window}'] = historical_data['pm25'].mean()
                row_features[f'pm25_roll_std_{window}'] = historical_data['pm25'].std()
                row_features[f'pm25_roll_min_{window}'] = historical_data['pm25'].min()
                row_features[f'pm25_roll_max_{window}'] = historical_data['pm25'].max()
        
        # Get weather features
        row_features['Temp'] = future_data.iloc[idx]['Temp']
        row_features['Wind'] = future_data.iloc[idx]['Wind']
        row_features['Humidity'] = future_data.iloc[idx]['Humidity']
        row_features['Precip'] = future_data.iloc[idx]['Precip']
        row_features['month_sin'] = future_data.iloc[idx]['month_sin']
        row_features['month_cos'] = future_data.iloc[idx]['month_cos']
        row_features['day_sin'] = future_data.iloc[idx]['day_sin']
        row_features['day_cos'] = future_data.iloc[idx]['day_cos']
        
        # Create feature array in correct order
        X_pred = pd.DataFrame([row_features])[feature_cols]
        
        # Predict
        pred = model.predict(X_pred)[0]
        pred = max(0, pred)  # PM2.5 can't be negative
        predictions.append(pred)
    
    future_data['pm25_forecast'] = predictions
    
    print(f"{city} forecasts generated!")
    print(f"   Mean PM2.5 (2026-2030): {future_data['pm25_forecast'].mean():.2f} μg/m³")
    print(f"   Max PM2.5 (2026-2030): {future_data['pm25_forecast'].max():.2f} μg/m³")
    
    return future_data

def create_annual_forecast_plots():
    """Create year-by-year forecast plots for all cities"""
    
    cities = ['Beijing', 'NewDelhi', 'Kathmandu']
    city_labels = ['Beijing', 'New Delhi', 'Kathmandu']
    all_forecasts = {}
    
    # Train models and generate forecasts
    for city in cities:
        model, feature_cols, historical_data = train_xgboost_model(city)
        forecast = generate_future_predictions(city, model, feature_cols, historical_data)
        all_forecasts[city] = forecast
    
    # Plot 1: Long-term trend (2026-2030) for all cities
    print("\nCreating long-term trend plot...")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = {'Beijing': '#2E86AB', 'NewDelhi': '#A23B72', 'Kathmandu': '#F18F01'}
    
    for city, city_label in zip(cities, city_labels):
        forecast = all_forecasts[city]
        ax.plot(forecast['Date'], forecast['pm25_forecast'], 
               linewidth=2, label=city_label, color=colors[city], alpha=0.8)
        
        # Add mean line
        mean_val = forecast['pm25_forecast'].mean()
        ax.axhline(y=mean_val, color=colors[city], linestyle='--', alpha=0.4,
                  linewidth=1.5)
    
    ax.set_title('PM2.5 Long-Term Forecast - All Cities (2026-2030)\nXGBoost Model Predictions',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('PM2.5 (μg/m³)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Straight Dates
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # --- CHANGED: DPI set to 500 ---
    output_path = 'Results/Forecast_2026_2030_LongTerm.png'
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path} (DPI=500)")
    plt.close()
    
    # Plot 2: Year-by-year breakdown
    years = [2026, 2027, 2028, 2029, 2030]
    
    for city, city_label in zip(cities, city_labels):
        print(f"\nCreating annual plots for {city_label}...")
        
        forecast = all_forecasts[city]
        
        # Create 5-panel plot (one per year)
        fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=False)
        fig.suptitle(f'{city_label} - Annual PM2.5 Forecasts (2026-2030)\nXGBoost Predictions',
                    fontsize=18, fontweight='bold', y=0.995)
        
        for idx, year in enumerate(years):
            ax = axes[idx]
            year_data = forecast[forecast['Date'].dt.year == year]
            
            # Straight lines (no markers)
            ax.plot(year_data['Date'], year_data['pm25_forecast'],
                    color=colors[city], linewidth=2, marker=None)
            
            # Add mean line
            mean_val = year_data['pm25_forecast'].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f} μg/m³')
            
            ax.set_title(f'{year} Forecast', fontsize=14, fontweight='bold', pad=10)
            ax.set_ylabel('PM2.5 (μg/m³)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right')
            
            if idx == 4:  # Last plot
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            
            # Straight Dates
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        plt.tight_layout()
        
        # --- CHANGED: DPI set to 500 ---
        output_path = f'Results/{city}_Annual_Forecasts_2026_2030.png'
        plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path} (DPI=500)")
        plt.close()
    
    # Save forecast CSVs
    print("\nSaving forecast data...")
    for city, city_label in zip(cities, city_labels):
        forecast = all_forecasts[city]
        output_csv = f'Results/{city}_Forecast_2026_2030.csv'
        forecast[['Date', 'pm25_forecast', 'Temp', 'Wind', 'Humidity', 'Precip']].to_csv(
            output_csv, index=False)
        print(f"Saved: {output_csv}")
    
    print("\n" + "="*80)
    print("ALL FORECAST PLOTS CREATED!")
    print("="*80)

if __name__ == "__main__":
    create_annual_forecast_plots()
    print("\nForecasting complete!")