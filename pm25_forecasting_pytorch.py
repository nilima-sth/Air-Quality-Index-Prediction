"""
PM2.5 Air Quality Forecasting - PyTorch Implementation
Comparative Analysis: XGBoost, LSTM (PyTorch), iTransformer (PyTorch), SARIMAX

Train: 2017-2023 | Test: Jan 2024 - March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# PyTorch for Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

import os

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'cities': {
        'Beijing': 'Data/Beijing_Ready.csv',
        'New Delhi': 'Data/NewDelhi_Ready.csv',
        'Kathmandu': 'Data/Kathmandu_Ready.csv'
    },
    'train_start': '2017-01-01',
    'train_end': '2023-12-31',
    'test_start': '2024-01-01',
    'test_end': '2025-03-31',
    'weather_features': ['Temp', 'Wind', 'Humidity', 'Precip'],
    'lag_features': [1, 2, 3, 7, 14, 30],
    'rolling_windows': [7, 14, 30],
    'lookback': 30,
    'lstm_epochs': 100,
    'lstm_batch_size': 32,
    'lstm_lr': 0.001,
    'transformer_epochs': 100,
    'transformer_batch_size': 32,
    'transformer_lr': 0.001
}

print("="*100)
print("PM2.5 AIR QUALITY FORECASTING - PYTORCH IMPLEMENTATION")
print("="*100)
print(f"\nConfiguration:")
print(f"  Training: {CONFIG['train_start']} to {CONFIG['train_end']}")
print(f"  Testing: {CONFIG['test_start']} to {CONFIG['test_end']}")
print(f"  Models: XGBoost, LSTM (PyTorch), iTransformer (PyTorch), SARIMAX")
print("="*100)

# ================================================================================================
# SECTION 1: DATA LOADING
# ================================================================================================

def load_data(city_name, file_path):
    """Load and prepare data"""
    print(f"\n[{city_name}] Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date'])  # Remove any rows with invalid dates
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Ensure we have required columns
    required = ['Date', 'pm25'] + CONFIG['weather_features']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    print(f"  Data range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Total records: {len(df)}")
    print(f"  PM2.5 range: {df['pm25'].min():.2f} to {df['pm25'].max():.2f}")
    
    return df

# ================================================================================================
# SECTION 2: FEATURE ENGINEERING
# ================================================================================================

def create_temporal_features(df):
    """Create temporal features with cyclical encoding"""
    df = df.copy()
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Cyclical encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    return df

def create_lag_features(df, lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for PM2.5"""
    df = df.copy()
    for lag in lags:
        df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 14, 30]):
    """Create rolling statistics"""
    df = df.copy()
    for window in windows:
        df[f'pm25_rolling_mean_{window}'] = df['pm25'].shift(1).rolling(window=window, min_periods=1).mean()
        df[f'pm25_rolling_std_{window}'] = df['pm25'].shift(1).rolling(window=window, min_periods=1).std()
        df[f'pm25_rolling_min_{window}'] = df['pm25'].shift(1).rolling(window=window, min_periods=1).min()
        df[f'pm25_rolling_max_{window}'] = df['pm25'].shift(1).rolling(window=window, min_periods=1).max()
    return df

def engineer_features(df):
    """Apply all feature engineering"""
    df = create_temporal_features(df)
    df = create_lag_features(df, CONFIG['lag_features'])
    df = create_rolling_features(df, CONFIG['rolling_windows'])
    df = df.bfill().ffill()
    return df

def split_data(df):
    """Split into train and test sets"""
    train = df[(df['Date'] >= CONFIG['train_start']) & (df['Date'] <= CONFIG['train_end'])].copy()
    test = df[(df['Date'] >= CONFIG['test_start']) & (df['Date'] <= CONFIG['test_end'])].copy()
    
    print(f"\n  Train: {len(train)} records ({train['Date'].min()} to {train['Date'].max()})")
    print(f"  Test: {len(test)} records ({test['Date'].min()} to {test['Date'].max()})")
    
    return train, test

# ================================================================================================
# SECTION 3: XGBOOST MODEL
# ================================================================================================

def train_xgboost(train_df, test_df, city_name):
    """Train XGBoost model"""
    print(f"\n{'='*80}")
    print(f"[{city_name}] Training XGBoost")
    print(f"{'='*80}")
    
    # Define features
    feature_cols = (
        CONFIG['weather_features'] +
        ['Month', 'DayOfYear', 'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Quarter', 'DayOfWeek'] +
        [f'pm25_lag_{lag}' for lag in CONFIG['lag_features']] +
        [f'pm25_rolling_mean_{w}' for w in CONFIG['rolling_windows']] +
        [f'pm25_rolling_std_{w}' for w in CONFIG['rolling_windows']] +
        [f'pm25_rolling_min_{w}' for w in CONFIG['rolling_windows']] +
        [f'pm25_rolling_max_{w}' for w in CONFIG['rolling_windows']]
    )
    
    X_train = train_df[feature_cols].values
    y_train = train_df['pm25'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['pm25'].values
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train)}")
    
    # Train
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  Test Performance:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    return y_pred, {'RMSE': rmse, 'MAE': mae}

# ================================================================================================
# SECTION 4: PYTORCH DATASET & DATALOADER
# ================================================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    def __init__(self, data, lookback=30):
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.lookback, :-1]  # All features except last (target)
        y = self.data[idx+self.lookback, -1]  # Target (pm25)
        return X, y

def prepare_sequences(df, lookback=30):
    """Prepare sequences for PyTorch models"""
    feature_cols = (
        CONFIG['weather_features'] +
        ['Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos']
    )
    
    data = df[feature_cols + ['pm25']].values
    
    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler, feature_cols

# ================================================================================================
# SECTION 5: LSTM MODEL (PYTORCH)
# ================================================================================================

class LSTMModel(nn.Module):
    """LSTM model using PyTorch"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.lstm3 = nn.LSTM(hidden_dim//2, hidden_dim//4, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_dim//4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def train_lstm_pytorch(train_df, test_df, city_name):
    """Train LSTM using PyTorch"""
    print(f"\n{'='*80}")
    print(f"[{city_name}] Training LSTM (PyTorch)")
    print(f"{'='*80}")
    
    lookback = CONFIG['lookback']
    
    # Prepare data
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    data_scaled, scaler, feature_cols = prepare_sequences(full_df, lookback)
    
    # Split
    train_size = len(train_df) - lookback
    
    # Create datasets
    train_data = data_scaled[:train_size+lookback]
    test_data = data_scaled[train_size:]
    
    train_dataset = TimeSeriesDataset(train_data, lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['lstm_batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['lstm_batch_size'], shuffle=False)
    
    print(f"  Lookback: {lookback}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Model
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=128, num_layers=3, dropout=0.2)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lstm_lr'])
    
    # Training
    print("\n  Training LSTM...")
    model.train()
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(CONFIG['lstm_epochs']):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{CONFIG['lstm_epochs']}, Loss: {avg_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform
    dummy = np.zeros((len(predictions), len(feature_cols) + 1))
    dummy[:, -1] = predictions
    y_pred = scaler.inverse_transform(dummy)[:, -1]
    
    dummy[:, -1] = actuals
    y_test = scaler.inverse_transform(dummy)[:, -1]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  Test Performance:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    return y_pred, {'RMSE': rmse, 'MAE': mae}

# ================================================================================================
# SECTION 6: iTRANSFORMER MODEL (PYTORCH)
# ================================================================================================

class iTransformerModel(nn.Module):
    """iTransformer model for time series using PyTorch"""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super(iTransformerModel, self).__init__()
        
        # Project input to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)
        
        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x

def train_itransformer_pytorch(train_df, test_df, city_name):
    """Train iTransformer using PyTorch"""
    print(f"\n{'='*80}")
    print(f"[{city_name}] Training iTransformer (PyTorch)")
    print(f"{'='*80}")
    
    lookback = CONFIG['lookback']
    
    # Prepare data (same as LSTM)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    data_scaled, scaler, feature_cols = prepare_sequences(full_df, lookback)
    
    train_size = len(train_df) - lookback
    
    train_data = data_scaled[:train_size+lookback]
    test_data = data_scaled[train_size:]
    
    train_dataset = TimeSeriesDataset(train_data, lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['transformer_batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['transformer_batch_size'], shuffle=False)
    
    print(f"  Lookback: {lookback}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Model
    model = iTransformerModel(input_dim=len(feature_cols), d_model=128, nhead=4, num_layers=2, dropout=0.1)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['transformer_lr'])
    
    # Training
    print("\n  Training iTransformer...")
    model.train()
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(CONFIG['transformer_epochs']):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{CONFIG['transformer_epochs']}, Loss: {avg_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform
    dummy = np.zeros((len(predictions), len(feature_cols) + 1))
    dummy[:, -1] = predictions
    y_pred = scaler.inverse_transform(dummy)[:, -1]
    
    dummy[:, -1] = actuals
    y_test = scaler.inverse_transform(dummy)[:, -1]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  Test Performance:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    return y_pred, {'RMSE': rmse, 'MAE': mae}

# ================================================================================================
# SECTION 7: SARIMAX MODEL
# ================================================================================================

def train_sarimax(train_df, test_df, city_name):
    """Train SARIMAX model"""
    print(f"\n{'='*80}")
    print(f"[{city_name}] Training SARIMAX")
    print(f"{'='*80}")
    
    exog_cols = CONFIG['weather_features'] + ['Month', 'DayOfYear']
    
    y_train = train_df['pm25'].values
    X_train = train_df[exog_cols].values
    y_test = test_df['pm25'].values
    X_test = test_df[exog_cols].values
    
    print(f"  Exogenous features: {len(exog_cols)}")
    print(f"  Training samples: {len(y_train)}")
    
    try:
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        model_fit = model.fit(disp=False, maxiter=200)
        y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)
        y_pred = np.maximum(y_pred, 0)
        
    except Exception as e:
        print(f"  SARIMAX failed with full config, using simplified: {str(e)[:50]}")
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False, maxiter=100)
        y_pred = model_fit.forecast(steps=len(y_test), exog=X_test)
        y_pred = np.maximum(y_pred, 0)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n  Test Performance:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    
    return y_pred, {'RMSE': rmse, 'MAE': mae}

# ================================================================================================
# SECTION 8: VISUALIZATION
# ================================================================================================

def plot_predictions(test_df, predictions_dict, city_name, output_dir='Results'):
    """Plot actual vs predicted"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 6))
    
    # Actual
    dates = test_df['Date'].values
    actual = test_df['pm25'].values
    
    plt.plot(dates, actual, label='Actual PM2.5', color='black', linewidth=2, alpha=0.7)
    
    # Predictions
    colors = {'XGBoost': 'blue', 'LSTM': 'red', 'iTransformer': 'green', 'SARIMAX': 'purple'}
    
    for model_name, pred_values in predictions_dict.items():
        # Handle different lengths (LSTM/Transformer have lookback offset)
        if model_name in ['LSTM', 'iTransformer']:
            lookback = CONFIG['lookback']
            pred_dates = dates[lookback:]
            min_len = min(len(pred_dates), len(pred_values))
            pred_dates = pred_dates[:min_len]
            pred_vals = pred_values[:min_len]
        else:
            pred_dates = dates
            min_len = min(len(pred_dates), len(pred_values))
            pred_dates = pred_dates[:min_len]
            pred_vals = pred_values[:min_len]
        
        plt.plot(pred_dates, pred_vals, label=model_name, 
                color=colors.get(model_name, 'gray'), linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('PM2.5 (μg/m³)', fontsize=12)
    plt.title(f'{city_name} - Model Comparison (Test: Jan 2024 - Mar 2025)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'{output_dir}/{city_name}_Predictions_PyTorch.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n  Plot saved: {filename}")
    plt.close()

# ================================================================================================
# SECTION 9: MAIN EXECUTION
# ================================================================================================

def run_analysis():
    """Main analysis function"""
    results = {}
    
    for city_name, file_path in CONFIG['cities'].items():
        print(f"\n{'#'*100}")
        print(f"# {city_name}")
        print(f"{'#'*100}")
        
        # Load data
        df = load_data(city_name, file_path)
        
        # Feature engineering
        print(f"\n[{city_name}] Engineering features...")
        df = engineer_features(df)
        
        # Split data
        train_df, test_df = split_data(df)
        
        # Store results
        results[city_name] = {}
        predictions = {}
        
        # XGBoost
        xgb_pred, xgb_metrics = train_xgboost(train_df, test_df, city_name)
        results[city_name]['XGBoost'] = xgb_metrics
        predictions['XGBoost'] = xgb_pred
        
        # LSTM (PyTorch)
        lstm_pred, lstm_metrics = train_lstm_pytorch(train_df, test_df, city_name)
        results[city_name]['LSTM'] = lstm_metrics
        predictions['LSTM'] = lstm_pred
        
        # iTransformer (PyTorch)
        trans_pred, trans_metrics = train_itransformer_pytorch(train_df, test_df, city_name)
        results[city_name]['iTransformer'] = trans_metrics
        predictions['iTransformer'] = trans_pred
        
        # SARIMAX
        sarimax_pred, sarimax_metrics = train_sarimax(train_df, test_df, city_name)
        results[city_name]['SARIMAX'] = sarimax_metrics
        predictions['SARIMAX'] = sarimax_pred
        
        # Plot
        print(f"\n[{city_name}] Generating plot...")
        plot_predictions(test_df, predictions, city_name)
    
    # Summary table
    print("\n" + "="*100)
    print("FINAL RESULTS - TEST PERIOD (JAN 2024 - MARCH 2025)")
    print("="*100)
    
    data = []
    for city, models in results.items():
        for model, metrics in models.items():
            data.append({
                'City': city,
                'Model': model,
                'RMSE': f"{metrics['RMSE']:.4f}",
                'MAE': f"{metrics['MAE']:.4f}"
            })
    
    df_results = pd.DataFrame(data)
    print(df_results.to_string(index=False))
    
    # Save results
    df_results.to_csv('Results/Model_Comparison_PyTorch.csv', index=False)
    print("\nResults saved to: Results/Model_Comparison_PyTorch.csv")
    print("="*100)
    
    return results

if __name__ == "__main__":
    results = run_analysis()
    print("\nAnalysis complete!")
