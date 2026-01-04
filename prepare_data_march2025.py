import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
CUTOFF_DATE = "2025-03-31"

# INPUT FILES ( The files you ALREADY have )
raw_files = {
    "Beijing": "Data/Beijing_final_datas.csv",
    "NewDelhi": "Data/Newdelhi_final_datas.csv",
    "Kathmandu": "Data/Kathmandu_Training_Merged.csv" 
}

# OUTPUT FILES ( The files the Agent needs )
output_files = {
    "Beijing": "Data/Beijing_Ready.csv",
    "NewDelhi": "Data/NewDelhi_Ready.csv",
    "Kathmandu": "Data/Kathmandu_Ready.csv"
}

def clean_and_save(city_name, input_path, output_path):
    print(f"Processing {city_name}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f" ERROR: Could not find {input_path}")
        return

    # 2. Standardize Date Column
    # Check if 'Date' exists, if not, try to find it
    if 'Date' not in df.columns:
        # Sometimes it's lowercase 'date'
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
        else:
            print(f" Error: No 'Date' column in {city_name}")
            return

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 3. Cut Data at March 31, 2025
    df = df[df['Date'] <= CUTOFF_DATE]
    
    # 4. Fix "In-Between" Gaps (Interpolation)
    # Create a full daily range from start to cutoff
    full_idx = pd.date_range(start=df['Date'].min(), end=CUTOFF_DATE, freq='D')
    df = df.set_index('Date').reindex(full_idx)
    df.index.name = 'Date'
    
    # Fill numeric columns (interpolate) - keep DatetimeIndex for interpolation
    cols_to_fix = ['pm25', 'Temp', 'Wind', 'Humidity', 'Precip']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time') # Connect the dots
            df[col] = df[col].bfill().ffill() # Fill edges
    
    # Reset index after interpolation
    df = df.reset_index()

    # 5. Save
    df.to_csv(output_path, index=False)
    print(f" Saved: {output_path} (Rows: {len(df)})")

# --- EXECUTION ---
if __name__ == "__main__":
    for city, path in raw_files.items():
        clean_and_save(city, path, output_files[city])
    
    print("\n DONE! '_Ready.csv' files are ready.")
