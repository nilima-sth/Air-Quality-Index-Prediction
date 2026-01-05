import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def plot_annual_trends():
    cities = ['Beijing', 'NewDelhi', 'Kathmandu']
    
    # Define distinct colors for each city
    colors = {
        'Beijing': '#008080',   # Teal
        'NewDelhi': '#800080',  # Deep Purple
        'Kathmandu': '#D35400'  # Dark Orange
    }
    
    # Create a figure with 3 subplots (one for each city)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    for idx, city in enumerate(cities):
        ax = axes[idx]
        file_path = f'Data/{city}_Ready.csv'
        
        print(f"Processing {city}...")
        
        try:
            # 1. Load Data
            df = pd.read_csv(file_path)
            
            # 2. Process Dates
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            elif 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
            
            df = df.dropna(subset=['Date'])
            
            # 3. Extract Year
            df['Year'] = df['Date'].dt.year
            
            # 4. Filter Data (Ensure we stop at 2025 if future data exists)
            df = df[df['Year'] <= 2025]
            
            # 5. Calculate Annual Average
            annual_data = df.groupby('Year')['pm25'].mean().reset_index()
            
            # 6. Plot
            # Line plot for trend
            sns.lineplot(data=annual_data, x='Year', y='pm25', ax=ax, 
                         color=colors[city], linewidth=3, marker='o', markersize=10)
            
            # Bar plot (optional, behind the line for emphasis) - uncomment if you want bars too
            # sns.barplot(data=annual_data, x='Year', y='pm25', ax=ax, color=colors[city], alpha=0.3)
            
            # 7. Formatting
            ax.set_title(f"Annual PM2.5 concentration trend in {city}", fontsize=16, fontweight='bold')
            ax.set_ylabel("PM2.5 Concentration (μg/m³)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Year", fontsize=12, fontweight='bold')
            
            # Annotate values on points
            for x, y in zip(annual_data['Year'], annual_data['pm25']):
                ax.text(x, y + 5, f'{y:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Ensure integer years on X-axis
            ax.set_xticks(annual_data['Year'].unique())
            
        except Exception as e:
            print(f"Error processing {city}: {e}")
            ax.text(0.5, 0.5, f"Data error for {city}", ha='center', va='center')

    plt.tight_layout()
    
    # Save the plot
    output_file = 'Results/Annual_PM25_Trends_AllCities.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Annual Trend Plot saved: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_annual_trends()