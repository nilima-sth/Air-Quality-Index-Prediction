"""
Quick Visualization Script - View PM2.5 PyTorch Results
Similar style to view_results.py but for PyTorch implementation
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

print("Loading PyTorch results...")

# Load model comparison table
comparison_df = pd.read_csv('Results/Model_Comparison_PyTorch.csv')

# Load actual test data for time series
beijing_data = pd.read_csv('Data/Beijing_Ready.csv')
beijing_data['Date'] = pd.to_datetime(beijing_data['Date'], format='mixed', errors='coerce')
beijing_test = beijing_data[beijing_data['Date'] >= '2024-01-01'].copy()

newdelhi_data = pd.read_csv('Data/NewDelhi_Ready.csv')
newdelhi_data['Date'] = pd.to_datetime(newdelhi_data['Date'], format='mixed', errors='coerce')
newdelhi_test = newdelhi_data[newdelhi_data['Date'] >= '2024-01-01'].copy()

kathmandu_data = pd.read_csv('Data/Kathmandu_Ready.csv')
kathmandu_data['Date'] = pd.to_datetime(kathmandu_data['Date'], format='mixed', errors='coerce')
kathmandu_test = kathmandu_data[kathmandu_data['Date'] >= '2024-01-01'].copy()

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============= ROW 1: PERFORMANCE COMPARISONS =============
# Plot 1: RMSE Comparison
ax1 = fig.add_subplot(gs[0, 0])
pivot_rmse = comparison_df.pivot(index='City', columns='Model', values='RMSE')
pivot_rmse.plot(kind='bar', ax=ax1, width=0.75, 
                color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
ax1.set_title('RMSE Comparison by City and Model', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('RMSE (μg/m³)', fontsize=11, fontweight='bold')
ax1.set_xlabel('City', fontsize=11, fontweight='bold')
ax1.legend(title='Model', fontsize=9, title_fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(axis='x', rotation=0)

# Plot 2: MAE Comparison
ax2 = fig.add_subplot(gs[0, 1])
pivot_mae = comparison_df.pivot(index='City', columns='Model', values='MAE')
pivot_mae.plot(kind='bar', ax=ax2, width=0.75,
               color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
ax2.set_title('MAE Comparison by City and Model', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylabel('MAE (μg/m³)', fontsize=11, fontweight='bold')
ax2.set_xlabel('City', fontsize=11, fontweight='bold')
ax2.legend(title='Model', fontsize=9, title_fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=0)

# Plot 3: Winner Summary
ax3 = fig.add_subplot(gs[0, 2])
winners = comparison_df.loc[comparison_df.groupby('City')['RMSE'].idxmin()]
colors_map = {'XGBoost': '#2ecc71', 'LSTM': '#e74c3c', 'iTransformer': '#3498db', 'SARIMAX': '#e67e22'}
colors = [colors_map.get(model, 'gray') for model in winners['Model']]
bars = ax3.bar(winners['City'], winners['RMSE'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_title('Best Model Performance by City', fontsize=13, fontweight='bold', pad=10)
ax3.set_ylabel('RMSE (μg/m³)', fontsize=11, fontweight='bold')
ax3.set_xlabel('City', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add model labels on bars
for i, (bar, city, rmse, model) in enumerate(zip(bars, winners['City'], winners['RMSE'], winners['Model'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'★ {model}\n{rmse:.2f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============= ROW 2: TIME SERIES - 2024 =============
# Beijing 2024
ax4 = fig.add_subplot(gs[1, 0])
beijing_2024 = beijing_test[beijing_test['Date'].dt.year == 2024]
ax4.plot(beijing_2024['Date'], beijing_2024['pm25'], 
         color='#2E86AB', linewidth=1.5, marker='o', markersize=2)
ax4.set_title('Beijing PM2.5 - Test Period 2024', fontsize=13, fontweight='bold', pad=10)
ax4.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
mean_val = beijing_2024['pm25'].mean()
ax4.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax4.legend(fontsize=9)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# New Delhi 2024
ax5 = fig.add_subplot(gs[1, 1])
newdelhi_2024 = newdelhi_test[newdelhi_test['Date'].dt.year == 2024]
ax5.plot(newdelhi_2024['Date'], newdelhi_2024['pm25'], 
         color='#A23B72', linewidth=1.5, marker='o', markersize=2)
ax5.set_title('New Delhi PM2.5 - Test Period 2024', fontsize=13, fontweight='bold', pad=10)
ax5.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Date', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
mean_val = newdelhi_2024['pm25'].mean()
ax5.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax5.legend(fontsize=9)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Kathmandu 2024
ax6 = fig.add_subplot(gs[1, 2])
kathmandu_2024 = kathmandu_test[kathmandu_test['Date'].dt.year == 2024]
ax6.plot(kathmandu_2024['Date'], kathmandu_2024['pm25'], 
         color='#F18F01', linewidth=1.5, marker='o', markersize=2)
ax6.set_title('Kathmandu PM2.5 - Test Period 2024', fontsize=13, fontweight='bold', pad=10)
ax6.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax6.set_xlabel('Date', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)
mean_val = kathmandu_2024['pm25'].mean()
ax6.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax6.legend(fontsize=9)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============= ROW 3: TIME SERIES - 2025 (Jan-March) =============
# Beijing 2025
ax7 = fig.add_subplot(gs[2, 0])
beijing_2025 = beijing_test[beijing_test['Date'].dt.year == 2025]
ax7.plot(beijing_2025['Date'], beijing_2025['pm25'], 
         color='#2E86AB', linewidth=2, marker='o', markersize=4)
ax7.set_title('Beijing PM2.5 - Test Period 2025 (Jan-Mar)', fontsize=13, fontweight='bold', pad=10)
ax7.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax7.set_xlabel('Date', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)
mean_val = beijing_2025['pm25'].mean()
ax7.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax7.legend(fontsize=9)
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

# New Delhi 2025
ax8 = fig.add_subplot(gs[2, 1])
newdelhi_2025 = newdelhi_test[newdelhi_test['Date'].dt.year == 2025]
ax8.plot(newdelhi_2025['Date'], newdelhi_2025['pm25'], 
         color='#A23B72', linewidth=2, marker='o', markersize=4)
ax8.set_title('New Delhi PM2.5 - Test Period 2025 (Jan-Mar)', fontsize=13, fontweight='bold', pad=10)
ax8.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax8.set_xlabel('Date', fontsize=11, fontweight='bold')
ax8.grid(True, alpha=0.3)
mean_val = newdelhi_2025['pm25'].mean()
ax8.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax8.legend(fontsize=9)
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Kathmandu 2025
ax9 = fig.add_subplot(gs[2, 2])
kathmandu_2025 = kathmandu_test[kathmandu_test['Date'].dt.year == 2025]
ax9.plot(kathmandu_2025['Date'], kathmandu_2025['pm25'], 
         color='#F18F01', linewidth=2, marker='o', markersize=4)
ax9.set_title('Kathmandu PM2.5 - Test Period 2025 (Jan-Mar)', fontsize=13, fontweight='bold', pad=10)
ax9.set_ylabel('PM2.5 (μg/m³)', fontsize=11, fontweight='bold')
ax9.set_xlabel('Date', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)
mean_val = kathmandu_2025['pm25'].mean()
ax9.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f} μg/m³')
ax9.legend(fontsize=9)
plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Main title
fig.suptitle('PM2.5 Forecasting - PyTorch Implementation Dashboard\nTest Period: January 2024 - March 2025', 
            fontsize=18, fontweight='bold', y=0.995)

# Save
output_path = 'Results/PyTorch_Dashboard.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Dashboard saved: {output_path}")

plt.show()

# Print detailed results
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON - PyTorch Implementation")
print("Test Period: January 2024 - March 2025")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# Calculate statistics
print("\n📊 TEST PERIOD STATISTICS:")
print("\nBeijing (2024-2025):")
print(f"  Mean PM2.5: {beijing_test['pm25'].mean():.2f} μg/m³")
print(f"  Max PM2.5: {beijing_test['pm25'].max():.2f} μg/m³")
print(f"  Min PM2.5: {beijing_test['pm25'].min():.2f} μg/m³")
print(f"  Std Dev: {beijing_test['pm25'].std():.2f} μg/m³")

print("\nNew Delhi (2024-2025):")
print(f"  Mean PM2.5: {newdelhi_test['pm25'].mean():.2f} μg/m³")
print(f"  Max PM2.5: {newdelhi_test['pm25'].max():.2f} μg/m³")
print(f"  Min PM2.5: {newdelhi_test['pm25'].min():.2f} μg/m³")
print(f"  Std Dev: {newdelhi_test['pm25'].std():.2f} μg/m³")

print("\nKathmandu (2024-2025):")
print(f"  Mean PM2.5: {kathmandu_test['pm25'].mean():.2f} μg/m³")
print(f"  Max PM2.5: {kathmandu_test['pm25'].max():.2f} μg/m³")
print(f"  Min PM2.5: {kathmandu_test['pm25'].min():.2f} μg/m³")
print(f"  Std Dev: {kathmandu_test['pm25'].std():.2f} μg/m³")

print("\n🏆 KEY FINDINGS:")
print("  • XGBoost wins for ALL 3 cities (★★★)")
print("  • Beijing: XGBoost RMSE = 28.82 μg/m³")
print("  • New Delhi: XGBoost RMSE = 36.05 μg/m³")
print("  • Kathmandu: XGBoost RMSE = 13.53 μg/m³")
print("  • SARIMAX significantly underperforms (5-7x worse)")
print("\n💡 RECOMMENDATION:")
print("  Deploy XGBoost for production PM2.5 forecasting")
print("  Use PyTorch LSTM as backup for longer sequences")
print("\n✅ Analysis complete!")
