"""
Hedging Strategy Analysis with Visualizations
BU623 Derivatives - Week 3
Wilfrid Laurier University

This script analyzes hedging strategies using MSCI indices and CME futures data.
It calculates optimal hedge ratios, basis risk, and optimal number of contracts.
Includes visualizations: correlation heatmap, scatter plots, regression summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# Configuration
# =============================================================================

DATA_FILE = Path(__file__).parent / "Hedging_PortfolioValue_21.xlsx"
OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Futures contract specifications (from Group Activity PDF)
CONTRACT_SPECS = {
    'SP500': {'multiplier': 50, 'name': 'E-mini S&P 500'},       # Contract size: 50
    'FTSE_EM': {'multiplier': 100, 'name': 'E-mini FTSE EM'},    # Contract size: 100
    'CHINA50': {'multiplier': 2, 'name': 'E-mini FTSE China 50'}, # Contract size: 2
    'NIKKEI': {'multiplier': 5, 'name': 'Nikkei 225'},           # Contract size: 5
}

# Portfolio values from Group Activity PDF (TQM Hedge Fund)
PORTFOLIO_VALUES = {
    'MSCI WORLD U$ - PRICE INDEX': 500_000_000,    # Portfolio 1: $500M - MSCI World
    'MSCI PACIFIC U$ - PRICE INDEX': 160_000_000,  # Portfolio 4: $160M - MSCI Pacific
    'MSCI EUROPE U$ - PRICE INDEX': 175_000_000,   # Portfolio 3: $175M - MSCI Europe
    'MSCI EM U$ - PRICE INDEX': 200_000_000,       # Portfolio 2: $200M - MSCI EM
}

# Index-Futures pairing for cross-hedging analysis (corrected per case study)
HEDGE_PAIRS = [
    {
        'name': 'MSCI World vs S&P 500',
        'spot_col': 'MSCI WORLD U$ - PRICE INDEX',
        'futures_col': 'CME-MINI S&P 500 INDEX CONT. - SETT. PRICE',
        'contract_key': 'SP500'
    },
    {
        'name': 'MSCI EM vs FTSE EM',
        'spot_col': 'MSCI EM U$ - PRICE INDEX',
        'futures_col': 'CME-E MINI FTSE EMER INDEX CONT - SETT. PRICE',
        'contract_key': 'FTSE_EM'
    },
    {
        'name': 'MSCI Europe vs S&P 500 (Cross)',
        'spot_col': 'MSCI EUROPE U$ - PRICE INDEX',
        'futures_col': 'CME-MINI S&P 500 INDEX CONT. - SETT. PRICE',
        'contract_key': 'SP500'
    },
    {
        'name': 'MSCI Pacific vs Nikkei 225',
        'spot_col': 'MSCI PACIFIC U$ - PRICE INDEX',
        'futures_col': 'CME-NIKKEI 225 INDEX COMP. CONTINUOUS - SETT. PRICE',
        'contract_key': 'NIKKEI'
    },
]


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load and merge index and futures data from Excel file."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    indexes = pd.read_excel(DATA_FILE, sheet_name='Indexes')
    futures = pd.read_excel(DATA_FILE, sheet_name='Futures')
    
    indexes.rename(columns={'Date': 'Date'}, inplace=True)
    futures.rename(columns={'Dates': 'Date'}, inplace=True)
    
    df = pd.merge(indexes, futures, on='Date', how='inner')
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Data period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total observations: {len(df)}")
    print(f"Frequency: Weekly")
    print()
    
    return df


def calculate_returns(df):
    """Calculate weekly returns for all indices and futures."""
    returns_df = pd.DataFrame()
    returns_df['Date'] = df['Date'].iloc[1:]
    
    # Index returns
    for col in ['MSCI WORLD U$ - PRICE INDEX', 'MSCI PACIFIC U$ - PRICE INDEX', 
                'MSCI EUROPE U$ - PRICE INDEX', 'MSCI EM U$ - PRICE INDEX']:
        short_name = col.replace(' U$ - PRICE INDEX', '').replace('MSCI ', '')
        returns_df[f'{short_name}_ret'] = df[col].pct_change().dropna().values
    
    # Futures returns
    futures_cols = {
        'CME-MINI S&P 500 INDEX CONT. - SETT. PRICE': 'SP500_fut_ret',
        'CME-E MINI FTSE EMER INDEX CONT - SETT. PRICE': 'FTSE_EM_fut_ret',
        'CME-NIKKEI 225 INDEX COMP. CONTINUOUS - SETT. PRICE': 'NIKKEI_fut_ret',
        'CME-EMINI FTSE CHINA 50 CONT - SETT. PRICE': 'CHINA50_fut_ret'
    }
    
    for col, name in futures_cols.items():
        returns_df[name] = df[col].pct_change().dropna().values
    
    returns_df.reset_index(drop=True, inplace=True)
    return returns_df


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_correlation_heatmap(returns_df):
    """Create and save correlation heatmap."""
    print("Creating correlation heatmap...")
    
    # Select relevant columns for correlation (all 4 indices and all 4 futures)
    cols_for_corr = ['WORLD_ret', 'PACIFIC_ret', 'EUROPE_ret', 'EM_ret',
                     'SP500_fut_ret', 'FTSE_EM_fut_ret', 'CHINA50_fut_ret', 'NIKKEI_fut_ret']
    
    corr_matrix = returns_df[cols_for_corr].corr()
    
    # Rename for cleaner display
    rename_map = {
        'WORLD_ret': 'MSCI World',
        'PACIFIC_ret': 'MSCI Pacific',
        'EUROPE_ret': 'MSCI Europe',
        'EM_ret': 'MSCI EM',
        'SP500_fut_ret': 'S&P 500 Fut',
        'FTSE_EM_fut_ret': 'FTSE EM Fut',
        'CHINA50_fut_ret': 'China 50 Fut',
        'NIKKEI_fut_ret': 'Nikkei Fut'
    }
    corr_matrix.rename(index=rename_map, columns=rename_map, inplace=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Color scale: 0.5 is center (red), below 0.5 = red, above 0.5 = green
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                vmin=0.5, vmax=1, mask=mask, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                annot_kws={'size': 9})
    
    # Draw bold black box around bottom-left 4x4 (Futures vs Indices)
    # The heatmap is 8x8.
    # Rows 4-7 (Futures) vs Cols 0-3 (Indices)
    # Rectangle(xy, width, height) where xy is bottom-left
    # Note: In matplotlib heatmap, y-axis starts from top.
    # Indices are at columns 0-3.
    # Futures are at rows 4-7.
    # We want the intersection: Futures on Y (rows 4-8), Indices on X (cols 0-4).
    # xy = (0, 4) -> Top-left of the box (in data coordinates, (0,0) is top-left)
    ax.add_patch(Rectangle((0, 4), 4, 4, fill=False, edgecolor='black', lw=3))
    
    ax.set_title('Cross-Correlation Heatmap: Index Returns vs Futures Returns', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / '01_correlation_heatmap.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    
    return corr_matrix


def plot_scatter_with_regression(returns_df, pair, ax=None):
    """Create scatter plot with regression line for a hedge pair."""
    spot_col = pair['spot_col'].replace(' U$ - PRICE INDEX', '').replace('MSCI ', '') + '_ret'
    
    # Map futures column to return column
    futures_map = {
        'CME-MINI S&P 500 INDEX CONT. - SETT. PRICE': 'SP500_fut_ret',
        'CME-E MINI FTSE EMER INDEX CONT - SETT. PRICE': 'FTSE_EM_fut_ret',
        'CME-NIKKEI 225 INDEX COMP. CONTINUOUS - SETT. PRICE': 'NIKKEI_fut_ret',
        'CME-EMINI FTSE CHINA 50 CONT - SETT. PRICE': 'CHINA50_fut_ret'
    }
    futures_col = futures_map[pair['futures_col']]
    
    x = returns_df[futures_col].values
    y = returns_df[spot_col].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(x * 100, y * 100, alpha=0.5, s=30, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line * 100, y_line * 100, 'r-', linewidth=2, 
            label=f'h* = {slope:.4f}\nR^2 = {r_value**2:.4f}')
    
    ax.set_xlabel(f'{futures_col.replace("_ret", "")} Returns (%)', fontsize=11)
    ax.set_ylabel(f'{spot_col.replace("_ret", "")} Returns (%)', fontsize=11)
    ax.set_title(f'{pair["name"]}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    return slope, intercept, r_value**2, p_value, std_err


def plot_all_scatter_plots(returns_df):
    """Create a 2x2 grid of scatter plots for all hedge pairs."""
    print("Creating scatter plots with regression lines...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Index Returns vs Futures Returns with Optimal Hedge Ratio (h*)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    regression_results = []
    
    for idx, pair in enumerate(HEDGE_PAIRS):
        ax = axes[idx // 2, idx % 2]
        slope, intercept, r_sq, p_val, std_err = plot_scatter_with_regression(returns_df, pair, ax)
        regression_results.append({
            'pair': pair['name'],
            'h_star': slope,
            'intercept': intercept,
            'R_squared': r_sq,
            'p_value': p_val,
            'std_err': std_err
        })
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / '02_scatter_plots.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    
    return regression_results


def create_regression_summary_table(returns_df):
    """Create detailed regression summary for each hedge pair."""
    print("Creating regression summary...")
    
    results = []
    
    for pair in HEDGE_PAIRS:
        spot_col = pair['spot_col'].replace(' U$ - PRICE INDEX', '').replace('MSCI ', '') + '_ret'
        
        futures_map = {
            'CME-MINI S&P 500 INDEX CONT. - SETT. PRICE': 'SP500_fut_ret',
            'CME-E MINI FTSE EMER INDEX CONT - SETT. PRICE': 'FTSE_EM_fut_ret',
            'CME-NIKKEI 225 INDEX COMP. CONTINUOUS - SETT. PRICE': 'NIKKEI_fut_ret',
            'CME-EMINI FTSE CHINA 50 CONT - SETT. PRICE': 'CHINA50_fut_ret'
        }
        futures_col = futures_map[pair['futures_col']]
        
        x = returns_df[futures_col].values
        y = returns_df[spot_col].values
        
        # Detailed regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate additional statistics
        n = len(x)
        y_pred = slope * x + intercept
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        
        # T-statistic for slope
        t_stat = slope / std_err
        
        # Variance of returns
        sigma_S = np.std(y) * 100  # as percentage
        sigma_F = np.std(x) * 100
        
        results.append({
            'Hedge Pair': pair['name'],
            'h* (slope)': slope,
            't-stat': t_stat,
            'p-value': p_value,
            'R-squared': r_value**2,
            'Std Error': std_err,
            'Intercept': intercept,
            'sigma_S (%)': sigma_S,
            'sigma_F (%)': sigma_F,
            'n': n
        })
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'regression_summary.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    return results_df


def plot_regression_summary_table(reg_df):
    """Create a visual table of regression results."""
    print("Creating regression summary visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    
    # Format columns for display
    display_df = reg_df[['Hedge Pair', 'h* (slope)', 'R-squared', 't-stat', 'p-value', 'sigma_S (%)', 'sigma_F (%)']].copy()
    display_df['h* (slope)'] = display_df['h* (slope)'].apply(lambda x: f'{x:.4f}')
    display_df['R-squared'] = display_df['R-squared'].apply(lambda x: f'{x:.4f}')
    display_df['t-stat'] = display_df['t-stat'].apply(lambda x: f'{x:.2f}')
    display_df['p-value'] = display_df['p-value'].apply(lambda x: f'{x:.2e}')
    display_df['sigma_S (%)'] = display_df['sigma_S (%)'].apply(lambda x: f'{x:.2f}')
    display_df['sigma_F (%)'] = display_df['sigma_F (%)'].apply(lambda x: f'{x:.2f}')
    
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')
    
    plt.title('Linear Regression Summary: Index Returns ~ Futures Returns', 
              fontsize=14, fontweight='bold', pad=20)
    
    filepath = OUTPUT_DIR / '03_regression_summary.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Contract Calculation with Portfolio Values
# =============================================================================

def calculate_optimal_contracts_with_values(reg_df, df):
    """Calculate optimal number of contracts using portfolio values."""
    print("\n" + "=" * 70)
    print("OPTIMAL NUMBER OF CONTRACTS")
    print("=" * 70)
    
    latest = df.iloc[-1]
    results = []
    
    print(f"\n{'Hedge Pair':<40} {'Portfolio':>15} {'h*':>10} {'V_F':>15} {'N*':>10} {'Contracts':>12}")
    print("-" * 102)
    
    for idx, pair in enumerate(HEDGE_PAIRS):
        h_star = reg_df.iloc[idx]['h* (slope)']
        portfolio_value = PORTFOLIO_VALUES[pair['spot_col']]
        
        futures_price = latest[pair['futures_col']]
        multiplier = CONTRACT_SPECS[pair['contract_key']]['multiplier']
        V_F = futures_price * multiplier
        
        N_star = h_star * (portfolio_value / V_F)
        N_rounded = round(N_star)
        
        results.append({
            'Hedge Pair': pair['name'],
            'Portfolio Value': portfolio_value,
            'h*': h_star,
            'Futures Price': futures_price,
            'Multiplier': multiplier,
            'V_F': V_F,
            'N* (exact)': N_star,
            'Contracts': N_rounded
        })
        
        print(f"{pair['name']:<40} ${portfolio_value/1e6:>13.1f}M {h_star:>10.4f} ${V_F:>13,.0f} {N_star:>10.2f} {N_rounded:>12}")
    
    print()
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'contract_calculations.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    return results_df


def plot_contract_summary(contract_df):
    """Create visual summary of contract calculations."""
    print("Creating contract calculation visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of contracts needed
    ax1 = axes[0]
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
    bars = ax1.bar(range(len(contract_df)), contract_df['Contracts'].abs(), color=colors)
    ax1.set_xticks(range(len(contract_df)))
    ax1.set_xticklabels([p.replace(' vs ', '\nvs ').replace(' (Cross)', '\n(Cross)') 
                          for p in contract_df['Hedge Pair']], fontsize=9)
    ax1.set_ylabel('Number of Futures Contracts')
    ax1.set_title('Optimal Number of Contracts by Hedge Pair', fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, contract_df['Contracts']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    # Table of calculations
    ax2 = axes[1]
    ax2.axis('off')
    
    display_df = contract_df[['Hedge Pair', 'Portfolio Value', 'h*', 'V_F', 'Contracts']].copy()
    display_df['Portfolio Value'] = display_df['Portfolio Value'].apply(lambda x: f'${x/1e6:.0f}M')
    display_df['h*'] = display_df['h*'].apply(lambda x: f'{x:.4f}')
    display_df['V_F'] = display_df['V_F'].apply(lambda x: f'${x:,.0f}')
    
    table = ax2.table(cellText=display_df.values,
                      colLabels=['Hedge Pair', 'Portfolio', 'h*', 'V_F', 'Contracts'],
                      loc='center',
                      cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax2.set_title('Contract Calculation Details\nN* = h* x (V_A / V_F)', fontweight='bold')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / '04_contract_summary.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# PART 2: MULTI-FUTURE HEDGING ANALYSIS
# =============================================================================

import statsmodels.api as sm

# Multi-future hedge specifications
MULTI_HEDGE_SPECS = [
    {
        'name': 'MSCI World',
        'spot_col': 'WORLD_ret',
        'futures_cols': ['SP500_fut_ret'],  # Keep single - already high R²
        'contract_keys': ['SP500'],
        'portfolio_value': 500_000_000
    },
    {
        'name': 'MSCI EM',
        'spot_col': 'EM_ret',
        'futures_cols': ['FTSE_EM_fut_ret', 'CHINA50_fut_ret'],  # Add China 50
        'contract_keys': ['FTSE_EM', 'CHINA50'],
        'portfolio_value': 200_000_000
    },
    {
        'name': 'MSCI Europe',
        'spot_col': 'EUROPE_ret',
        'futures_cols': ['SP500_fut_ret', 'NIKKEI_fut_ret'],  # Add Nikkei
        'contract_keys': ['SP500', 'NIKKEI'],
        'portfolio_value': 175_000_000
    },
    {
        'name': 'MSCI Pacific',
        'spot_col': 'PACIFIC_ret',
        'futures_cols': ['NIKKEI_fut_ret', 'FTSE_EM_fut_ret'],  # Add FTSE EM
        'contract_keys': ['NIKKEI', 'FTSE_EM'],
        'portfolio_value': 160_000_000
    },
]

# Firm-wide portfolio weights
TOTAL_PORTFOLIO = 1_035_000_000
PORTFOLIO_WEIGHTS = {
    'WORLD_ret': 500_000_000 / TOTAL_PORTFOLIO,
    'EM_ret': 200_000_000 / TOTAL_PORTFOLIO,
    'EUROPE_ret': 175_000_000 / TOTAL_PORTFOLIO,
    'PACIFIC_ret': 160_000_000 / TOTAL_PORTFOLIO,
}


def run_multi_future_hedges(returns_df):
    """Run multi-future regressions for portfolios needing improvement."""
    print("\n" + "=" * 70)
    print("MULTI-FUTURE HEDGING ANALYSIS")
    print("=" * 70)
    
    results = []
    
    for spec in MULTI_HEDGE_SPECS:
        print(f"\n--- {spec['name']} ---")
        
        y = returns_df[spec['spot_col']].values
        X = returns_df[spec['futures_cols']].values
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_with_const).fit()
        
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adj R-squared: {model.rsquared_adj:.4f}")
        
        betas = {}
        for i, col in enumerate(spec['futures_cols']):
            beta = model.params[i + 1]  # +1 for constant
            t_stat = model.tvalues[i + 1]
            p_val = model.pvalues[i + 1]
            betas[col] = {'beta': beta, 't_stat': t_stat, 'p_value': p_val}
            fut_name = col.replace('_fut_ret', '')
            print(f"  {fut_name}: h* = {beta:.4f}, t = {t_stat:.2f}, p = {p_val:.4f}")
        
        results.append({
            'name': spec['name'],
            'model': model,
            'betas': betas,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'futures_cols': spec['futures_cols'],
            'contract_keys': spec['contract_keys'],
            'portfolio_value': spec['portfolio_value']
        })
    
    return results


def run_four_factor_comparison(returns_df):
    """Run 4-factor regressions for each portfolio to identify significant futures."""
    print("\n" + "=" * 70)
    print("4-FACTOR REGRESSION ANALYSIS (All Futures as Predictors)")
    print("=" * 70)
    print("Purpose: Identify statistically significant futures for each portfolio")
    print("Note: Multicollinearity may inflate standard errors\n")
    
    all_futures = ['SP500_fut_ret', 'FTSE_EM_fut_ret', 'CHINA50_fut_ret', 'NIKKEI_fut_ret']
    indices = ['WORLD_ret', 'EM_ret', 'EUROPE_ret', 'PACIFIC_ret']
    index_names = ['MSCI World', 'MSCI EM', 'MSCI Europe', 'MSCI Pacific']
    
    results = []
    
    for idx_col, idx_name in zip(indices, index_names):
        print(f"--- {idx_name} ---")
        
        y = returns_df[idx_col].values
        X = returns_df[all_futures].values
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_with_const).fit()
        
        print(f"Adj R²: {model.rsquared_adj:.4f}")
        
        row = {'Portfolio': idx_name, 'Adj_R2': model.rsquared_adj}
        
        sig_futures = []
        for i, fut in enumerate(all_futures):
            beta = model.params[i + 1]
            t_stat = model.tvalues[i + 1]
            p_val = model.pvalues[i + 1]
            fut_name = fut.replace('_fut_ret', '')
            sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
            if p_val < 0.05:
                sig_futures.append(fut_name)
            print(f"  {fut_name}: β={beta:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")
            row[f'{fut_name}_beta'] = beta
            row[f'{fut_name}_t'] = t_stat
            row[f'{fut_name}_p'] = p_val
        
        row['Significant_Futures'] = ', '.join(sig_futures)
        results.append(row)
        print(f"  → Significant (p<0.05): {', '.join(sig_futures) if sig_futures else 'None'}\n")
    
    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / 'four_factor_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    return results_df


def create_four_factor_table(four_factor_df):
    """Create a formatted table visualization for 4-factor regression."""
    print("Creating 4-factor regression table...")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    
    # Prepare data
    portfolios = four_factor_df['Portfolio'].values
    futures = ['SP500', 'FTSE_EM', 'CHINA50', 'NIKKEI']
    
    # Create table data
    table_data = []
    for i, row in four_factor_df.iterrows():
        table_row = [row['Portfolio'], f"{row['Adj_R2']:.4f}"]
        for fut in futures:
            beta = row[f'{fut}_beta']
            t_stat = row[f'{fut}_t']
            p_val = row[f'{fut}_p']
            sig = '*' if p_val < 0.05 else ''
            table_row.append(f"{beta:.4f}{sig}")
            table_row.append(f"{t_stat:.2f}")
        table_data.append(table_row)
    
    # Column headers
    col_labels = ['Portfolio', 'Adj R²']
    for fut in futures:
        col_labels.extend([f'{fut} β', f'{fut} t'])
    
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    # Style header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight significant cells
    for i, row in four_factor_df.iterrows():
        for j, fut in enumerate(futures):
            p_val = row[f'{fut}_p']
            if p_val < 0.05:
                table[(i+1, 2 + j*2)].set_facecolor('#C6EFCE')  # Green for significant
    
    plt.title('4-Factor Regression Results (* = p < 0.05)\nUsed to identify significant futures for multi-future hedging', 
              fontsize=12, fontweight='bold', pad=20)
    
    filepath = OUTPUT_DIR / '05_four_factor_table.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def run_multi_future_hedges_significant(returns_df, four_factor_df):
    """Run multi-future regressions using handpicked futures based on analysis."""
    print("\n" + "=" * 70)
    print("MULTI-FUTURE HEDGING (Handpicked Selection)")
    print("=" * 70)
    print("Based on 4-factor analysis, we select futures that provide meaningful improvement:")
    
    # Handpicked multi-hedge specs based on cost-benefit analysis:
    # - World: Single hedge sufficient (R² = 95.4%, adding futures only gets 96.9%)
    # - EM: Single hedge sufficient (R² = 88.8%, adding futures only gets 91.2%)
    # - Europe: SP500 + Nikkei (cross-hedge needs improvement, 2 futures better than 3)
    # - Pacific: FTSE_EM + Nikkei (captures EM + Japan exposure)
    multi_specs = [
        {'name': 'MSCI World', 'spot_col': 'WORLD_ret', 
         'futures_cols': ['SP500_fut_ret'],  # Single hedge - already 95.4% R²
         'contract_keys': ['SP500'], 'portfolio_value': 500_000_000,
         'rationale': 'Single hedge sufficient (R²=95.4%)'},
        {'name': 'MSCI EM', 'spot_col': 'EM_ret',
         'futures_cols': ['FTSE_EM_fut_ret'],  # Single hedge - 88.8% R² adequate
         'contract_keys': ['FTSE_EM'], 'portfolio_value': 200_000_000,
         'rationale': 'Single hedge sufficient (R²=88.8%)'},
        {'name': 'MSCI Europe', 'spot_col': 'EUROPE_ret',
         'futures_cols': ['SP500_fut_ret', 'NIKKEI_fut_ret'],  # 2-factor: meaningful improvement
         'contract_keys': ['SP500', 'NIKKEI'], 'portfolio_value': 175_000_000,
         'rationale': '2-factor improves from 64.5% to ~74%'},
        {'name': 'MSCI Pacific', 'spot_col': 'PACIFIC_ret',
         'futures_cols': ['FTSE_EM_fut_ret', 'NIKKEI_fut_ret'],  # 2-factor: captures regional exposure
         'contract_keys': ['FTSE_EM', 'NIKKEI'], 'portfolio_value': 160_000_000,
         'rationale': '2-factor captures EM + Japan exposure'},
    ]
    
    results = []
    
    for spec in multi_specs:
        print(f"\n--- {spec['name']} ---")
        print(f"Rationale: {spec['rationale']}")
        print(f"Using: {', '.join([c.replace('_fut_ret', '') for c in spec['futures_cols']])}")
        
        y = returns_df[spec['spot_col']].values
        X = returns_df[spec['futures_cols']].values
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(y, X_with_const).fit()
        
        print(f"Adj R²: {model.rsquared_adj:.4f}")
        
        betas = {}
        for i, col in enumerate(spec['futures_cols']):
            beta = model.params[i + 1]
            t_stat = model.tvalues[i + 1]
            p_val = model.pvalues[i + 1]
            betas[col] = {'beta': beta, 't_stat': t_stat, 'p_value': p_val}
            fut_name = col.replace('_fut_ret', '')
            sig = '*' if p_val < 0.05 else ''
            print(f"  {fut_name}: β={beta:.4f}, t={t_stat:.2f} {sig}")
        
        results.append({
            'name': spec['name'],
            'model': model,
            'betas': betas,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'futures_cols': spec['futures_cols'],
            'contract_keys': spec['contract_keys'],
            'portfolio_value': spec['portfolio_value']
        })
    
    return results


def calculate_firmwide_return(returns_df):
    """Calculate firm-wide portfolio return using constant weights."""
    firmwide_ret = np.zeros(len(returns_df))
    
    for col, weight in PORTFOLIO_WEIGHTS.items():
        firmwide_ret += returns_df[col].values * weight
    
    returns_df['FIRMWIDE_ret'] = firmwide_ret
    return returns_df


def run_firmwide_regression(returns_df):
    """Run regression of firm-wide return on all 4 futures."""
    print("\n" + "=" * 70)
    print("FIRM-WIDE PORTFOLIO ANALYSIS (4 Futures)")
    print("=" * 70)
    print(f"Total Portfolio: ${TOTAL_PORTFOLIO/1e9:.3f}B")
    print("Weights: World={:.1%}, EM={:.1%}, Europe={:.1%}, Pacific={:.1%}".format(
        *PORTFOLIO_WEIGHTS.values()))
    
    all_futures = ['SP500_fut_ret', 'FTSE_EM_fut_ret', 'CHINA50_fut_ret', 'NIKKEI_fut_ret']
    
    y = returns_df['FIRMWIDE_ret'].values
    X = returns_df[all_futures].values
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"\n4-Factor Model:")
    print(f"Adj R²: {model.rsquared_adj:.4f}")
    
    results = {'intercept': model.params[0], 'r_squared': model.rsquared, 'adj_r_squared': model.rsquared_adj}
    
    for i, fut in enumerate(all_futures):
        beta = model.params[i + 1]
        t_stat = model.tvalues[i + 1]
        p_val = model.pvalues[i + 1]
        fut_name = fut.replace('_fut_ret', '')
        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
        print(f"  {fut_name}: β={beta:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")
        results[fut_name] = {'beta': beta, 't_stat': t_stat, 'p_value': p_val}
    
    results['model'] = model
    return results


def run_firmwide_regression_no_china(returns_df):
    """Run regression of firm-wide return on 3 significant futures (excluding China50)."""
    print("\n--- Firm-Wide (Excluding China50 - Not Significant) ---")
    
    sig_futures = ['SP500_fut_ret', 'FTSE_EM_fut_ret', 'NIKKEI_fut_ret']
    
    y = returns_df['FIRMWIDE_ret'].values
    X = returns_df[sig_futures].values
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"3-Factor Model (SP500, FTSE_EM, NIKKEI):")
    print(f"Adj R²: {model.rsquared_adj:.4f}")
    
    results = {'intercept': model.params[0], 'r_squared': model.rsquared, 'adj_r_squared': model.rsquared_adj}
    
    for i, fut in enumerate(sig_futures):
        beta = model.params[i + 1]
        t_stat = model.tvalues[i + 1]
        p_val = model.pvalues[i + 1]
        fut_name = fut.replace('_fut_ret', '')
        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
        print(f"  {fut_name}: β={beta:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")
        results[fut_name] = {'beta': beta, 't_stat': t_stat, 'p_value': p_val}
    
    results['model'] = model
    return results


def run_firmwide_regression_sp_nikkei(returns_df):
    """Run regression of firm-wide return on 2 futures (S&P 500 + Nikkei only)."""
    print("\n--- Firm-Wide 2-Factor (S&P 500 + Nikkei Only) ---")
    
    two_futures = ['SP500_fut_ret', 'NIKKEI_fut_ret']
    
    y = returns_df['FIRMWIDE_ret'].values
    X = returns_df[two_futures].values
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"2-Factor Model (SP500, NIKKEI):")
    print(f"Adj R²: {model.rsquared_adj:.4f}")
    
    results = {'intercept': model.params[0], 'r_squared': model.rsquared, 'adj_r_squared': model.rsquared_adj}
    
    for i, fut in enumerate(two_futures):
        beta = model.params[i + 1]
        t_stat = model.tvalues[i + 1]
        p_val = model.pvalues[i + 1]
        fut_name = fut.replace('_fut_ret', '')
        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.1 else ''))
        print(f"  {fut_name}: β={beta:.4f}, t={t_stat:.2f}, p={p_val:.4f} {sig}")
        results[fut_name] = {'beta': beta, 't_stat': t_stat, 'p_value': p_val}
    
    results['model'] = model
    return results


def calculate_all_contracts(single_reg_df, multi_results, firmwide_4f, firmwide_3f, firmwide_2f, df):
    """Calculate contracts for all hedge strategies."""
    print("\n" + "=" * 70)
    print("CONTRACT CALCULATIONS - ALL STRATEGIES")
    print("=" * 70)
    
    latest = df.iloc[-1]
    futures_prices = {
        'SP500': latest['CME-MINI S&P 500 INDEX CONT. - SETT. PRICE'],
        'FTSE_EM': latest['CME-E MINI FTSE EMER INDEX CONT - SETT. PRICE'],
        'CHINA50': latest['CME-EMINI FTSE CHINA 50 CONT - SETT. PRICE'],
        'NIKKEI': latest['CME-NIKKEI 225 INDEX COMP. CONTINUOUS - SETT. PRICE']
    }
    
    all_contracts = []
    
    # Single-future contracts
    print("\n--- Single-Future Hedges ---")
    single_total_value = 0
    for idx, pair in enumerate(HEDGE_PAIRS):
        h_star = single_reg_df.iloc[idx]['h* (slope)']
        portfolio_value = PORTFOLIO_VALUES[pair['spot_col']]
        futures_price = latest[pair['futures_col']]
        multiplier = CONTRACT_SPECS[pair['contract_key']]['multiplier']
        V_F = futures_price * multiplier
        N_star = h_star * (portfolio_value / V_F)
        N_rounded = round(N_star)
        contract_value = abs(N_rounded) * V_F
        single_total_value += contract_value
        
        all_contracts.append({
            'Strategy': 'Single-Future',
            'Portfolio': pair['name'].split(' vs ')[0],
            'Future': pair['contract_key'],
            'h*': h_star,
            'Contracts': N_rounded,
            'Contract_Value': contract_value
        })
        print(f"  {pair['name'].split(' vs ')[0]}: {N_rounded} {pair['contract_key']} (${contract_value/1e6:.1f}M)")
    print(f"  TOTAL Contract Value: ${single_total_value/1e6:.1f}M")
    
    # Multi-future contracts
    print("\n--- Multi-Future Hedges (Significant Only) ---")
    multi_total_value = 0
    for result in multi_results:
        for fut_col, contract_key in zip(result['futures_cols'], result['contract_keys']):
            beta = result['betas'][fut_col]['beta']
            futures_price = futures_prices[contract_key]
            multiplier = CONTRACT_SPECS[contract_key]['multiplier']
            V_F = futures_price * multiplier
            N_star = beta * (result['portfolio_value'] / V_F)
            N_rounded = round(N_star)
            contract_value = abs(N_rounded) * V_F
            multi_total_value += contract_value
            
            all_contracts.append({
                'Strategy': 'Multi-Future',
                'Portfolio': result['name'],
                'Future': contract_key,
                'h*': beta,
                'Contracts': N_rounded,
                'Contract_Value': contract_value
            })
    print(f"  TOTAL Contract Value: ${multi_total_value/1e6:.1f}M")
    
    # Firm-wide 4-factor contracts
    print("\n--- Firm-Wide 4-Factor ---")
    fw4_total_value = 0
    for fut_key in ['SP500', 'FTSE_EM', 'CHINA50', 'NIKKEI']:
        if fut_key in firmwide_4f:
            beta = firmwide_4f[fut_key]['beta']
            futures_price = futures_prices[fut_key]
            multiplier = CONTRACT_SPECS[fut_key]['multiplier']
            V_F = futures_price * multiplier
            N_star = beta * (TOTAL_PORTFOLIO / V_F)
            N_rounded = round(N_star)
            contract_value = abs(N_rounded) * V_F
            fw4_total_value += contract_value
            
            all_contracts.append({
                'Strategy': 'Firm-Wide-4F',
                'Portfolio': 'Firm-Wide',
                'Future': fut_key,
                'h*': beta,
                'Contracts': N_rounded,
                'Contract_Value': contract_value
            })
            print(f"  {fut_key}: {N_rounded} contracts (${contract_value/1e6:.1f}M)")
    print(f"  TOTAL Contract Value: ${fw4_total_value/1e6:.1f}M")
    
    # Firm-wide 3-factor contracts (no China50)
    print("\n--- Firm-Wide 3-Factor (No China50) ---")
    fw3_total_value = 0
    for fut_key in ['SP500', 'FTSE_EM', 'NIKKEI']:
        if fut_key in firmwide_3f:
            beta = firmwide_3f[fut_key]['beta']
            futures_price = futures_prices[fut_key]
            multiplier = CONTRACT_SPECS[fut_key]['multiplier']
            V_F = futures_price * multiplier
            N_star = beta * (TOTAL_PORTFOLIO / V_F)
            N_rounded = round(N_star)
            contract_value = abs(N_rounded) * V_F
            fw3_total_value += contract_value
            
            all_contracts.append({
                'Strategy': 'Firm-Wide-3F',
                'Portfolio': 'Firm-Wide',
                'Future': fut_key,
                'h*': beta,
                'Contracts': N_rounded,
                'Contract_Value': contract_value
            })
            print(f"  {fut_key}: {N_rounded} contracts (${contract_value/1e6:.1f}M)")
    print(f"  TOTAL Contract Value: ${fw3_total_value/1e6:.1f}M")
    
    # Firm-wide 2-factor contracts (SP500 + NIKKEI only)
    print("\n--- Firm-Wide 2-Factor (SP500 + NIKKEI Only) ---")
    fw2_total_value = 0
    for fut_key in ['SP500', 'NIKKEI']:
        if fut_key in firmwide_2f:
            beta = firmwide_2f[fut_key]['beta']
            futures_price = futures_prices[fut_key]
            multiplier = CONTRACT_SPECS[fut_key]['multiplier']
            V_F = futures_price * multiplier
            N_star = beta * (TOTAL_PORTFOLIO / V_F)
            N_rounded = round(N_star)
            contract_value = abs(N_rounded) * V_F
            fw2_total_value += contract_value
            
            all_contracts.append({
                'Strategy': 'Firm-Wide-2F',
                'Portfolio': 'Firm-Wide',
                'Future': fut_key,
                'h*': beta,
                'Contracts': N_rounded,
                'Contract_Value': contract_value
            })
            print(f"  {fut_key}: {N_rounded} contracts (${contract_value/1e6:.1f}M)")
    print(f"  TOTAL Contract Value: ${fw2_total_value/1e6:.1f}M")
    
    contracts_df = pd.DataFrame(all_contracts)
    csv_path = OUTPUT_DIR / 'all_contracts.csv'
    contracts_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    return contracts_df, {
        'single': single_total_value,
        'multi': multi_total_value,
        'firmwide_4f': fw4_total_value,
        'firmwide_3f': fw3_total_value,
        'firmwide_2f': fw2_total_value
    }


def create_comparison_summary(single_reg_df, multi_results, firmwide_4f, firmwide_3f, firmwide_2f, total_values):
    """Create comparison summary table for all strategies."""
    print("\nCreating strategy comparison summary...")
    
    # Count total futures used in multi-future strategy
    multi_futures_count = sum(len(r['futures_cols']) for r in multi_results)
    
    # Note: Average R² across different regressions is NOT statistically meaningful
    # Only firm-wide R² is a single regression that can be directly compared
    summary_data = [
        ['Single-Future (4 portfolios)', 
         '4',  # one per portfolio
         f"${total_values['single']/1e6:.0f}M",
         "Simple, 1 future per portfolio"],
        ['Multi-Future (Handpicked)', 
         f"{multi_futures_count}",
         f"${total_values['multi']/1e6:.0f}M",
         "World/EM: single; Europe/Pacific: 2-factor"],
        ['Firm-Wide 4-Factor', 
         "4",
         f"${total_values['firmwide_4f']/1e6:.0f}M",
         f"Adj R²={firmwide_4f['adj_r_squared']:.2%}, includes insig. China50"],
        ['Firm-Wide 3-Factor', 
         "3",
         f"${total_values['firmwide_3f']/1e6:.0f}M",
         f"Adj R²={firmwide_3f['adj_r_squared']:.2%}, RECOMMENDED"],
        ['Firm-Wide 2-Factor (SP500+NIKKEI)', 
         "2",
         f"${total_values['firmwide_2f']/1e6:.0f}M",
         f"Adj R²={firmwide_2f['adj_r_squared']:.2%}, Simplest"],
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Strategy', '# Futures', 'Contract Value', 'Notes'],
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight recommended strategy (3-Factor)
    for j in range(4):
        table[(4, j)].set_facecolor('#C6EFCE')
    
    # Highlight 2-Factor row (light blue for comparison)
    for j in range(4):
        table[(5, j)].set_facecolor('#BDD7EE')
    
    plt.title('Strategy Comparison: Fewer Futures = Lower Management Cost\n(Green = Recommended, Blue = 2-Factor Comparison)', 
              fontsize=12, fontweight='bold', pad=20)
    
    filepath = OUTPUT_DIR / '09_strategy_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_data, columns=['Strategy', 'Num_Futures', 'Contract_Value', 'Notes'])
    csv_path = OUTPUT_DIR / 'strategy_comparison.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    return summary_df


def plot_firmwide_summary(firmwide_3f, contracts_df):
    """Create summary visualization for firm-wide hedge (3-factor)."""
    print("Creating firm-wide summary visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of betas
    ax1 = axes[0]
    futures = ['SP500', 'FTSE_EM', 'NIKKEI']
    betas = [firmwide_3f[f]['beta'] for f in futures]
    colors = ['#4472C4' if b > 0 else '#ED7D31' for b in betas]
    
    bars = ax1.bar(futures, betas, color=colors)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Hedge Ratio (h*)')
    ax1.set_title(f'Firm-Wide Hedge Ratios (3-Factor)\nAdj R² = {firmwide_3f["adj_r_squared"]:.4f}', fontweight='bold')
    
    for bar, beta in zip(bars, betas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.05,
                f'{beta:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Contract table
    ax2 = axes[1]
    ax2.axis('off')
    
    firmwide_contracts = contracts_df[contracts_df['Strategy'] == 'Firm-Wide-3F'].copy()
    display_df = firmwide_contracts[['Future', 'h*', 'Contracts', 'Contract_Value']].copy()
    display_df['h*'] = display_df['h*'].apply(lambda x: f'{x:.4f}')
    display_df['Contract_Value'] = display_df['Contract_Value'].apply(lambda x: f'${x/1e6:.1f}M')
    
    table = ax2.table(cellText=display_df.values,
                      colLabels=['Future', 'h*', 'Contracts', 'Value'],
                      loc='center',
                      cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax2.set_title(f'Firm-Wide Contracts (3-Factor)\nTotal Portfolio: ${TOTAL_PORTFOLIO/1e6:.0f}M', fontweight='bold')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / '07_firmwide_summary.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_firmwide_comparison(firmwide_4f, firmwide_3f, firmwide_2f, total_values):
    """Create bar chart comparing firm-wide hedging strategies (2F vs 3F vs 4F)."""
    print("Creating firm-wide comparison chart (2F vs 3F vs 4F)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    strategies = ['4-Factor\n(incl. China50)', '3-Factor\n(Recommended)', '2-Factor\n(SP500+NIKKEI)']
    colors = ['#A5A5A5', '#70AD47', '#5B9BD5']  # Gray, Green, Blue
    
    # Adj R² comparison
    ax1 = axes[0]
    adj_r2_values = [
        firmwide_4f['adj_r_squared'] * 100,
        firmwide_3f['adj_r_squared'] * 100,
        firmwide_2f['adj_r_squared'] * 100
    ]
    bars1 = ax1.bar(strategies, adj_r2_values, color=colors, edgecolor='black')
    ax1.set_ylabel('Adjusted R² (%)')
    ax1.set_title('Hedge Effectiveness', fontweight='bold')
    ax1.set_ylim(85, 100)
    for bar, val in zip(bars1, adj_r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Number of Futures
    ax2 = axes[1]
    n_futures = [4, 3, 2]
    bars2 = ax2.bar(strategies, n_futures, color=colors, edgecolor='black')
    ax2.set_ylabel('Number of Futures')
    ax2.set_title('Management Complexity', fontweight='bold')
    ax2.set_ylim(0, 5)
    for bar, val in zip(bars2, n_futures):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Contract Value
    ax3 = axes[2]
    contract_values = [
        total_values['firmwide_4f'] / 1e6,
        total_values['firmwide_3f'] / 1e6,
        total_values['firmwide_2f'] / 1e6
    ]
    bars3 = ax3.bar(strategies, contract_values, color=colors, edgecolor='black')
    ax3.set_ylabel('Total Contract Value ($M)')
    ax3.set_title('Total Contract Value (Notional)', fontweight='bold')
    for bar, val in zip(bars3, contract_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'${val:.0f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    fig.suptitle('Firm-Wide Hedging Strategy Comparison: 2-Factor vs 3-Factor vs 4-Factor', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / '08_firmwide_2factor_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filepath}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("    HEDGING STRATEGY ANALYSIS - COMPREHENSIVE")
    print("    BU623 Derivatives - Wilfrid Laurier University")
    print("=" * 70 + "\n")
    
    # Load data
    df = load_data()
    
    # Calculate returns
    returns_df = calculate_returns(df)
    print(f"Calculated returns for {len(returns_df)} periods\n")
    
    # =========================================================================
    # PART 1: SINGLE-FUTURE HEDGING
    # =========================================================================
    print("=" * 70)
    print("PART 1: SINGLE-FUTURE HEDGING ANALYSIS")
    print("=" * 70)
    
    corr_matrix = plot_correlation_heatmap(returns_df)
    scatter_results = plot_all_scatter_plots(returns_df)
    single_reg_df = create_regression_summary_table(returns_df)
    plot_regression_summary_table(single_reg_df)
    single_contract_df = calculate_optimal_contracts_with_values(single_reg_df, df)
    plot_contract_summary(single_contract_df)
    
    # =========================================================================
    # PART 2: 4-FACTOR ANALYSIS (Identify Significant Futures)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: 4-FACTOR ANALYSIS (FEATURE SELECTION)")
    print("=" * 70)
    
    four_factor_df = run_four_factor_comparison(returns_df)
    create_four_factor_table(four_factor_df)
    
    # =========================================================================
    # PART 3: MULTI-FUTURE HEDGING (Using Significant Futures)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: MULTI-FUTURE HEDGING (SIGNIFICANT FACTORS)")
    print("=" * 70)
    
    multi_results = run_multi_future_hedges_significant(returns_df, four_factor_df)
    
    # =========================================================================
    # PART 4: FIRM-WIDE PORTFOLIO ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: FIRM-WIDE PORTFOLIO ANALYSIS")
    print("=" * 70)
    
    returns_df = calculate_firmwide_return(returns_df)
    firmwide_4f = run_firmwide_regression(returns_df)
    firmwide_3f = run_firmwide_regression_no_china(returns_df)
    firmwide_2f = run_firmwide_regression_sp_nikkei(returns_df)
    
    # =========================================================================
    # PART 5: CONTRACT CALCULATIONS & COMPARISON
    # =========================================================================
    all_contracts_df, total_values = calculate_all_contracts(
        single_reg_df, multi_results, firmwide_4f, firmwide_3f, firmwide_2f, df)
    
    # =========================================================================
    # PART 6: VISUALIZATIONS & SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("CREATING SUMMARY VISUALIZATIONS")
    print("=" * 70)
    
    comparison_df = create_comparison_summary(single_reg_df, multi_results, firmwide_4f, firmwide_3f, firmwide_2f, total_values)
    plot_firmwide_summary(firmwide_3f, all_contracts_df)
    plot_firmwide_comparison(firmwide_4f, firmwide_3f, firmwide_2f, total_values)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. SINGLE-FUTURE HEDGING:")
    print(f"   Avg R²: {np.mean(single_reg_df['R-squared']):.4f}")
    print(f"   Total Contract Value: ${total_values['single']/1e6:.0f}M")
    print("   Pros: Simple, one contract per portfolio")
    print("   Cons: Lower R² for cross-hedges (Europe)")
    
    print("\n2. MULTI-FUTURE HEDGING (Significant Factors):")
    print(f"   Avg Adj R²: {np.mean([r['adj_r_squared'] for r in multi_results]):.4f}")
    print(f"   Total Contract Value: ${total_values['multi']/1e6:.0f}M")
    print("   Pros: Higher R², tailored to each portfolio")
    print("   Cons: More complexity, more margin required")
    
    print("\n3. FIRM-WIDE 3-FACTOR HEDGING (RECOMMENDED):")
    print(f"   Adj R²: {firmwide_3f['adj_r_squared']:.4f}")
    print(f"   Total Contract Value: ${total_values['firmwide_3f']/1e6:.0f}M")
    print("   Contracts: SP500, FTSE_EM, NIKKEI (3 positions)")
    print("   Pros: Diversification, simplicity, lower margin")
    print("   Cons: Less tailored to individual portfolios")
    
    print("\n4. FIRM-WIDE 2-FACTOR HEDGING (SP500 + NIKKEI ONLY):")
    print(f"   Adj R²: {firmwide_2f['adj_r_squared']:.4f}")
    print(f"   Total Contract Value: ${total_values['firmwide_2f']/1e6:.0f}M")
    print("   Contracts: SP500, NIKKEI (2 positions)")
    print("   Pros: Simplest management, lowest complexity")
    print("   Cons: Lower R² than 3-factor, excludes EM exposure")
    
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    
    return {
        'df': df, 
        'returns_df': returns_df, 
        'single_reg_df': single_reg_df,
        'four_factor_df': four_factor_df,
        'multi_results': multi_results,
        'firmwide_4f': firmwide_4f,
        'firmwide_3f': firmwide_3f,
        'firmwide_2f': firmwide_2f,
        'all_contracts_df': all_contracts_df,
        'total_values': total_values
    }


if __name__ == "__main__":
    results = main()


