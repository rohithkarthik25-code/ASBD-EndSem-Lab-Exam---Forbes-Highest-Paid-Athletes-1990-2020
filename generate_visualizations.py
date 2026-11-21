"""
================================================================================
                    FORBES HIGHEST-PAID ATHLETES (1990-2020)
                  PHASE 2: DESCRIPTIVE ANALYTICS & VISUALIZATION
================================================================================
Generates 11 publication-grade visualizations from the cleaned dataset.
Output Directory: hackathon_live/visualizations/
================================================================================
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-resolution styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

CURRENT_DIR = Path(__file__).resolve().parent
DATA_PATH = CURRENT_DIR / "data" / "forbes_athletes_cleaned.csv"
VIS_DIR = CURRENT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded cleaned dataset with {len(df)} records for visualization.")

# ------------------------------------------------------------------------------
# PLOT 1: EARNINGS DISTRIBUTION HISTOGRAM & KDE
# ------------------------------------------------------------------------------
def plot_01_earnings_distribution():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.histplot(df['earnings ($ million)'], kde=True, color='#1f77b4', bins=25,
                 edgecolor='black', alpha=0.65, ax=ax)

    mean_val = df['earnings ($ million)'].mean()
    median_val = df['earnings ($ million)'].median()
    q1_val = df['earnings ($ million)'].quantile(0.25)
    q3_val = df['earnings ($ million)'].quantile(0.75)

    ax.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean: ${mean_val:.1f}M')
    ax.axvline(median_val, color='#2ca02c', linestyle='-', linewidth=2, label=f'Median: ${median_val:.1f}M')
    ax.axvline(q1_val, color='#7f7f7f', linestyle=':', linewidth=1.5, label=f'Q1: ${q1_val:.1f}M')
    ax.axvline(q3_val, color='#7f7f7f', linestyle=':', linewidth=1.5, label=f'Q3: ${q3_val:.1f}M')

    # Annotate skewness
    ax.text(0.68, 0.65, f"Skewness: +{df['earnings ($ million)'].skew():.2f}\nKurtosis: +{df['earnings ($ million)'].kurt():.2f}\nMax: $300.0M (Mayweather)",
            transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    ax.set_title("Plot 1: Distribution of Athlete Earnings (1990–2020)\n[Nominal USD, Unadjusted for Inflation]", fontweight='bold')
    ax.set_xlabel("Earnings ($ Million Nominal USD)")
    ax.set_ylabel("Frequency (Count of Recorded Appearances)")
    ax.legend(loc='upper right', frameon=True)

    out_file = VIS_DIR / "plot_01_earnings_distribution.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 2: EARNINGS BOXPLOT BY SPORT
# ------------------------------------------------------------------------------
def plot_02_earnings_by_sport():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Order sports by median earnings
    sport_order = df.groupby('Sport_Clean')['earnings ($ million)'].median().sort_values(ascending=False).index

    sns.boxplot(data=df, x='Sport_Clean', y='earnings ($ million)', order=sport_order,
                palette='Set2', ax=ax, width=0.6, flierprops=dict(marker='o', markersize=6, markerfacecolor='red', alpha=0.7))

    # Add sample counts below sport labels
    counts = df['Sport_Clean'].value_counts()
    new_labels = [f"{sport}\n(N={counts[sport]})" for sport in sport_order]
    ax.set_xticklabels(new_labels, rotation=25, ha='right')

    ax.set_title("Plot 2: Athlete Earnings Distribution by Sport (1990–2020)\n[Ordered by Median Earnings; Sample Sizes (N) Indicated]", fontweight='bold')
    ax.set_xlabel("Canonical Sport Category")
    ax.set_ylabel("Earnings ($ Million Nominal USD)")

    # Annotate key outliers
    ax.text(0.72, 0.88, "Key Outliers:\n• Mayweather (Boxing): $300M, $285M\n• Pacquiao (Boxing): $160M\n• Messi (Soccer): $127M\n• Woods (Golf): $115M\n*Note: Cycling (N=1) and MMA (N=1) are singletons",
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="orange", alpha=0.9))

    out_file = VIS_DIR / "plot_02_earnings_by_sport.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 3: YEAR-WISE TOTAL & AVERAGE EARNINGS LINE PLOT
# ------------------------------------------------------------------------------
def plot_03_yearwise_total_earnings():
    yearly = df.groupby('Year')['earnings ($ million)'].agg(['sum', 'mean', 'median', 'max']).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # Total earnings bar / line
    color1 = '#1f77b4'
    ax1.plot(yearly['Year'], yearly['sum'], color=color1, marker='o', linewidth=2.5, label='Total Top-10 Earnings ($M)')
    ax1.fill_between(yearly['Year'], yearly['sum'], color=color1, alpha=0.15)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Earnings of Recorded Top-10 ($ Million)", color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Highlight year 2001 missing and year 2018 peak
    ax1.axvspan(2000.5, 2001.5, color='gray', alpha=0.25, label='2001: Missing Forbes Publication')
    ax1.annotate('Peak: $1,059M\n(2018 Mayweather fight)', xy=(2018, 1059.4), xytext=(2012, 1000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight='bold')

    ax1.set_title("Plot 3: Annual Aggregate Earnings of Forbes Top-10 Athletes (1990–2020)\n[Illustrates 30-Year Secular Growth in Elite Sports Compensation]", fontweight='bold')
    ax1.set_xticks(range(1990, 2021, 2))
    ax1.set_xticklabels(range(1990, 2021, 2), rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True)

    out_file = VIS_DIR / "plot_03_yearwise_total_earnings.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 4: YEAR-WISE ATHLETE RECORD COUNT LINE PLOT
# ------------------------------------------------------------------------------
def plot_04_yearwise_record_count():
    all_years = pd.DataFrame({'Year': range(1990, 2021)})
    yearly_counts = df.groupby('Year').size().reset_index(name='Record_Count')
    merged_counts = pd.merge(all_years, yearly_counts, on='Year', how='left').fillna(0)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.plot(merged_counts['Year'], merged_counts['Record_Count'], color='#2ca02c', marker='s', linewidth=2, markersize=6)

    # Annotate anomalies
    ax.annotate('2001: 0 Records\n(List omitted by Forbes)', xy=(2001, 0), xytext=(1996, 3),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', bbox=dict(boxstyle="round", fc="yellow", alpha=0.5))

    ax.annotate('2002: 11 Records\n(Rank 10 tie: Villeneuve & Gordon)', xy=(2002, 11), xytext=(2004, 11.8),
                arrowprops=dict(facecolor='blue', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))

    ax.set_ylim(-0.5, 13)
    ax.set_yticks(range(0, 14, 2))
    ax.set_xticks(range(1990, 2021, 2))
    ax.set_xticklabels(range(1990, 2021, 2), rotation=45)
    ax.set_title("Plot 4: Number of Recorded Athlete Records Per Year (1990–2020)\n[Reveals List Completeness and Identified Dataset Anomalies]", fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Records in Dataset")
    ax.grid(True, linestyle='--', alpha=0.5)

    out_file = VIS_DIR / "plot_04_yearwise_record_count.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 5: SPORT-WISE RECORD COUNT BAR CHART
# ------------------------------------------------------------------------------
def plot_05_sport_record_count():
    counts = df['Sport_Clean'].value_counts()

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax.bar(counts.index, counts.values, color='#3470a3', edgecolor='black', alpha=0.85)

    # Annotate percentages on top of bars
    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(counts.values) + 12)
    ax.set_title("Plot 5: Frequency Distribution of Top-10 Appearances by Sport (1990–2020)\n[Total Records = 301]", fontweight='bold')
    ax.set_xlabel("Canonical Sport Category")
    ax.set_ylabel("Total Number of Recorded Appearances")
    plt.xticks(rotation=30, ha='right')

    out_file = VIS_DIR / "plot_05_sport_record_count.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 6: NATIONALITY DISTRIBUTION BAR CHART (TOP + OTHER)
# ------------------------------------------------------------------------------
def plot_06_nationality_distribution():
    nat_counts = df['Nationality_Clean'].value_counts()
    top_nats = nat_counts.head(7)
    other_count = nat_counts.iloc[7:].sum()

    combined_nats = pd.concat([top_nats, pd.Series({'Other (14 Nations)': other_count})])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#1f77b4' if k == 'USA' else ('#ff7f0e' if k != 'Other (14 Nations)' else '#7f7f7f') for k in combined_nats.index]
    bars = ax.bar(combined_nats.index, combined_nats.values, color=colors, edgecolor='black', alpha=0.85)

    total = len(df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 3, f"{h} ({pct:.1f}%)", ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(combined_nats.values) + 25)
    ax.set_title("Plot 6: Athlete Nationality Distribution (Top Nationalities + Grouped Other)\n[Highlights Marked Geographical Concentration in USA]", fontweight='bold')
    ax.set_xlabel("Athlete Nationality (Standardized Country)")
    ax.set_ylabel("Recorded Appearances Count")
    plt.xticks(rotation=25, ha='right')

    out_file = VIS_DIR / "plot_06_nationality_distribution.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 7: CURRENT RANK VS EARNINGS SCATTERPLOT
# ------------------------------------------------------------------------------
def plot_07_rank_vs_earnings():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Scatter colored by decade
    palette = {'1990s': '#1f77b4', '2000s': '#2ca02c', '2010s': '#d62728', '2020s': '#9467bd'}
    sns.scatterplot(data=df, x='Current Rank', y='earnings ($ million)', hue='Decade',
                    palette=palette, s=75, alpha=0.75, edgecolor='black', ax=ax)

    # Plot mean per rank trendline
    mean_by_rank = df.groupby('Current Rank')['earnings ($ million)'].mean().reset_index()
    ax.plot(mean_by_rank['Current Rank'], mean_by_rank['earnings ($ million)'],
            color='black', linestyle='--', linewidth=2, marker='D', markersize=6, label='Mean Earnings by Rank')

    ax.set_xticks(range(1, 11))
    ax.set_title("Plot 7: Current Rank vs. Earnings Across Decades\n[Shows Rank #1 Premium and Increasing Spread in Modern Eras]", fontweight='bold')
    ax.set_xlabel("Current Rank on Forbes List (1 = Top Earner, 10 = Lowest Top-10)")
    ax.set_ylabel("Earnings ($ Million Nominal USD)")
    ax.legend(title='Decade', loc='upper right', frameon=True)

    out_file = VIS_DIR / "plot_07_rank_vs_earnings.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 8: EARNINGS DISTRIBUTION BY DECADE BOXPLOT
# ------------------------------------------------------------------------------
def plot_08_earnings_by_decade():
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    decade_order = ['1990s', '2000s', '2010s', '2020s']
    sns.boxplot(data=df, x='Decade', y='earnings ($ million)', order=decade_order,
                palette='Blues', ax=ax, width=0.5, flierprops=dict(marker='o', markersize=6, markerfacecolor='red', alpha=0.7))

    # Add median labels on top of boxes
    medians = df.groupby('Decade')['earnings ($ million)'].median()
    iqrs = df.groupby('Decade')['earnings ($ million)'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    counts = df.groupby('Decade').size()

    for idx, dec in enumerate(decade_order):
        med = medians[dec]
        n_cnt = counts[dec]
        ax.text(idx, med + 4, f"Med: ${med:.1f}M\n(N={n_cnt})", ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    ax.set_title("Plot 8: Athlete Earnings Distribution Across Decades (1990s–2020s)\n[Demonstrates Progressive Wage Inflation and Increasing Dispersion]", fontweight='bold')
    ax.set_xlabel("Decade")
    ax.set_ylabel("Earnings ($ Million Nominal USD)")

    out_file = VIS_DIR / "plot_08_earnings_by_decade.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 9: CORRELATION HEATMAP OF NUMERICAL VARIABLES
# ------------------------------------------------------------------------------
def plot_09_correlation_heatmap():
    # Prepare suitable numerical variables
    num_df = df[['Year', 'Current Rank', 'earnings ($ million)']].copy()

    # Create numeric encoding of ordinal rank tier and decade for correlation analysis
    tier_map = {'Podium (1-3)': 1, 'Mid-Tier (4-7)': 2, 'Lower Top-10 (8-10)': 3}
    num_df['Rank_Tier_Code'] = df['Rank_Tier'].map(tier_map)
    num_df['Decade_Code'] = df['Decade'].map({'1990s': 1, '2000s': 2, '2010s': 3, '2020s': 4})
    num_df['Is_USA'] = (df['Nationality_Clean'] == 'USA').astype(int)

    # Rename for publication display
    corr_df = num_df.rename(columns={
        'Year': 'Year',
        'Current Rank': 'Current_Rank',
        'earnings ($ million)': 'Earnings_USD_M',
        'Rank_Tier_Code': 'Rank_Tier_Ordinal',
        'Decade_Code': 'Decade_Ordinal',
        'Is_USA': 'Nationality_Is_USA'
    })

    corr_matrix = corr_df.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=1, linecolor='white', cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)

    ax.set_title("Plot 9: Correlation Matrix Heatmap of Numerical & Ordinal Attributes\n[Evaluates Linear Associations Across Features]", fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    out_file = VIS_DIR / "plot_09_correlation_heatmap.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 10: TOP ATHLETES BY CUMULATIVE EARNINGS ACROSS APPEARANCES
# ------------------------------------------------------------------------------
def plot_10_top_athletes_cumulative_earnings():
    # Calculate cumulative recorded earnings
    top_athletes = df.groupby('Name').agg(
        Total_Earnings=('earnings ($ million)', 'sum'),
        Appearances=('Year', 'count'),
        Primary_Sport=('Sport_Clean', lambda x: x.mode()[0])
    ).sort_values(by='Total_Earnings', ascending=False).head(12)

    top_athletes = top_athletes.sort_values(by='Total_Earnings', ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    bars = ax.barh(top_athletes.index, top_athletes['Total_Earnings'], color='#2b5c8f', edgecolor='black', alpha=0.85)

    # Add text labels on the bars
    for bar, (_, row) in zip(bars, top_athletes.iterrows()):
        w = bar.get_width()
        ax.text(w + 15, bar.get_y() + bar.get_height()/2.,
                f"${w:.1f}M ({row['Appearances']} apps, {row['Primary_Sport']})",
                ha='left', va='center', fontsize=9)

    ax.set_xlim(0, max(top_athletes['Total_Earnings']) + 280)
    ax.set_title("Plot 10: All-Time Leaders in Cumulative Earnings Across Recorded Forbes Appearances\n[*Disclaimer: Reflects Total of Top-10 Annual Listings (1990–2020), Not Lifetime Career Income]", fontweight='bold')
    ax.set_xlabel("Cumulative Recorded Earnings ($ Million Nominal USD)")
    ax.set_ylabel("Athlete Name")

    out_file = VIS_DIR / "plot_10_top_athletes_cumulative_earnings.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# PLOT 11: SPORT COMPOSITION ACROSS DECADES (STACKED CHART)
# ------------------------------------------------------------------------------
def plot_11_sport_composition_by_decade():
    # Cross-tabulate Decade vs Sport_Clean
    ct = pd.crosstab(df['Decade'], df['Sport_Clean'], normalize='index') * 100
    decade_order = ['1990s', '2000s', '2010s', '2020s']
    ct = ct.reindex(decade_order)

    # Consolidate smaller sports into Other for clean stacked representation
    major_sports = ['Basketball', 'Boxing', 'Golf', 'Auto Racing', 'Soccer', 'Tennis', 'American Football']
    ct_major = ct[major_sports].copy()
    ct_major['Other Sports'] = ct[[col for col in ct.columns if col not in major_sports]].sum(axis=1)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    colors = sns.color_palette('tab10', n_colors=8)
    ct_major.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black', width=0.6, alpha=0.9)

    ax.set_title("Plot 11: Sport Composition Dynamics Across Decades (1990s–2020s)\n[100% Stacked Chart Showing Transition from Boxing/Auto Racing to Basketball/Soccer]", fontweight='bold')
    ax.set_xlabel("Decade")
    ax.set_ylabel("Percentage of Annual Top-10 Spots (%)")
    ax.legend(title='Sport Category', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)

    out_file = VIS_DIR / "plot_11_sport_composition_by_decade.png"
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_file.name}")


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING ALL 11 DESCRIPTIVE VISUALIZATIONS")
    print("=" * 60)

    plot_01_earnings_distribution()
    plot_02_earnings_by_sport()
    plot_03_yearwise_total_earnings()
    plot_04_yearwise_record_count()
    plot_05_sport_record_count()
    plot_06_nationality_distribution()
    plot_07_rank_vs_earnings()
    plot_08_earnings_by_decade()
    plot_09_correlation_heatmap()
    plot_10_top_athletes_cumulative_earnings()
    plot_11_sport_composition_by_decade()

    print("\n" + "=" * 60)
    print(f"COMPLETED: All 11 plots saved to: {VIS_DIR}")
    print("=" * 60)
