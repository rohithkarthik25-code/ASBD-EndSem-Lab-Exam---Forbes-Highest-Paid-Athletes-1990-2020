"""
================================================================================
                    FORBES HIGHEST-PAID ATHLETES (1990-2020)
                       DATA CLEANING & PREPROCESSING PIPELINE
================================================================================
Phase 1: Data Preprocessing
- Input:  hackathon_live/data/forbes_athletes.csv (unmodified raw data)
- Output: hackathon_live/data/forbes_athletes_cleaned.csv
================================================================================
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "forbes_athletes.csv"
CLEANED_DATA_PATH = DATA_DIR / "forbes_athletes_cleaned.csv"


# ------------------------------------------------------------------------------
# 1. SPORT NORMALIZATION TAXONOMY
# ------------------------------------------------------------------------------
# Maps 29 raw variations (case sensitivity, synonyms, league abbreviations,
# and sub-disciplines) into 11 canonical, mutually exclusive sport categories.
SPORT_MAPPING = {
    # Basketball (82 rows)
    'basketball': 'Basketball',
    'Basketball': 'Basketball',
    'nba': 'Basketball',
    'NBA': 'Basketball',

    # Boxing (46 rows)
    'boxing': 'Boxing',
    'Boxing': 'Boxing',

    # Golf (44 rows)
    'golf': 'Golf',
    'Golf': 'Golf',

    # Auto Racing & Motorsports (40 rows)
    'auto racing': 'Auto Racing',
    'Auto Racing': 'Auto Racing',
    'Auto racing': 'Auto Racing',
    'auto racing (nascar)': 'Auto Racing',
    'Auto Racing (Nascar)': 'Auto Racing',
    'nascar': 'Auto Racing',
    'NASCAR': 'Auto Racing',
    'f1 racing': 'Auto Racing',
    'F1 racing': 'Auto Racing',
    'f1 motorsports': 'Auto Racing',
    'F1 Motorsports': 'Auto Racing',
    'motorcycle gp': 'Auto Racing',

    # Soccer (33 rows)
    'soccer': 'Soccer',
    'Soccer': 'Soccer',

    # Tennis (23 rows)
    'tennis': 'Tennis',
    'Tennis': 'Tennis',

    # American Football (21 rows)
    'american football': 'American Football',
    'American Football': 'American Football',
    'nfl': 'American Football',
    'NFL': 'American Football',
    'american football / baseball': 'American Football',  # Deion Sanders (primary NFL contract)
    'American Football / Baseball': 'American Football',

    # Baseball (6 rows)
    'baseball': 'Baseball',
    'Baseball': 'Baseball',

    # Hockey (4 rows)
    'hockey': 'Hockey',
    'Hockey': 'Hockey',
    'ice hockey': 'Hockey',
    'Ice Hockey': 'Hockey',

    # Cycling (1 row - Lance Armstrong)
    'cycling': 'Cycling',

    # MMA (1 row - Conor McGregor)
    'mma': 'MMA',
    'MMA': 'MMA'
}


# ------------------------------------------------------------------------------
# 2. NATIONALITY NORMALIZATION
# ------------------------------------------------------------------------------
# Standardizes demonyms to official country names.
NATIONALITY_MAPPING = {
    'Filipino': 'Philippines',           # Standardize demonym to country name (Manny Pacquiao)
    'Dominican': 'Dominican Republic'    # Standardize demonym to country name (Alex Rodriguez 2011)
}


# ------------------------------------------------------------------------------
# 3. PREVIOUS YEAR RANK CLEANING & STATUS PRESERVATION
# ------------------------------------------------------------------------------
def clean_previous_year_rank(val):
    """
    Cleans Previous Year Rank without corrupting categorical and bound statuses.
    Preserves:
      - 'Unranked' for NaN, 'not ranked', 'none', '?', '??'
      - '>10', '>20', '>30', '>40', '>100' for range boundaries
      - Exact numeric strings '1'..'40' for verified prior ranks
    """
    if pd.isna(val):
        return 'Unranked'
    val_str = str(val).strip()
    if val_str.lower() in ['not ranked', 'none', '?', '??']:
        return 'Unranked'
    return val_str


# ------------------------------------------------------------------------------
# 4. DERIVED LOGICAL FEATURES
# ------------------------------------------------------------------------------
def derive_rank_trajectory(row):
    """
    Determines year-over-year rank movement relative to the Top 10 list:
      - 'Improved': Current rank is strictly better (lower number) than prior numeric rank.
      - 'Maintained': Current rank is identical to prior numeric rank.
      - 'Dropped': Current rank is worse (higher number) than prior numeric rank.
      - 'Promoted to Top 10': Prior rank was outside Top 10 (e.g. '>10', '>20', '>40', '>100').
      - 'New Entry': Prior rank was Unranked / first appearance.
    """
    curr = row['Current Rank']
    prev_str = row['Previous_Year_Rank_Clean']

    if prev_str == 'Unranked':
        return 'New Entry'
    elif prev_str.startswith('>'):
        return 'Promoted to Top 10'

    try:
        prev_num = int(prev_str)
        if curr < prev_num:
            return 'Improved'
        elif curr == prev_num:
            return 'Maintained'
        else:
            return 'Dropped'
    except ValueError:
        return 'Unknown'


def derive_decade(year):
    """Categorizes historical era by decade."""
    if year < 2000:
        return '1990s'
    elif year < 2010:
        return '2000s'
    elif year < 2020:
        return '2010s'
    else:
        return '2020s'


def derive_earnings_category(earnings):
    """
    Discretizes continuous earnings based on empirical quartiles:
      - Moderate ($8M-$25M): Bottom quartile (Q1 = $24.0M)
      - High ($25M-$60M): Interquartile range (Median = $39.0M, Q3 = $59.4M)
      - Ultra-High (>$60M): Top quartile megastar peak earnings
    """
    if earnings <= 25.0:
        return 'Moderate ($8M-$25M)'
    elif earnings <= 60.0:
        return 'High ($25M-$60M)'
    else:
        return 'Ultra-High (>$60M)'


def derive_rank_tier(rank):
    """Groups ranks into competitive tiers."""
    if rank <= 3:
        return 'Podium (1-3)'
    elif rank <= 7:
        return 'Mid-Tier (4-7)'
    else:
        return 'Lower Top-10 (8-10)'


# ------------------------------------------------------------------------------
# 5. MAIN PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
def run_preprocessing(input_path=RAW_DATA_PATH, output_path=CLEANED_DATA_PATH):
    print("=" * 75)
    print("EXECUTING PREPROCESSING PIPELINE FOR FORBES ATHLETES DATASET")
    print("=" * 75)

    # A. Ingest Raw Dataset (do not mutate source)
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    raw_df = pd.read_csv(input_path)
    df = raw_df.copy()

    initial_shape = df.shape
    print(f"\n[1] Ingested raw dataset: {initial_shape[0]} rows x {initial_shape[1]} columns")

    # B. Sport Normalization
    df['Sport_Clean'] = df['Sport'].str.strip().map(SPORT_MAPPING)
    unmapped_sports = df[df['Sport_Clean'].isna()]['Sport'].unique()
    if len(unmapped_sports) > 0:
        raise ValueError(f"Found unmapped sports: {unmapped_sports}")
    print(f"[2] Normalized Sport column: 29 raw variants -> {df['Sport_Clean'].nunique()} canonical sports")

    # C. Nationality Normalization
    df['Nationality_Clean'] = df['Nationality'].str.strip().replace(NATIONALITY_MAPPING)
    print(f"[3] Normalized Nationality column: unified demonyms (e.g. Filipino -> Philippines)")

    # D. Previous Year Rank Normalization
    df['Previous_Year_Rank_Clean'] = df['Previous Year Rank'].apply(clean_previous_year_rank)
    print(f"[4] Cleaned Previous Year Rank: preserved unranked/unknown as 'Unranked' (0 lost statuses)")

    # E. Feature Engineering / Derived Attributes
    df['Rank_Trajectory'] = df.apply(derive_rank_trajectory, axis=1)
    df['Decade'] = df['Year'].apply(derive_decade)
    df['Earnings_Category'] = df['earnings ($ million)'].apply(derive_earnings_category)
    df['Nationality_Group'] = df['Nationality_Clean'].apply(lambda x: 'USA' if x == 'USA' else 'International')
    df['Rank_Tier'] = df['Current Rank'].apply(derive_rank_tier)
    print(f"[5] Engineered 5 derived features: Rank_Trajectory, Decade, Earnings_Category, Nationality_Group, Rank_Tier")

    # F. Model Features vs Identifier Specification
    # S.NO, Name, Nationality (raw), Sport (raw), Previous Year Rank (raw) are retained
    # for lineage and traceability, but flagged as metadata rather than direct model features.
    metadata_cols = ['S.NO', 'Name', 'Nationality', 'Sport', 'Previous Year Rank']
    analytical_feature_cols = [
        'Year', 'Decade', 'earnings ($ million)', 'Earnings_Category',
        'Current Rank', 'Rank_Tier', 'Previous_Year_Rank_Clean', 'Rank_Trajectory',
        'Sport_Clean', 'Nationality_Clean', 'Nationality_Group'
    ]

    # Reorder columns logically: Identifiers first, then core attributes, then clean/derived
    final_cols = [
        'S.NO', 'Name', 'Year', 'Decade',
        'Current Rank', 'Rank_Tier', 'Previous_Year_Rank_Clean', 'Rank_Trajectory',
        'earnings ($ million)', 'Earnings_Category',
        'Sport_Clean', 'Nationality_Clean', 'Nationality_Group',
        'Nationality', 'Sport', 'Previous Year Rank'  # Preserved raw reference columns at end
    ]
    df_final = df[final_cols]

    # G. Integrity Verification
    final_shape = df_final.shape
    print(f"\n[6] Final cleaned shape: {final_shape[0]} rows x {final_shape[1]} columns")

    assert final_shape[0] == initial_shape[0], f"Row count mismatch! Initial {initial_shape[0]} vs Final {final_shape[0]}"
    assert df_final['earnings ($ million)'].isna().sum() == 0, "Nulls found in earnings!"
    assert df_final['Year'].isna().sum() == 0, "Nulls found in Year!"
    assert df_final['Current Rank'].isna().sum() == 0, "Nulls found in Current Rank!"
    assert df_final['Sport_Clean'].isna().sum() == 0, "Nulls found in Sport_Clean!"
    assert df_final['Nationality_Clean'].isna().sum() == 0, "Nulls found in Nationality_Clean!"
    assert df_final['Previous_Year_Rank_Clean'].isna().sum() == 0, "Nulls found in Previous_Year_Rank_Clean!"
    print("[+] All integrity assertions passed successfully!")

    # H. Export Cleaned Dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"[7] Successfully saved cleaned dataset to: {output_path} ({os.path.getsize(output_path):,} bytes)")

    return df_final


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    df_cleaned = run_preprocessing()
    print("\n" + "=" * 50)
    print("SAMPLE OF CLEANED DATASET (FIRST 5 ROWS)")
    print("=" * 50)
    print(df_cleaned[['Name', 'Year', 'Decade', 'Current Rank', 'earnings ($ million)', 'Sport_Clean', 'Nationality_Clean', 'Rank_Trajectory']].head())
