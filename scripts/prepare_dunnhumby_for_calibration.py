"""
Prepare Dunnhumby Data for Calibration

Transforms raw Dunnhumby transaction_data.csv into the format required by
the calibration scripts.

Raw Dunnhumby columns:
- household_key (customer ID)
- BASKET_ID (transaction ID)
- DAY (transaction date)
- PRODUCT_ID
- QUANTITY
- SALES_VALUE (line total)
- STORE_ID
- RETAIL_DISC (discount)
- TRANS_TIME
- WEEK_NO
- COUPON_DISC
- COUPON_MATCH_DISC

Required calibration columns:
- transaction_id
- customer_id
- transaction_date
- line_total
- quantity

Usage:
    python scripts/prepare_dunnhumby_for_calibration.py \
        --input data/raw/dunnhumby/transaction_data.csv \
        --output data/processed/dunnhumby_calibration.csv

Author: RetailSynth Team
Date: November 2024
"""

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def prepare_dunnhumby_data(input_path: str, output_path: str, sample_frac: float = 1.0):
    """
    Transform raw Dunnhumby data into calibration format
    
    Args:
        input_path: Path to raw transaction_data.csv
        output_path: Path to save processed data
        sample_frac: Fraction of data to sample (1.0 = all data)
    """
    print(f"\n📊 Preparing Dunnhumby data for calibration")
    print(f"   • Input: {input_path}")
    print(f"   • Output: {output_path}")
    
    # Load raw data
    print(f"\n📂 Loading raw data...")
    df = pd.read_csv(input_path)
    print(f"   ✅ Loaded {len(df):,} rows")
    
    # Show original columns
    print(f"\n📋 Original columns: {list(df.columns)}")
    
    # Sample if requested
    if sample_frac < 1.0:
        print(f"\n🎲 Sampling {sample_frac*100:.1f}% of data...")
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"   ✅ Sampled to {len(df):,} rows")
    
    # Rename columns to match calibration format
    print(f"\n🔄 Transforming columns...")
    
    # Map Dunnhumby columns to calibration format
    column_mapping = {
        'BASKET_ID': 'transaction_id',
        'household_key': 'customer_id',
        'DAY': 'transaction_date',
        'SALES_VALUE': 'line_total',
        'QUANTITY': 'quantity',
        'PRODUCT_ID': 'product_id',
        'STORE_ID': 'store_id',
        'WEEK_NO': 'week_number'
    }
    
    # Check which columns exist
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing_cols = set(column_mapping.keys()) - set(available_cols.keys())
    
    if missing_cols:
        print(f"   ⚠️  Missing columns: {missing_cols}")
    
    # Rename available columns
    df_processed = df[list(available_cols.keys())].copy()
    df_processed.columns = [available_cols[col] for col in df_processed.columns]
    
    # Convert transaction_date to datetime
    if 'transaction_date' in df_processed.columns:
        print(f"   🗓️  Converting dates...")
        # Dunnhumby DAY is days since start of study
        # Assume study starts on 2008-01-01 (typical for Dunnhumby dataset)
        start_date = datetime(2008, 1, 1)
        df_processed['transaction_date'] = df_processed['transaction_date'].apply(
            lambda x: start_date + timedelta(days=int(x))
        )
    
    # Ensure required columns exist
    required_cols = ['transaction_id', 'customer_id', 'transaction_date', 'line_total', 'quantity']
    missing_required = [col for col in required_cols if col not in df_processed.columns]
    
    if missing_required:
        raise ValueError(f"❌ Missing required columns after transformation: {missing_required}")
    
    # Sort by customer and date
    print(f"   📊 Sorting data...")
    df_processed = df_processed.sort_values(['customer_id', 'transaction_date', 'transaction_id'])
    
    # Basic statistics
    print(f"\n📊 Processed Data Statistics:")
    print(f"   • Total rows: {len(df_processed):,}")
    print(f"   • Unique customers: {df_processed['customer_id'].nunique():,}")
    print(f"   • Unique transactions: {df_processed['transaction_id'].nunique():,}")
    print(f"   • Date range: {df_processed['transaction_date'].min()} to {df_processed['transaction_date'].max()}")
    print(f"   • Avg basket size: {df_processed.groupby('transaction_id').size().mean():.2f}")
    print(f"   • Avg revenue per transaction: ${df_processed.groupby('transaction_id')['line_total'].sum().mean():.2f}")
    print(f"   • Avg quantity per line: {df_processed['quantity'].mean():.2f}")
    
    # Save processed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving processed data...")
    if output_path.suffix == '.parquet':
        df_processed.to_parquet(output_path, index=False)
    else:
        df_processed.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved to: {output_path}")
    print(f"   📦 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return df_processed


def main():
    parser = argparse.ArgumentParser(description='Prepare Dunnhumby data for calibration')
    parser.add_argument('--input', type=str, 
                       default='data/raw/dunnhumby/transaction_data.csv',
                       help='Path to raw transaction_data.csv')
    parser.add_argument('--output', type=str,
                       default='data/processed/dunnhumby_calibration.csv',
                       help='Path to save processed data')
    parser.add_argument('--sample', type=float, default=1.0,
                       help='Fraction of data to sample (0.0-1.0, default: 1.0)')
    
    args = parser.parse_args()
    
    # Prepare data
    df = prepare_dunnhumby_data(args.input, args.output, args.sample)
    
    print(f"\n✅ Data preparation complete!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Run calibration:")
    print(f"      python scripts/tune_parameters_optuna.py \\")
    print(f"          --real-data {args.output} \\")
    print(f"          --tier 1 \\")
    print(f"          --n-trials 50")


if __name__ == '__main__':
    main()
