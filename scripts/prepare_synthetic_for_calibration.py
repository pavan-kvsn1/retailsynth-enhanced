"""
Prepare Synthetic Data for Calibration

Merges transaction_items.csv with transactions.csv to create the format
required for calibration (matching Dunnhumby format).

Synthetic data structure:
- transaction_items.csv: line_number, product_id, quantity, unit_price, line_total, transaction_id
- transactions.csv: transaction_id, customer_id, store_id, transaction_date, etc.

Required calibration format:
- transaction_id
- customer_id
- transaction_date
- line_total
- quantity

Usage:
    python scripts/prepare_synthetic_for_calibration.py \
        --items outputs/synthetic_data_with_elasticity/csvs/transaction_items.csv \
        --transactions outputs/synthetic_data_with_elasticity/csvs/transactions.csv \
        --output data/processed/synthetic_calibration.csv

Author: RetailSynth Team
Date: November 2024
"""

import argparse
import pandas as pd
from pathlib import Path


def prepare_synthetic_data(items_path: str, transactions_path: str, output_path: str):
    """
    Merge transaction items with transactions to create calibration format
    
    Args:
        items_path: Path to transaction_items.csv
        transactions_path: Path to transactions.csv
        output_path: Path to save merged data
    """
    print(f"\n📊 Preparing synthetic data for calibration")
    print(f"   • Items: {items_path}")
    print(f"   • Transactions: {transactions_path}")
    print(f"   • Output: {output_path}")
    
    # Load transaction items
    print(f"\n📂 Loading transaction items...")
    items_df = pd.read_csv(items_path)
    print(f"   ✅ Loaded {len(items_df):,} line items")
    print(f"   📋 Columns: {list(items_df.columns)}")
    
    # Load transactions
    print(f"\n📂 Loading transactions...")
    trans_df = pd.read_csv(transactions_path)
    print(f"   ✅ Loaded {len(trans_df):,} transactions")
    print(f"   📋 Columns: {list(trans_df.columns)}")
    
    # Merge on transaction_id
    print(f"\n🔗 Merging data...")
    merged_df = items_df.merge(
        trans_df[['transaction_id', 'customer_id', 'transaction_date', 'store_id', 'week_number']],
        on='transaction_id',
        how='left'
    )
    print(f"   ✅ Merged to {len(merged_df):,} rows")
    
    # Select and rename columns to match calibration format
    print(f"\n🔄 Formatting columns...")
    calibration_df = merged_df[[
        'transaction_id',
        'customer_id', 
        'transaction_date',
        'line_total',
        'quantity'
    ]].copy()
    
    # Convert transaction_date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(calibration_df['transaction_date']):
        calibration_df['transaction_date'] = pd.to_datetime(calibration_df['transaction_date'])
    
    # Sort by customer and date
    print(f"   📊 Sorting data...")
    calibration_df = calibration_df.sort_values(['customer_id', 'transaction_date', 'transaction_id'])
    
    # Basic statistics
    print(f"\n📊 Processed Data Statistics:")
    print(f"   • Total rows: {len(calibration_df):,}")
    print(f"   • Unique customers: {calibration_df['customer_id'].nunique():,}")
    print(f"   • Unique transactions: {calibration_df['transaction_id'].nunique():,}")
    print(f"   • Date range: {calibration_df['transaction_date'].min()} to {calibration_df['transaction_date'].max()}")
    print(f"   • Avg basket size: {calibration_df.groupby('transaction_id').size().mean():.2f}")
    print(f"   • Avg revenue per transaction: ${calibration_df.groupby('transaction_id')['line_total'].sum().mean():.2f}")
    print(f"   • Avg quantity per line: {calibration_df['quantity'].mean():.2f}")
    
    # Check for missing values
    missing = calibration_df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"   • {col}: {count:,} missing")
    
    # Save processed data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving processed data...")
    if output_path.suffix == '.parquet':
        calibration_df.to_parquet(output_path, index=False)
    else:
        calibration_df.to_csv(output_path, index=False)
    
    print(f"   ✅ Saved to: {output_path}")
    print(f"   📦 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return calibration_df


def main():
    parser = argparse.ArgumentParser(description='Prepare synthetic data for calibration')
    parser.add_argument('--items', type=str,
                       default='outputs/synthetic_data_with_elasticity/csvs/transaction_items.csv',
                       help='Path to transaction_items.csv')
    parser.add_argument('--transactions', type=str,
                       default='outputs/synthetic_data_with_elasticity/csvs/transactions.csv',
                       help='Path to transactions.csv')
    parser.add_argument('--output', type=str,
                       default='data/processed/synthetic_calibration.csv',
                       help='Path to save merged calibration data')
    
    args = parser.parse_args()
    
    # Prepare data
    df = prepare_synthetic_data(args.items, args.transactions, args.output)
    
    print(f"\n✅ Synthetic data preparation complete!")
    print(f"\n🎯 Next steps:")
    print(f"   1. Compare with real data:")
    print(f"      python scripts/calibrate_synth_data.py \\")
    print(f"          --real-data data/processed/dunnhumby_calibration.csv \\")
    print(f"          --synthetic-data {args.output}")


if __name__ == '__main__':
    main()
