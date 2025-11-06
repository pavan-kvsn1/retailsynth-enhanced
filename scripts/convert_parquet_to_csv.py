"""
Convert parquet files to CSV format.

This script reads all parquet files from the synthetic_data_with_elasticity directory
and saves them as CSV files in a csvs subfolder.
"""

import os
from pathlib import Path
import pandas as pd


def convert_parquet_to_csv(input_dir: str, output_subdir: str = "csvs"):
    """
    Convert all parquet files in input_dir to CSV files in output_subdir.
    
    Args:
        input_dir: Directory containing parquet files
        output_subdir: Name of subdirectory to create for CSV files
    """
    input_path = Path(input_dir)
    output_path = input_path / output_subdir
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all parquet files
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to convert")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Convert each parquet file to CSV
    for parquet_file in parquet_files:
        try:
            print(f"Converting {parquet_file.name}...")
            
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            # Create CSV filename
            csv_filename = parquet_file.stem + ".csv"
            csv_path = output_path / csv_filename
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            
            print(f"  ✓ Saved {csv_filename} ({len(df):,} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            print(f"  ✗ Error converting {parquet_file.name}: {e}")
    
    print("-" * 60)
    print(f"Conversion complete! CSV files saved in: {output_path}")


if __name__ == "__main__":
    # Default path relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_directory = project_root / "outputs" / "synthetic_data_with_elasticity"
    
    print("Parquet to CSV Converter")
    print("=" * 60)
    
    convert_parquet_to_csv(str(input_directory))
