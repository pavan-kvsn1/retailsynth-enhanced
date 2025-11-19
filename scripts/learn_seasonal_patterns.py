"""
Learn Seasonal Patterns from Dunnhumby Data (Phase 2.7)

This script extracts seasonal demand patterns from historical Dunnhumby transactions
and creates product-specific and category-level seasonal indices.

Output:
- seasonal_patterns.pkl: Learned seasonal patterns for products and categories

Author: RetailSynth Team
Sprint: 2, Phase: 2.7
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from collections import defaultdict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.engines.seasonality_learning import LearnedSeasonalityEngine, SeasonalPattern

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_dunnhumby_data(transactions_path: str, 
                        products_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Dunnhumby transaction and product data
    
    Args:
        transactions_path: Path to transaction_data.csv
        products_path: Path to product.csv
    
    Returns:
        Tuple of (transactions_df, products_df)
    """
    logger.info("Loading Dunnhumby data...")
    
    # Load transactions
    transactions = pd.read_csv(transactions_path)
    logger.info(f"  • Loaded {len(transactions):,} transactions")
    
    # Load products
    products = pd.read_csv(products_path)
    logger.info(f"  • Loaded {len(products):,} products")
    
    # Convert date to datetime
    if 'DAY' in transactions.columns:
        # Dunnhumby uses DAY as integer offset
        # Convert to proper dates (assuming day 1 = 2011-01-01 for example)
        reference_date = pd.Timestamp('2011-01-01')
        transactions['transaction_date'] = reference_date + pd.to_timedelta(transactions['DAY'], unit='D')
    
    return transactions, products


def compute_week_of_year(dates: pd.Series) -> pd.Series:
    """
    Compute week of year (1-52) from dates
    
    Args:
        dates: Series of datetime values
    
    Returns:
        Series of week numbers
    """
    # Use ISO week numbers (1-52/53, treat 53 as 52)
    week_of_year = dates.dt.isocalendar().week
    week_of_year = week_of_year.clip(upper=52)
    return week_of_year


def learn_product_seasonal_patterns(transactions: pd.DataFrame,
                                    min_observations: int = 100,
                                    smoothing_window: int = 3) -> dict:
    """
    Learn product-specific seasonal patterns
    
    Args:
        transactions: Transactions DataFrame with columns:
                     - PRODUCT_ID, QUANTITY, transaction_date, week_of_year
        min_observations: Minimum observations to compute pattern
        smoothing_window: Window for moving average smoothing
    
    Returns:
        Dict mapping product_id to SeasonalPattern
    """

    logger.info("\n" + "="*70)
    logger.info("LEARNING PRODUCT-SPECIFIC SEASONAL PATTERNS")
    logger.info("="*70)
    
    product_patterns = {}
    
    # Group by product and week
    logger.info("Aggregating by product and week...")
    grouped = transactions.groupby(['PRODUCT_ID', 'week_of_year'])['QUANTITY'].sum().reset_index()
    
    # Get products with sufficient data
    product_totals = transactions.groupby('PRODUCT_ID')['QUANTITY'].agg(['sum', 'count']).reset_index()
    valid_products = product_totals[product_totals['count'] >= min_observations]['PRODUCT_ID'].values
    
    logger.info(f"  • Total products: {len(product_totals):,}")
    logger.info(f"  • Products with ≥{min_observations} observations: {len(valid_products):,}")
    
    # Learn pattern for each product
    logger.info("Computing seasonal indices...")
    
    for i, product_id in enumerate(valid_products):
        if (i + 1) % 1000 == 0:
            logger.info(f"  • Processed {i+1:,}/{len(valid_products):,} products...")
        
        # Get weekly sales for this product
        product_data = grouped[grouped['PRODUCT_ID'] == product_id].copy()
        
        # Create full 52-week series (fill missing with mean)
        weekly_sales = np.zeros(52)
        for week in range(1, 53):
            week_data = product_data[product_data['week_of_year'] == week]
            if len(week_data) > 0:
                weekly_sales[week - 1] = week_data['QUANTITY'].sum()
            else:
                # Fill with mean of available data
                weekly_sales[week - 1] = product_data['QUANTITY'].mean()
        
        # Compute baseline (overall average)
        baseline = np.mean(weekly_sales)
        
        if baseline < 1e-6:  # Avoid division by zero
            continue
        
        # Compute seasonal indices (multiplicative)
        seasonal_indices = weekly_sales / baseline
        
        # Apply smoothing (moving average)
        if smoothing_window > 1:
            seasonal_indices = pd.Series(seasonal_indices).rolling(
                window=smoothing_window, 
                center=True, 
                min_periods=1
            ).mean().values
        
        # Compute confidence based on data variability and sample size
        n_obs = len(product_data)
        variability = np.std(seasonal_indices)
        confidence = min(1.0, (n_obs / 500) * (1.0 - min(variability / 2.0, 0.5)))
        
        # Store pattern
        product_patterns[int(product_id)] = SeasonalPattern(
            entity_id=int(product_id),
            entity_type='product',
            weekly_indices=seasonal_indices,
            baseline=float(baseline),
            n_observations=int(n_obs),
            confidence=float(confidence)
        )
    
    logger.info(f"\n✅ Learned patterns for {len(product_patterns):,} products")
    
    # Show statistics
    confidences = [p.confidence for p in product_patterns.values()]
    logger.info(f"   Confidence distribution:")
    logger.info(f"   • Mean: {np.mean(confidences):.3f}")
    logger.info(f"   • Median: {np.median(confidences):.3f}")
    logger.info(f"   • High confidence (>0.7): {sum(1 for c in confidences if c > 0.7):,}")
    
    return product_patterns


def learn_category_seasonal_patterns(transactions: pd.DataFrame,
                                     products: pd.DataFrame,
                                     min_observations: int = 500,
                                     smoothing_window: int = 3) -> dict:
    """
    Learn category-level seasonal patterns
    
    Args:
        transactions: Transactions DataFrame
        products: Products DataFrame with PRODUCT_ID and category columns
        min_observations: Minimum observations per category
        smoothing_window: Window for moving average smoothing
    
    Returns:
        Dict mapping category to SeasonalPattern
    """
    logger.info("\n" + "="*70)
    logger.info("LEARNING CATEGORY-LEVEL SEASONAL PATTERNS")
    logger.info("="*70)
    
    category_patterns = {}
    
    # Merge transactions with product categories
    logger.info("Merging transactions with product categories...")
    
    # Try different category column names
    category_col = None
    for col in ['DEPARTMENT', 'COMMODITY_DESC', 'category', 'CATEGORY']:
        if col in products.columns:
            category_col = col
            break
    
    if category_col is None:
        logger.warning("⚠️  No category column found in products")
        return {}
    
    # Merge
    trans_with_cat = transactions.merge(
        products[['PRODUCT_ID', category_col]],
        on='PRODUCT_ID',
        how='left'
    )
    
    # Group by category and week
    logger.info(f"Aggregating by {category_col} and week...")
    grouped = trans_with_cat.groupby([category_col, 'week_of_year'])['QUANTITY'].sum().reset_index()
    
    # Get categories with sufficient data
    category_totals = trans_with_cat.groupby(category_col)['QUANTITY'].agg(['sum', 'count']).reset_index()
    valid_categories = category_totals[category_totals['count'] >= min_observations][category_col].values
    
    logger.info(f"  • Total categories: {len(category_totals)}")
    logger.info(f"  • Categories with ≥{min_observations} observations: {len(valid_categories)}")
    
    # Learn pattern for each category
    logger.info("Computing seasonal indices...")
    
    for category in valid_categories:
        # Get weekly sales for this category
        category_data = grouped[grouped[category_col] == category].copy()
        
        # Create full 52-week series
        weekly_sales = np.zeros(52)
        for week in range(1, 53):
            week_data = category_data[category_data['week_of_year'] == week]
            if len(week_data) > 0:
                weekly_sales[week - 1] = week_data['QUANTITY'].sum()
            else:
                weekly_sales[week - 1] = category_data['QUANTITY'].mean()
        
        # Compute baseline
        baseline = np.mean(weekly_sales)
        
        if baseline < 1e-6:
            continue
        
        # Compute seasonal indices
        seasonal_indices = weekly_sales / baseline
        
        # Apply smoothing
        if smoothing_window > 1:
            seasonal_indices = pd.Series(seasonal_indices).rolling(
                window=smoothing_window,
                center=True,
                min_periods=1
            ).mean().values
        
        # Compute confidence
        n_obs = len(category_data)
        variability = np.std(seasonal_indices)
        confidence = min(1.0, (n_obs / 1000) * (1.0 - min(variability / 2.0, 0.5)))
        
        # Store pattern (use hash of category name as ID)
        category_patterns[str(category)] = SeasonalPattern(
            entity_id=hash(category),
            entity_type='category',
            weekly_indices=seasonal_indices,
            baseline=float(baseline),
            n_observations=int(n_obs),
            confidence=float(confidence)
        )
    
    logger.info(f"\n✅ Learned patterns for {len(category_patterns)} categories")
    
    return category_patterns


def save_patterns(product_patterns: dict, 
                 category_patterns: dict,
                 output_path: str):
    """
    Save learned patterns to file
    
    Args:
        product_patterns: Dict of product patterns
        category_patterns: Dict of category patterns
        output_path: Path to save (.pkl)
    """
    logger.info(f"\n💾 Saving patterns to {output_path}...")
    
    # Create engine and save
    engine = LearnedSeasonalityEngine()
    engine.product_patterns = product_patterns
    engine.category_patterns = category_patterns
    engine.save_patterns(output_path)
    
    logger.info("✅ Patterns saved successfully")


def generate_report(product_patterns: dict,
                   category_patterns: dict,
                   output_path: str):
    """
    Generate seasonality learning report
    
    Args:
        product_patterns: Product patterns
        category_patterns: Category patterns
        output_path: Path to save report
    """
    logger.info("\n📊 Generating seasonality report...")
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("SEASONALITY LEARNING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*70)
    
    # Product patterns summary
    report_lines.append("\n1. PRODUCT-SPECIFIC PATTERNS")
    report_lines.append("-"*70)
    report_lines.append(f"Total products with patterns: {len(product_patterns):,}")
    
    if product_patterns:
        confidences = [p.confidence for p in product_patterns.values()]
        report_lines.append(f"\nConfidence distribution:")
        report_lines.append(f"  Mean: {np.mean(confidences):.3f}")
        report_lines.append(f"  Median: {np.median(confidences):.3f}")
        report_lines.append(f"  High confidence (>0.7): {sum(1 for c in confidences if c > 0.7):,} ({sum(1 for c in confidences if c > 0.7)/len(confidences)*100:.1f}%)")
        
        # Sample patterns
        report_lines.append(f"\nSample patterns (top 5 by confidence):")
        top_patterns = sorted(product_patterns.items(), key=lambda x: x[1].confidence, reverse=True)[:5]
        for prod_id, pattern in top_patterns:
            peak_week = np.argmax(pattern.weekly_indices) + 1
            trough_week = np.argmin(pattern.weekly_indices) + 1
            report_lines.append(f"  Product {prod_id}:")
            report_lines.append(f"    Confidence: {pattern.confidence:.3f}")
            report_lines.append(f"    Peak: Week {peak_week} ({pattern.weekly_indices[peak_week-1]:.2f}x)")
            report_lines.append(f"    Trough: Week {trough_week} ({pattern.weekly_indices[trough_week-1]:.2f}x)")
    
    # Category patterns summary
    report_lines.append("\n2. CATEGORY-LEVEL PATTERNS")
    report_lines.append("-"*70)
    report_lines.append(f"Total categories with patterns: {len(category_patterns)}")
    
    if category_patterns:
        confidences = [p.confidence for p in category_patterns.values()]
        report_lines.append(f"\nConfidence distribution:")
        report_lines.append(f"  Mean: {np.mean(confidences):.3f}")
        report_lines.append(f"  Median: {np.median(confidences):.3f}")
        
        # List all categories
        report_lines.append(f"\nCategories with patterns:")
        for cat_name in sorted(category_patterns.keys()):
            pattern = category_patterns[cat_name]
            peak_week = np.argmax(pattern.weekly_indices) + 1
            report_lines.append(f"  {cat_name}: peak week {peak_week}, confidence {pattern.confidence:.3f}")
    
    # Write report
    report_path = Path(output_path)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✅ Report saved to {report_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Learn seasonal patterns from Dunnhumby data")
    parser.add_argument('--transactions', type=str, 
                       default='data/raw/dunnhumby/transaction_data.csv',
                       help='Path to transaction data CSV')
    parser.add_argument('--products', type=str,
                       default='data/raw/dunnhumby/product.csv',
                       help='Path to products CSV')
    parser.add_argument('--output', type=str,
                       default='data/processed/seasonal_patterns/seasonal_patterns.pkl',
                       help='Output path for patterns')
    parser.add_argument('--min-product-obs', type=int, default=100,
                       help='Minimum observations per product')
    parser.add_argument('--min-category-obs', type=int, default=500,
                       help='Minimum observations per category')
    parser.add_argument('--smoothing-window', type=int, default=3,
                       help='Smoothing window size')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("SEASONALITY LEARNING PIPELINE (Phase 2.7)")
    logger.info("="*70)
    
    # Load data
    transactions, products = load_dunnhumby_data(args.transactions, args.products)
    
    # Add week of year
    logger.info("\nComputing week of year...")
    transactions['week_of_year'] = compute_week_of_year(transactions['transaction_date'])
    
    print(transactions[transactions['PRODUCT_ID'] == 928786])

    # Learn product patterns
    product_patterns = learn_product_seasonal_patterns(
        transactions,
        min_observations=args.min_product_obs,
        smoothing_window=args.smoothing_window
    )
    
    # Learn category patterns
    category_patterns = learn_category_seasonal_patterns(
        transactions,
        products,
        min_observations=args.min_category_obs,
        smoothing_window=args.smoothing_window
    )
    
    # Save patterns
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_patterns(product_patterns, category_patterns, args.output)
    
    # Generate report
    report_path = output_path.parent / 'seasonality_report.txt'
    generate_report(product_patterns, category_patterns, report_path)
    
    logger.info("\n" + "="*70)
    logger.info("✅ SEASONALITY LEARNING COMPLETE")
    logger.info("="*70)
    logger.info(f"\nOutput files:")
    logger.info(f"  • Patterns: {output_path}")
    logger.info(f"  • Report: {report_path}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review seasonality_report.txt for validation")
    logger.info(f"  2. Integrate patterns into transaction generation")
    logger.info(f"  3. Set seasonal_patterns_path in config")


if __name__ == '__main__':
    main()
