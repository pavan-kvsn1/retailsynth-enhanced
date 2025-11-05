"""
Product Catalog Builder for RetailSynth Enhanced.

Creates a representative 20K SKU sample from Dunnhumby's 92K products
using stratified sampling to preserve category structure, brand mix,
and purchase frequency distribution.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

class ProductCatalogBuilder:
    """
    Build representative 20K SKU catalog from Dunnhumby's 92K products.
    
    Sampling Strategy:
    - Tier A: Top 20% of products (80% of volume) → Sample 40% (8,000 SKUs)
    - Tier B: Next 30% (15% of volume) → Sample 35% (7,000 SKUs)
    - Tier C: Remaining 50% (5% of volume) → Sample 25% (5,000 SKUs)
    
    This preserves the Pareto distribution while ensuring long-tail coverage.
    """
    
    def __init__(self, n_target_skus: int = 20000, random_seed: int = 42):
        """
        Initialize ProductCatalogBuilder.
        
        Args:
            n_target_skus: Target number of SKUs in sample (default: 20,000)
            random_seed: Random seed for reproducibility
        """
        self.n_target_skus = n_target_skus
        self.random_seed = random_seed
        self.full_catalog = None
        self.representative_catalog = None
        
        np.random.seed(random_seed)
    
    def load_dunnhumby_data(self, 
                           products_path: str,
                           transactions_path: str) -> pd.DataFrame:
        """
        Load and enrich Dunnhumby product catalog with transaction statistics.
        
        Args:
            products_path: Path to product.csv
            transactions_path: Path to transaction_data.csv
            
        Returns:
            Enriched product DataFrame
        """
        print("Loading Dunnhumby data...")
        
        # Load products
        products_df = pd.read_csv(products_path)
        print(f"  Loaded {len(products_df):,} products")
        
        # Load transactions (sample if too large)
        print("  Loading transactions...")
        transactions_df = pd.read_csv(transactions_path)
        print(f"  Loaded {len(transactions_df):,} transactions")
        
        # Calculate product statistics
        print("  Calculating product statistics...")
        product_stats = transactions_df.groupby('PRODUCT_ID').agg({
            'SALES_VALUE': ['sum', 'mean', 'std'],
            'QUANTITY': ['sum', 'mean'],
            'BASKET_ID': 'nunique',
            'household_key': 'nunique'
        }).reset_index()
        
        # Flatten column names
        product_stats.columns = [
            'PRODUCT_ID', 
            'total_revenue', 'avg_revenue_per_transaction', 'std_revenue',
            'total_quantity', 'avg_quantity_per_transaction',
            'total_baskets', 'total_customers'
        ]
        
        # Merge with products
        products_enriched = products_df.merge(
            product_stats, 
            on='PRODUCT_ID', 
            how='left'
        )
        
        # Fill missing values (products with no transactions)
        products_enriched['total_revenue'] = products_enriched['total_revenue'].fillna(0)
        products_enriched['total_quantity'] = products_enriched['total_quantity'].fillna(0)
        products_enriched['total_baskets'] = products_enriched['total_baskets'].fillna(0)
        products_enriched['total_customers'] = products_enriched['total_customers'].fillna(0)
        
        # Calculate average price
        products_enriched['avg_price'] = (
            products_enriched['total_revenue'] / 
            products_enriched['total_quantity'].replace(0, 1)
        )
        
        # Calculate purchase frequency (baskets per product)
        products_enriched['purchase_frequency'] = products_enriched['total_baskets']
        
        self.full_catalog = products_enriched
        
        print(f"✅ Loaded and enriched {len(products_enriched):,} products")
        return products_enriched
    
    def create_representative_sample(self) -> pd.DataFrame:
        """
        Create 20K SKU sample that preserves:
        1. Category structure (proportional sampling)
        2. Brand mix (major brands + private label)
        3. Price distribution (low/mid/high)
        4. Purchase frequency distribution (movers vs. long-tail)
        
        Returns:
            Representative catalog DataFrame
        """
        if self.full_catalog is None:
            raise ValueError("Must call load_dunnhumby_data() first")
        
        print(f"\nCreating representative {self.n_target_skus:,} SKU sample...")
        
        # Step 1: Classify products by popularity tier
        print("  Step 1: Classifying products by popularity...")
        self._classify_popularity_tiers()
        
        # Step 2: Stratified sampling by department + tier
        print("  Step 2: Stratified sampling by department and tier...")
        sample_skus = self._stratified_sampling()
        
        # Step 3: Ensure major brands are included
        print("  Step 3: Ensuring major brand coverage...")
        sample_skus = self._ensure_major_brands(sample_skus)
        
        # Step 4: Ensure category diversity
        print("  Step 4: Ensuring category coverage...")
        sample_skus = self._ensure_category_coverage(sample_skus)
        
        # Step 5: Remove duplicates and finalize
        representative_catalog = sample_skus.drop_duplicates(
            subset='PRODUCT_ID'
        ).reset_index(drop=True)
        
        # Trim to exact target if needed
        if len(representative_catalog) > self.n_target_skus:
            representative_catalog = representative_catalog.head(self.n_target_skus)
        
        self.representative_catalog = representative_catalog
        
        print(f"✅ Created representative catalog with {len(representative_catalog):,} SKUs")
        return representative_catalog
    
    def _classify_popularity_tiers(self):
        """Classify products into popularity tiers (A/B/C)."""
        # Create tiers using Pareto distribution
        # Tier A: Top 20% of products (80% of volume)
        # Tier B: Next 30% (15% of volume)
        # Tier C: Remaining 50% (5% of volume)
        
        tier_a_cutoff = self.full_catalog['purchase_frequency'].quantile(0.80)
        tier_b_cutoff = self.full_catalog['purchase_frequency'].quantile(0.50)
        
        def assign_tier(freq):
            if freq >= tier_a_cutoff:
                return 'A'
            elif freq >= tier_b_cutoff:
                return 'B'
            else:
                return 'C'
        
        self.full_catalog['tier'] = self.full_catalog['purchase_frequency'].apply(assign_tier)
        
        tier_counts = self.full_catalog['tier'].value_counts()
        print(f"    Tier A: {tier_counts.get('A', 0):,} products")
        print(f"    Tier B: {tier_counts.get('B', 0):,} products")
        print(f"    Tier C: {tier_counts.get('C', 0):,} products")
    
    def _stratified_sampling(self) -> pd.DataFrame:
        """Perform stratified sampling by department and tier."""
        sample_skus = []
        
        # Get all columns that need to be preserved
        columns_to_preserve = [
            'PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC',
            'BRAND', 'MANUFACTURER', 'tier', 'purchase_frequency',
            'total_revenue', 'total_baskets', 'avg_price'
        ]
        
        departments = self.full_catalog['DEPARTMENT'].unique()
        
        for department in tqdm(departments, desc="    Sampling departments"):
            dept_products = self.full_catalog[
                (self.full_catalog['DEPARTMENT'] == department) & 
                (self.full_catalog['COMMODITY_DESC'].notna())  # Ensure COMMODITY_DESC is not null
            ][columns_to_preserve]  # Only select the columns we need
            
            if dept_products.empty:
                continue
                
            # Proportional allocation
            dept_share = len(dept_products) / len(self.full_catalog)
            dept_target = int(self.n_target_skus * dept_share)
            
            if dept_target == 0:
                continue
            
            # Sample by tier within department
            for tier, tier_weight in [('A', 0.4), ('B', 0.35), ('C', 0.25)]:
                tier_products = dept_products[dept_products['tier'] == tier]
                tier_sample_size = int(dept_target * tier_weight)
                
                if len(tier_products) > 0 and tier_sample_size > 0:
                    # Weighted sampling by purchase frequency
                    weights = tier_products['purchase_frequency'].values
                    weights = weights / weights.sum() if weights.sum() > 0 else None
                    
                    sample_size = min(tier_sample_size, len(tier_products))
                    
                    if weights is not None:
                        sample = tier_products.sample(
                            n=sample_size,
                            weights=weights,
                            random_state=self.random_seed,
                            replace=False
                        )
                    else:
                        sample = tier_products.sample(
                            n=sample_size,
                            random_state=self.random_seed,
                            replace=False
                        )
                    
                    sample_skus.append(sample)
        
        return pd.concat(sample_skus, ignore_index=True)
    
    def _ensure_major_brands(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Ensure top brands by revenue are represented."""
        # Get top 100 brands by total revenue
        brand_revenues = self.full_catalog.groupby('BRAND')['total_revenue'].sum()
        top_brands = brand_revenues.nlargest(100).index
        
        # Check coverage
        covered_brands = set(catalog['BRAND'].unique())
        missing_brands = set(top_brands) - covered_brands
        
        print(f"    Top 100 brands: {len(covered_brands)}/100 covered")
        
        # Add representative products for missing major brands
        if missing_brands:
            print(f"    Adding {len(missing_brands)} missing major brands...")
            for brand in missing_brands:
                brand_products = self.full_catalog[self.full_catalog['BRAND'] == brand]
                if len(brand_products) > 0:
                    # Add most popular product from this brand
                    top_product = brand_products.nlargest(1, 'purchase_frequency')
                    catalog = pd.concat([catalog, top_product], ignore_index=True)
        
        return catalog
    
    def _ensure_category_coverage(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Ensure all major categories have minimum representation."""
        min_per_commodity = 3
        
        all_commodities = self.full_catalog['COMMODITY_DESC'].unique()
        
        for commodity in all_commodities:
            commodity_count = (catalog['COMMODITY_DESC'] == commodity).sum()
            
            if commodity_count < min_per_commodity:
                commodity_products = self.full_catalog[
                    self.full_catalog['COMMODITY_DESC'] == commodity
                ]
                
                if len(commodity_products) > 0:
                    additional_needed = min_per_commodity - commodity_count
                    additional = commodity_products.nlargest(
                        min(additional_needed, len(commodity_products)),
                        'purchase_frequency'
                    )
                    catalog = pd.concat([catalog, additional], ignore_index=True)
        
        return catalog
    
    def save_catalog(self, output_dir: str):
        """
        Save processed product catalog and metadata.
        
        Args:
            output_dir: Directory to save catalog files
        """
        if self.representative_catalog is None:
            raise ValueError("Must call create_representative_sample() first")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nSaving product catalog to {output_dir}...")
        
        # Save full catalog as parquet
        catalog_file = output_path / 'product_catalog_20k.parquet'
        self.representative_catalog.to_parquet(catalog_file, index=False)
        print(f"  ✅ Saved catalog: {catalog_file}")
        
        # Save summary statistics
        summary = {
            'n_products': len(self.representative_catalog),
            'n_departments': self.representative_catalog['DEPARTMENT'].nunique(),
            'n_brands': self.representative_catalog['BRAND'].nunique(),
            'n_manufacturers': self.representative_catalog['MANUFACTURER'].nunique(),
            'n_commodities': self.representative_catalog['COMMODITY_DESC'].nunique(),
            'total_revenue': float(self.representative_catalog['total_revenue'].sum()),
            'total_transactions': int(self.representative_catalog['total_baskets'].sum()),
            'avg_price': float(self.representative_catalog['avg_price'].mean()),
        }
        
        summary_file = output_path / 'catalog_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✅ Saved summary: {summary_file}")
        
        print(f"\n{'='*70}")
        print("CATALOG SUMMARY")
        print(f"{'='*70}")
        print(f"  Products: {summary['n_products']:,}")
        print(f"  Departments: {summary['n_departments']}")
        print(f"  Brands: {summary['n_brands']:,}")
        print(f"  Manufacturers: {summary['n_manufacturers']:,}")
        print(f"  Commodities: {summary['n_commodities']}")
        print(f"  Avg Price: ${summary['avg_price']:.2f}")
        print(f"{'='*70}")
    
    def validate_sample(self) -> Dict[str, float]:
        """
        Validate that sample preserves original distributions.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.representative_catalog is None or self.full_catalog is None:
            raise ValueError("Must create sample first")
        
        print("\n" + "="*70)
        print("VALIDATION REPORT")
        print("="*70)
        
        metrics = {}
        
        # 1. Department distribution
        orig_dept_dist = self.full_catalog['DEPARTMENT'].value_counts(normalize=True)
        sample_dept_dist = self.representative_catalog['DEPARTMENT'].value_counts(normalize=True)
        
        dept_error = 0
        print("\nDepartment Distribution Match:")
        for dept in orig_dept_dist.index[:10]:  # Top 10 departments
            orig_pct = orig_dept_dist.get(dept, 0) * 100
            sample_pct = sample_dept_dist.get(dept, 0) * 100
            diff = abs(orig_pct - sample_pct)
            dept_error += diff
            print(f"  {dept[:30]:30s}: {orig_pct:5.1f}% → {sample_pct:5.1f}% (Δ{diff:4.1f}%)")
        
        metrics['dept_distribution_error'] = dept_error / len(orig_dept_dist)
        
        # 2. Price distribution
        from scipy import stats
        orig_prices = self.full_catalog['avg_price'].dropna()
        sample_prices = self.representative_catalog['avg_price'].dropna()
        
        ks_stat, p_value = stats.ks_2samp(orig_prices, sample_prices)
        metrics['price_ks_statistic'] = ks_stat
        metrics['price_ks_pvalue'] = p_value
        
        print(f"\nPrice Distribution:")
        print(f"  Original: mean=${orig_prices.mean():.2f}, std=${orig_prices.std():.2f}")
        print(f"  Sample:   mean=${sample_prices.mean():.2f}, std=${sample_prices.std():.2f}")
        print(f"  KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        
        # 3. Brand coverage
        top_brands = self.full_catalog.groupby('BRAND')['total_revenue'].sum().nlargest(100).index
        covered_brands = set(self.representative_catalog['BRAND'].unique())
        brand_coverage = len(set(top_brands) & covered_brands) / len(top_brands)
        metrics['top_100_brand_coverage'] = brand_coverage
        
        print(f"\nBrand Coverage:")
        print(f"  Top 100 brands covered: {brand_coverage*100:.1f}%")
        
        # 4. Purchase frequency distribution
        orig_freq = self.full_catalog['purchase_frequency'].dropna()
        sample_freq = self.representative_catalog['purchase_frequency'].dropna()
        
        ks_stat_freq, p_value_freq = stats.ks_2samp(orig_freq, sample_freq)
        metrics['frequency_ks_statistic'] = ks_stat_freq
        metrics['frequency_ks_pvalue'] = p_value_freq
        
        print(f"\nPurchase Frequency Distribution:")
        print(f"  KS test: statistic={ks_stat_freq:.4f}, p-value={p_value_freq:.4f}")
        
        # Overall assessment
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        passed = 0
        total = 0
        
        checks = [
            ("Department distribution error < 2%", metrics['dept_distribution_error'] < 2.0),
            ("Price distribution KS p-value > 0.05", metrics['price_ks_pvalue'] > 0.05),
            ("Top 100 brand coverage > 90%", metrics['top_100_brand_coverage'] > 0.90),
            ("Frequency distribution KS p-value > 0.05", metrics['frequency_ks_pvalue'] > 0.05),
        ]
        
        for check_name, check_result in checks:
            total += 1
            if check_result:
                passed += 1
                print(f"  ✅ {check_name}")
            else:
                print(f"  ❌ {check_name}")
        
        print(f"\nValidation Score: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
        print(f"{'='*70}\n")
        
        metrics['validation_score'] = passed / total
        
        return metrics