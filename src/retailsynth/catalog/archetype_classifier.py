"""
Archetype Classifier for RetailSynth Enhanced.

Classifies products into behavioral archetypes based on:
- Price tier (economy/mid/premium)
- Purchase frequency (occasional/regular/staple)
- Category role (lpg_line/front_basket/mid_basket/back_basket)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

class ArchetypeClassifier:
    """
    Classify products into behavioral archetypes.
    
    Archetypes combine:
    1. Price Tier: economy (0-33%), mid_tier (33-67%), premium (67-100%)
    2. Frequency Tier: occasional, regular, staple
    3. Category Role: lpg_line, front_basket, mid_basket, back_basket
    
    Example archetype: "GROCERY_premium_staple_front_basket"
    """
    
    def __init__(self):
        """Initialize ArchetypeClassifier."""
        self.archetypes = None
        self.archetype_definitions = {}
    
    def classify_products(self, catalog_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify all products into archetypes.
        
        Args:
            catalog_df: Product catalog with price and frequency data
            
        Returns:
            Catalog with archetype columns added
        """
        print("Classifying products into archetypes...")
        
        catalog = catalog_df.copy()
        
        # Step 1: Classify price tier (within category)
        print("  Step 1: Classifying price tiers...")
        catalog = self._classify_price_tier(catalog)
        
        # Step 2: Classify purchase frequency tier
        print("  Step 2: Classifying frequency tiers...")
        catalog = self._classify_frequency_tier(catalog)
        
        # Step 3: Classify category role
        print("  Step 3: Classifying category roles...")
        catalog = self._classify_category_role(catalog)
        
        # Step 4: Create archetype ID
        print("  Step 4: Creating archetype IDs...")
        catalog['archetype'] = (
            catalog['DEPARTMENT'].astype(str) + '_' +
            catalog['price_tier'].astype(str) + '_' +
            catalog['frequency_tier'].astype(str) + '_' +
            catalog['category_role'].astype(str)
        )
        
        self.archetypes = catalog
        
        # Generate archetype definitions
        self._generate_archetype_definitions(catalog)
        
        print(f" Classified {len(catalog):,} products into {catalog['archetype'].nunique()} archetypes")
        
        return catalog
    
    def _classify_price_tier(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Classify products by price tier within their commodity category.
        
        Uses percentiles within commodity to handle different price ranges.
        """
        def assign_price_tier(group):
            # Calculate percentiles within this commodity
            group['price_percentile'] = group['avg_price'].rank(pct=True)
            
            # Classify into tiers
            group['price_tier'] = pd.cut(
                group['price_percentile'],
                bins=[0, 0.33, 0.67, 1.0],
                labels=['economy', 'mid_tier', 'premium'],
                include_lowest=True
            )
            
            return group
        
        # Apply within each commodity - don't use include_groups=False as it drops the grouping column
        catalog = catalog.groupby('COMMODITY_DESC', group_keys=False).apply(assign_price_tier)
        
        # Fill any missing values with 'mid_tier'
        catalog['price_tier'] = catalog['price_tier'].fillna('mid_tier')
        
        tier_counts = catalog['price_tier'].value_counts()
        print(f"    Economy: {tier_counts.get('economy', 0):,}")
        print(f"    Mid-tier: {tier_counts.get('mid_tier', 0):,}")
        print(f"    Premium: {tier_counts.get('premium', 0):,}")
        
        return catalog
    
    def _classify_frequency_tier(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Classify products by purchase frequency.
        
        - Staple: Top 15% (bought very frequently)
        - Regular: Next 35% (bought regularly)
        - Occasional: Bottom 50% (bought infrequently)
        """
        # Calculate frequency percentiles
        catalog['frequency_percentile'] = catalog['purchase_frequency'].rank(pct=True)
        
        # Classify into tiers
        def assign_frequency_tier(percentile):
            if percentile >= 0.85:
                return 'staple'
            elif percentile >= 0.50:
                return 'regular'
            else:
                return 'occasional'
        
        catalog['frequency_tier'] = catalog['frequency_percentile'].apply(assign_frequency_tier)
        
        tier_counts = catalog['frequency_tier'].value_counts()
        print(f"    Staple: {tier_counts.get('staple', 0):,}")
        print(f"    Regular: {tier_counts.get('regular', 0):,}")
        print(f"    Occasional: {tier_counts.get('occasional', 0):,}")
        
        return catalog
    
    def _classify_category_role(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Classify products by their assortment role in retail strategy.
        
        Uses standard retail assortment roles:
        - lpg_line (Low Price Guarantee): High frequency staples, price-sensitive (15%)
        - front_basket: Planned purchases, high frequency (25%)
        - mid_basket: Regular purchases, medium frequency (40%)
        - back_basket: Occasional/impulse purchases, low frequency (20%)
        """
        def assign_role(row):
            price_pct = row['price_percentile']
            freq_pct = row['frequency_percentile']
            
            # LPG Line: High frequency + low price (staples, price-sensitive)
            if freq_pct >= 0.75 and price_pct <= 0.35:
                return 'lpg_line'
            
            # Front Basket: High frequency + any price (planned purchases)
            elif freq_pct >= 0.65:
                return 'front_basket'
            
            # Back Basket: Low frequency (impulse/occasional)
            elif freq_pct < 0.40:
                return 'back_basket'
            
            # Mid Basket: Everything else (regular purchases)
            else:
                return 'mid_basket'
        
        catalog['category_role'] = catalog.apply(assign_role, axis=1)
        
        # Also create assortment_role column (same as category_role for real catalog)
        catalog['assortment_role'] = catalog['category_role']
        
        role_counts = catalog['category_role'].value_counts()
        print(f"    LPG Line: {role_counts.get('lpg_line', 0):,}")
        print(f"    Front Basket: {role_counts.get('front_basket', 0):,}")
        print(f"    Mid Basket: {role_counts.get('mid_basket', 0):,}")
        print(f"    Back Basket: {role_counts.get('back_basket', 0):,}")
        
        return catalog
    
    def _generate_archetype_definitions(self, catalog: pd.DataFrame):
        """Generate summary statistics for each archetype."""
        archetype_stats = catalog.groupby('archetype').agg({
            'PRODUCT_ID': 'count',
            'avg_price': ['mean', 'std'],
            'purchase_frequency': ['mean', 'std'],
            'total_revenue': 'sum',
            'total_customers': 'sum'
        }).reset_index()
        
        # Flatten column names
        archetype_stats.columns = [
            'archetype',
            'n_products',
            'avg_price_mean', 'avg_price_std',
            'purchase_freq_mean', 'purchase_freq_std',
            'total_revenue', 'total_customers'
        ]
        
        self.archetype_definitions = archetype_stats.set_index('archetype').to_dict('index')
    
    def get_archetype_info(self, archetype_id: str) -> Dict:
        """
        Get information about a specific archetype.
        
        Args:
            archetype_id: Archetype identifier
            
        Returns:
            Dictionary with archetype statistics
        """
        return self.archetype_definitions.get(archetype_id, {})
    
    def get_products_by_archetype(self, archetype_id: str) -> List[int]:
        """
        Get all product IDs for a specific archetype.
        
        Args:
            archetype_id: Archetype identifier
            
        Returns:
            List of product IDs
        """
        if self.archetypes is None:
            return []
        
        products = self.archetypes[
            self.archetypes['archetype'] == archetype_id
        ]['PRODUCT_ID'].tolist()
        
        return products
    
    def save_archetypes(self, output_path: str):
        """
        Save archetype classifications and definitions.
        
        Args:
            output_path: Path to save CSV file
        """
        if self.archetypes is None:
            raise ValueError("Must call classify_products() first")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Save archetype summary
        archetype_summary = pd.DataFrame.from_dict(
            self.archetype_definitions,
            orient='index'
        ).reset_index()
        archetype_summary.columns = ['archetype'] + list(archetype_summary.columns[1:])
        
        archetype_summary.to_csv(output_file, index=False)
        
        print(f" Saved archetype definitions to {output_path}")
    
    def print_archetype_summary(self):
        """Print summary of archetypes."""
        if self.archetypes is None:
            print("No archetypes classified yet.")
            return
        
        print("\n" + "="*70)
        print("ARCHETYPE SUMMARY")
        print("="*70)
        
        # Overall stats
        n_archetypes = self.archetypes['archetype'].nunique()
        print(f"\nTotal Archetypes: {n_archetypes}")
        
        # Top 10 archetypes by product count
        top_archetypes = self.archetypes['archetype'].value_counts().head(10)
        
        print("\nTop 10 Archetypes by Product Count:")
        for archetype, count in top_archetypes.items():
            print(f"  {archetype[:50]:50s}: {count:4d} products")
        
        # Distribution by dimensions
        print("\nDistribution by Price Tier:")
        for tier, count in self.archetypes['price_tier'].value_counts().items():
            pct = count / len(self.archetypes) * 100
            print(f"  {tier:10s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nDistribution by Frequency Tier:")
        for tier, count in self.archetypes['frequency_tier'].value_counts().items():
            pct = count / len(self.archetypes) * 100
            print(f"  {tier:10s}: {count:5d} ({pct:5.1f}%)")
        
        print("\nDistribution by Category Role:")
        for role, count in self.archetypes['category_role'].value_counts().items():
            pct = count / len(self.archetypes) * 100
            print(f"  {role:12s}: {count:5d} ({pct:5.1f}%)")
        
        print("="*70 + "\n")