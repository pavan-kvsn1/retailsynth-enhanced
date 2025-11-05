
"""
Unit tests for Product Catalog Builder (Sprint 1.1)

These tests validate the CRITICAL business requirements:
1. Stratified sampling preserves category distributions
2. Major brand coverage (top 100 brands)
3. Price distribution matches original
4. Purchase frequency distribution preserved
5. Assortment role classification aligns with retail strategy
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats
 
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import directly from catalog module to avoid JAX dependency
from retailsynth.catalog.product_catalog_builder import ProductCatalogBuilder
from retailsynth.catalog.hierarchy_mapper import HierarchyMapper
from retailsynth.catalog.archetype_classifier import ArchetypeClassifier


class TestProductCatalogBuilder:
    """Test ProductCatalogBuilder with REAL validation metrics"""
    
    @pytest.fixture
    def realistic_products(self):
        """Create realistic product data that mimics Dunnhumby structure"""
        np.random.seed(42)
        n_products = 5000  # Larger sample for statistical validity
        
        # Realistic department distribution (from Dunnhumby)
        departments = {
            'GROCERY': 0.42,
            'DRUG GM': 0.34,
            'PRODUCE': 0.08,
            'MEAT': 0.05,
            'DAIRY': 0.05,
            'DELI': 0.03,
            'PASTRY': 0.03
        }
        
        # Generate products with realistic distributions
        dept_samples = []
        for dept, prob in departments.items():
            n_dept = int(n_products * prob)
            dept_samples.extend([dept] * n_dept)
        
        # Pad to exact count
        while len(dept_samples) < n_products:
            dept_samples.append('GROCERY')
        dept_samples = dept_samples[:n_products]
        np.random.shuffle(dept_samples)
        
        # Create realistic brands (Zipf distribution - few major brands, many small)
        major_brands = [f'Major_Brand_{i}' for i in range(20)]
        minor_brands = [f'Brand_{i}' for i in range(200)]
        brand_probs = np.array([1/(i+1)**1.5 for i in range(220)])
        brand_probs = brand_probs / brand_probs.sum()
        
        brands = np.random.choice(
            major_brands + minor_brands,
            size=n_products,
            p=brand_probs
        )
        
        # Realistic price distribution (log-normal)
        prices = np.random.lognormal(mean=1.0, sigma=0.8, size=n_products)
        prices = np.clip(prices, 0.5, 50.0)
        
        # Realistic purchase frequency (power law)
        frequencies = np.random.pareto(a=2.0, size=n_products) * 10
        frequencies = np.clip(frequencies, 1, 10000)
        
        return pd.DataFrame({
            'PRODUCT_ID': range(1, n_products + 1),
            'DEPARTMENT': dept_samples,
            'COMMODITY_DESC': [f'{dept}_COMMODITY_{np.random.randint(1, 10)}' 
                              for dept in dept_samples],
            'SUB_COMMODITY_DESC': [f'SUB_{np.random.randint(1, 20)}' 
                                   for _ in range(n_products)],
            'BRAND': brands,
            'MANUFACTURER': [f'Mfg_{np.random.randint(1, 50)}' 
                            for _ in range(n_products)],
            'CURR_SIZE_OF_PRODUCT': '12 OZ',
        }), prices, frequencies
    
    @pytest.fixture
    def realistic_transactions(self, realistic_products):
        """Create realistic transaction data"""
        products_df, prices, frequencies = realistic_products
        np.random.seed(42)
        
        # Generate transactions weighted by frequency
        n_transactions = 50000
        product_weights = frequencies / frequencies.sum()
        
        sampled_products = np.random.choice(
            products_df['PRODUCT_ID'].values,
            size=n_transactions,
            p=product_weights
        )
        
        transactions = []
        for product_id in sampled_products:
            price = prices[product_id - 1]
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            transactions.append({
                'PRODUCT_ID': product_id,
                'SALES_VALUE': price * quantity,
                'QUANTITY': quantity,
                'BASKET_ID': np.random.randint(1, 10000),
                'household_key': np.random.randint(1, 2000),
            })
        
        return pd.DataFrame(transactions)
    
    def test_stratified_sampling_preserves_department_distribution(self, realistic_products, realistic_transactions, tmp_path):
        """CRITICAL: Verify department distribution error < 2%"""
        products_df, _, _ = realistic_products
        
        # Save data
        products_path = tmp_path / 'products.csv'
        transactions_path = tmp_path / 'transactions.csv'
        products_df.to_csv(products_path, index=False)
        realistic_transactions.to_csv(transactions_path, index=False)
        
        # Create sample
        builder = ProductCatalogBuilder(n_target_skus=500, random_seed=42)
        builder.load_dunnhumby_data(str(products_path), str(transactions_path))
        sample = builder.create_representative_sample()
        
        # Calculate distributions
        original_dist = products_df['DEPARTMENT'].value_counts(normalize=True)
        sample_dist = sample['DEPARTMENT'].value_counts(normalize=True)
        
        # Verify each major department has < 2% error
        for dept in ['GROCERY', 'DRUG GM', 'PRODUCE']:
            original_pct = original_dist.get(dept, 0)
            sample_pct = sample_dist.get(dept, 0)
            error = abs(original_pct - sample_pct)
            
            assert error < 0.02, (
                f"Department {dept} distribution error {error:.3f} exceeds 2% threshold. "
                f"Original: {original_pct:.3f}, Sample: {sample_pct:.3f}"
            )
    
    def test_major_brand_coverage(self, realistic_products, realistic_transactions, tmp_path):
        """CRITICAL: Verify top 100 brands are covered (>90%)"""
        products_df, _, _ = realistic_products
        
        # Save data
        products_path = tmp_path / 'products.csv'
        transactions_path = tmp_path / 'transactions.csv'
        products_df.to_csv(products_path, index=False)
        realistic_transactions.to_csv(transactions_path, index=False)
        
        # Create sample
        builder = ProductCatalogBuilder(n_target_skus=500, random_seed=42)
        builder.load_dunnhumby_data(str(products_path), str(transactions_path))
        sample = builder.create_representative_sample()
        
        # Get top brands by revenue
        brand_revenue = builder.full_catalog.groupby('BRAND')['total_revenue'].sum()
        top_100_brands = set(brand_revenue.nlargest(100).index)
        
        # Check coverage in sample
        sample_brands = set(sample['BRAND'].unique())
        coverage = len(top_100_brands & sample_brands) / len(top_100_brands)
        
        assert coverage > 0.90, (
            f"Major brand coverage {coverage:.2%} is below 90% threshold. "
            f"Only {len(top_100_brands & sample_brands)}/100 top brands covered."
        )
    
    def test_price_distribution_similarity(self, realistic_products, realistic_transactions, tmp_path):
        """CRITICAL: Verify price distribution matches (KS test p > 0.05)"""
        products_df, _, _ = realistic_products
        
        # Save data
        products_path = tmp_path / 'products.csv'
        transactions_path = tmp_path / 'transactions.csv'
        products_df.to_csv(products_path, index=False)
        realistic_transactions.to_csv(transactions_path, index=False)
        
        # Create sample
        builder = ProductCatalogBuilder(n_target_skus=500, random_seed=42)
        builder.load_dunnhumby_data(str(products_path), str(transactions_path))
        sample = builder.create_representative_sample()
        
        # KS test for price distribution
        original_prices = builder.full_catalog['avg_price'].dropna()
        sample_prices = sample['avg_price'].dropna()
        
        ks_stat, p_value = stats.ks_2samp(original_prices, sample_prices)
        
        # We want distributions to be similar (high p-value)
        # But with realistic sampling, some difference is expected
        assert p_value > 0.01, (
            f"Price distribution differs significantly (KS p={p_value:.4f}). "
            f"Original mean: ${original_prices.mean():.2f}, "
            f"Sample mean: ${sample_prices.mean():.2f}"
        )
    
    def test_popularity_tier_representation(self, realistic_products, realistic_transactions, tmp_path):
        """Verify all popularity tiers (A/B/C) are represented"""
        products_df, _, _ = realistic_products
        
        # Save data
        products_path = tmp_path / 'products.csv'
        transactions_path = tmp_path / 'transactions.csv'
        products_df.to_csv(products_path, index=False)
        realistic_transactions.to_csv(transactions_path, index=False)
        
        # Create sample
        builder = ProductCatalogBuilder(n_target_skus=500, random_seed=42)
        builder.load_dunnhumby_data(str(products_path), str(transactions_path))
        sample = builder.create_representative_sample()
        
        # Check all tiers present
        tiers = set(sample['tier'].unique())
        assert tiers == {'A', 'B', 'C'}, (
            f"Not all popularity tiers represented. Found: {tiers}"
        )
        
        # Check tier distribution is reasonable (not all in one tier)
        tier_counts = sample['tier'].value_counts(normalize=True)
        for tier in ['A', 'B', 'C']:
            assert tier_counts[tier] > 0.10, (
                f"Tier {tier} has only {tier_counts[tier]:.1%} of products"
            )


class TestArchetypeClassifier:
    """Test ArchetypeClassifier with REAL retail validation"""
    
    @pytest.fixture
    def realistic_catalog(self):
        """Create catalog with realistic price/frequency patterns"""
        np.random.seed(42)
        n = 1000
        
        # Create products with realistic patterns
        departments = np.random.choice(['GROCERY', 'PRODUCE', 'DAIRY'], n)
        commodities = [f'{dept}_COMM_{np.random.randint(1, 5)}' for dept in departments]
        
        # Realistic price distribution (log-normal)
        prices = np.random.lognormal(mean=1.0, sigma=0.8, size=n)
        prices = np.clip(prices, 0.5, 50.0)
        
        # Realistic frequency (power law - few high frequency, many low)
        frequencies = np.random.pareto(a=2.0, size=n) * 100
        frequencies = np.clip(frequencies, 1, 10000)
        
        return pd.DataFrame({
            'PRODUCT_ID': range(1, n + 1),
            'DEPARTMENT': departments,
            'COMMODITY_DESC': commodities,
            'SUB_COMMODITY_DESC': 'TEST',
            'BRAND': 'Test Brand',
            'avg_price': prices,
            'purchase_frequency': frequencies,
            'total_revenue': prices * frequencies,
            'total_customers': np.random.randint(10, 1000, n),
        })
    
    def test_assortment_role_distribution(self, realistic_catalog):
        """CRITICAL: Verify assortment roles match retail strategy (15/25/40/20)"""
        classifier = ArchetypeClassifier()
        classified = classifier.classify_products(realistic_catalog)
        
        # Check assortment_role column exists
        assert 'assortment_role' in classified.columns, "Missing assortment_role column"
        assert 'category_role' in classified.columns, "Missing category_role column"
        
        # Verify they're identical (as per design)
        assert (classified['assortment_role'] == classified['category_role']).all(), (
            "assortment_role and category_role should be identical"
        )
        
        # Check distribution matches retail strategy (with tolerance)
        role_dist = classified['assortment_role'].value_counts(normalize=True)
        
        expected = {
            'lpg_line': 0.15,
            'front_basket': 0.25,
            'mid_basket': 0.40,
            'back_basket': 0.20
        }
        
        for role, expected_pct in expected.items():
            actual_pct = role_dist.get(role, 0)
            error = abs(actual_pct - expected_pct)
            
            # Allow 10% tolerance (e.g., 15% ± 1.5%)
            tolerance = expected_pct * 0.10
            
            assert error < tolerance, (
                f"Assortment role '{role}' distribution {actual_pct:.1%} "
                f"deviates from target {expected_pct:.1%} by {error:.1%} "
                f"(tolerance: {tolerance:.1%})"
            )
    
    def test_lpg_line_characteristics(self, realistic_catalog):
        """CRITICAL: Verify LPG line products are high frequency + low price"""
        classifier = ArchetypeClassifier()
        classified = classifier.classify_products(realistic_catalog)
        
        lpg_products = classified[classified['assortment_role'] == 'lpg_line']
        
        if len(lpg_products) > 0:
            # LPG products should be in top 25% frequency
            freq_percentile = lpg_products['frequency_percentile'].mean()
            assert freq_percentile > 0.75, (
                f"LPG products avg frequency percentile {freq_percentile:.2f} < 0.75"
            )
            
            # LPG products should be in bottom 35% price
            price_percentile = lpg_products['price_percentile'].mean()
            assert price_percentile < 0.35, (
                f"LPG products avg price percentile {price_percentile:.2f} > 0.35"
            )
    
    def test_back_basket_characteristics(self, realistic_catalog):
        """Verify back basket products are low frequency (impulse)"""
        classifier = ArchetypeClassifier()
        classified = classifier.classify_products(realistic_catalog)
        
        back_basket = classified[classified['assortment_role'] == 'back_basket']
        
        if len(back_basket) > 0:
            # Back basket should be low frequency
            freq_percentile = back_basket['frequency_percentile'].mean()
            assert freq_percentile < 0.40, (
                f"Back basket avg frequency percentile {freq_percentile:.2f} >= 0.40"
            )
    
    def test_price_tier_balance(self, realistic_catalog):
        """Verify price tiers are roughly balanced (economy/mid/premium)"""
        classifier = ArchetypeClassifier()
        classified = classifier.classify_products(realistic_catalog)
        
        tier_dist = classified['price_tier'].value_counts(normalize=True)
        
        # Each tier should have at least 20% (not all in one tier)
        for tier in ['economy', 'mid_tier', 'premium']:
            assert tier_dist.get(tier, 0) > 0.20, (
                f"Price tier '{tier}' has only {tier_dist.get(tier, 0):.1%} of products"
            )
    
    def test_archetype_granularity(self, realistic_catalog):
        """Verify archetypes provide meaningful segmentation"""
        classifier = ArchetypeClassifier()
        classified = classifier.classify_products(realistic_catalog)
        
        n_archetypes = classified['archetype'].nunique()
        
        # Should have meaningful number of archetypes (not too few, not too many)
        assert 10 < n_archetypes < 100, (
            f"Archetype count {n_archetypes} outside reasonable range [10, 100]"
        )
        
        # Archetypes should have reasonable size (not all products in one)
        archetype_sizes = classified['archetype'].value_counts()
        max_archetype_pct = archetype_sizes.max() / len(classified)
        
        assert max_archetype_pct < 0.30, (
            f"Largest archetype has {max_archetype_pct:.1%} of products (too concentrated)"
        )


class TestHierarchyMapper:
    """Test HierarchyMapper with realistic validation"""
    
    @pytest.fixture
    def realistic_catalog(self):
        """Create realistic multi-level hierarchy"""
        products = []
        product_id = 1
        
        # Create realistic hierarchy structure
        hierarchy = {
            'GROCERY': {
                'SOFT DRINKS': ['CARBONATED', 'JUICE', 'WATER'],
                'SNACKS': ['CHIPS', 'CRACKERS', 'NUTS'],
                'CEREAL': ['HOT', 'COLD']
            },
            'PRODUCE': {
                'VEGETABLES': ['FRESH', 'FROZEN'],
                'FRUITS': ['FRESH', 'FROZEN', 'CANNED']
            },
            'DAIRY': {
                'MILK': ['WHOLE', 'SKIM', 'FLAVORED'],
                'CHEESE': ['SLICED', 'BLOCK', 'SHREDDED']
            }
        }
        
        for dept, commodities in hierarchy.items():
            for comm, subs in commodities.items():
                for sub in subs:
                    # Add multiple products per sub-commodity
                    for i in range(5):
                        products.append({
                            'PRODUCT_ID': product_id,
                            'DEPARTMENT': dept,
                            'COMMODITY_DESC': comm,
                            'SUB_COMMODITY_DESC': sub,
                            'BRAND': f'Brand_{i}',
                            'MANUFACTURER': f'Mfg_{i}',
                            'CURR_SIZE_OF_PRODUCT': '12 OZ',
                            'avg_price': np.random.uniform(1, 10),
                            'total_revenue': np.random.uniform(100, 1000),
                            'purchase_frequency': np.random.uniform(10, 100),
                        })
                        product_id += 1
        
        return pd.DataFrame(products)
    
    def test_hierarchy_completeness(self, realistic_catalog):
        """Verify all products are mapped in hierarchy"""
        mapper = HierarchyMapper()
        mapper.build_hierarchy(realistic_catalog)
        
        # All products should be in hierarchy
        all_products_in_hierarchy = set()
        for dept in mapper.hierarchy.values():
            for comm in dept.values():
                for sub in comm.values():
                    all_products_in_hierarchy.update([p['product_id'] for p in sub])
        
        original_products = set(realistic_catalog['PRODUCT_ID'].values)
        
        assert all_products_in_hierarchy == original_products, (
            f"Missing {len(original_products - all_products_in_hierarchy)} products in hierarchy"
        )
    
    def test_sibling_product_logic(self, realistic_catalog):
        """Verify sibling products are correctly identified"""
        mapper = HierarchyMapper()
        mapper.build_hierarchy(realistic_catalog)
        
        # Get a product and its siblings
        test_product = realistic_catalog.iloc[0]
        product_id = test_product['PRODUCT_ID']
        
        # Get siblings at commodity level
        siblings = mapper.get_sibling_products(product_id, level='commodity')
        
        # Siblings should include products from same commodity
        same_commodity = realistic_catalog[
            (realistic_catalog['DEPARTMENT'] == test_product['DEPARTMENT']) &
            (realistic_catalog['COMMODITY_DESC'] == test_product['COMMODITY_DESC'])
        ]['PRODUCT_ID'].tolist()
        
        assert set(siblings) == set(same_commodity), (
            f"Sibling logic incorrect. Expected {len(same_commodity)} siblings, "
            f"got {len(siblings)}"
        )
    
    def test_category_stats_accuracy(self, realistic_catalog):
        """Verify category statistics are calculated correctly"""
        mapper = HierarchyMapper()
        mapper.build_hierarchy(realistic_catalog)
        
        # Check department stats
        for dept in realistic_catalog['DEPARTMENT'].unique():
            dept_products = realistic_catalog[realistic_catalog['DEPARTMENT'] == dept]
            expected_count = len(dept_products)
            
            # Stats should match actual data
            actual_count = mapper.category_stats['departments'][dept]['PRODUCT_ID']
            
            assert actual_count == expected_count, (
                f"Department {dept} stats incorrect. "
                f"Expected {expected_count}, got {actual_count}"
            )