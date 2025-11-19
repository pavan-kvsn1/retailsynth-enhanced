"""
Test Suite for Phase 2.4: Customer Heterogeneity
Tests individual parameter generation and distributions
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.customer_heterogeneity import (
    CustomerHeterogeneityEngine,
    CustomerParameters
)


class TestPhase2_4:
    """Test suite for Phase 2.4: Customer Heterogeneity"""
    
    def __init__(self):
        self.engine = CustomerHeterogeneityEngine(random_seed=42)
        self.n_test_customers = 1000
    
    def test_1_engine_initialization(self):
        """Test 1: Engine initializes correctly"""
        print("\n" + "="*70)
        print("TEST 1: Customer Heterogeneity Engine Initialization")
        print("="*70)
        
        # Test initialization with seed
        engine_seeded = CustomerHeterogeneityEngine(random_seed=42)
        assert engine_seeded.random_seed == 42, "❌ Random seed not set"
        print("✅ Engine initialized with seed")
        
        # Test initialization without seed
        engine_unseeded = CustomerHeterogeneityEngine()
        assert engine_unseeded.random_seed is None, "❌ Unseeded engine has seed"
        print("✅ Engine initialized without seed")
        
        # Check distributions are defined
        assert hasattr(engine_seeded, 'price_sensitivity_dist'), "❌ Missing price sensitivity dist"
        assert hasattr(engine_seeded, 'quality_preference_dist'), "❌ Missing quality preference dist"
        assert hasattr(engine_seeded, 'promo_responsiveness_dist'), "❌ Missing promo responsiveness dist"
        print("✅ All parameter distributions defined")
        
        return True
    
    def test_2_single_customer_generation(self):
        """Test 2: Generate parameters for single customer"""
        print("\n" + "="*70)
        print("TEST 2: Single Customer Parameter Generation")
        print("="*70)
        
        params = self.engine.generate_customer_parameters(customer_id=1)
        
        # Check all parameters exist
        assert params.customer_id == 1, "❌ Customer ID mismatch"
        assert params.price_sensitivity > 0, "❌ Invalid price sensitivity"
        assert params.quality_preference > 0, "❌ Invalid quality preference"
        assert params.promo_responsiveness > 0, "❌ Invalid promo responsiveness"
        print("✅ All parameters generated")
        
        # Check parameter ranges
        assert 0.5 <= params.price_sensitivity <= 2.5, "❌ Price sensitivity out of range"
        assert 0.3 <= params.quality_preference <= 1.5, "❌ Quality preference out of range"
        assert 0.5 <= params.promo_responsiveness <= 2.0, "❌ Promo responsiveness out of range"
        assert 0.3 <= params.display_sensitivity <= 1.2, "❌ Display sensitivity out of range"
        assert 0.3 <= params.advertising_receptivity <= 1.5, "❌ Advertising receptivity out of range"
        assert 0.3 <= params.variety_seeking <= 1.2, "❌ Variety seeking out of range"
        assert 0.2 <= params.brand_loyalty <= 1.5, "❌ Brand loyalty out of range"
        assert 0.3 <= params.store_loyalty <= 1.3, "❌ Store loyalty out of range"
        print("✅ All parameters within valid ranges")
        
        # Print example parameters
        print(f"\n   Example Customer Parameters:")
        print(f"   Price Sensitivity: {params.price_sensitivity:.3f}")
        print(f"   Quality Preference: {params.quality_preference:.3f}")
        print(f"   Promo Responsiveness: {params.promo_responsiveness:.3f}")
        print(f"   Display Sensitivity: {params.display_sensitivity:.3f}")
        print(f"   Advertising Receptivity: {params.advertising_receptivity:.3f}")
        print(f"   Variety Seeking: {params.variety_seeking:.3f}")
        print(f"   Brand Loyalty: {params.brand_loyalty:.3f}")
        print(f"   Store Loyalty: {params.store_loyalty:.3f}")
        
        return True
    
    def test_3_population_generation(self):
        """Test 3: Generate parameters for population"""
        print("\n" + "="*70)
        print("TEST 3: Population Parameter Generation")
        print("="*70)
        
        n_customers = self.n_test_customers
        df = self.engine.generate_population_parameters(n_customers)
        
        # Check DataFrame structure
        assert len(df) == n_customers, f"❌ Expected {n_customers} customers, got {len(df)}"
        print(f"✅ Generated {n_customers:,} customers")
        
        # Check all columns present
        required_cols = [
            'customer_id', 'price_sensitivity', 'quality_preference',
            'promo_responsiveness', 'display_sensitivity', 'advertising_receptivity',
            'variety_seeking', 'brand_loyalty', 'store_loyalty',
            'basket_size_preference', 'impulsivity', 'segment_label'
        ]
        for col in required_cols:
            assert col in df.columns, f"❌ Missing column: {col}"
        print("✅ All required columns present")
        
        # Check no missing values
        assert df.isnull().sum().sum() == 0, "❌ Missing values found"
        print("✅ No missing values")
        
        return True
    
    def test_4_parameter_distributions(self):
        """Test 4: Check parameter distributions are realistic"""
        print("\n" + "="*70)
        print("TEST 4: Parameter Distribution Validation")
        print("="*70)
        
        df = self.engine.generate_population_parameters(self.n_test_customers)
        
        # Check price sensitivity distribution
        ps_mean = df['price_sensitivity'].mean()
        ps_std = df['price_sensitivity'].std()
        print(f"\n   Price Sensitivity: μ={ps_mean:.3f}, σ={ps_std:.3f}")
        assert 0.8 <= ps_mean <= 1.6, f"❌ Price sensitivity mean unusual: {ps_mean:.3f}"
        assert 0.2 <= ps_std <= 0.6, f"❌ Price sensitivity std unusual: {ps_std:.3f}"
        print("   ✅ Price sensitivity distribution realistic")
        
        # Check quality preference distribution
        qp_mean = df['quality_preference'].mean()
        qp_std = df['quality_preference'].std()
        print(f"\n   Quality Preference: μ={qp_mean:.3f}, σ={qp_std:.3f}")
        assert 0.6 <= qp_mean <= 1.2, f"❌ Quality preference mean unusual: {qp_mean:.3f}"
        assert 0.15 <= qp_std <= 0.4, f"❌ Quality preference std unusual: {qp_std:.3f}"
        print("   ✅ Quality preference distribution realistic")
        
        # Check promo responsiveness distribution
        pr_mean = df['promo_responsiveness'].mean()
        pr_std = df['promo_responsiveness'].std()
        print(f"\n   Promo Responsiveness: μ={pr_mean:.3f}, σ={pr_std:.3f}")
        assert 0.9 <= pr_mean <= 1.5, f"❌ Promo responsiveness mean unusual: {pr_mean:.3f}"
        assert 0.2 <= pr_std <= 0.5, f"❌ Promo responsiveness std unusual: {pr_std:.3f}"
        print("   ✅ Promo responsiveness distribution realistic")
        
        return True
    
    def test_5_heterogeneity_verification(self):
        """Test 5: Verify customers are actually heterogeneous (not all the same)"""
        print("\n" + "="*70)
        print("TEST 5: Heterogeneity Verification")
        print("="*70)
        
        df = self.engine.generate_population_parameters(self.n_test_customers)
        
        # Check that we have variation in parameters
        params_to_check = [
            'price_sensitivity', 'quality_preference', 'promo_responsiveness',
            'display_sensitivity', 'advertising_receptivity'
        ]
        
        for param in params_to_check:
            unique_values = df[param].nunique()
            std_dev = df[param].std()
            
            assert unique_values > self.n_test_customers * 0.9, \
                f"❌ Too few unique values for {param}: {unique_values}"
            assert std_dev > 0.1, \
                f"❌ Insufficient variation in {param}: σ={std_dev:.3f}"
            
            print(f"   {param}: {unique_values} unique values, σ={std_dev:.3f}")
        
        print("✅ Customers are heterogeneous (sufficient variation)")
        
        return True
    
    def test_6_reproducibility(self):
        """Test 6: Seeded generation is reproducible"""
        print("\n" + "="*70)
        print("TEST 6: Reproducibility Test")
        print("="*70)
        
        # Generate twice with same seed
        engine1 = CustomerHeterogeneityEngine(random_seed=123)
        df1 = engine1.generate_population_parameters(100)
        
        engine2 = CustomerHeterogeneityEngine(random_seed=123)
        df2 = engine2.generate_population_parameters(100)
        
        # Check they're identical
        for col in df1.columns:
            if col != 'segment_label':  # segment label uses different seed logic
                if df1[col].dtype in [np.float64, np.float32]:
                    assert np.allclose(df1[col], df2[col]), \
                        f"❌ Column {col} not reproducible"
                else:
                    assert (df1[col] == df2[col]).all(), \
                        f"❌ Column {col} not reproducible"
        
        print("✅ Seeded generation is reproducible")
        
        # Check different seeds produce different results
        engine3 = CustomerHeterogeneityEngine(random_seed=456)
        df3 = engine3.generate_population_parameters(100)
        
        assert not np.allclose(df1['price_sensitivity'], df3['price_sensitivity']), \
            "❌ Different seeds produced same results"
        print("✅ Different seeds produce different results")
        
        return True
    
    def test_7_no_extreme_outliers(self):
        """Test 7: Check for unrealistic extreme outliers"""
        print("\n" + "="*70)
        print("TEST 7: Outlier Detection")
        print("="*70)
        
        df = self.engine.generate_population_parameters(self.n_test_customers)
        
        # Define acceptable ranges (should match distribution bounds)
        ranges = {
            'price_sensitivity': (0.5, 2.5),
            'quality_preference': (0.3, 1.5),
            'promo_responsiveness': (0.5, 2.0),
            'display_sensitivity': (0.3, 1.2),
            'advertising_receptivity': (0.3, 1.5),
            'variety_seeking': (0.3, 1.2),
            'brand_loyalty': (0.2, 1.5),
            'store_loyalty': (0.3, 1.3),
            'basket_size_preference': (0.5, 2.0),
            'impulsivity': (0.2, 1.5)
        }
        
        outliers_found = False
        for param, (min_val, max_val) in ranges.items():
            below = (df[param] < min_val).sum()
            above = (df[param] > max_val).sum()
            
            if below > 0 or above > 0:
                print(f"   ⚠️  {param}: {below} below {min_val}, {above} above {max_val}")
                outliers_found = True
            else:
                print(f"   ✅ {param}: all values in range [{min_val}, {max_val}]")
        
        assert not outliers_found, "❌ Outliers found outside acceptable ranges"
        print("\n✅ No extreme outliers detected")
        
        return True
    
    def test_8_correlation_check(self):
        """Test 8: Check parameter correlations (should be mostly independent)"""
        print("\n" + "="*70)
        print("TEST 8: Parameter Independence Check")
        print("="*70)
        
        df = self.engine.generate_population_parameters(self.n_test_customers)
        
        # Calculate correlation matrix
        numeric_cols = [
            'price_sensitivity', 'quality_preference', 'promo_responsiveness',
            'display_sensitivity', 'advertising_receptivity', 'variety_seeking'
        ]
        corr_matrix = df[numeric_cols].corr()
        
        # Check for high correlations (excluding diagonal)
        high_corrs = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.3:
                        high_corrs.append((col1, col2, corr))
                        print(f"   ⚠️  High correlation: {col1} <-> {col2}: {corr:.3f}")
        
        if not high_corrs:
            print("   ✅ All parameters are approximately independent (|r| < 0.3)")
        else:
            print(f"   ℹ️  Found {len(high_corrs)} correlations > 0.3 (acceptable if < 0.5)")
        
        # Fail only if correlation > 0.5 (very high)
        very_high = [corr for _, _, corr in high_corrs if abs(corr) > 0.5]
        assert len(very_high) == 0, "❌ Very high correlations found (|r| > 0.5)"
        
        return True
    
    def test_9_distribution_summary(self):
        """Test 9: Generate and validate distribution summary"""
        print("\n" + "="*70)
        print("TEST 9: Distribution Summary")
        print("="*70)
        
        df = self.engine.generate_population_parameters(self.n_test_customers)
        summary = self.engine.get_distribution_summary(df)
        
        # Check summary structure
        assert isinstance(summary, dict), "❌ Summary should be a dict"
        assert len(summary) > 5, "❌ Summary should have multiple parameters"
        print(f"✅ Summary generated for {len(summary)} parameters")
        
        # Check each parameter has required statistics
        for param, stats in summary.items():
            required_stats = ['mean', 'std', 'min', 'max', 'q25', 'q50', 'q75']
            for stat in required_stats:
                assert stat in stats, f"❌ Missing {stat} for {param}"
            
            # Sanity checks
            assert stats['min'] <= stats['q25'] <= stats['q50'] <= stats['q75'] <= stats['max'], \
                f"❌ Quantiles out of order for {param}"
            assert stats['std'] > 0, f"❌ Zero standard deviation for {param}"
            
            print(f"\n   {param}:")
            print(f"      Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"      Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"      Quartiles: [{stats['q25']:.3f}, {stats['q50']:.3f}, {stats['q75']:.3f}]")
        
        print("\n✅ Distribution summary valid")
        
        return True


def run_all_tests():
    """Run complete Phase 2.4 test suite"""
    print("\n" + "="*70)
    print("PHASE 2.4 TEST SUITE: Customer Heterogeneity")
    print("="*70)
    
    test_suite = TestPhase2_4()
    
    tests = [
        test_suite.test_1_engine_initialization,
        test_suite.test_2_single_customer_generation,
        test_suite.test_3_population_generation,
        test_suite.test_4_parameter_distributions,
        test_suite.test_5_heterogeneity_verification,
        test_suite.test_6_reproducibility,
        test_suite.test_7_no_extreme_outliers,
        test_suite.test_8_correlation_check,
        test_suite.test_9_distribution_summary
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 2.4 heterogeneity engine is working!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
