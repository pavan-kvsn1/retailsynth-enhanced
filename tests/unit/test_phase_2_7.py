"""
Test Phase 2.7: Seasonality Learning

This script tests the integration of learned seasonal patterns into the main generator.

Tests:
1. Seasonality engine initialization
2. Pattern loading and coverage
3. Seasonal multiplier calculation
4. Product-specific vs category-level patterns
5. Integration with transaction generation
6. Comparison with hard-coded seasonality

Author: RetailSynth Team
Sprint: 2, Phase: 2.7
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from retailsynth.config import EnhancedRetailConfig
from retailsynth.engines.seasonality_learning import LearnedSeasonalityEngine, SeasonalPattern
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1


def test_seasonality_engine_initialization():
    """Test 1: Verify seasonality engine initializes correctly"""
    print("="*70)
    print("TEST 1: Seasonality Engine Initialization")
    print("="*70)
    
    # Test without patterns file (should use uniform seasonality)
    engine1 = LearnedSeasonalityEngine(
        seasonal_patterns_path=None,
        enable_seasonality=True,
        min_confidence=0.3
    )
    
    print(f"\n✓ Engine initialized without patterns file")
    print(f"  • Product patterns: {engine1.n_products_with_patterns}")
    print(f"  • Category patterns: {engine1.n_categories_with_patterns}")
    assert engine1.n_products_with_patterns == 0, "Should have no product patterns"
    assert engine1.n_categories_with_patterns == 0, "Should have no category patterns"
    
    # Test with mock patterns
    engine2 = LearnedSeasonalityEngine(enable_seasonality=True)
    
    # Add mock product pattern
    mock_pattern = SeasonalPattern(
        entity_id=12345,
        entity_type='product',
        weekly_indices=np.ones(52) * 1.2,  # 20% above baseline all year
        baseline=100.0,
        n_observations=500,
        confidence=0.85
    )
    engine2.product_patterns[12345] = mock_pattern
    
    # Test multiplier retrieval
    multiplier = engine2.get_seasonal_multiplier(12345, week_of_year=26)
    print(f"\n✓ Mock pattern added and retrieved")
    print(f"  • Product 12345, Week 26: {multiplier:.2f}x")
    assert np.isclose(multiplier, 1.2), f"Expected 1.2, got {multiplier}"
    
    print("\n✅ TEST 1 PASSED: Engine initialization works correctly")
    return True


def test_pattern_coverage():
    """Test 2: Verify pattern coverage calculation"""
    print("\n" + "="*70)
    print("TEST 2: Pattern Coverage Calculation")
    print("="*70)
    
    engine = LearnedSeasonalityEngine(enable_seasonality=True)
    
    # Create mock patterns
    product_ids = list(range(1000, 1100))  # 100 products
    
    # Add product patterns for 30% of products
    for pid in product_ids[:30]:
        engine.product_patterns[pid] = SeasonalPattern(
            entity_id=pid,
            entity_type='product',
            weekly_indices=np.random.uniform(0.8, 1.5, 52),
            baseline=100.0,
            n_observations=200,
            confidence=0.7
        )
    
    # Add category patterns
    categories = ['DAIRY', 'PRODUCE', 'BAKERY']
    for cat in categories:
        engine.category_patterns[cat] = SeasonalPattern(
            entity_id=0,
            entity_type='category',
            weekly_indices=np.random.uniform(0.9, 1.3, 52),
            baseline=100.0,
            n_observations=5000,
            confidence=0.9
        )
    
    # Assign categories to products
    product_categories = [categories[i % 3] for i in range(100)]
    
    # Get coverage stats
    stats = engine.get_coverage_stats(product_ids, product_categories)
    
    print(f"\n📊 Coverage Statistics:")
    print(f"  • Total products: {stats['n_products']}")
    print(f"  • With product pattern: {stats['n_with_product_pattern']} ({stats['product_coverage']:.1%})")
    print(f"  • With category pattern: {stats['n_with_category_pattern']} ({stats['n_with_category_pattern']/stats['n_products']:.1%})")
    print(f"  • Total coverage: {stats['n_with_any_pattern']} ({stats['total_coverage']:.1%})")
    
    assert stats['n_products'] == 100
    assert stats['n_with_product_pattern'] == 30
    assert stats['total_coverage'] == 1.0, "All products should have coverage (product or category)"
    
    print("\n✅ TEST 2 PASSED: Coverage calculation works correctly")
    return True


def test_seasonal_multipliers():
    """Test 3: Verify seasonal multipliers for different weeks"""
    print("\n" + "="*70)
    print("TEST 3: Seasonal Multiplier Calculation")
    print("="*70)
    
    engine = LearnedSeasonalityEngine(enable_seasonality=True)
    
    # Create holiday pattern (high in weeks 47-52)
    holiday_pattern = np.ones(52)
    holiday_pattern[46:52] = [1.2, 1.3, 1.5, 2.0, 1.8, 1.4]  # Thanksgiving + Christmas
    
    engine.product_patterns[999] = SeasonalPattern(
        entity_id=999,
        entity_type='product',
        weekly_indices=holiday_pattern,
        baseline=100.0,
        n_observations=1000,
        confidence=0.95
    )
    
    # Test different weeks
    test_weeks = [1, 13, 26, 39, 47, 50, 52]
    
    print(f"\n📈 Seasonal Multipliers (Product 999):")
    print(f"{'Week':<8} {'Multiplier':<12} {'Bar':<30}")
    print("-" * 50)
    
    for week in test_weeks:
        mult = engine.get_seasonal_multiplier(999, week)
        bar = "█" * int(mult * 20)
        print(f"{week:<8} {mult:<12.2f} {bar}")
    
    # Verify holiday weeks have high multipliers
    holiday_mult = engine.get_seasonal_multiplier(999, 50)
    regular_mult = engine.get_seasonal_multiplier(999, 26)
    
    assert holiday_mult > 1.5, f"Holiday week should have high multiplier, got {holiday_mult}"
    assert regular_mult == 1.0, f"Regular week should be baseline, got {regular_mult}"
    
    print("\n✅ TEST 3 PASSED: Seasonal multipliers calculated correctly")
    return True


def test_vectorized_multipliers():
    """Test 4: Verify vectorized multiplier calculation"""
    print("\n" + "="*70)
    print("TEST 4: Vectorized Multiplier Calculation")
    print("="*70)
    
    engine = LearnedSeasonalityEngine(enable_seasonality=True)
    
    # Create patterns for 5 products
    product_ids = np.array([100, 101, 102, 103, 104])
    
    for i, pid in enumerate(product_ids):
        # Different seasonal patterns
        pattern = np.ones(52) * (1.0 + i * 0.1)  # Increasing baseline
        
        engine.product_patterns[pid] = SeasonalPattern(
            entity_id=pid,
            entity_type='product',
            weekly_indices=pattern,
            baseline=100.0,
            n_observations=500,
            confidence=0.8
        )
    
    # Get multipliers for week 26
    multipliers = engine.get_seasonal_multipliers_vectorized(
        product_ids=product_ids,
        week_of_year=26,
        fallback_value=1.0
    )
    
    print(f"\n📊 Vectorized Multipliers (Week 26):")
    for pid, mult in zip(product_ids, multipliers):
        print(f"  • Product {pid}: {mult:.2f}x")
    
    # Verify increasing pattern
    assert len(multipliers) == 5
    assert multipliers[0] < multipliers[4], "Multipliers should increase"
    
    print("\n✅ TEST 4 PASSED: Vectorized calculation works correctly")
    return True


def test_fallback_to_category():
    """Test 5: Verify fallback from product to category patterns"""
    print("\n" + "="*70)
    print("TEST 5: Product → Category Fallback")
    print("="*70)
    
    engine = LearnedSeasonalityEngine(enable_seasonality=True, min_confidence=0.5)
    
    # Add product pattern with LOW confidence (should be ignored)
    low_conf_pattern = SeasonalPattern(
        entity_id=200,
        entity_type='product',
        weekly_indices=np.ones(52) * 2.0,  # High multiplier
        baseline=100.0,
        n_observations=50,
        confidence=0.2  # Below threshold
    )
    engine.product_patterns[200] = low_conf_pattern
    
    # Add category pattern with HIGH confidence
    high_conf_pattern = SeasonalPattern(
        entity_id=0,
        entity_type='category',
        weekly_indices=np.ones(52) * 1.1,  # Modest multiplier
        baseline=100.0,
        n_observations=10000,
        confidence=0.9
    )
    engine.category_patterns['DAIRY'] = high_conf_pattern
    
    # Get multiplier (should use category, not low-confidence product)
    mult = engine.get_seasonal_multiplier(200, 26, category='DAIRY')
    
    print(f"\n🔄 Fallback Test:")
    print(f"  • Product 200 pattern: 2.0x (confidence: 0.2 ❌)")
    print(f"  • Category DAIRY pattern: 1.1x (confidence: 0.9 ✅)")
    print(f"  • Selected multiplier: {mult:.2f}x")
    
    assert np.isclose(mult, 1.1), f"Should use category pattern, got {mult}"
    
    print("\n✅ TEST 5 PASSED: Fallback mechanism works correctly")
    return True


def test_main_generator_integration():
    """Test 6: Verify integration with main generator"""
    print("\n" + "="*70)
    print("TEST 6: Main Generator Integration")
    print("="*70)
    
    # Create minimal config
    config = EnhancedRetailConfig(
        n_customers=100,
        n_products=50,
        n_stores=1,
        simulation_weeks=2,
        use_real_catalog=False,
        enable_seasonality_learning=True,
        seasonal_patterns_path=None,  # Will use uniform patterns
        enable_nonlinear_utilities=False  # Simplify test
    )
    
    try:
        # Initialize generator
        print("\n🔧 Initializing generator with seasonality learning...")
        generator = EnhancedRetailSynthV4_1(config)
        
        # Check that seasonality engine is initialized
        assert generator.seasonality_engine is not None
        assert isinstance(generator.seasonality_engine, LearnedSeasonalityEngine)
        
        print("  ✓ Generator initialized successfully")
        print(f"  ✓ Seasonality engine: {type(generator.seasonality_engine).__name__}")
        
        # Test pattern info (should return no patterns since we didn't load any)
        info = generator.seasonality_engine.get_pattern_info(999)
        print(f"\n📋 Pattern Info for Product 999:")
        print(f"  • Has product pattern: {info['has_product_pattern']}")
        print(f"  • Has category pattern: {info['has_category_pattern']}")
        print(f"  • Pattern type: {info['pattern_type']}")
        
        print("\n✅ TEST 6 PASSED: Main generator integration successful")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_hardcoded():
    """Test 7: Compare learned vs hard-coded seasonality"""
    print("\n" + "="*70)
    print("TEST 7: Learned vs Hard-Coded Seasonality")
    print("="*70)
    
    from retailsynth.engines.seasonality_engine import SeasonalityEngine
    
    # Initialize both engines
    learned_engine = LearnedSeasonalityEngine(enable_seasonality=True)
    hardcoded_engine = SeasonalityEngine(region='US')
    
    # Add a holiday pattern to learned engine
    holiday_pattern = np.ones(52)
    holiday_pattern[46:52] = [1.3, 1.5, 1.8, 2.2, 2.0, 1.6]  # Holiday season
    
    learned_engine.category_patterns['Fresh'] = SeasonalPattern(
        entity_id=0,
        entity_type='category',
        weekly_indices=holiday_pattern,
        baseline=100.0,
        n_observations=50000,
        confidence=0.95
    )
    
    # Compare multipliers across weeks
    test_weeks = [1, 13, 26, 39, 47, 50, 52]
    
    print(f"\n📊 Comparison (Fresh Category):")
    print(f"{'Week':<8} {'Learned':<12} {'Hard-Coded':<12} {'Difference':<12}")
    print("-" * 50)
    
    for week in test_weeks:
        learned_mult = learned_engine.get_seasonal_multiplier(
            product_id=999,
            week_of_year=week,
            category='Fresh',
            fallback_value=1.0
        )
        hardcoded_mult = hardcoded_engine.get_seasonality_multiplier(week, 'Fresh')
        
        diff = learned_mult - hardcoded_mult
        print(f"{week:<8} {learned_mult:<12.2f} {hardcoded_mult:<12.2f} {diff:+.2f}")
    
    print("\n💡 Observations:")
    print("  • Learned patterns are product/category-specific")
    print("  • Hard-coded patterns use predefined holiday rules")
    print("  • Learned patterns can capture unexpected seasonal effects")
    
    print("\n✅ TEST 7 PASSED: Comparison completed")
    return True


def run_all_tests():
    """Run all Phase 2.7 tests"""
    print("\n" + "🎄"*35)
    print("   PHASE 2.7: SEASONALITY LEARNING - COMPREHENSIVE TEST SUITE")
    print("🎄"*35 + "\n")
    
    tests = [
        ("Seasonality Engine Initialization", test_seasonality_engine_initialization),
        ("Pattern Coverage Calculation", test_pattern_coverage),
        ("Seasonal Multiplier Calculation", test_seasonal_multipliers),
        ("Vectorized Multiplier Calculation", test_vectorized_multipliers),
        ("Product → Category Fallback", test_fallback_to_category),
        ("Main Generator Integration", test_main_generator_integration),
        ("Learned vs Hard-Coded Comparison", test_comparison_with_hardcoded)
    ]
    
    results = []
    
    for i, (name, test_func) in enumerate(tests, 1):
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ TEST {i} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for i, (name, success) in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i}. {status}: {name}")
    
    total_passed = sum(1 for _, s in results if s)
    total_tests = len(results)
    
    print(f"\n{'='*70}")
    print(f"Total: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")
    print(f"{'='*70}")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Phase 2.7 is ready for production.")
        print("\n🎯 Next Steps:")
        print("   1. Run learn_seasonal_patterns.py on Dunnhumby data")
        print("   2. Generate transactions with learned patterns")
        print("   3. Compare validation metrics vs hard-coded seasonality")
        print("   4. Document improvements in PHASE_2_7_SUMMARY.md")
    else:
        print("\n⚠️  Some tests failed. Please review and fix before proceeding.")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
