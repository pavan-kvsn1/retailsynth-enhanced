"""
Test Suite for Phase 2.3: Marketing Signal Integration
Tests marketing signal calculation and visit probability boost
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.promotional_engine import StorePromoContext
from retailsynth.engines.marketing_signal import MarketingSignalCalculator


def create_mock_promo_context(n_promotions=20, scenario='moderate'):
    """
    Create mock promotional context for testing
    
    Scenarios:
    - 'weak': Low promotional intensity
    - 'moderate': Medium promotional intensity
    - 'strong': High promotional intensity
    """
    context = StorePromoContext(
        store_id=1,
        week_number=1,
        promoted_products=list(range(1, n_promotions + 1))
    )
    
    if scenario == 'weak':
        # Weak promotions: Small discounts, mostly shelf tags, no ads
        context.promo_depths = {i: np.random.uniform(0.05, 0.15) for i in context.promoted_products}
        context.display_types = {i: 'shelf_tag' if i <= 15 else 'none' for i in context.promoted_products}
        context.end_cap_products = [i for i in context.promoted_products if i <= 3]
        context.feature_display_products = []
        context.in_ad_products = []
        context.mailer_products = []
        
    elif scenario == 'moderate':
        # Moderate promotions: Medium discounts, some end caps, some ads
        context.promo_depths = {i: np.random.uniform(0.15, 0.30) for i in context.promoted_products}
        context.display_types = {i: 'end_cap' if i <= 5 else ('shelf_tag' if i <= 15 else 'none') 
                               for i in context.promoted_products}
        context.end_cap_products = [i for i in context.promoted_products if i <= 5]
        context.feature_display_products = [i for i in context.promoted_products if i <= 2]
        context.in_ad_products = [i for i in context.promoted_products if i <= 8]
        context.mailer_products = [i for i in context.promoted_products if i <= 4]
        
    elif scenario == 'strong':
        # Strong promotions: Deep discounts, many end caps/features, heavy advertising
        context.promo_depths = {i: np.random.uniform(0.30, 0.50) for i in context.promoted_products}
        context.display_types = {i: 'feature_display' if i <= 3 else ('end_cap' if i <= 10 else 'shelf_tag')
                               for i in context.promoted_products}
        context.end_cap_products = [i for i in context.promoted_products if 4 <= i <= 10]
        context.feature_display_products = [i for i in context.promoted_products if i <= 3]
        context.in_ad_products = [i for i in context.promoted_products if i <= 15]
        context.mailer_products = [i for i in context.promoted_products if i <= 10]
    
    # Compute metrics
    context.compute_metrics()
    
    return context


class TestPhase2_3:
    """Test suite for Phase 2.3: Marketing Signal"""
    
    def __init__(self):
        self.calculator = MarketingSignalCalculator()
    
    def test_1_signal_calculator_initialization(self):
        """Test 1: Marketing signal calculator initializes correctly"""
        print("\n" + "="*70)
        print("TEST 1: Marketing Signal Calculator Initialization")
        print("="*70)
        
        # Default configuration
        calc = MarketingSignalCalculator()
        assert calc.weights['discount_depth'] > 0, "❌ Discount weight should be > 0"
        assert calc.weights['display_prominence'] > 0, "❌ Display weight should be > 0"
        assert calc.weights['advertising_reach'] > 0, "❌ Advertising weight should be > 0"
        print("✅ Default weights initialized")
        
        # Custom configuration
        custom_config = {
            'discount_weight': 0.5,
            'display_weight': 0.3,
            'advertising_weight': 0.2
        }
        calc_custom = MarketingSignalCalculator(config=custom_config)
        assert calc_custom.weights['discount_depth'] == 0.5, "❌ Custom discount weight not applied"
        assert calc_custom.weights['display_prominence'] == 0.3, "❌ Custom display weight not applied"
        assert calc_custom.weights['advertising_reach'] == 0.2, "❌ Custom advertising weight not applied"
        print("✅ Custom weights applied correctly")
        
        # Display multipliers
        assert calc.display_multipliers['feature_display'] > calc.display_multipliers['end_cap'], \
            "❌ Feature display should have higher signal than end cap"
        assert calc.display_multipliers['end_cap'] > calc.display_multipliers['shelf_tag'], \
            "❌ End cap should have higher signal than shelf tag"
        print("✅ Display multipliers properly ordered")
        
        return True
    
    def test_2_signal_strength_no_promotions(self):
        """Test 2: Signal is zero when no promotions"""
        print("\n" + "="*70)
        print("TEST 2: Signal Strength with No Promotions")
        print("="*70)
        
        context = StorePromoContext(store_id=1, week_number=1)
        signal = self.calculator.calculate_signal_strength(context)
        
        assert signal == 0.0, "❌ Signal should be 0.0 when no promotions"
        print(f"   Signal strength: {signal:.3f}")
        print("✅ Zero signal for no promotions")
        
        return True
    
    def test_3_signal_strength_scenarios(self):
        """Test 3: Signal strength varies by promotional intensity"""
        print("\n" + "="*70)
        print("TEST 3: Signal Strength Across Scenarios")
        print("="*70)
        
        # Test weak, moderate, and strong scenarios
        weak_context = create_mock_promo_context(20, 'weak')
        moderate_context = create_mock_promo_context(20, 'moderate')
        strong_context = create_mock_promo_context(20, 'strong')
        
        weak_signal = self.calculator.calculate_signal_strength(weak_context)
        moderate_signal = self.calculator.calculate_signal_strength(moderate_context)
        strong_signal = self.calculator.calculate_signal_strength(strong_context)
        
        print(f"   Weak promotional signal: {weak_signal:.3f}")
        print(f"   Moderate promotional signal: {moderate_signal:.3f}")
        print(f"   Strong promotional signal: {strong_signal:.3f}")
        
        # Verify ordering
        assert weak_signal < moderate_signal < strong_signal, \
            "❌ Signal strength should increase with promotional intensity"
        print("✅ Signal strength properly ordered by intensity")
        
        # Verify ranges
        assert 0.0 <= weak_signal <= 1.0, "❌ Weak signal out of range"
        assert 0.0 <= moderate_signal <= 1.0, "❌ Moderate signal out of range"
        assert 0.0 <= strong_signal <= 1.0, "❌ Strong signal out of range"
        print("✅ All signals in valid range [0, 1]")
        
        return True
    
    def test_4_discount_signal_component(self):
        """Test 4: Discount depth component works correctly"""
        print("\n" + "="*70)
        print("TEST 4: Discount Depth Signal Component")
        print("="*70)
        
        # Create contexts with varying discount depths
        shallow_context = create_mock_promo_context(10, 'weak')
        deep_context = create_mock_promo_context(10, 'strong')
        
        shallow_signal = self.calculator._calculate_discount_signal(shallow_context)
        deep_signal = self.calculator._calculate_discount_signal(deep_context)
        
        print(f"   Shallow discount signal: {shallow_signal:.3f} (avg: {shallow_context.avg_discount_depth:.1%})")
        print(f"   Deep discount signal: {deep_signal:.3f} (avg: {deep_context.avg_discount_depth:.1%})")
        
        assert deep_signal > shallow_signal, "❌ Deep discounts should have stronger signal"
        print("✅ Discount signal increases with discount depth")
        
        return True
    
    def test_5_display_signal_component(self):
        """Test 5: Display prominence component works correctly"""
        print("\n" + "="*70)
        print("TEST 5: Display Prominence Signal Component")
        print("="*70)
        
        # Create contexts with varying display types
        weak_displays = create_mock_promo_context(10, 'weak')
        strong_displays = create_mock_promo_context(10, 'strong')
        
        weak_signal = self.calculator._calculate_display_signal(weak_displays)
        strong_signal = self.calculator._calculate_display_signal(strong_displays)
        
        print(f"   Weak display signal: {weak_signal:.3f} ")
        print(f"     End caps: {len(weak_displays.end_cap_products)}, Features: {len(weak_displays.feature_display_products)}")
        print(f"   Strong display signal: {strong_signal:.3f}")
        print(f"     End caps: {len(strong_displays.end_cap_products)}, Features: {len(strong_displays.feature_display_products)}")
        
        assert strong_signal > weak_signal, "❌ Prominent displays should have stronger signal"
        print("✅ Display signal increases with prominence")
        
        return True
    
    def test_6_advertising_signal_component(self):
        """Test 6: Advertising reach component works correctly"""
        print("\n" + "="*70)
        print("TEST 6: Advertising Reach Signal Component")
        print("="*70)
        
        # No advertising
        no_ads = create_mock_promo_context(10, 'weak')
        no_ads_signal = self.calculator._calculate_advertising_signal(no_ads)
        
        # Heavy advertising
        heavy_ads = create_mock_promo_context(10, 'strong')
        heavy_ads_signal = self.calculator._calculate_advertising_signal(heavy_ads)
        
        print(f"   No advertising signal: {no_ads_signal:.3f}")
        print(f"     In-ad: {len(no_ads.in_ad_products)}, Mailer: {len(no_ads.mailer_products)}")
        print(f"   Heavy advertising signal: {heavy_ads_signal:.3f}")
        print(f"     In-ad: {len(heavy_ads.in_ad_products)}, Mailer: {len(heavy_ads.mailer_products)}")
        
        assert heavy_ads_signal > no_ads_signal, "❌ Heavy advertising should have stronger signal"
        print("✅ Advertising signal increases with reach")
        
        return True
    
    def test_7_visit_probability_boost(self):
        """Test 7: Marketing signal boosts visit probability"""
        print("\n" + "="*70)
        print("TEST 7: Visit Probability Boost")
        print("="*70)
        
        base_prob = 0.3
        
        # Test different signal strengths
        weak_boost = self.calculator.calculate_visit_probability_boost(0.2, base_prob)
        moderate_boost = self.calculator.calculate_visit_probability_boost(0.5, base_prob)
        strong_boost = self.calculator.calculate_visit_probability_boost(0.8, base_prob)
        
        print(f"   Base visit probability: {base_prob:.1%}")
        print(f"   Weak signal boost (0.2): {weak_boost:.1%} (+{(weak_boost - base_prob)/base_prob:.1%})")
        print(f"   Moderate signal boost (0.5): {moderate_boost:.1%} (+{(moderate_boost - base_prob)/base_prob:.1%})")
        print(f"   Strong signal boost (0.8): {strong_boost:.1%} (+{(strong_boost - base_prob)/base_prob:.1%})")
        
        # Verify ordering
        assert base_prob < weak_boost < moderate_boost < strong_boost, \
            "❌ Boosted probabilities should increase with signal strength"
        print("✅ Visit probability increases with signal strength")
        
        # Verify reasonable ranges
        assert weak_boost <= base_prob * 1.2, "❌ Weak boost too large"
        assert moderate_boost <= base_prob * 1.35, "❌ Moderate boost too large"
        assert strong_boost <= 0.95, "❌ Strong boost should cap at 0.95"
        print("✅ Boost magnitudes are reasonable")
        
        return True
    
    def test_8_signal_breakdown(self):
        """Test 8: Signal breakdown provides detailed metrics"""
        print("\n" + "="*70)
        print("TEST 8: Signal Breakdown")
        print("="*70)
        
        context = create_mock_promo_context(20, 'moderate')
        breakdown = self.calculator.get_signal_breakdown(context)
        
        print(f"   Store: {breakdown['store_id']}, Week: {breakdown['week_number']}")
        print(f"   Total promotions: {breakdown['n_promotions']}")
        print(f"   Avg discount: {breakdown['avg_discount']:.1%}")
        print(f"   End caps: {breakdown['n_end_caps']}, Features: {breakdown['n_features']}")
        print(f"   In-ad: {breakdown['n_in_ad']}, Mailer: {breakdown['n_mailer']}")
        print(f"   ---")
        print(f"   Discount signal: {breakdown['discount_signal']:.3f}")
        print(f"   Display signal: {breakdown['display_signal']:.3f}")
        print(f"   Advertising signal: {breakdown['advertising_signal']:.3f}")
        print(f"   Total signal: {breakdown['total_signal']:.3f}")
        
        # Verify all keys present
        required_keys = ['store_id', 'week_number', 'discount_signal', 'display_signal', 
                        'advertising_signal', 'total_signal', 'n_promotions']
        for key in required_keys:
            assert key in breakdown, f"❌ Missing key: {key}"
        print("✅ All breakdown metrics present")
        
        # Verify total is combination of components
        manual_total = (
            self.calculator.weights['discount_depth'] * breakdown['discount_signal'] +
            self.calculator.weights['display_prominence'] * breakdown['display_signal'] +
            self.calculator.weights['advertising_reach'] * breakdown['advertising_signal']
        )
        assert abs(breakdown['total_signal'] - manual_total) < 0.001, \
            "❌ Total signal doesn't match component combination"
        print("✅ Total signal correctly combines components")
        
        return True
    
    def test_9_multi_store_variation(self):
        """Test 9: Different stores can have different signals"""
        print("\n" + "="*70)
        print("TEST 9: Multi-Store Signal Variation")
        print("="*70)
        
        # Simulate 5 stores with varying promotional intensity
        signals = []
        for store_id in range(1, 6):
            scenario = ['weak', 'weak', 'moderate', 'moderate', 'strong'][store_id - 1]
            context = create_mock_promo_context(20, scenario)
            context.store_id = store_id
            signal = self.calculator.calculate_signal_strength(context)
            signals.append(signal)
            print(f"   Store {store_id} ({scenario}): signal = {signal:.3f}")
        
        # Check for variation
        signal_std = np.std(signals)
        assert signal_std > 0.05, "❌ Insufficient variation across stores"
        print(f"   Signal std dev: {signal_std:.3f}")
        print("✅ Signals vary appropriately across stores")
        
        return True


def run_all_tests():
    """Run complete Phase 2.3 test suite"""
    print("\n" + "="*70)
    print("PHASE 2.3 TEST SUITE: Marketing Signal Integration")
    print("="*70)
    
    test_suite = TestPhase2_3()
    
    tests = [
        test_suite.test_1_signal_calculator_initialization,
        test_suite.test_2_signal_strength_no_promotions,
        test_suite.test_3_signal_strength_scenarios,
        test_suite.test_4_discount_signal_component,
        test_suite.test_5_display_signal_component,
        test_suite.test_6_advertising_signal_component,
        test_suite.test_7_visit_probability_boost,
        test_suite.test_8_signal_breakdown,
        test_suite.test_9_multi_store_variation
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
        print("\n🎉 ALL TESTS PASSED! Phase 2.3 is working correctly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
