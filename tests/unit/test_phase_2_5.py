"""
Test Suite for Phase 2.5: Promotional Response + Arc Elasticity
Tests individual promotional response and elasticity calculations
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.promo_response import (
    PromoResponseCalculator,
    PromoResponse
)
from retailsynth.engines.customer_heterogeneity import CustomerHeterogeneityEngine


class TestPhase2_5:
    """Test suite for Phase 2.5: Promotional Response"""
    
    def __init__(self):
        self.calculator = PromoResponseCalculator()
        self.hetero_engine = CustomerHeterogeneityEngine(random_seed=42)
        
        # Generate test customers
        self.test_customers = self.hetero_engine.generate_population_parameters(100)
    
    def test_1_calculator_initialization(self):
        """Test 1: Calculator initializes correctly"""
        print("\n" + "="*70)
        print("TEST 1: Promo Response Calculator Initialization")
        print("="*70)
        
        # Test default initialization
        calc = PromoResponseCalculator()
        assert hasattr(calc, 'base_price_elasticity'), "❌ Missing base_price_elasticity"
        assert hasattr(calc, 'base_promo_elasticity'), "❌ Missing base_promo_elasticity"
        assert hasattr(calc, 'display_boosts'), "❌ Missing display_boosts"
        assert hasattr(calc, 'advertising_boosts'), "❌ Missing advertising_boosts"
        print("✅ Calculator initialized with default config")
        
        # Test custom config
        custom_config = {
            'base_price_elasticity': -2.5,
            'base_promo_elasticity': -4.0
        }
        calc_custom = PromoResponseCalculator(config=custom_config)
        assert calc_custom.base_price_elasticity == -2.5, "❌ Custom config not applied"
        print("✅ Calculator initialized with custom config")
        
        return True
    
    def test_2_single_promo_response(self):
        """Test 2: Calculate response for single customer"""
        print("\n" + "="*70)
        print("TEST 2: Single Customer Promotional Response")
        print("="*70)
        
        # Get a test customer
        customer = self.test_customers.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        # Test promotion
        base_utility = 5.0
        discount_depth = 0.20  # 20% off
        marketing_signal = 0.6
        
        response = self.calculator.calculate_promo_response(
            customer_params=customer_params,
            base_utility=base_utility,
            discount_depth=discount_depth,
            marketing_signal=marketing_signal,
            display_type='end_cap',
            advertising_type='in_ad_only',
            product_id=1001
        )
        
        # Verify response structure
        assert isinstance(response, PromoResponse), "❌ Invalid response type"
        assert response.customer_id == customer_params['customer_id'], "❌ Customer ID mismatch"
        assert response.base_utility == base_utility, "❌ Base utility mismatch"
        print("✅ Response calculated successfully")
        
        # Verify utility boost
        assert response.promo_boost > 0, "❌ Promo boost should be positive"
        assert response.final_utility > base_utility, "❌ Final utility should exceed base"
        print(f"✅ Utility boost: {response.promo_boost:.3f}")
        
        # Verify elasticity
        assert response.elasticity < 0, "❌ Elasticity should be negative"
        print(f"✅ Arc elasticity: {response.elasticity:.2f}")
        
        # Verify probability
        assert 0.0 <= response.response_probability <= 1.0, "❌ Invalid probability"
        print(f"✅ Response probability: {response.response_probability:.3f}")
        
        # Print breakdown
        print(f"\n   Response Breakdown:")
        print(f"      Discount boost: {response.discount_boost:.3f}")
        print(f"      Display boost: {response.display_boost:.3f}")
        print(f"      Advertising boost: {response.advertising_boost:.3f}")
        print(f"      Signal multiplier: {response.signal_multiplier:.3f}")
        
        return True
    
    def test_3_discount_sensitivity(self):
        """Test 3: Verify discount depth affects response"""
        print("\n" + "="*70)
        print("TEST 3: Discount Sensitivity Test")
        print("="*70)
        
        customer = self.test_customers.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        base_utility = 5.0
        discounts = [0.05, 0.10, 0.20, 0.30, 0.40]
        
        responses = []
        for discount in discounts:
            response = self.calculator.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utility,
                discount_depth=discount,
                marketing_signal=0.5
            )
            responses.append(response)
        
        # Verify increasing discount → increasing boost
        boosts = [r.promo_boost for r in responses]
        for i in range(len(boosts) - 1):
            assert boosts[i] < boosts[i+1], \
                f"❌ Deeper discount should increase boost: {boosts[i]:.3f} >= {boosts[i+1]:.3f}"
        
        print("✅ Deeper discounts produce higher boosts")
        
        # Print discount curve
        print("\n   Discount Response Curve:")
        for discount, response in zip(discounts, responses):
            print(f"      {discount:.0%} off → Boost: {response.promo_boost:.3f}, "
                  f"Elasticity: {response.elasticity:.2f}")
        
        return True
    
    def test_4_individual_heterogeneity(self):
        """Test 4: Different customers respond differently to same promo"""
        print("\n" + "="*70)
        print("TEST 4: Individual Heterogeneity in Response")
        print("="*70)
        
        # Same promotion, different customers
        base_utility = 5.0
        discount_depth = 0.20
        marketing_signal = 0.6
        
        responses = []
        for i in range(10):  # Test 10 customers
            customer = self.test_customers.iloc[i]
            customer_params = {
                'customer_id': int(customer['customer_id']),
                'promo_responsiveness_param': customer['promo_responsiveness'],
                'display_sensitivity_param': customer['display_sensitivity'],
                'advertising_receptivity_param': customer['advertising_receptivity'],
                'price_sensitivity_param': customer['price_sensitivity']
            }
            
            response = self.calculator.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utility,
                discount_depth=discount_depth,
                marketing_signal=marketing_signal
            )
            responses.append(response)
        
        # Verify variation in responses
        boosts = [r.promo_boost for r in responses]
        elasticities = [r.elasticity for r in responses]
        
        boost_std = np.std(boosts)
        elasticity_std = np.std(elasticities)
        
        assert boost_std > 0.01, f"❌ Insufficient boost variation: σ={boost_std:.4f}"
        assert elasticity_std > 0.1, f"❌ Insufficient elasticity variation: σ={elasticity_std:.4f}"
        
        print(f"✅ Boost variation: mean={np.mean(boosts):.3f}, σ={boost_std:.3f}")
        print(f"✅ Elasticity variation: mean={np.mean(elasticities):.2f}, σ={elasticity_std:.2f}")
        
        # Show range
        print(f"\n   Response Range:")
        print(f"      Boost: [{min(boosts):.3f}, {max(boosts):.3f}]")
        print(f"      Elasticity: [{min(elasticities):.2f}, {max(elasticities):.2f}]")
        
        return True
    
    def test_5_display_effects(self):
        """Test 5: Display types affect response"""
        print("\n" + "="*70)
        print("TEST 5: Display Type Effects")
        print("="*70)
        
        customer = self.test_customers.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        base_utility = 5.0
        discount_depth = 0.20
        marketing_signal = 0.5
        
        display_types = ['none', 'shelf_tag', 'end_cap', 'feature_display']
        responses = {}
        
        for display in display_types:
            response = self.calculator.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utility,
                discount_depth=discount_depth,
                marketing_signal=marketing_signal,
                display_type=display
            )
            responses[display] = response
        
        # Verify display hierarchy: feature_display > end_cap > shelf_tag > none
        assert responses['feature_display'].display_boost > responses['end_cap'].display_boost, \
            "❌ Feature display should boost more than end cap"
        assert responses['end_cap'].display_boost > responses['shelf_tag'].display_boost, \
            "❌ End cap should boost more than shelf tag"
        assert responses['shelf_tag'].display_boost > responses['none'].display_boost, \
            "❌ Shelf tag should boost more than none"
        
        print("✅ Display hierarchy verified: feature > end_cap > shelf_tag > none")
        
        print("\n   Display Effects:")
        for display in display_types:
            boost = responses[display].display_boost
            print(f"      {display:20s}: {boost:.3f}")
        
        return True
    
    def test_6_advertising_effects(self):
        """Test 6: Advertising types affect response"""
        print("\n" + "="*70)
        print("TEST 6: Advertising Type Effects")
        print("="*70)
        
        customer = self.test_customers.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        base_utility = 5.0
        discount_depth = 0.20
        marketing_signal = 0.5
        
        ad_types = ['none', 'mailer_only', 'in_ad_only', 'in_ad_and_mailer']
        responses = {}
        
        for ad_type in ad_types:
            response = self.calculator.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utility,
                discount_depth=discount_depth,
                marketing_signal=marketing_signal,
                advertising_type=ad_type
            )
            responses[ad_type] = response
        
        # Verify advertising hierarchy
        assert responses['in_ad_and_mailer'].advertising_boost > responses['in_ad_only'].advertising_boost, \
            "❌ Both channels should boost more than single channel"
        assert responses['in_ad_only'].advertising_boost > responses['mailer_only'].advertising_boost, \
            "❌ In-ad should boost more than mailer only"
        assert responses['mailer_only'].advertising_boost > responses['none'].advertising_boost, \
            "❌ Mailer should boost more than none"
        
        print("✅ Advertising hierarchy verified: both > in_ad > mailer > none")
        
        print("\n   Advertising Effects:")
        for ad_type in ad_types:
            boost = responses[ad_type].advertising_boost
            print(f"      {ad_type:20s}: {boost:.3f}")
        
        return True
    
    def test_7_marketing_signal_amplification(self):
        """Test 7: Marketing signals amplify promotional response"""
        print("\n" + "="*70)
        print("TEST 7: Marketing Signal Amplification")
        print("="*70)
        
        customer = self.test_customers.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        base_utility = 5.0
        discount_depth = 0.20
        
        signals = [0.0, 0.25, 0.50, 0.75, 1.0]
        responses = []
        
        for signal in signals:
            response = self.calculator.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utility,
                discount_depth=discount_depth,
                marketing_signal=signal
            )
            responses.append(response)
        
        # Verify stronger signal → higher boost
        boosts = [r.promo_boost for r in responses]
        for i in range(len(boosts) - 1):
            assert boosts[i] < boosts[i+1], \
                f"❌ Stronger signal should increase boost: {boosts[i]:.3f} >= {boosts[i+1]:.3f}"
        
        print("✅ Stronger marketing signals amplify response")
        
        print("\n   Signal Amplification:")
        for signal, response in zip(signals, responses):
            print(f"      Signal {signal:.2f} → Boost: {response.promo_boost:.3f}, "
                  f"Multiplier: {response.signal_multiplier:.3f}")
        
        return True
    
    def test_8_population_response(self):
        """Test 8: Calculate response for entire population"""
        print("\n" + "="*70)
        print("TEST 8: Population-Level Response Calculation")
        print("="*70)
        
        # Generate base utilities
        n_customers = len(self.test_customers)
        base_utilities = np.random.uniform(4.0, 6.0, size=n_customers)
        
        # Calculate population response
        discount_depth = 0.20
        marketing_signal = 0.6
        
        response_df = self.calculator.calculate_population_response(
            customers_df=self.test_customers,
            base_utilities=base_utilities,
            discount_depth=discount_depth,
            marketing_signal=marketing_signal
        )
        
        # Verify DataFrame structure
        assert len(response_df) == n_customers, f"❌ Expected {n_customers} responses"
        print(f"✅ Calculated responses for {n_customers} customers")
        
        # Check columns
        required_cols = ['customer_id', 'base_utility', 'promo_boost', 'final_utility',
                        'elasticity', 'response_probability']
        for col in required_cols:
            assert col in response_df.columns, f"❌ Missing column: {col}"
        print("✅ All required columns present")
        
        # Verify statistics
        print(f"\n   Population Response Statistics:")
        print(f"      Promo Boost: mean={response_df['promo_boost'].mean():.3f}, "
              f"std={response_df['promo_boost'].std():.3f}")
        print(f"      Elasticity: mean={response_df['elasticity'].mean():.2f}, "
              f"std={response_df['elasticity'].std():.2f}")
        print(f"      Response Prob: mean={response_df['response_probability'].mean():.3f}, "
              f"std={response_df['response_probability'].std():.3f}")
        
        return True
    
    def test_9_elasticity_curves(self):
        """Test 9: Generate elasticity curves across discount range"""
        print("\n" + "="*70)
        print("TEST 9: Elasticity Curves")
        print("="*70)
        
        # Use small sample for speed
        sample_customers = self.test_customers.head(10)
        
        elasticity_df = self.calculator.get_elasticity_summary(
            customers_df=sample_customers,
            discount_range=(0.05, 0.50),
            n_points=5
        )
        
        # Verify structure
        assert len(elasticity_df) == 10 * 5, "❌ Wrong number of elasticity points"
        print(f"✅ Generated elasticity curves for {len(sample_customers)} customers")
        
        # Check columns
        required_cols = ['customer_id', 'discount_depth', 'elasticity',
                        'promo_responsiveness', 'price_sensitivity']
        for col in required_cols:
            assert col in elasticity_df.columns, f"❌ Missing column: {col}"
        print("✅ Elasticity data structure valid")
        
        # Print sample
        print("\n   Sample Elasticity Curve (Customer 1):")
        customer_1 = elasticity_df[elasticity_df['customer_id'] == sample_customers.iloc[0]['customer_id']]
        for _, row in customer_1.iterrows():
            print(f"      Discount {row['discount_depth']:.1%} → Elasticity {row['elasticity']:.2f}")
        
        return True


def run_all_tests():
    """Run complete Phase 2.5 test suite"""
    print("\n" + "="*70)
    print("PHASE 2.5 TEST SUITE: Promotional Response + Arc Elasticity")
    print("="*70)
    
    test_suite = TestPhase2_5()
    
    tests = [
        test_suite.test_1_calculator_initialization,
        test_suite.test_2_single_promo_response,
        test_suite.test_3_discount_sensitivity,
        test_suite.test_4_individual_heterogeneity,
        test_suite.test_5_display_effects,
        test_suite.test_6_advertising_effects,
        test_suite.test_7_marketing_signal_amplification,
        test_suite.test_8_population_response,
        test_suite.test_9_elasticity_curves
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
        print("\n🎉 ALL TESTS PASSED! Phase 2.5 promotional response is working!")
        print("\n   Key Features Validated:")
        print("   ✅ Individual promotional response")
        print("   ✅ Arc elasticity calculations")
        print("   ✅ Display and advertising effects")
        print("   ✅ Marketing signal amplification")
        print("   ✅ Customer heterogeneity integration")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
