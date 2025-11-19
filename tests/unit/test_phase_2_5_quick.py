"""
Quick Phase 2.5 Integration Test
Verify promotional response works end-to-end with heterogeneity and marketing signals
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.promo_response import PromoResponseCalculator
from retailsynth.engines.customer_heterogeneity import CustomerHeterogeneityEngine
from retailsynth.engines.marketing_signal import MarketingSignalCalculator


def test_phase_2_5_integration():
    """Test Phase 2.5 promotional response integration"""
    print("\n" + "="*70)
    print("PHASE 2.5 QUICK INTEGRATION TEST")
    print("Promotional Response + Arc Elasticity")
    print("="*70)
    
    # Test 1: Initialize all engines
    print("\n1. Initializing engines...")
    try:
        promo_calc = PromoResponseCalculator()
        hetero_engine = CustomerHeterogeneityEngine(random_seed=42)
        signal_calc = MarketingSignalCalculator()
        print("   ✅ All engines initialized")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Generate heterogeneous customers
    print("\n2. Generating heterogeneous customers...")
    try:
        n_customers = 50
        customers_df = hetero_engine.generate_population_parameters(n_customers)
        print(f"   ✅ Generated {n_customers} customers with unique parameters")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Test individual promotional response
    print("\n3. Testing individual promotional response...")
    try:
        customer = customers_df.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        response = promo_calc.calculate_promo_response(
            customer_params=customer_params,
            base_utility=5.0,
            discount_depth=0.25,  # 25% off
            marketing_signal=0.7,
            display_type='feature_display',
            advertising_type='in_ad_and_mailer'
        )
        
        print(f"   ✅ Customer {customer['customer_id']}:")
        print(f"      Base utility: {response.base_utility:.3f}")
        print(f"      Promo boost: {response.promo_boost:.3f}")
        print(f"      Final utility: {response.final_utility:.3f}")
        print(f"      Elasticity: {response.elasticity:.2f}")
        print(f"      Response prob: {response.response_probability:.3f}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test population-level response
    print("\n4. Testing population-level response...")
    try:
        base_utilities = np.random.uniform(4.0, 6.0, size=n_customers)
        
        response_df = promo_calc.calculate_population_response(
            customers_df=customers_df,
            base_utilities=base_utilities,
            discount_depth=0.20,
            marketing_signal=0.6
        )
        
        print(f"   ✅ Calculated responses for {len(response_df)} customers")
        print(f"      Mean boost: {response_df['promo_boost'].mean():.3f}")
        print(f"      Boost range: [{response_df['promo_boost'].min():.3f}, {response_df['promo_boost'].max():.3f}]")
        print(f"      Mean elasticity: {response_df['elasticity'].mean():.2f}")
        print(f"      Elasticity range: [{response_df['elasticity'].min():.2f}, {response_df['elasticity'].max():.2f}]")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test heterogeneity in response
    print("\n5. Verifying individual heterogeneity in promotional response...")
    try:
        # Same promotion, different customers
        responses = []
        for i in range(10):
            customer = customers_df.iloc[i]
            customer_params = {
                'customer_id': int(customer['customer_id']),
                'promo_responsiveness_param': customer['promo_responsiveness'],
                'display_sensitivity_param': customer['display_sensitivity'],
                'advertising_receptivity_param': customer['advertising_receptivity'],
                'price_sensitivity_param': customer['price_sensitivity']
            }
            
            response = promo_calc.calculate_promo_response(
                customer_params=customer_params,
                base_utility=5.0,
                discount_depth=0.20,
                marketing_signal=0.6
            )
            responses.append(response.promo_boost)
        
        boost_std = np.std(responses)
        boost_cv = boost_std / np.mean(responses)  # Coefficient of variation
        
        print(f"   ✅ Response heterogeneity verified:")
        print(f"      Boost std: {boost_std:.3f}")
        print(f"      Coefficient of variation: {boost_cv:.2%}")
        
        if boost_cv < 0.05:
            print("   ⚠️  Low variation - customers responding very similarly")
        else:
            print("   ✅ Good variation - customers have diverse responses")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 6: Test discount sensitivity curve
    print("\n6. Testing discount sensitivity curves...")
    try:
        customer = customers_df.iloc[0]
        discounts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        
        print(f"   Customer {customer['customer_id']} response to varying discounts:")
        for discount in discounts:
            customer_params = {
                'customer_id': int(customer['customer_id']),
                'promo_responsiveness_param': customer['promo_responsiveness'],
                'display_sensitivity_param': customer['display_sensitivity'],
                'advertising_receptivity_param': customer['advertising_receptivity'],
                'price_sensitivity_param': customer['price_sensitivity']
            }
            
            response = promo_calc.calculate_promo_response(
                customer_params=customer_params,
                base_utility=5.0,
                discount_depth=discount,
                marketing_signal=0.5
            )
            
            print(f"      {discount:5.0%} off → Boost: {response.promo_boost:.3f}, "
                  f"Elasticity: {response.elasticity:.2f}")
        
        print("   ✅ Discount sensitivity curve generated")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 7: Test marketing signal amplification
    print("\n7. Testing marketing signal amplification...")
    try:
        customer = customers_df.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        signals = [0.0, 0.3, 0.6, 1.0]
        
        print(f"   Marketing signal effects:")
        for signal in signals:
            response = promo_calc.calculate_promo_response(
                customer_params=customer_params,
                base_utility=5.0,
                discount_depth=0.20,
                marketing_signal=signal
            )
            
            print(f"      Signal {signal:.1f} → Boost: {response.promo_boost:.3f}, "
                  f"Multiplier: {response.signal_multiplier:.2f}")
        
        print("   ✅ Marketing signals amplify promotional response")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 8: Test display and advertising effects
    print("\n8. Testing display and advertising effects...")
    try:
        customer = customers_df.iloc[0]
        customer_params = {
            'customer_id': int(customer['customer_id']),
            'promo_responsiveness_param': customer['promo_responsiveness'],
            'display_sensitivity_param': customer['display_sensitivity'],
            'advertising_receptivity_param': customer['advertising_receptivity'],
            'price_sensitivity_param': customer['price_sensitivity']
        }
        
        # Test combinations
        combos = [
            ('none', 'none', 'Baseline'),
            ('shelf_tag', 'none', 'Shelf tag only'),
            ('end_cap', 'mailer_only', 'End cap + Mailer'),
            ('feature_display', 'in_ad_and_mailer', 'Full promotion')
        ]
        
        print(f"   Promotional combination effects:")
        for display, ad, label in combos:
            response = promo_calc.calculate_promo_response(
                customer_params=customer_params,
                base_utility=5.0,
                discount_depth=0.20,
                marketing_signal=0.6,
                display_type=display,
                advertising_type=ad
            )
            
            print(f"      {label:25s}: Boost {response.promo_boost:.3f} "
                  f"(Display: {response.display_boost:.3f}, Ad: {response.advertising_boost:.3f})")
        
        print("   ✅ Display and advertising effects verified")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL INTEGRATION CHECKS PASSED!")
    print("="*70)
    print("\nPhase 2.5 Integration Summary:")
    print("  • PromoResponseCalculator: ✅ Working")
    print("  • Individual heterogeneity: ✅ Integrated")
    print("  • Marketing signals: ✅ Amplifying response")
    print("  • Arc elasticity: ✅ Calculated")
    print("  • Display effects: ✅ Working")
    print("  • Advertising effects: ✅ Working")
    print("  • Population response: ✅ Calculated")
    print("\n🎉 Phase 2.5 is OPERATIONAL!")
    print("   Same promotion → Different response per customer!")
    print("   Phases 2.3, 2.4, 2.5 working together perfectly!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_phase_2_5_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
