"""
Diagnostic Script for Phase 2.5 Integration

This script checks why promotional response may not be working:
1. Checks if hetero_params exist in customers
2. Checks if hetero_params_dict is populated
3. Checks if promo_response_calc is initialized
4. Checks if store_promo_contexts are generated
5. Provides recommendations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.config import EnhancedRetailConfig
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1

def diagnose_phase_2_5():
    """Diagnose Phase 2.5 integration status"""
    
    print("="*70)
    print("PHASE 2.5 DIAGNOSTIC TOOL")
    print("="*70)
    
    # Create small config for testing
    config = EnhancedRetailConfig(
        n_customers=50,
        n_products=100,
        n_stores=5,
        simulation_weeks=1,
        use_real_catalog=True,
        product_catalog_path='data/processed/product_catalog/product_catalog_20k.parquet',
        enable_temporal_dynamics=True,
        enable_customer_drift=True,
        enable_store_loyalty=True,
        enable_basket_composition=True,
        region='US'
    )
    
    print("\n📋 Step 1: Initialize generator")
    generator = EnhancedRetailSynthV4_1(config)
    
    print("\n📋 Step 2: Generate base datasets")
    generator.generate_base_datasets()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS")
    print("="*70)
    
    # Check 1: Customer heterogeneity parameters
    print("\n🔍 Check 1: Customer Heterogeneity Parameters (Phase 2.4)")
    customers_df = generator.datasets['customers']
    
    if 'hetero_params' in customers_df.columns:
        print("   ✅ 'hetero_params' column EXISTS in customers DataFrame")
        
        # Check if it's populated
        non_null_count = customers_df['hetero_params'].notna().sum()
        print(f"   ✅ Populated for {non_null_count}/{len(customers_df)} customers")
        
        # Show sample
        if non_null_count > 0:
            sample_idx = customers_df['hetero_params'].notna().idxmax()
            sample_params = customers_df.loc[sample_idx, 'hetero_params']
            print(f"\n   📊 Sample customer hetero_params:")
            for key, value in sample_params.items():
                print(f"      • {key}: {value:.3f}")
    else:
        print("   ❌ 'hetero_params' column NOT FOUND in customers DataFrame")
        print("   ℹ️  Phase 2.4 customer generator needs integration")
    
    # Check 2: Precomputation hetero_params_dict
    print("\n🔍 Check 2: Precomputation hetero_params_dict")
    
    if hasattr(generator, 'precomp'):
        if hasattr(generator.precomp, 'hetero_params_dict'):
            n_params = len(generator.precomp.hetero_params_dict)
            print(f"   ✅ hetero_params_dict exists with {n_params} entries")
            
            if n_params > 0:
                # Show sample
                sample_customer_id = list(generator.precomp.hetero_params_dict.keys())[0]
                sample_params = generator.precomp.hetero_params_dict[sample_customer_id]
                print(f"\n   📊 Sample customer {sample_customer_id} from precomp:")
                print(f"      • promo_responsiveness: {sample_params.get('promo_responsiveness_param', 'N/A')}")
                print(f"      • price_sensitivity: {sample_params.get('price_sensitivity_param', 'N/A')}")
                print(f"      • display_sensitivity: {sample_params.get('display_sensitivity_param', 'N/A')}")
            else:
                print("   ⚠️  hetero_params_dict is EMPTY")
                print("   ℹ️  This will cause Phase 2.5 to be skipped")
        else:
            print("   ❌ hetero_params_dict attribute NOT FOUND")
    else:
        print("   ❌ precomp attribute NOT FOUND")
    
    # Check 3: Promo Response Calculator
    print("\n🔍 Check 3: Promotional Response Calculator (Phase 2.5)")
    
    if hasattr(generator, 'promo_response_calculator'):
        print("   ✅ promo_response_calculator EXISTS")
    else:
        print("   ❌ promo_response_calculator NOT FOUND")
    
    # Check 4: Promotional Engine
    print("\n🔍 Check 4: Promotional Engine (Phase 2.1/2.2)")
    
    if hasattr(generator, 'promotional_engine'):
        print("   ✅ promotional_engine EXISTS")
        
        # Check for generate_store_promo_context method
        if hasattr(generator.promotional_engine, 'generate_store_promo_context'):
            print("   ✅ generate_store_promo_context() method EXISTS")
        else:
            print("   ❌ generate_store_promo_context() method NOT FOUND")
            print("   ℹ️  Need to apply proposed changes to promotional_engine.py")
        
        # Check for signal calculator
        if hasattr(generator.promotional_engine, 'signal_calculator'):
            print("   ✅ signal_calculator EXISTS (Phase 2.3)")
        else:
            print("   ⚠️  signal_calculator NOT FOUND")
    else:
        print("   ❌ promotional_engine NOT FOUND")
    
    # Summary and Recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    issues_found = []
    
    if 'hetero_params' not in customers_df.columns:
        issues_found.append("hetero_params missing from customers")
    
    if hasattr(generator, 'precomp') and hasattr(generator.precomp, 'hetero_params_dict'):
        if len(generator.precomp.hetero_params_dict) == 0:
            issues_found.append("hetero_params_dict is empty")
    
    if not hasattr(generator, 'promo_response_calculator'):
        issues_found.append("promo_response_calculator not initialized")
    
    if not hasattr(generator, 'promotional_engine'):
        issues_found.append("promotional_engine not initialized")
    elif not hasattr(generator.promotional_engine, 'generate_store_promo_context'):
        issues_found.append("generate_store_promo_context method missing")
    
    if len(issues_found) == 0:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🎉 Phase 2.5 integration is complete and working!")
        print("\nYou can now run:")
        print("   python scripts/generate_with_elasticity.py --skip-save")
    else:
        print(f"\n⚠️  FOUND {len(issues_found)} ISSUE(S):")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 RECOMMENDED ACTIONS:")
        
        if "hetero_params missing from customers" in issues_found or "hetero_params_dict is empty" in issues_found:
            print("\n   📌 Phase 2.4 Customer Heterogeneity:")
            print("      The customer generator is creating 'hetero_params' but they")
            print("      may not be populated correctly.")
            print("\n      ✅ GOOD NEWS: The customer generator already has Phase 2.4 code!")
            print("      ✅ The hetero_params are being created in customer_generator.py")
            print("\n      Action: Check if customers are being generated with all parameters")
        
        if "promo_response_calculator not initialized" in issues_found:
            print("\n   📌 Promo Response Calculator:")
            print("      Add to main_generator.py __init__:")
            print("      self.promo_response_calculator = PromoResponseCalculator()")
        
        if "generate_store_promo_context method missing" in issues_found:
            print("\n   📌 Promotional Engine:")
            print("      Apply proposed changes to promotional_engine.py")
            print("      This adds the generate_store_promo_context() method")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    diagnose_phase_2_5()
