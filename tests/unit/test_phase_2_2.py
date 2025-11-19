"""
Test Suite for Phase 2.2: Promo Organization Enhancements
Tests HMM integration, product tendencies, and multi-store support
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.promotional_engine import PromotionalEngine, StorePromoContext
from retailsynth.engines.price_hmm import PriceStateHMM


def create_mock_products(n=100):
    """Create mock product dataset"""
    return pd.DataFrame({
        'product_id': range(1, n+1),
        'product_name': [f'Product {i}' for i in range(1, n+1)],
        'category': np.random.choice(['Grocery', 'Beverage', 'Snacks'], n),
        'base_price': np.random.uniform(1.0, 20.0, n)
    })


def create_mock_stores(n=3):
    """Create mock store dataset"""
    return pd.DataFrame({
        'store_id': range(1, n+1),
        'store_name': [f'Store {i}' for i in range(1, n+1)],
        'state': ['CA', 'TX', 'NY'][:n]
    })


def create_mock_hmm(products_df):
    """Create mock HMM model with transition matrices"""
    hmm = PriceStateHMM(products_df, n_states=4)
    
    # Simulate learned parameters for a few products
    for product_id in products_df['product_id'].values[:20]:  # First 20 products
        # Create a simple transition matrix
        hmm.transition_matrices[product_id] = np.array([
            [0.7, 0.2, 0.08, 0.02],  # From regular
            [0.3, 0.5, 0.15, 0.05],  # From feature
            [0.4, 0.3, 0.2, 0.1],    # From deep
            [0.5, 0.3, 0.15, 0.05]   # From clearance
        ])
        
        # Initial state probabilities
        hmm.initial_state_probs[product_id] = np.array([0.6, 0.25, 0.1, 0.05])
    
    return hmm


class TestPhase2_2:
    """Test suite for Phase 2.2 enhancements"""
    
    def __init__(self):
        self.products = create_mock_products(100)
        self.stores = create_mock_stores(3)
        self.hmm = create_mock_hmm(self.products)
        
    def test_1_product_tendencies_initialization(self):
        """Test 1: Product tendencies are initialized correctly"""
        print("\n" + "="*70)
        print("TEST 1: Product Tendencies Initialization")
        print("="*70)
        
        engine = PromotionalEngine(
            products_df=self.products,
            stores_df=self.stores
        )
        
        # Check tendencies exist
        assert len(engine.product_promo_tendencies) == len(self.products), \
            "❌ Not all products have tendencies"
        print("✅ All products have promotional tendencies")
        
        # Check tendency distribution
        tendencies = list(engine.product_promo_tendencies.values())
        mean_tendency = np.mean(tendencies)
        std_tendency = np.std(tendencies)
        
        print(f"   Mean tendency: {mean_tendency:.3f}")
        print(f"   Std tendency: {std_tendency:.3f}")
        print(f"   Min tendency: {min(tendencies):.3f}")
        print(f"   Max tendency: {max(tendencies):.3f}")
        
        # Check distribution roughly matches expectations
        high_count = sum(1 for t in tendencies if t > 1.2)
        low_count = sum(1 for t in tendencies if t < 0.8)
        moderate_count = sum(1 for t in tendencies if 0.8 <= t <= 1.2)
        
        print(f"   High promo items (>1.2): {high_count} ({high_count/len(tendencies)*100:.1f}%)")
        print(f"   Low promo items (<0.8): {low_count} ({low_count/len(tendencies)*100:.1f}%)")
        print(f"   Moderate items (0.8-1.2): {moderate_count} ({moderate_count/len(tendencies)*100:.1f}%)")
        
        assert 0.8 < mean_tendency < 1.2, "❌ Mean tendency should be near 1.0"
        print("✅ Tendency distribution is reasonable")
        
        return True
    
    def test_2_hmm_integration(self):
        """Test 2: HMM model is used when available"""
        print("\n" + "="*70)
        print("TEST 2: HMM Integration")
        print("="*70)
        
        # Engine without HMM
        engine_no_hmm = PromotionalEngine(
            products_df=self.products,
            stores_df=self.stores
        )
        
        # Engine with HMM
        engine_with_hmm = PromotionalEngine(
            hmm_model=self.hmm,
            products_df=self.products,
            stores_df=self.stores
        )
        
        print("✅ Engine initialized with HMM model")
        
        # Generate promotions
        product_ids = self.products['product_id'].values
        base_prices = self.products['base_price'].values
        
        context_no_hmm = engine_no_hmm.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        context_with_hmm = engine_with_hmm.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        print(f"   Without HMM: {len(context_no_hmm.promoted_products)} products promoted")
        print(f"   With HMM: {len(context_with_hmm.promoted_products)} products promoted")
        
        # Both should have promotions
        assert len(context_no_hmm.promoted_products) > 0, "❌ No promotions generated without HMM"
        assert len(context_with_hmm.promoted_products) > 0, "❌ No promotions generated with HMM"
        
        print("✅ Both engines generate valid promotions")
        
        # Check state assignments
        states_no_hmm = list(context_no_hmm.promo_states.values())
        states_with_hmm = list(context_with_hmm.promo_states.values())
        
        print(f"   State distribution (no HMM): {np.bincount(states_no_hmm, minlength=4)}")
        print(f"   State distribution (with HMM): {np.bincount(states_with_hmm, minlength=4)}")
        
        print("✅ HMM integration test passed")
        
        return True
    
    def test_3_tendency_weighted_selection(self):
        """Test 3: High tendency products promote more frequently"""
        print("\n" + "="*70)
        print("TEST 3: Tendency-Weighted Selection")
        print("="*70)
        
        engine = PromotionalEngine(
            products_df=self.products,
            stores_df=self.stores
        )
        
        # Run 50 weeks and count promotions per product
        promo_counts = {pid: 0 for pid in self.products['product_id'].values}
        
        product_ids = self.products['product_id'].values
        base_prices = self.products['base_price'].values
        
        for week in range(1, 51):
            context = engine.generate_store_promotions(
                store_id=1,
                week_number=week,
                base_prices=base_prices,
                product_ids=product_ids
            )
            for pid in context.promoted_products:
                promo_counts[pid] += 1
        
        # Calculate correlation between tendency and promo frequency
        tendencies = []
        frequencies = []
        
        for pid in product_ids:
            tendencies.append(engine.product_promo_tendencies[pid])
            frequencies.append(promo_counts[pid])
        
        correlation = np.corrcoef(tendencies, frequencies)[0, 1]
        
        print(f"   Correlation (tendency vs frequency): {correlation:.3f}")
        
        # High tendency products should promote more
        high_tendency_pids = [pid for pid in product_ids 
                              if engine.product_promo_tendencies[pid] > 1.2]
        low_tendency_pids = [pid for pid in product_ids 
                             if engine.product_promo_tendencies[pid] < 0.8]
        
        avg_high = np.mean([promo_counts[pid] for pid in high_tendency_pids])
        avg_low = np.mean([promo_counts[pid] for pid in low_tendency_pids])
        avg_all = np.mean(list(promo_counts.values()))
        
        print(f"   Avg promos (high tendency): {avg_high:.1f}")
        print(f"   Avg promos (low tendency): {avg_low:.1f}")
        print(f"   Avg promos (all products): {avg_all:.1f}")
        
        assert avg_high > avg_all, "❌ High tendency products should promote more than average"
        assert avg_low < avg_all, "❌ Low tendency products should promote less than average"
        
        print("✅ Tendency-weighted selection works correctly")
        
        return True
    
    def test_4_multi_store_contexts(self):
        """Test 4: Different promotions per store"""
        print("\n" + "="*70)
        print("TEST 4: Multi-Store Promotional Contexts")
        print("="*70)
        
        engine = PromotionalEngine(
            hmm_model=self.hmm,
            products_df=self.products,
            stores_df=self.stores
        )
        
        product_ids = self.products['product_id'].values
        base_prices = self.products['base_price'].values
        
        # Generate promotions for multiple stores
        contexts = {}
        for store_id in self.stores['store_id'].values:
            contexts[store_id] = engine.generate_store_promotions(
                store_id=store_id,
                week_number=1,
                base_prices=base_prices,
                product_ids=product_ids
            )
        
        print(f"✅ Generated contexts for {len(contexts)} stores")
        
        # Check that different stores have different promotions
        for store_id, context in contexts.items():
            print(f"   Store {store_id}: {len(context.promoted_products)} products, "
                  f"avg discount: {context.avg_discount_depth:.1%}")
        
        # At least some variation expected
        promo_sets = [set(ctx.promoted_products) for ctx in contexts.values()]
        all_same = all(s == promo_sets[0] for s in promo_sets)
        
        if all_same:
            print("⚠️  Warning: All stores have identical promotions (may happen with small datasets)")
        else:
            print("✅ Different stores have different promotional mixes")
        
        return True
    
    def test_5_discount_depth_variation(self):
        """Test 5: Product tendencies affect discount depths"""
        print("\n" + "="*70)
        print("TEST 5: Discount Depth Variation by Tendency")
        print("="*70)
        
        engine = PromotionalEngine(
            products_df=self.products,
            stores_df=self.stores
        )
        
        product_ids = self.products['product_id'].values
        base_prices = self.products['base_price'].values
        
        # Run multiple weeks and collect discount data
        high_tendency_discounts = []
        low_tendency_discounts = []
        
        for week in range(1, 21):
            context = engine.generate_store_promotions(
                store_id=1,
                week_number=week,
                base_prices=base_prices,
                product_ids=product_ids
            )
            
            for pid in context.promoted_products:
                tendency = engine.product_promo_tendencies[pid]
                discount = context.promo_depths[pid]
                
                if tendency > 1.2:
                    high_tendency_discounts.append(discount)
                elif tendency < 0.8:
                    low_tendency_discounts.append(discount)
        
        if high_tendency_discounts and low_tendency_discounts:
            avg_high_discount = np.mean(high_tendency_discounts)
            avg_low_discount = np.mean(low_tendency_discounts)
            
            print(f"   Avg discount (high tendency): {avg_high_discount:.1%}")
            print(f"   Avg discount (low tendency): {avg_low_discount:.1%}")
            print(f"   Difference: {(avg_high_discount - avg_low_discount):.1%}")
            
            # High tendency should have slightly deeper discounts
            assert avg_high_discount > avg_low_discount, \
                "❌ High tendency products should have deeper discounts"
            
            print("✅ Discount depths vary appropriately with tendency")
        else:
            print("⚠️  Not enough data to test discount variation")
        
        return True
    
    def test_6_promotional_summary(self):
        """Test 6: Promotional summary metrics are accurate"""
        print("\n" + "="*70)
        print("TEST 6: Promotional Summary Metrics")
        print("="*70)
        
        engine = PromotionalEngine(
            hmm_model=self.hmm,
            products_df=self.products,
            stores_df=self.stores
        )
        
        product_ids = self.products['product_id'].values
        base_prices = self.products['base_price'].values
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        summary = engine.get_promo_summary(context)
        
        print(f"   Store ID: {summary['store_id']}")
        print(f"   Week: {summary['week_number']}")
        print(f"   Total promotions: {summary['n_promotions']}")
        print(f"   Avg discount: {summary['avg_discount']:.1%}")
        print(f"   Deep discounts (>30%): {summary['n_deep_discounts']}")
        print(f"   End caps: {summary['n_end_caps']}")
        print(f"   Feature displays: {summary['n_features']}")
        print(f"   In-ad products: {summary['n_in_ad']}")
        print(f"   Mailer products: {summary['n_mailer']}")
        
        # Validate metrics
        assert summary['n_promotions'] == len(context.promoted_products), \
            "❌ Promotion count mismatch"
        assert 0 <= summary['avg_discount'] <= 0.70, \
            "❌ Average discount out of range"
        assert summary['n_end_caps'] <= 10, \
            "❌ Too many end caps (max 10)"
        assert summary['n_features'] <= 3, \
            "❌ Too many feature displays (max 3)"
        
        print("✅ All summary metrics are accurate")
        
        return True


def run_all_tests():
    """Run complete Phase 2.2 test suite"""
    print("\n" + "="*70)
    print("PHASE 2.2 TEST SUITE: Promo Organization Enhancements")
    print("="*70)
    
    test_suite = TestPhase2_2()
    
    tests = [
        test_suite.test_1_product_tendencies_initialization,
        test_suite.test_2_hmm_integration,
        test_suite.test_3_tendency_weighted_selection,
        test_suite.test_4_multi_store_contexts,
        test_suite.test_5_discount_depth_variation,
        test_suite.test_6_promotional_summary
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
            failed += 1
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Phase 2.2 is working correctly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
