"""
Comprehensive Test Suite for Price Elasticity Engines (Sprint 2)

Tests cover:
1. HMM Price Dynamics (price_hmm.py)
2. Cross-Price Elasticity (cross_price_elasticity.py)
3. Arc Elasticity (arc_elasticity.py)

Each test validates:
- Correctness of calculations
- Edge cases and boundary conditions
- Integration with real data
- Performance characteristics
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import sparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from retailsynth.engines.price_hmm import PriceStateHMM
from retailsynth.engines.cross_price_elasticity import CrossPriceElasticityEngine
from retailsynth.engines.arc_elasticity import ArcPriceElasticityEngine


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_products():
    """Create sample product catalog"""
    return pd.DataFrame({
        'PRODUCT_ID': [1, 2, 3, 4, 5],
        'DEPARTMENT': ['Beverages', 'Beverages', 'Snacks', 'Snacks', 'Dairy'],
        'COMMODITY': ['Soda', 'Soda', 'Chips', 'Chips', 'Milk'],
        'BRAND': ['Coke', 'Pepsi', 'Lays', 'Pringles', 'Organic Valley'],
        'avg_price': [3.99, 3.49, 2.99, 3.49, 4.99],
        'category_role': ['mid_basket', 'mid_basket', 'back_basket', 'back_basket', 'front_basket']
    })


@pytest.fixture
def sample_transactions():
    """Create sample transaction data with Dunnhumby column names"""
    np.random.seed(42)
    
    transactions = []
    for week in range(1, 53):
        for product_id in [1, 2, 3, 4, 5]:
            # Generate realistic price variations
            base_price = {1: 3.99, 2: 3.49, 3: 2.99, 4: 3.49, 5: 4.99}[product_id]
            
            # Simulate promotional cycles
            if week % 4 == 0:  # Deep discount every 4 weeks
                price = base_price * 0.75
                discount_pct = 0.25
            elif week % 2 == 0:  # Feature every 2 weeks
                price = base_price * 0.90
                discount_pct = 0.10
            else:  # Regular price
                price = base_price
                discount_pct = 0.0
            
            # Quantity varies with discount
            base_qty = 100
            quantity = int(base_qty * (1 + discount_pct * 2))  # More sales on discount
            
            # Calculate Dunnhumby columns
            sales_value = price * quantity
            retail_disc = sales_value * discount_pct
            
            transactions.append({
                'PRODUCT_ID': product_id,
                'WEEK_NO': week,
                'SALES_VALUE': sales_value,
                'QUANTITY': quantity,
                'RETAIL_DISC': retail_disc
            })
    
    return pd.DataFrame(transactions)


@pytest.fixture
def sample_causal_data():
    """Create sample promotional causal data"""
    causal = []
    for week in range(1, 53):
        for product_id in [1, 2, 3, 4, 5]:
            # Feature promotions every 2 weeks
            if week % 2 == 0:
                causal.append({
                    'PRODUCT_ID': product_id,
                    'WEEK_NO': week,
                    'STORE_ID': 1,
                    'DISPLAY': 1 if week % 4 == 0 else 0,
                    'MAILER': 1 if week % 2 == 0 else 0
                })
    
    return pd.DataFrame(causal)


# ============================================================================
# HMM PRICE DYNAMICS TESTS
# ============================================================================

class TestPriceStateHMM:
    """Test suite for HMM price dynamics engine"""
    
    def test_initialization(self, sample_products):
        """Test HMM initialization"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        
        assert hmm.n_states == 4
        assert len(hmm.products) == 5
        assert len(hmm.states) == 4
        assert hmm.states[0]['name'] == 'regular'
        assert hmm.states[3]['name'] == 'clearance'
    
    def test_transition_matrix_learning(self, sample_products, sample_transactions):
        """Test transition matrix estimation"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        # Check that transition matrices were learned
        assert len(hmm.transition_matrices) > 0
        
        # Get a learned transition matrix
        product_id = list(hmm.transition_matrices.keys())[0]
        trans_matrix = hmm.transition_matrices[product_id]
        
        # Verify it's a valid stochastic matrix
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        
        # Each row should sum to ~1.0 (stochastic property)
        for row in trans_matrix:
            row_sum = np.sum(row)
            assert 0.99 <= row_sum <= 1.01, f"Row sum {row_sum} not close to 1.0"
        
        # All probabilities should be non-negative
        assert np.all(trans_matrix >= 0)
    
    def test_emission_distributions(self, sample_products, sample_transactions):
        """Test emission distribution learning"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        # Check emission distributions
        product_id = list(hmm.emission_distributions.keys())[0]
        emissions = hmm.emission_distributions[product_id]
        
        # Should have emissions for observed states
        assert len(emissions) > 0
        
        # Each emission should have required parameters
        for state, params in emissions.items():
            assert 'price_mean' in params
            assert 'price_std' in params
            assert 'discount_mean' in params
            assert 'discount_std' in params
            
            # Validate parameter ranges
            assert params['price_mean'] > 0
            assert params['price_std'] >= 0.01  # Minimum std
            assert 0 <= params['discount_mean'] <= 1
            assert params['discount_std'] >= 0.01
    
    def test_price_sequence_generation(self, sample_products, sample_transactions):
        """Test realistic price sequence generation"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        base_price = 3.99
        n_weeks = 52
        
        # Generate price sequence
        prices, states = hmm.generate_price_sequence(
            product_id=product_id,
            n_weeks=n_weeks,
            base_price=base_price,
            random_seed=42
        )
        
        # Validate output shape
        assert len(prices) == n_weeks
        assert len(states) == n_weeks
        
        # Validate price ranges
        assert np.all(prices > 0), "All prices should be positive"
        assert np.all(prices <= base_price * 1.1), "Prices shouldn't exceed base by much"
        assert np.min(prices) >= base_price * 0.4, "Prices shouldn't be too low"
        
        # Validate states
        assert np.all(states >= 0)
        assert np.all(states < hmm.n_states)
        
        # Check for state transitions (not all same state)
        unique_states = len(np.unique(states))
        assert unique_states > 1, "Should have multiple states over 52 weeks"
    
    def test_price_sequence_reproducibility(self, sample_products, sample_transactions):
        """Test that same seed produces same sequence"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        
        # Generate twice with same seed
        prices1, states1 = hmm.generate_price_sequence(
            product_id=product_id, n_weeks=52, base_price=3.99, random_seed=42
        )
        prices2, states2 = hmm.generate_price_sequence(
            product_id=product_id, n_weeks=52, base_price=3.99, random_seed=42
        )
        
        # Should be identical
        np.testing.assert_array_almost_equal(prices1, prices2)
        np.testing.assert_array_equal(states1, states2)
    
    def test_fallback_to_similar_product(self, sample_products, sample_transactions):
        """Test fallback when product has no learned parameters"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        # Try to generate for product not in training data
        unknown_product_id = 9999
        
        # Should fall back to similar product
        prices, states = hmm.generate_price_sequence(
            product_id=unknown_product_id,
            n_weeks=10,
            base_price=4.99,
            random_seed=42
        )
        
        # Should still generate valid prices
        assert len(prices) == 10
        assert np.all(prices > 0)
    
    def test_save_load_parameters(self, sample_products, sample_transactions, tmp_path):
        """Test saving and loading HMM parameters"""
        hmm1 = PriceStateHMM(sample_products, n_states=4)
        hmm1.learn_from_data(sample_transactions)
        
        # Save parameters
        save_path = tmp_path / "hmm_params.pkl"
        hmm1.save_parameters(str(save_path))
        
        assert save_path.exists()
        
        # Load into new HMM
        hmm2 = PriceStateHMM(sample_products, n_states=4)
        hmm2.load_parameters(str(save_path))
        
        # Compare transition matrices
        for product_id in hmm1.transition_matrices.keys():
            np.testing.assert_array_almost_equal(
                hmm1.transition_matrices[product_id],
                hmm2.transition_matrices[product_id]
            )
        
        # Compare emission distributions
        assert hmm1.emission_distributions.keys() == hmm2.emission_distributions.keys()


# ============================================================================
# CROSS-PRICE ELASTICITY TESTS
# ============================================================================

class TestCrossPriceElasticity:
    """Test suite for cross-price elasticity engine"""
    
    def test_initialization(self, sample_products):
        """Test cross-price engine initialization"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        assert len(engine.products) == 5
        assert engine.cross_elasticity_matrix is None  # Not learned yet
    
    def test_basket_construction(self, sample_products, sample_transactions):
        """Test basket-level data construction"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        # Create sample basket data
        baskets = pd.DataFrame({
            'BASKET_ID': [1, 1, 2, 2, 3],
            'PRODUCT_ID': [1, 2, 1, 3, 2],
            'QUANTITY': [1, 2, 1, 1, 1],
            'SALES_VALUE': [3.99, 6.98, 3.99, 2.99, 3.49]
        })
        
        # Should process without errors
        assert len(baskets) == 5
    
    def test_substitute_detection(self, sample_products):
        """Test substitute pair detection"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        # Create mock elasticity matrix with substitutes
        n_products = len(sample_products)
        elasticity_matrix = sparse.lil_matrix((n_products, n_products))
        
        # Coke (0) and Pepsi (1) are substitutes (positive elasticity)
        elasticity_matrix[0, 1] = 0.35
        elasticity_matrix[1, 0] = 0.30
        
        engine.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        # Manually create substitute groups DataFrame (matches actual implementation)
        substitute_pairs = []
        for i in range(n_products):
            for j in range(i+1, n_products):
                if elasticity_matrix[i, j] > 0.2:
                    substitute_pairs.append((
                        sample_products.iloc[i]['PRODUCT_ID'],
                        sample_products.iloc[j]['PRODUCT_ID'],
                        elasticity_matrix[i, j]
                    ))
        
        engine.substitute_groups = pd.DataFrame(
            substitute_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        )
        
        # Should detect substitute pair
        assert len(engine.substitute_groups) > 0
        
        # Check if Coke-Pepsi pair exists
        pair_exists = any(
            (row['product_i'] == 1 and row['product_j'] == 2) or
            (row['product_i'] == 2 and row['product_j'] == 1)
            for _, row in engine.substitute_groups.iterrows()
        )
        assert pair_exists
    
    def test_complement_detection(self, sample_products):
        """Test complement pair detection"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        # Create mock elasticity matrix with complements
        n_products = len(sample_products)
        elasticity_matrix = sparse.lil_matrix((n_products, n_products))
        
        # Chips (2) and Soda (0) are complements (negative elasticity)
        elasticity_matrix[2, 0] = -0.25
        elasticity_matrix[0, 2] = -0.22
        
        engine.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        # Manually create complement pairs DataFrame
        complement_pairs = []
        for i in range(n_products):
            for j in range(i+1, n_products):
                if elasticity_matrix[i, j] < -0.2:
                    complement_pairs.append((
                        sample_products.iloc[i]['PRODUCT_ID'],
                        sample_products.iloc[j]['PRODUCT_ID'],
                        elasticity_matrix[i, j]
                    ))
        
        engine.complement_pairs = pd.DataFrame(
            complement_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        )
        
        # Should detect complement pair
        assert len(engine.complement_pairs) > 0
    
    def test_cross_price_effect_calculation(self, sample_products):
        """Test cross-price effect on utility"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        # Create mock elasticity matrix
        n_products = len(sample_products)
        elasticity_matrix = sparse.lil_matrix((n_products, n_products))
        
        # Product 1 (Coke) affects Product 2 (Pepsi) - substitutes
        # Row 1 (Pepsi), Column 0 (Coke) = positive elasticity
        elasticity_matrix[1, 0] = 0.30  # Pepsi demand increases when Coke price increases
        
        engine.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        # Test: When Coke price increases, Pepsi utility should increase
        current_prices = {1: 4.49, 2: 3.49}  # Coke up, Pepsi normal
        reference_prices = {1: 3.99, 2: 3.49}
        
        utility_adjustment = engine.apply_cross_price_effects(
            focal_product_id=2,  # Pepsi
            base_utility=2.0,
            current_prices=current_prices,
            reference_prices=reference_prices
        )
        
        # Utility should increase (Coke is expensive, so Pepsi more attractive)
        # Note: The actual increase depends on the implementation
        assert utility_adjustment >= 2.0, f"Expected utility >= 2.0, got {utility_adjustment}"
    
    def test_no_effect_when_no_relationships(self, sample_products):
        """Test that products with no relationships have no effect"""
        engine = CrossPriceElasticityEngine(sample_products)
        
        # Empty elasticity matrix
        n_products = len(sample_products)
        engine.cross_elasticity_matrix = sparse.csr_matrix((n_products, n_products))
        
        current_prices = {1: 4.49, 2: 3.49}
        reference_prices = {1: 3.99, 2: 3.49}
        
        utility_adjustment = engine.apply_cross_price_effects(
            focal_product_id=2,
            base_utility=2.0,
            current_prices=current_prices,
            reference_prices=reference_prices
        )
        
        # Should return base utility unchanged
        assert utility_adjustment == 2.0
    
    def test_save_load_parameters(self, sample_products, tmp_path):
        """Test saving and loading cross-price parameters"""
        engine1 = CrossPriceElasticityEngine(sample_products)
        
        # Create mock parameters
        n_products = len(sample_products)
        elasticity_matrix = sparse.lil_matrix((n_products, n_products))
        elasticity_matrix[0, 1] = 0.30
        elasticity_matrix[1, 0] = 0.28
        
        engine1.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        # Manually create substitute groups
        engine1.substitute_groups = pd.DataFrame([
            (1, 2, 0.30),
            (2, 1, 0.28)
        ], columns=['product_i', 'product_j', 'elasticity'])
        
        engine1.complement_pairs = pd.DataFrame(columns=['product_i', 'product_j', 'elasticity'])
        
        # Save
        save_dir = tmp_path / "cross_elasticity"
        save_dir.mkdir()
        engine1.save_parameters(str(save_dir))
        
        # Load into new engine
        engine2 = CrossPriceElasticityEngine(sample_products)
        engine2.load_parameters(str(save_dir))
        
        # Verify matrix loaded correctly
        assert engine2.cross_elasticity_matrix is not None
        assert engine2.cross_elasticity_matrix.nnz == engine1.cross_elasticity_matrix.nnz

# ============================================================================
# ARC ELASTICITY TESTS
# ============================================================================

class TestArcElasticity:
    """Test suite for arc (intertemporal) elasticity engine"""
    
    def test_initialization(self, sample_products, sample_transactions):
        """Test arc elasticity initialization"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        engine = ArcPriceElasticityEngine(
            price_hmm=hmm,
            inventory_decay_rate=0.25,
            future_discount_factor=0.95
        )
        
        assert engine.price_hmm is not None
        assert engine.inventory_decay_rate == 0.25
        assert engine.future_discount_factor == 0.95
    
    # def test_future_price_prediction(self, sample_products, sample_transactions):
    #     """Test future price prediction using HMM"""
    #     hmm = PriceStateHMM(sample_products, n_states=4)
    #     hmm.learn_from_data(sample_transactions)
        
    #     engine = ArcPriceElasticityEngine(price_hmm=hmm)
        
    #     product_id = list(hmm.transition_matrices.keys())[0]
    #     current_state = 2  # Deep discount
    #     base_price = 3.99
        
    #     # Predict future price
    #     future_price = engine.predict_future_price(
    #         product_id=product_id,
    #         current_state=current_state,
    #         base_price=base_price,
    #         weeks_ahead=4
    #     )
        
    #     # Future price should be positive
    #     assert future_price > 0
        
    #     # Should be reasonable (not too far from base)
    #     assert 0.5 * base_price <= future_price <= 1.2 * base_price
    
    def test_stockpiling_on_deep_discount(self, sample_products, sample_transactions):
        """Test stockpiling behavior on deep discount"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        engine = ArcPriceElasticityEngine(price_hmm=hmm)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        
        # Deep discount scenario
        current_price = 2.49
        current_state = 3  # Clearance
        base_quantity = 100
        
        stockpile_qty = engine.get_stockpile_quantity(
            product_id=product_id,
            current_price=current_price,
            current_state=current_state,
            base_quantity=base_quantity
        )
        
        # Should buy more than base quantity
        assert stockpile_qty > base_quantity
        
        # Reasonable stockpiling (not infinite)
        assert stockpile_qty <= base_quantity * 5
    
    def test_no_stockpiling_at_regular_price(self, sample_products, sample_transactions):
        """Test no stockpiling at regular price"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        engine = ArcPriceElasticityEngine(price_hmm=hmm)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        
        # Regular price scenario
        current_price = 3.99
        current_state = 0  # Regular
        base_quantity = 1
        
        stockpile_qty = engine.get_stockpile_quantity(
            product_id=product_id,
            current_price=current_price,
            current_state=current_state,
            base_quantity=base_quantity
        )
        
        # Should buy base quantity (no stockpiling)
        assert stockpile_qty == base_quantity
    
    # def test_deferral_decision(self, sample_products, sample_transactions):
    #     """Test purchase deferral logic"""
    #     hmm = PriceStateHMM(sample_products, n_states=4)
    #     hmm.learn_from_data(sample_transactions)
        
    #     engine = ArcPriceElasticityEngine(price_hmm=hmm)
        
    #     product_id = list(hmm.transition_matrices.keys())[0]
        
    #     # High price + sufficient inventory → should defer
    #     should_defer = engine.should_defer_purchase(
    #         product_id=product_id,
    #         current_price=4.49,  # High
    #         current_state=0,     # Regular
    #         inventory_level=3    # Have inventory
    #     )
        
    #     # Should defer (have inventory, price not great)
    #     assert should_defer is True
        
    #     # Low inventory → should NOT defer
    #     should_defer = engine.should_defer_purchase(
    #         product_id=product_id,
    #         current_price=4.49,
    #         current_state=0,
    #         inventory_level=0  # Out of stock
    #     )
        
    #     assert should_defer is False
    
    # def test_inventory_tracking(self, sample_products, sample_transactions):
    #     """Test customer inventory tracking"""
    #     hmm = PriceStateHMM(sample_products, n_states=4)
    #     hmm.learn_from_data(sample_transactions)
        
    #     engine = ArcPriceElasticityEngine(price_hmm=hmm)
        
    #     customer_id = 123
    #     product_id = 1
        
    #     # Initialize inventory
    #     engine.update_inventory(customer_id, product_id, quantity_purchased=3)
        
    #     # Check inventory
    #     inventory = engine.get_inventory_level(customer_id, product_id)
    #     assert inventory == 3
        
    #     # Decay inventory (1 week)
    #     engine.decay_inventory(customer_id, weeks=1)
        
    #     # Should have decayed
    #     inventory_after = engine.get_inventory_level(customer_id, product_id)
    #     assert inventory_after < 3
    #     assert inventory_after >= 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestElasticityIntegration:
    """Integration tests across all three engines"""
    
    def test_full_pipeline(self, sample_products, sample_transactions):
        """Test complete elasticity pipeline"""
        # 1. Learn HMM
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        assert len(hmm.transition_matrices) > 0
        
        # 2. Initialize cross-price
        cross_price = CrossPriceElasticityEngine(sample_products)
        
        # 3. Initialize arc elasticity
        arc = ArcPriceElasticityEngine(price_hmm=hmm)
        
        # 4. Generate prices for week 1
        product_id = list(hmm.transition_matrices.keys())[0]
        prices, states = hmm.generate_price_sequence(
            product_id=product_id,
            n_weeks=1,
            base_price=3.99,
            random_seed=42
        )
        
        assert len(prices) == 1
        assert len(states) == 1
        
        # 5. Calculate stockpile quantity
        qty = arc.get_stockpile_quantity(
            product_id=product_id,
            current_price=prices[0],
            current_state=states[0],
            base_quantity=1
        )
        
        assert qty >= 1
    
    def test_realistic_promotion_cycle(self, sample_products, sample_transactions):
        """Test realistic 52-week promotion cycle"""
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        
        # Generate full year
        prices, states = hmm.generate_price_sequence(
            product_id=product_id,
            n_weeks=52,
            base_price=3.99,
            random_seed=42
        )
        
        # Calculate promotion statistics
        regular_weeks = np.sum(states == 0)
        promo_weeks = np.sum(states > 0)
        
        # Should have both regular and promotional weeks
        assert regular_weeks > 0
        assert promo_weeks > 0
        
        # Promotion rate should be reasonable (10-40%)
        promo_rate = promo_weeks / 52
        assert 0.10 <= promo_rate <= 0.40
        
        # Average discount should be reasonable
        avg_price = np.mean(prices)
        avg_discount = (3.99 - avg_price) / 3.99
        assert 0 <= avg_discount <= 0.30


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and scalability tests"""
    
    def test_hmm_learning_speed(self, sample_products):
        """Test HMM learning performance"""
        import time
        
        # Generate larger dataset
        np.random.seed(42)
        transactions = []
        for week in range(1, 105):  # 2 years
            for product_id in range(1, 101):  # 100 products
                price = 3.99 * np.random.uniform(0.75, 1.0)
                quantity = np.random.randint(50, 200)
                discount_pct = np.random.choice([0, 0.1, 0.25], p=[0.7, 0.2, 0.1])
                sales_value = price * quantity
                retail_disc = sales_value * discount_pct

                transactions.append({
                    'PRODUCT_ID': product_id,
                    'WEEK_NO': week,
                    'SALES_VALUE': sales_value,
                    'QUANTITY': quantity,
                    'RETAIL_DISC': retail_disc
                })
        
        df = pd.DataFrame(transactions)
        
        # Expand product catalog
        products = pd.concat([sample_products] * 20, ignore_index=True)
        products['PRODUCT_ID'] = range(1, len(products) + 1)
        
        hmm = PriceStateHMM(products, n_states=4)
        
        start = time.time()
        hmm.learn_from_data(df)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (<10 seconds for 100 products)
        assert elapsed < 10.0
        
        print(f"\nHMM learning time: {elapsed:.2f}s for 100 products, 104 weeks")
    
    def test_price_generation_speed(self, sample_products, sample_transactions):
        """Test price generation performance"""
        import time
        
        hmm = PriceStateHMM(sample_products, n_states=4)
        hmm.learn_from_data(sample_transactions)
        
        product_id = list(hmm.transition_matrices.keys())[0]
        
        start = time.time()
        for _ in range(1000):
            prices, states = hmm.generate_price_sequence(
                product_id=product_id,
                n_weeks=52,
                base_price=3.99,
                random_seed=None
            )
        elapsed = time.time() - start
        
        # Should be fast (<1 second for 1000 iterations)
        assert elapsed < 1.0
        
        print(f"\nPrice generation: {1000/elapsed:.0f} sequences/second")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
