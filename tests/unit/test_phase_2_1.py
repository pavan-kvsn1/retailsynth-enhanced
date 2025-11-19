"""
Unit tests for Sprint 2.1: Pricing-Promo Separation
Tests for PricingEvolutionEngine and PromotionalEngine
"""

import numpy as np
import pytest
from retailsynth.engines.pricing_engine import PricingEvolutionEngine
from retailsynth.engines.promotional_engine import PromotionalEngine, StorePromoContext


class TestPricingEvolutionEngine:
    """Tests for refactored PricingEvolutionEngine (base prices only)"""
    
    def test_initialization(self):
        """Test engine initializes correctly"""
        engine = PricingEvolutionEngine(n_products=100)
        
        assert engine.n_products == 100
        assert engine.inflation_rate == 0.0005
        assert engine.competitive_pressure == 0.001
        assert engine.min_price == 0.50
    
    def test_initialization_with_config(self):
        """Test engine initializes with custom config"""
        config = {
            'inflation_rate': 0.001,
            'competitive_pressure': 0.002,
            'price_volatility': 0.05
        }
        engine = PricingEvolutionEngine(n_products=100, config=config)
        
        assert engine.inflation_rate == 0.001
        assert engine.competitive_pressure == 0.002
        assert engine.price_volatility == 0.05
    
    def test_evolve_prices_returns_correct_shape(self):
        """Test evolve_prices returns array of correct shape"""
        engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 5.0
        
        current_prices = engine.evolve_prices(base_prices, week_number=1)
        
        assert isinstance(current_prices, np.ndarray)
        assert len(current_prices) == 100
        assert current_prices.dtype == np.float64
    
    def test_evolve_prices_no_promotions(self):
        """Test that evolve_prices does not return promotion flags"""
        engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 5.0
        
        result = engine.evolve_prices(base_prices, week_number=1)
        
        # Should return only prices (not a tuple)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, tuple)
    
    def test_inflation_effect(self):
        """Test that inflation increases prices over time"""
        engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 5.0
        
        # Disable competitive pressure and volatility for clean test
        engine.competitive_pressure = 0.0
        engine.price_volatility = 0.0
        
        prices_week_1 = engine.evolve_prices(base_prices.copy(), week_number=1)
        prices_week_52 = engine.evolve_prices(base_prices.copy(), week_number=52)
        
        # Prices should increase due to inflation
        assert np.mean(prices_week_52) > np.mean(prices_week_1)
        
        # Calculate expected inflation
        expected_factor = 1.0 + (0.0005 * 52)
        assert abs(np.mean(prices_week_52) / np.mean(base_prices) - expected_factor) < 0.01
    
    def test_competitive_pressure_effect(self):
        """Test that competitive pressure reduces prices over time"""
        engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 5.0
        
        # Disable inflation and volatility for clean test
        engine.inflation_rate = 0.0
        engine.price_volatility = 0.0
        
        prices_week_1 = engine.evolve_prices(base_prices.copy(), week_number=1)
        prices_week_10 = engine.evolve_prices(base_prices.copy(), week_number=10)
        
        # Prices should decrease due to competitive pressure
        assert np.mean(prices_week_10) < np.mean(prices_week_1)
    
    def test_minimum_price_enforced(self):
        """Test that minimum price is enforced"""
        engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 0.30  # Below minimum
        
        current_prices = engine.evolve_prices(base_prices, week_number=1)
        
        assert np.all(current_prices >= engine.min_price)
    
    def test_get_base_price_at_week(self):
        """Test single product price calculation"""
        engine = PricingEvolutionEngine(n_products=100)
        engine.inflation_rate = 0.001
        engine.competitive_pressure = 0.0  # Disable for clean test
        
        initial_price = 5.0
        price_week_10 = engine.get_base_price_at_week(initial_price, week_number=10)
        
        expected_price = 5.0 * (1.0 + 0.001 * 10)
        assert abs(price_week_10 - expected_price) < 0.01
    
    def test_price_dynamics_summary(self):
        """Test price dynamics summary"""
        engine = PricingEvolutionEngine(n_products=100)
        summary = engine.get_price_dynamics_summary(week_number=10)
        
        assert 'week_number' in summary
        assert 'inflation_factor' in summary
        assert 'competitive_factor' in summary
        assert summary['week_number'] == 10
        assert summary['inflation_factor'] > 1.0  # Inflation increases prices


class TestPromotionalEngine:
    """Tests for new PromotionalEngine"""
    
    def test_initialization(self):
        """Test promotional engine initializes correctly"""
        engine = PromotionalEngine()
        
        assert engine.depth_ranges is not None
        assert len(engine.depth_ranges) == 4  # 4 HMM states
        assert engine.display_capacity['end_cap'] == 10
        assert engine.display_capacity['feature_display'] == 3
    
    def test_initialization_with_config(self):
        """Test engine initializes with custom config"""
        config = {
            'promo_frequency': {'min': 0.05, 'max': 0.15}
        }
        engine = PromotionalEngine(config=config)
        
        assert engine.promo_frequency['min'] == 0.05
        assert engine.promo_frequency['max'] == 0.15
    
    def test_generate_store_promotions(self):
        """Test generating promotional context for a store"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        # Check context structure
        assert isinstance(context, StorePromoContext)
        assert context.store_id == 1
        assert context.week_number == 1
        assert len(context.promoted_products) > 0
        assert len(context.promo_depths) > 0
    
    def test_promo_frequency_in_range(self):
        """Test that promotion frequency is within configured range"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        promo_percentage = len(context.promoted_products) / 100
        assert 0.10 <= promo_percentage <= 0.30
    
    def test_discount_depths_valid(self):
        """Test that discount depths are valid percentages"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        for discount in context.promo_depths.values():
            assert 0.0 <= discount <= 0.70
    
    def test_display_allocation(self):
        """Test that display types are allocated correctly"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        # Check display capacity constraints
        assert len(context.end_cap_products) <= 10
        assert len(context.feature_display_products) <= 3
        
        # Check that display types are valid
        for display_type in context.display_types.values():
            assert display_type in ['none', 'shelf_tag', 'end_cap', 'feature_display']
    
    def test_feature_advertising(self):
        """Test that feature advertising is assigned"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        # In-ad and mailer products should be subsets of promoted products
        for product_id in context.in_ad_products:
            assert product_id in context.promoted_products
        
        for product_id in context.mailer_products:
            assert product_id in context.promoted_products
    
    def test_get_promotional_price(self):
        """Test calculating promotional price"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        # Test promoted product
        if context.promoted_products:
            product_id = context.promoted_products[0]
            base_price = 5.0
            discount = context.promo_depths[product_id]
            
            promo_price = engine.get_promotional_price(product_id, base_price, context)
            
            expected_price = base_price * (1.0 - discount)
            assert abs(promo_price - expected_price) < 0.01
        
        # Test non-promoted product
        non_promo_id = 9999
        promo_price = engine.get_promotional_price(non_promo_id, 5.0, context)
        assert promo_price == 5.0  # No discount
    
    def test_promo_summary(self):
        """Test promotional summary statistics"""
        engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        summary = engine.get_promo_summary(context)
        
        assert 'store_id' in summary
        assert 'n_promotions' in summary
        assert 'avg_discount' in summary
        assert summary['n_promotions'] == len(context.promoted_products)


class TestPricingPromoSeparation:
    """Integration tests to verify pricing and promo are truly separate"""
    
    def test_pricing_engine_independent(self):
        """Test that pricing engine works without promotional engine"""
        pricing_engine = PricingEvolutionEngine(n_products=100)
        base_prices = np.ones(100) * 5.0
        
        # Should work fine without promotional engine
        current_prices = pricing_engine.evolve_prices(base_prices, week_number=1)
        
        assert current_prices is not None
        assert len(current_prices) == 100
    
    def test_promo_engine_independent(self):
        """Test that promotional engine works without pricing engine"""
        promo_engine = PromotionalEngine()
        
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        # Should work fine without pricing engine
        context = promo_engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        
        assert context is not None
        assert len(context.promoted_products) > 0
    
    def test_combined_flow(self):
        """Test pricing and promo engines work together"""
        pricing_engine = PricingEvolutionEngine(n_products=100)
        promo_engine = PromotionalEngine()
        
        # Initial prices
        base_prices = np.ones(100) * 5.0
        product_ids = np.arange(100)
        
        # Step 1: Get base prices
        current_base_prices = pricing_engine.evolve_prices(base_prices, week_number=1)
        
        # Step 2: Generate promotions
        promo_context = promo_engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=current_base_prices,
            product_ids=product_ids
        )
        
        # Step 3: Apply promotions
        final_prices = []
        for i, product_id in enumerate(product_ids):
            base_price = current_base_prices[i]
            final_price = promo_engine.get_promotional_price(
                product_id, base_price, promo_context
            )
            final_prices.append(final_price)
        
        final_prices = np.array(final_prices)
        
        # Verify
        assert len(final_prices) == 100
        
        # Some products should have discounts
        n_discounted = np.sum(final_prices < current_base_prices - 0.01)
        assert n_discounted == len(promo_context.promoted_products)
        
        # Promoted products should have lower prices
        for product_id in promo_context.promoted_products:
            idx = np.where(product_ids == product_id)[0][0]
            assert final_prices[idx] < current_base_prices[idx]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
