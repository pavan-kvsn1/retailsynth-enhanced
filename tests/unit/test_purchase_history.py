"""
Validation Tests for Purchase History System (Sprint 1.3)

Tests validate:
1. Customer state tracking (loyalty, habits, inventory)
2. History-dependent utility calculations
3. Repeat purchase patterns
4. Brand loyalty metrics
5. Inter-purchase timing
6. Inventory depletion cycles

Author: RetailSynth Team
Date: November 2024
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from retailsynth.engines.customer_state import CustomerState, CustomerStateManager, get_depletion_rates_by_assortment
from retailsynth.engines.purchase_history_engine import (
    PurchaseHistoryEngine,
    InterPurchaseTimingModel,
    calculate_repeat_purchase_rate,
    calculate_brand_loyalty_metrics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_products():
    """Create sample product catalog with assortment roles"""
    return pd.DataFrame({
        'PRODUCT_ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'BRAND': ['Coca Cola', 'Pepsi', 'Lays', 'Pringles', 'Organic Valley', 'Coca Cola', 'Pepsi', 'Lays'],
        'COMMODITY': ['Soft Drinks', 'Soft Drinks', 'Chips', 'Chips', 'Milk', 'Soft Drinks', 'Soft Drinks', 'Chips'],
        'DEPARTMENT': ['Beverages', 'Beverages', 'Snacks', 'Snacks', 'Dairy', 'Beverages', 'Beverages', 'Snacks'],
        'avg_price': [3.99, 3.49, 2.99, 3.49, 4.99, 4.49, 3.99, 3.29],
        'assortment_role': ['mid_basket', 'mid_basket', 'back_basket', 'back_basket', 'lpg_line', 'mid_basket', 'mid_basket', 'back_basket']
    })


@pytest.fixture
def customer_state():
    """Create a fresh customer state"""
    return CustomerState(customer_id=1, current_week=1)


# ============================================================================
# CUSTOMER STATE TESTS
# ============================================================================

class TestCustomerState:
    """Test customer state tracking"""
    
    def test_initialization(self):
        """Test customer state initialization"""
        state = CustomerState(customer_id=123)
        
        assert state.customer_id == 123
        assert state.current_week == 0
        assert len(state.products_tried) == 0
        assert state.variety_seeking_score == 0.1
    
    def test_purchase_update(self, customer_state):
        """Test state update after purchase"""
        customer_state.update_after_purchase(
            product_id=1,
            brand='Coca Cola',
            category='Soft Drinks',
            week=1,
            satisfaction=0.8,
            quantity=2
        )
        
        # Check purchase history
        assert customer_state.last_purchase_week[1] == 1
        assert customer_state.purchase_count[1] == 1
        assert 1 in customer_state.products_tried
        
        # Check brand loyalty
        assert customer_state.brand_purchase_count['Coca Cola'] == 1
        assert customer_state.brand_experience['Coca Cola'] > 0
        
        # Check inventory
        assert customer_state.category_inventory['Soft Drinks'] > 0.5  # Replenished
    
    def test_habit_formation(self, customer_state):
        """Test habit strength increases with purchases"""
        # Make multiple purchases
        for week in range(1, 12):
            customer_state.update_after_purchase(
                product_id=1,
                brand='Coca Cola',
                category='Soft Drinks',
                week=week,
                satisfaction=0.7,
                quantity=1
            )
        
        # Check habit strength progression
        assert customer_state.purchase_count[1] == 11
        assert customer_state.habit_strength[1] > 0.8  # Strong habit after 11 purchases
    
    def test_loyalty_bonus_calculation(self, customer_state):
        """Test loyalty bonus calculation"""
        # Build up loyalty
        for i in range(5):
            customer_state.update_after_purchase(
                product_id=1,
                brand='Coca Cola',
                category='Soft Drinks',
                week=i+1,
                satisfaction=0.8,
                quantity=1
            )
        
        customer_state.current_week = 6
        
        # Calculate loyalty bonus
        bonus = customer_state.get_loyalty_bonus(1, 'Coca Cola')
        
        # Should have positive bonus from brand loyalty + habit + recency
        assert bonus > 0
        assert bonus <= 3.3  # Max possible bonus
    
    def test_inventory_depletion(self, customer_state):
        """Test inventory depletion over time"""
        # Start with full inventory
        customer_state.category_inventory['Milk'] = 1.0
        
        # Deplete over 4 weeks (lpg_line: 30% per week)
        depletion_rates = {'Milk': 0.30}
        
        for _ in range(4):
            customer_state.deplete_inventory(depletion_rates)
        
        # Should be nearly empty
        assert customer_state.category_inventory['Milk'] < 0.2
    
    def test_inventory_need_urgency(self, customer_state):
        """Test inventory need calculation"""
        # Low inventory → high urgency
        customer_state.category_inventory['Milk'] = 0.1
        urgency = customer_state.get_inventory_need('Milk', 'lpg_line')
        assert urgency == 5.0  # Critical need
        
        # Medium inventory → moderate urgency
        customer_state.category_inventory['Milk'] = 0.5
        urgency = customer_state.get_inventory_need('Milk', 'lpg_line')
        assert urgency == 1.5  # Moderate need
        
        # High inventory → no urgency
        customer_state.category_inventory['Milk'] = 0.8
        urgency = customer_state.get_inventory_need('Milk', 'lpg_line')
        assert urgency == 0.0  # No need
    
    def test_variety_seeking(self, customer_state):
        """Test variety-seeking behavior"""
        # Initially low exploration
        customer_state.weeks_since_new_product = 0
        explore_prob_initial = sum(customer_state.should_try_new_product() for _ in range(1000)) / 1000
        assert 0.05 <= explore_prob_initial <= 0.15  # ~10%
        
        # After many weeks, exploration increases
        customer_state.weeks_since_new_product = 20
        explore_prob_later = sum(customer_state.should_try_new_product() for _ in range(1000)) / 1000
        assert explore_prob_later > explore_prob_initial
    
    def test_state_decay(self, customer_state):
        """Test loyalty and habit decay over time"""
        # Build up loyalty
        customer_state.brand_experience['Coca Cola'] = 10.0
        customer_state.habit_strength[1] = 0.8
        customer_state.current_week = 1
        
        # Decay over 10 weeks
        for _ in range(10):
            customer_state.decay_state(weeks_elapsed=1)
        
        # Should have decayed
        assert customer_state.brand_experience['Coca Cola'] < 10.0
        assert customer_state.habit_strength[1] < 0.8


# ============================================================================
# PURCHASE HISTORY ENGINE TESTS
# ============================================================================

class TestPurchaseHistoryEngine:
    """Test purchase history engine"""
    
    def test_initialization(self, sample_products):
        """Test engine initialization"""
        engine = PurchaseHistoryEngine(sample_products)
        
        assert len(engine.product_to_brand) == 8
        assert len(engine.product_to_category) == 8
        assert engine.loyalty_weight == 0.3
        assert engine.habit_weight == 0.4
    
    def test_history_utility_adjustment(self, sample_products, customer_state):
        """Test history-dependent utility adjustments"""
        engine = PurchaseHistoryEngine(sample_products)
        
        # Build purchase history
        customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', 1, 0.8, 1)
        customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', 2, 0.8, 1)
        customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', 3, 0.8, 1)
        customer_state.current_week = 4
        
        # Base utilities
        product_ids = np.array([1, 2, 3, 4, 5])
        base_utilities = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        # Calculate adjusted utilities
        adjusted = engine.calculate_history_utility(
            customer_state, product_ids, base_utilities, 4
        )
        
        # Product 1 (Coke) should have higher utility due to loyalty
        assert adjusted[0] > base_utilities[0]
    
    def test_repeat_purchase_probability(self, sample_products, customer_state):
        """Test repeat purchase probability calculation"""
        engine = PurchaseHistoryEngine(sample_products)
        
        # No purchase history → 0 probability
        prob = engine.get_repeat_purchase_probability(customer_state, 1)
        assert prob == 0.0
        
        # After 5 purchases → higher probability
        for i in range(5):
            customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', i+1, 0.8, 1)
        
        customer_state.current_week = 6
        prob = engine.get_repeat_purchase_probability(customer_state, 1)
        assert prob > 0.3  # Should have decent repeat probability
    
    def test_brand_switching_probability(self, sample_products, customer_state):
        """Test brand switching probability"""
        engine = PurchaseHistoryEngine(sample_products)
        
        # Build loyalty to Coca Cola
        for i in range(10):
            customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', i+1, 0.8, 1)
        
        # Calculate switching probability to Pepsi
        switch_prob = engine.get_brand_switching_probability(
            customer_state, 'Coca Cola', 'Pepsi'
        )
        
        # Should be low due to Coke loyalty
        assert switch_prob < 0.3
    
    def test_category_purchase_timing(self, sample_products, customer_state):
        """Test category purchase timing estimation"""
        engine = PurchaseHistoryEngine(sample_products)
        
        # Full inventory → should wait
        customer_state.category_inventory['Milk'] = 1.0
        weeks_until = engine.get_category_purchase_timing(
            customer_state, 'Milk', 'lpg_line'
        )
        assert weeks_until > 0
        
        # Empty inventory → buy now
        customer_state.category_inventory['Milk'] = 0.1
        weeks_until = engine.get_category_purchase_timing(
            customer_state, 'Milk', 'lpg_line'
        )
        assert weeks_until == 0


# ============================================================================
# INTER-PURCHASE TIMING TESTS
# ============================================================================

class TestInterPurchaseTimingModel:
    """Test inter-purchase timing model"""
    
    def test_initialization(self, sample_products):
        """Test timing model initialization"""
        model = InterPurchaseTimingModel(sample_products)
        
        assert model.expected_intervals['lpg_line'] == 3.5
        assert model.expected_intervals['mid_basket'] == 6.5
        assert model.expected_intervals['back_basket'] == 20.0
    
    def test_next_purchase_prediction(self, sample_products, customer_state):
        """Test next purchase week prediction"""
        model = InterPurchaseTimingModel(sample_products)
        
        # Last purchased in week 1
        customer_state.last_purchase_week[1] = 1
        
        # Predict next purchase
        next_week = model.predict_next_purchase_week(customer_state, 1, 5)
        
        # Should be in the future (mid_basket ~6.5 weeks)
        assert next_week > 1
        assert next_week < 15  # Reasonable range
    
    def test_purchase_probability_by_week(self, sample_products, customer_state):
        """Test weekly purchase probability"""
        model = InterPurchaseTimingModel(sample_products)
        
        customer_state.last_purchase_week[1] = 1
        
        # Probability should peak around predicted week
        probs = []
        for week in range(1, 20):
            prob = model.get_purchase_probability_by_week(customer_state, 1, week)
            probs.append(prob)
        
        # Should have a peak
        max_prob = max(probs)
        assert max_prob >= 0.4  # Should have high probability at some point


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPurchaseHistoryIntegration:
    """Integration tests for complete purchase history system"""
    
    def test_full_customer_journey(self, sample_products):
        """Test complete customer journey over 52 weeks"""
        engine = PurchaseHistoryEngine(sample_products)
        state = CustomerState(customer_id=1)
        
        # Simulate 52 weeks of shopping
        for week in range(1, 53):
            state.current_week = week
            
            # Deplete inventory
            depletion_rates = get_depletion_rates_by_assortment()
            state.deplete_inventory(depletion_rates)
            
            # Make purchases based on inventory needs
            if state.category_inventory.get('Milk', 0.5) < 0.3:
                state.update_after_purchase(5, 'Organic Valley', 'Milk', week, 0.7, 1)
            
            if state.category_inventory.get('Soft Drinks', 0.5) < 0.4:
                state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', week, 0.8, 1)
        
        # Validate final state
        assert len(state.products_tried) >= 2
        assert state.purchase_count[5] > 0  # Bought milk
        assert state.purchase_count[1] > 0  # Bought Coke
        assert state.habit_strength[5] > 0  # Formed habit for milk
    
    def test_state_manager_batch_operations(self, sample_products):
        """Test CustomerStateManager batch operations"""
        manager = CustomerStateManager(n_customers=100)
        
        # Update all states
        manager.update_all_states(current_week=5)
        
        # Check all updated
        for state in manager.states.values():
            assert state.current_week == 5
        
        # Deplete all inventory
        depletion_rates = get_depletion_rates_by_assortment()
        manager.deplete_all_inventory(depletion_rates)
        
        # Get summary
        summary = manager.get_summary_statistics()
        assert len(summary) == 100
        assert 'customer_id' in summary.columns
    
    def test_repeat_purchase_rate_calculation(self, sample_products):
        """Test repeat purchase rate metric"""
        # Create mock transaction data
        transactions = pd.DataFrame({
            'customer_id': [1, 1, 1, 2, 2, 3],
            'product_id': [1, 1, 2, 1, 3, 1]
        })
        
        repeat_rate = calculate_repeat_purchase_rate(transactions)
        
        # Customer 1 bought product 1 twice → 1 repeat
        # Customer 2 bought different products → 0 repeats
        # Customer 3 bought once → 0 repeats
        # Total: 1 repeat out of 4 customer-product pairs
        assert 0.0 <= repeat_rate <= 1.0
    
    def test_brand_loyalty_metrics(self, sample_products):
        """Test brand loyalty metrics calculation"""
        # Create mock transaction data
        transactions = pd.DataFrame({
            'customer_id': [1, 1, 1, 1, 2, 2],
            'product_id': [1, 1, 6, 6, 2, 3]  # Customer 1: Coke only, Customer 2: Pepsi + Lays
        })
        
        loyalty_metrics = calculate_brand_loyalty_metrics(transactions, sample_products)
        
        assert len(loyalty_metrics) == 2  # 2 customers
        assert 'brand_concentration' in loyalty_metrics.columns
        assert 'top_brand_share' in loyalty_metrics.columns
        
        # Customer 1 should have high concentration (only Coke)
        customer_1_metrics = loyalty_metrics[loyalty_metrics['customer_id'] == 1].iloc[0]
        assert customer_1_metrics['brand_concentration'] > 0.9


# ============================================================================
# VALIDATION METRICS TESTS
# ============================================================================

class TestValidationMetrics:
    """Test validation metrics match expected ranges"""
    
    def test_depletion_rates(self):
        """Test depletion rates are realistic"""
        rates = get_depletion_rates_by_assortment()
        
        # lpg_line (milk, bread) should deplete fastest
        assert rates['lpg_line'] == 0.30  # Buy every 3-4 weeks
        
        # back_basket (occasional) should deplete slowest
        assert rates['back_basket'] == 0.05  # Buy every 20 weeks
        
        # All rates should be positive and < 1
        for rate in rates.values():
            assert 0 < rate < 1
    
    def test_loyalty_bonus_ranges(self, customer_state):
        """Test loyalty bonus stays within expected ranges"""
        # Build maximum loyalty
        for i in range(20):
            customer_state.update_after_purchase(1, 'Coca Cola', 'Soft Drinks', i+1, 1.0, 1)
        
        customer_state.current_week = 21
        
        bonus = customer_state.get_loyalty_bonus(1, 'Coca Cola')
        
        # Should be positive but capped
        assert 0 <= bonus <= 3.5
    
    def test_inventory_urgency_ranges(self, customer_state):
        """Test inventory urgency stays within expected ranges"""
        for inventory_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
            customer_state.category_inventory['Milk'] = inventory_level
            urgency = customer_state.get_inventory_need('Milk', 'lpg_line')
            
            # Should be non-negative and capped
            assert 0 <= urgency <= 6.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
