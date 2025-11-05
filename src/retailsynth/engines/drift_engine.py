import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

# ============================================================================
# TEMPORAL CUSTOMER DRIFT ENGINE (v3.6)
# ============================================================================

class TemporalCustomerDriftEngine:
    """
    Manages customer evolution over time with realistic demographic changes (v3.6).
    Includes life events, income mobility, and behavioral drift.
    """
    
    def __init__(self, customers_df: pd.DataFrame):
        print("   🔧 Initializing temporal customer drift engine...")
        self.customers = customers_df.copy()
        self.n_customers = len(customers_df)
        
        # Track customer state changes
        self.drift_history = []
        self.life_events_log = []
    
    def apply_weekly_drift(self, week_number: int) -> pd.DataFrame:
        """
        Apply realistic drift to all customers.
        Returns updated customers DataFrame.
        """
        # 1. Gradual behavioral drift (everyone)
        self._apply_gradual_drift(week_number)
        
        # 2. Life events (random subset)
        if week_number % 4 == 0:  # Check every 4 weeks
            self._apply_life_events(week_number)
        
        # 3. Economic shocks (rare)
        if np.random.random() < 0.02:  # 2% chance per week
            self._apply_economic_shock(week_number)
        
        return self.customers
    
    def _apply_gradual_drift(self, week_number: int):
        """Apply gradual drift to utility parameters and behaviors"""
        
        # Price sensitivity drift (economic pressure increasing)
        price_drift_rate = 0.0005  # Small weekly change
        for i in range(self.n_customers):
            utility_params = self.customers.iloc[i]['utility_params']
            current_beta = utility_params['beta_price']
            # Trend toward more price-sensitive (more negative)
            drift = np.random.normal(-price_drift_rate, price_drift_rate * 0.5)
            new_beta = current_beta * (1 + drift)
            utility_params['beta_price'] = new_beta
        
        # Sustainability preference growth (societal trend)
        sustainability_growth = 0.001  # 0.1% per week = 5% per year
        self.customers['sustainability_preference'] = np.minimum(
            0.95,
            self.customers['sustainability_preference'] * (1 + sustainability_growth)
        )
        
        # Mobile usage increase (technology adoption)
        mobile_growth = 0.002  # Faster tech adoption
        self.customers['mobile_usage_propensity'] = np.minimum(
            0.98,
            self.customers['mobile_usage_propensity'] * (1 + mobile_growth)
        )
        
        # Brand loyalty erosion (market fragmentation)
        loyalty_erosion = 0.0003
        for i in range(self.n_customers):
            utility_params = self.customers.iloc[i]['utility_params']
            current_beta = utility_params['beta_brand']
            new_beta = current_beta * (1 - loyalty_erosion)
            utility_params['beta_brand'] = new_beta
    
    def _apply_life_events(self, week_number: int):
        """Apply major life events to random subset of customers"""
        
        # 5% of customers experience life events per year = ~0.1% per week
        n_events = int(self.n_customers * 0.001)
        event_customers = np.random.choice(self.n_customers, size=n_events, replace=False)
        
        for customer_idx in event_customers:
            event_type = np.random.choice([
                'marriage', 'child_birth', 'job_change', 'relocation', 'major_purchase'
            ], p=[0.2, 0.2, 0.3, 0.2, 0.1])
            
            if event_type == 'marriage':
                self._apply_marriage(customer_idx, week_number)
            elif event_type == 'child_birth':
                self._apply_child_birth(customer_idx, week_number)
            elif event_type == 'job_change':
                self._apply_job_change(customer_idx, week_number)
            elif event_type == 'relocation':
                self._apply_relocation(customer_idx, week_number)
            elif event_type == 'major_purchase':
                self._apply_major_purchase(customer_idx, week_number)
            
            # Log event
            self.life_events_log.append({
                'week': week_number,
                'customer_id': self.customers.iloc[customer_idx]['customer_id'],
                'event_type': event_type
            })
    
    def _apply_marriage(self, customer_idx: int, week_number: int):
        """Customer gets married - increases household size, changes shopping"""
        if self.customers.iloc[customer_idx]['marital_status'] == 'Single':
            self.customers.at[customer_idx, 'marital_status'] = 'Married'
            self.customers.at[customer_idx, 'household_size'] = min(
                5, self.customers.iloc[customer_idx]['household_size'] + 1
            )
            # More planned shopping
            if np.random.random() < 0.6:
                self.customers.at[customer_idx, 'shopping_personality'] = 'planned'
    
    def _apply_child_birth(self, customer_idx: int, week_number: int):
        """Customer has child - major shopping behavior shift"""
        self.customers.at[customer_idx, 'children_count'] = (
            self.customers.iloc[customer_idx]['children_count'] + 1
        )
        self.customers.at[customer_idx, 'household_size'] = min(
            5, self.customers.iloc[customer_idx]['household_size'] + 1
        )
        # More price-sensitive, more convenience-oriented
        utility_params = self.customers.iloc[customer_idx]['utility_params']
        current_beta = utility_params['beta_price']
        utility_params['beta_price'] = current_beta * 1.2
        
        # Might shift to convenience shopping
        if np.random.random() < 0.4:
            self.customers.at[customer_idx, 'shopping_personality'] = 'convenience'
    
    def _apply_job_change(self, customer_idx: int, week_number: int):
        """Customer changes job - income and time constraints change"""
        current_income = self.customers.iloc[customer_idx]['income_bracket']
        
        # Income mobility (50% up, 30% same, 20% down)
        direction = np.random.choice(['up', 'same', 'down'], p=[0.5, 0.3, 0.2])
        
        income_ladder = ['<30K', '30-50K', '50-75K', '75-100K', '>100K']
        current_idx = income_ladder.index(current_income) if current_income in income_ladder else 2
        
        utility_params = self.customers.iloc[customer_idx]['utility_params']
        
        if direction == 'up' and current_idx < len(income_ladder) - 1:
            new_income = income_ladder[current_idx + 1]
            self.customers.at[customer_idx, 'income_bracket'] = new_income
            # Less price-sensitive
            current_beta = utility_params['beta_price']
            utility_params['beta_price'] = current_beta * 0.9
        elif direction == 'down' and current_idx > 0:
            new_income = income_ladder[current_idx - 1]
            self.customers.at[customer_idx, 'income_bracket'] = new_income
            # More price-sensitive
            current_beta = utility_params['beta_price']
            utility_params['beta_price'] = current_beta * 1.15
    
    def _apply_relocation(self, customer_idx: int, week_number: int):
        """Customer relocates - resets store loyalty"""
        self.customers.at[customer_idx, 'store_loyalty_level'] = 0.2
        self.customers.at[customer_idx, 'days_since_last_visit'] = 1
    
    def _apply_major_purchase(self, customer_idx: int, week_number: int):
        """Major purchase (house, car) - temporary price sensitivity spike"""
        utility_params = self.customers.iloc[customer_idx]['utility_params']
        current_beta = utility_params['beta_price']
        utility_params['beta_price'] = current_beta * 1.25
    
    def _apply_economic_shock(self, week_number: int):
        """Economic shock affects subset of population"""
        # 20% of population affected
        n_affected = int(self.n_customers * 0.2)
        affected_customers = np.random.choice(self.n_customers, size=n_affected, replace=False)
        
        print(f"   💥 Economic shock in week {week_number}: {n_affected:,} customers affected")
        
        for customer_idx in affected_customers:
            # Increase price sensitivity
            utility_params = self.customers.iloc[customer_idx]['utility_params']
            current_beta = utility_params['beta_price']
            utility_params['beta_price'] = current_beta * 1.3
            
            # Shift to price-anchor personality
            if np.random.random() < 0.4:
                self.customers.at[customer_idx, 'shopping_personality'] = 'price_anchor'
