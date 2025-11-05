import numpy as np
import pandas as pd
from retailsynth.calibration import CalibrationEngine
from retailsynth.utils import RealisticCategoryHierarchy
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig

# ============================================================================
# CUSTOMER GENERATOR (v3.2)
# ============================================================================

class CustomerGenerator:
    """
    Generates customer profiles with realistic demographics and purchase patterns.
    """
    
    @staticmethod
    def generate_customers_vectorized(config: EnhancedRetailConfig, 
                                     calibration_engine: CalibrationEngine) -> pd.DataFrame:
        """
        Generate customers with vectorized operations.
        Only unavoidable loops (utility param sampling) remain.
        """
        print(f"      Generating {config.n_customers:,} customers (vectorized)...")
        start_time = datetime.now()
        
        n = config.n_customers
        
        # Vectorized generation of basic attributes
        ages = np.random.choice([25, 35, 45, 55, 65], size=n, p=[0.2, 0.25, 0.25, 0.2, 0.1])
        household_sizes = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.28, 0.35, 0.16, 0.15, 0.06])
        personalities = np.random.choice(
            ['price_anchor', 'convenience', 'planned', 'impulse'],
            size=n,
            p=[config.price_anchor_customers, config.convenience_customers,
               config.planned_customers, config.impulse_customers]
        )
        
        # Vectorized income brackets (age-dependent)
        income_brackets = np.empty(n, dtype=object)
        young_mask = ages <= 35
        middle_mask = (ages > 35) & (ages <= 55)
        senior_mask = ages > 55
        
        income_brackets[young_mask] = np.random.choice(
            ['<30K', '30-50K', '50-75K'], 
            size=young_mask.sum(), 
            p=[0.35, 0.4, 0.25]
        )
        income_brackets[middle_mask] = np.random.choice(
            ['30-50K', '50-75K', '75-100K', '>100K'],
            size=middle_mask.sum(),
            p=[0.25, 0.35, 0.25, 0.15]
        )
        income_brackets[senior_mask] = np.random.choice(
            ['30-50K', '50-75K', '75-100K'],
            size=senior_mask.sum(),
            p=[0.4, 0.4, 0.2]
        )
        
        # Vectorized price sensitivity
        price_sensitivity = np.where(
            (np.isin(income_brackets, ['<30K', '30-50K'])) | (personalities == 'price_anchor'),
            'high',
            np.where(personalities == 'convenience', 'low', 'medium')
        )
        
        # Vectorized marital status
        marital_status = np.where(
            (ages < 30) | (np.random.random(n) < 0.3),
            'Single',
            'Married'
        )
        
        # Vectorized children count
        children_count = np.where(
            (ages < 25) | (marital_status == 'Single'),
            0,
            np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.35, 0.25, 0.1])
        )
        
        # Build customer list (only loop needed for utility params and brand prefs)
        all_brands = RealisticCategoryHierarchy.get_all_brands()
        customers = []
        
        for i in range(n):
            if (i + 1) % 10000 == 0:
                print(f"         {i+1:,}/{n:,} customers...", end='\r')
            
            customer_id = 100000 + i
            age = int(ages[i])
            household_size = int(household_sizes[i])
            shopping_personality = personalities[i]
            income_bracket = income_brackets[i]
            
            # Calculate modern attributes
            mobile_usage = CustomerGenerator._calc_mobile(age, income_bracket)
            sustainability = CustomerGenerator._calc_sustainability(age, income_bracket)
            
            # Sample utility parameters (unavoidable - needs calibration engine)
            customer_demographics = {
                'age_group': f"{age}-{age+9}",
                'income_bracket': income_bracket,
                'household_size': household_size,
                'shopping_personality': shopping_personality,
                'price_sensitivity': price_sensitivity[i]
            }
            utility_params = calibration_engine.sample_utility_parameters(customer_demographics)
            
            # Brand preferences
            brand_preferences = {}
            n_preferred_brands = np.random.choice([3, 4, 5, 6, 7], p=[0.15, 0.25, 0.30, 0.20, 0.10])
            preferred_brands = np.random.choice(all_brands, size=n_preferred_brands, replace=False)
            
            for brand in preferred_brands:
                if shopping_personality == 'convenience':
                    pref = np.random.uniform(0.6, 0.9)
                elif shopping_personality == 'price_anchor':
                    pref = np.random.uniform(0.2, 0.5)
                else:
                    pref = np.random.uniform(0.4, 0.7)
                brand_preferences[brand] = round(pref, 3)
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'household_size': household_size,
                'marital_status': marital_status[i],
                'children_count': int(children_count[i]),
                'income_bracket': income_bracket,
                'shopping_personality': shopping_personality,
                'price_sensitivity': price_sensitivity[i],
                'store_loyalty_level': round(np.random.beta(2, 3), 3),
                'days_since_last_visit': int(np.random.exponential(7)),
                'mobile_usage_propensity': round(mobile_usage, 3),
                'sustainability_preference': round(sustainability, 3),
                'utility_params': utility_params,
                'brand_preferences': brand_preferences,
                'created_at': datetime.now()
            })
        
        print(f"\n         ✅ Generated in {(datetime.now() - start_time).total_seconds():.1f}s")
        return pd.DataFrame(customers)
    
    @staticmethod
    def _calc_mobile(age: int, income: str) -> float:
        """Calculate mobile usage propensity"""
        base = 0.5
        if age < 35:
            base += 0.3
        elif age > 55:
            base -= 0.2
        if income in ['>100K', '75-100K']:
            base += 0.15
        return np.clip(base + np.random.uniform(-0.1, 0.1), 0, 1)
    
    @staticmethod
    def _calc_sustainability(age: int, income: str) -> float:
        """Calculate sustainability preference"""
        base = 0.4
        if age < 40:
            base += 0.25
        if income in ['>100K', '75-100K']:
            base += 0.15
        return np.clip(base + np.random.uniform(-0.15, 0.15), 0, 1)
    