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
        
        # Vectorized generation of basic attributes (NOW FROM CONFIG)
        ages = np.random.choice(config.age_values, size=n, p=config.age_probabilities)
        household_sizes = np.random.choice(config.household_sizes, size=n, p=config.household_size_probs)
        personalities = np.random.choice(
            ['price_anchor', 'convenience', 'planned', 'impulse'],
            size=n,
            p=[config.price_anchor_customers, config.convenience_customers,
               config.planned_customers, config.impulse_customers]
        )
        
        # Vectorized income brackets (age-dependent, NOW FROM CONFIG)
        income_brackets = np.empty(n, dtype=object)
        young_mask = ages <= 35
        middle_mask = (ages > 35) & (ages <= 55)
        senior_mask = ages > 55
        
        income_brackets[young_mask] = np.random.choice(
            ['<30K', '30-50K', '50-75K'], 
            size=young_mask.sum(), 
            p=config.young_income_probs
        )
        income_brackets[middle_mask] = np.random.choice(
            ['30-50K', '50-75K', '75-100K', '>100K'],
            size=middle_mask.sum(),
            p=config.middle_income_probs
        )
        income_brackets[senior_mask] = np.random.choice(
            ['30-50K', '50-75K', '75-100K'],
            size=senior_mask.sum(),
            p=config.senior_income_probs
        )
        
        # Vectorized price sensitivity (NOW USING CONFIG)
        price_sensitivity = np.empty(n, dtype=object)
        for i, (income, personality) in enumerate(zip(income_brackets, personalities)):
            # Use config-based price sensitivity
            income_sens = config.price_sensitivity_by_income.get(income, 0.5)
            personality_sens = config.price_sensitivity_by_personality.get(personality, 0.5)
            # Average the two factors
            combined_sens = (income_sens + personality_sens) / 2
            if combined_sens > 0.6:
                price_sensitivity[i] = 'high'
            elif combined_sens < 0.4:
                price_sensitivity[i] = 'low'
            else:
                price_sensitivity[i] = 'medium'
        
        # Vectorized marital status (NOW USING CONFIG)
        marital_status = np.where(
            (ages < 30) | (np.random.random(n) < config.single_probability),
            'Single',
            'Married'
        )
        
        # Vectorized children count (NOW FROM CONFIG)
        children_count = np.where(
            (ages < 25) | (marital_status == 'Single'),
            0,
            np.random.choice([0, 1, 2, 3], size=n, p=config.children_probs)
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
            
            # Brand preferences (NOW USING CONFIG)
            brand_preferences = {}
            n_preferred_brands = np.random.choice([3, 4, 5, 6, 7], p=[0.15, 0.25, 0.30, 0.20, 0.10])
            preferred_brands = np.random.choice(all_brands, size=n_preferred_brands, replace=False)
            
            # Use config-based brand loyalty
            base_loyalty = config.brand_loyalty_by_personality.get(shopping_personality, config.brand_loyalty_mean)
            loyalty_std = config.brand_loyalty_std
            
            for brand in preferred_brands:
                # Sample from normal distribution around personality-specific mean
                pref = np.random.normal(base_loyalty, loyalty_std)
                pref = np.clip(pref, 0.1, 0.95)  # Clip to valid range
                brand_preferences[brand] = round(pref, 3)
            
            # Trip purpose preferences (NEW - Sprint 1.4)
            trip_preferences = CustomerGenerator._calc_trip_preferences(
                shopping_personality, 
                household_size, 
                age
            )
            
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
                'trip_preferences': trip_preferences,  # NEW - Sprint 1.4
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
    
    @staticmethod
    def _calc_trip_preferences(shopping_personality: str, household_size: int, age: int) -> dict:
        """
        Calculate trip purpose preferences (NEW - Sprint 1.4)
        
        Determines customer's propensity for different shopping trip types:
        - stock_up: Large weekly shopping trips
        - fill_in: Quick top-up trips
        - meal_prep: Recipe-focused shopping
        - convenience: Grab-and-go trips
        - special_occasion: Party/holiday shopping
        
        Args:
            shopping_personality: Customer type (price_anchor, convenience, planned, impulse)
            household_size: Number of people in household
            age: Customer age
        
        Returns:
            Dictionary with trip type preferences and shopping frequency
        """
        # Base preferences by personality type
        if shopping_personality == 'price_anchor':
            # Price-conscious shoppers prefer stock-up trips (bulk buying)
            base_prefs = {
                'stock_up_propensity': 0.65,
                'fill_in_propensity': 0.25,
                'meal_prep_propensity': 0.15,
                'convenience_propensity': 0.05,
                'special_occasion_propensity': 0.05
            }
            trips_per_month = 3.5
        elif shopping_personality == 'convenience':
            # Convenience shoppers prefer frequent small trips
            base_prefs = {
                'stock_up_propensity': 0.15,
                'fill_in_propensity': 0.35,
                'meal_prep_propensity': 0.10,
                'convenience_propensity': 0.40,
                'special_occasion_propensity': 0.05
            }
            trips_per_month = 6.0
        elif shopping_personality == 'planned':
            # Planners prefer structured shopping (stock-up + meal prep)
            base_prefs = {
                'stock_up_propensity': 0.45,
                'fill_in_propensity': 0.20,
                'meal_prep_propensity': 0.30,
                'convenience_propensity': 0.05,
                'special_occasion_propensity': 0.05
            }
            trips_per_month = 4.5
        else:  # impulse
            # Impulse shoppers have varied trip types
            base_prefs = {
                'stock_up_propensity': 0.25,
                'fill_in_propensity': 0.30,
                'meal_prep_propensity': 0.15,
                'convenience_propensity': 0.20,
                'special_occasion_propensity': 0.10
            }
            trips_per_month = 5.0
        
        # Adjust for household size (larger households need more stock-up trips)
        if household_size >= 4:
            base_prefs['stock_up_propensity'] += 0.15
            base_prefs['convenience_propensity'] -= 0.10
            trips_per_month += 1.0
        elif household_size == 1:
            base_prefs['stock_up_propensity'] -= 0.15
            base_prefs['convenience_propensity'] += 0.10
            base_prefs['fill_in_propensity'] += 0.05
            trips_per_month -= 0.5
        
        # Adjust for age (younger people more convenience, older more planned)
        if age < 30:
            base_prefs['convenience_propensity'] += 0.10
            base_prefs['stock_up_propensity'] -= 0.05
        elif age > 60:
            base_prefs['stock_up_propensity'] += 0.05
            base_prefs['convenience_propensity'] -= 0.05
        
        # Normalize propensities to sum to 1.0
        total = sum(base_prefs.values())
        normalized_prefs = {k: v / total for k, v in base_prefs.items()}
        
        # Add some individual variation
        for key in normalized_prefs:
            normalized_prefs[key] = np.clip(
                normalized_prefs[key] + np.random.uniform(-0.05, 0.05),
                0.01,
                0.95
            )
        
        # Re-normalize after adding noise
        total = sum(normalized_prefs.values())
        normalized_prefs = {k: round(v / total, 3) for k, v in normalized_prefs.items()}
        
        # Add shopping frequency
        normalized_prefs['trips_per_month'] = round(
            np.clip(trips_per_month + np.random.uniform(-0.5, 0.5), 2.0, 10.0),
            1
        )
        
        return normalized_prefs