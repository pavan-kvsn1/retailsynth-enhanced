import numpy as np
import pandas as pd
from retailsynth.calibration import CalibrationEngine
from retailsynth.utils import RealisticCategoryHierarchy
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig
from retailsynth.engines.customer_heterogeneity import CustomerHeterogeneityEngine  # Phase 2.4

# ============================================================================
# CUSTOMER GENERATOR (v4.0 - Phase 2.4 Heterogeneity)
# Individual customer parameters replace discrete archetypes
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
        
        Phase 2.4: Now generates individual heterogeneous parameters
        instead of discrete customer types.
        """
        print(f"      Generating {config.n_customers:,} customers (Phase 2.4 - Heterogeneous)...")
        start_time = datetime.now()
        
        n = config.n_customers
        
        # Phase 2.4: Initialize heterogeneity engine
        heterogeneity_engine = CustomerHeterogeneityEngine(
            random_seed=config.random_seed if hasattr(config, 'random_seed') else None
        )
        
        # Generate heterogeneous parameters for all customers
        print("         Generating individual behavioral parameters...")
        customer_params_df = heterogeneity_engine.generate_population_parameters(n)
        
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
        
        # Phase 2.4: Extract heterogeneous parameters
        # No longer discrete categories - continuous values!
        price_sensitivity_continuous = customer_params_df['price_sensitivity'].values
        quality_preference_continuous = customer_params_df['quality_preference'].values
        promo_responsiveness_continuous = customer_params_df['promo_responsiveness'].values
        
        # For backward compatibility, create categorical labels (analysis only)
        price_sensitivity_category = np.empty(n, dtype=object)
        price_sensitivity_category[price_sensitivity_continuous < 1.0] = 'low'
        price_sensitivity_category[(price_sensitivity_continuous >= 1.0) & (price_sensitivity_continuous < 1.5)] = 'medium'
        price_sensitivity_category[price_sensitivity_continuous >= 1.5] = 'high'
        
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
        
        # Build customer list
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
                'price_sensitivity': price_sensitivity_category[i]  # Categorical for calibration
            }
            utility_params = calibration_engine.sample_utility_parameters(customer_demographics)
            
            # Phase 2.4: Override utility params with heterogeneous parameters
            # This ensures individual variation takes precedence
            utility_params['price_sensitivity'] = float(price_sensitivity_continuous[i])
            utility_params['quality_weight'] = float(quality_preference_continuous[i])
            
            # Brand preferences (NOW USING CONFIG + HETEROGENEITY)
            brand_preferences = {}
            n_preferred_brands = np.random.choice([3, 4, 5, 6, 7], p=[0.15, 0.25, 0.30, 0.20, 0.10])
            preferred_brands = np.random.choice(all_brands, size=n_preferred_brands, replace=False)
            
            # Phase 2.4: Use individual brand_loyalty parameter
            brand_loyalty_param = customer_params_df.loc[i, 'brand_loyalty']
            loyalty_std = 0.15  # Some variation around individual parameter
            
            for brand in preferred_brands:
                # Sample around individual brand loyalty
                pref = np.random.normal(brand_loyalty_param * 0.5, loyalty_std)  # Scale to [0, 1]
                pref = np.clip(pref, 0.1, 0.95)
                brand_preferences[brand] = round(pref, 3)
            
            # Trip purpose preferences (NEW - Sprint 1.4)
            trip_preferences = CustomerGenerator._calc_trip_preferences(
                shopping_personality, 
                household_size, 
                age
            )
            
            # Phase 2.4: Extract all heterogeneous parameters for this customer
            hetero_params = {
                'price_sensitivity_param': float(price_sensitivity_continuous[i]),
                'quality_preference_param': float(quality_preference_continuous[i]),
                'promo_responsiveness_param': float(promo_responsiveness_continuous[i]),
                'display_sensitivity_param': float(customer_params_df.loc[i, 'display_sensitivity']),
                'advertising_receptivity_param': float(customer_params_df.loc[i, 'advertising_receptivity']),
                'variety_seeking_param': float(customer_params_df.loc[i, 'variety_seeking']),
                'brand_loyalty_param': float(customer_params_df.loc[i, 'brand_loyalty']),
                'store_loyalty_param': float(customer_params_df.loc[i, 'store_loyalty']),
                'basket_size_preference_param': float(customer_params_df.loc[i, 'basket_size_preference']),
                'impulsivity_param': float(customer_params_df.loc[i, 'impulsivity'])
            }
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'household_size': household_size,
                'marital_status': marital_status[i],
                'children_count': int(children_count[i]),
                'income_bracket': income_bracket,
                'shopping_personality': shopping_personality,
                'price_sensitivity': price_sensitivity_category[i],  # Categorical (analysis only)
                'store_loyalty_level': float(customer_params_df.loc[i, 'store_loyalty']),  # Phase 2.4: From heterogeneity
                # Use Gamma distribution for habit formation (not memoryless Exponential)
                # Gamma captures consistent shopping patterns ("Saturday shopper")
                'days_since_last_visit': int(np.random.gamma(
                    config.days_since_last_visit_shape,
                    config.days_since_last_visit_scale
                )),
                'mobile_usage_propensity': round(mobile_usage, 3),
                'sustainability_preference': round(sustainability, 3),
                'utility_params': utility_params,
                'hetero_params': hetero_params,  # Phase 2.4: Individual parameters
                'brand_preferences': brand_preferences,
                'trip_preferences': trip_preferences,
                'created_at': datetime.now()
            })
        
        print(f"\n         ✅ Generated in {(datetime.now() - start_time).total_seconds():.1f}s")
        print(f"         ✅ Every customer has unique behavioral parameters!")
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