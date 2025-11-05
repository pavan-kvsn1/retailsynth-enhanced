import numpy as np
from typing import Dict

class CalibrationEngine:
    """
    Calibration parameters based on RetailSynth paper and retail research.
    Provides realistic ranges for all model parameters.
    """
    
    def __init__(self):
        self.parameters = {
            # Utility function parameters (from RetailSynth paper)
            'beta_price': {
                'mean': -2.5,
                'std': 0.8,
                'range': (-4.5, -0.5)
            },
            'beta_brand': {
                'mean': 1.2,
                'std': 0.6,
                'range': (0.0, 3.0)
            },
            'beta_promotion': {
                'mean': 0.8,
                'std': 0.3,
                'range': (0.0, 2.0)
            },
            'beta_assortment_role': {
                'mean': 0.6,
                'std': 0.25,
                'range': (0.0, 1.5)
            },
            
            # Shopping frequency parameters
            'visit_frequency': {
                'price_anchor': {'mean': 1.5, 'std': 0.5},  # visits per week
                'convenience': {'mean': 2.5, 'std': 0.7},
                'planned': {'mean': 1.2, 'std': 0.4},
                'impulse': {'mean': 2.0, 'std': 0.6}
            },
            
            # Basket size parameters
            'basket_size': {
                'price_anchor': {'mean': 25, 'std': 10},
                'convenience': {'mean': 8, 'std': 4},
                'planned': {'mean': 35, 'std': 12},
                'impulse': {'mean': 15, 'std': 8}
            },
            
            # Product role preferences by personality
            'role_preferences': {
                'price_anchor': {
                    'lpg_line': 0.8,
                    'front_basket': 0.3,
                    'mid_basket': 0.5,
                    'back_basket': 0.1
                },
                'convenience': {
                    'lpg_line': 0.3,
                    'front_basket': 0.6,
                    'mid_basket': 0.4,
                    'back_basket': 0.7
                },
                'planned': {
                    'lpg_line': 0.4,
                    'front_basket': 0.2,
                    'mid_basket': 0.8,
                    'back_basket': 0.2
                },
                'impulse': {
                    'lpg_line': 0.2,
                    'front_basket': 0.8,
                    'mid_basket': 0.4,
                    'back_basket': 0.9
                }
            }
        }
    
    def sample_utility_parameters(self, customer_demographics: Dict) -> Dict[str, float]:
        """
        Sample utility parameters for a customer based on demographics.
        Implements realistic correlations between parameters.
        """
        personality = customer_demographics['shopping_personality']
        income = customer_demographics['income_bracket']
        age_group = customer_demographics['age_group']
        
        # Base sampling
        beta_price = np.random.normal(
            self.parameters['beta_price']['mean'],
            self.parameters['beta_price']['std']
        )
        beta_brand = np.random.normal(
            self.parameters['beta_brand']['mean'],
            self.parameters['beta_brand']['std']
        )
        beta_promotion = np.random.normal(
            self.parameters['beta_promotion']['mean'],
            self.parameters['beta_promotion']['std']
        )
        beta_role = np.random.normal(
            self.parameters['beta_assortment_role']['mean'],
            self.parameters['beta_assortment_role']['std']
        )
        
        # Adjust based on personality
        if personality == 'price_anchor':
            beta_price *= 1.3  # More price-sensitive
            beta_promotion *= 1.4  # More promotion-responsive
        elif personality == 'convenience':
            beta_price *= 0.7  # Less price-sensitive
            beta_brand *= 1.2  # More brand-loyal
        elif personality == 'planned':
            beta_role *= 1.3  # More strategic
        elif personality == 'impulse':
            beta_promotion *= 1.3
            beta_role *= 0.7
        
        # Adjust based on income
        if income in ['<30K', '30-50K']:
            beta_price *= 1.2  # More price-sensitive
        elif income in ['>100K']:
            beta_price *= 0.8  # Less price-sensitive
            beta_brand *= 1.15  # More brand-loyal
        
        # Clip to valid ranges
        beta_price = np.clip(beta_price, *self.parameters['beta_price']['range'])
        beta_brand = np.clip(beta_brand, *self.parameters['beta_brand']['range'])
        beta_promotion = np.clip(beta_promotion, *self.parameters['beta_promotion']['range'])
        beta_role = np.clip(beta_role, *self.parameters['beta_assortment_role']['range'])
        
        return {
            'beta_price': beta_price,
            'beta_brand': beta_brand,
            'beta_promotion': beta_promotion,
            'beta_assortment_role': beta_role
        }

