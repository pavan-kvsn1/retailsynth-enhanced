import numpy as np
import pandas as pd
from retailsynth.calibration import CalibrationEngine
from retailsynth.engines import GPUUtilityEngine, ChoiceModel, SeasonalityEngine
from retailsynth.utils import RealisticCategoryHierarchy
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig

# ============================================================================
# STORE GENERATOR (v3.2)
# ============================================================================

class StoreGenerator:
    """
    Generates store locations and attributes.
    """
    
    @staticmethod
    def generate_stores(config: EnhancedRetailConfig) -> pd.DataFrame:
        """Generate store locations and attributes"""
        stores = []
        
        store_types = ['Supermarket', 'Hypermarket', 'Convenience', 'Supercenter']
        regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
        
        for i in range(1, config.n_stores + 1):
            # Store type based on distribution
            store_type = np.random.choice(store_types, p=[0.4, 0.3, 0.2, 0.1])
            
            # Store size based on type
            if store_type == 'Hypermarket':
                square_feet = np.random.randint(100000, 200000)
            elif store_type == 'Supercenter':
                square_feet = np.random.randint(120000, 180000)
            elif store_type == 'Supermarket':
                square_feet = np.random.randint(40000, 80000)
            else:  # Convenience
                square_feet = np.random.randint(5000, 15000)
            
            stores.append({
                'store_id': i,
                'store_name': f"Store {i}",
                'store_type': store_type,
                'region': np.random.choice(regions),
                'square_feet': square_feet,
                'established_year': np.random.randint(1995, 2023),
                'has_pharmacy': np.random.choice([True, False], p=[0.6, 0.4]),
                'has_deli': np.random.choice([True, False], p=[0.7, 0.3]),
                'has_bakery': np.random.choice([True, False], p=[0.8, 0.2]),
                'parking_spaces': int(square_feet / 200),
                'created_at': datetime.now()
            })
        
        return pd.DataFrame(stores)
 