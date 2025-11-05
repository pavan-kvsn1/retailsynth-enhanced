from dataclasses import dataclass, field
from datetime import date
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path


@dataclass
class EnhancedRetailConfig:
    """Configuration for Enhanced RetailSynth v4.1 (with real product catalog)"""
    
    # Scale parameters
    n_customers: int = 100000
    n_products: int = 20000  # Now from real Dunnhumby catalog
    n_stores: int = 20
    simulation_weeks: int = 52
    
    # Product catalog paths (NEW - Sprint 1.1)
    product_catalog_path: Optional[str] = 'data/processed/product_catalog/product_catalog_20k.parquet'
    category_hierarchy_path: Optional[str] = 'data/processed/product_catalog/category_hierarchy.json'
    product_mapping_path: Optional[str] = 'data/processed/product_catalog/product_to_category.json'
    archetype_definitions_path: Optional[str] = 'data/processed/product_catalog/product_archetypes.csv'
    
    # Use real catalog (if False, falls back to synthetic generation)
    use_real_catalog: bool = True
    
    # Brand and Department filter (NEW - Sprint 1.1)
    department_filter: Optional[List[str]] = None
    brand_filter: Optional[List[str]] = None

    # Temporal dynamics
    enable_temporal_dynamics: bool = True
    enable_customer_drift: bool = True
    enable_product_lifecycle: bool = False
    enable_store_loyalty: bool = True
    
    # Customer mix (must sum to 1.0)
    price_anchor_customers: float = 0.25
    convenience_customers: float = 0.25
    planned_customers: float = 0.30
    impulse_customers: float = 0.20
    
    # Performance optimization
    enable_gpu_acceleration: bool = True
    batch_size: int = 2000
    parallel_weeks: bool = True
    
    # Temporal settings
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    region: str = 'US'
    
    # Product lifecycle settings
    product_retirement_weeks: int = 8  # Weeks in decline before retirement
    new_product_launch_rate: float = 0.02  # 2% new products per week
    
    # Random seed
    random_seed: int = 42
    
    def validate(self):
        """Validate configuration parameters."""
        # Check customer mix sums to 1.0
        customer_mix_sum = (
            self.price_anchor_customers +
            self.convenience_customers +
            self.planned_customers +
            self.impulse_customers
        )
        
        if not np.isclose(customer_mix_sum, 1.0, atol=0.01):
            raise ValueError(
                f"Customer mix must sum to 1.0, got {customer_mix_sum:.3f}"
            )
        
        # Check catalog paths exist if using real catalog
        if self.use_real_catalog:
            if self.product_catalog_path:
                catalog_path = Path(self.product_catalog_path)
                if not catalog_path.exists():
                    raise FileNotFoundError(
                        f"Product catalog not found: {self.product_catalog_path}\n"
                        f"Run: python scripts/build_product_catalog.py"
                    )
        
        return True
