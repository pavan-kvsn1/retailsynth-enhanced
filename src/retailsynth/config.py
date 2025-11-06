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
    
    # Sprint 1.4: Basket composition
    enable_basket_composition: bool = True  # Trip-purpose driven basket generation
    
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
    
    # ========================================================================
    # TUNABLE PARAMETERS FOR CALIBRATION (Sprint 2.1)
    # ========================================================================
    
    # 1. CUSTOMER DEMOGRAPHICS
    # ------------------------
    
    # Age distribution
    age_values: List[int] = field(default_factory=lambda: [25, 35, 45, 55, 65])
    age_probabilities: List[float] = field(default_factory=lambda: [0.2, 0.25, 0.25, 0.2, 0.1])
    
    # Household size distribution
    household_sizes: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    household_size_probs: List[float] = field(default_factory=lambda: [0.28, 0.35, 0.16, 0.15, 0.06])
    
    # Income distribution by age group
    young_income_probs: List[float] = field(default_factory=lambda: [0.35, 0.4, 0.25])  # <30K, 30-50K, 50-75K
    middle_income_probs: List[float] = field(default_factory=lambda: [0.25, 0.35, 0.25, 0.15])  # 30-50K, 50-75K, 75-100K, >100K
    senior_income_probs: List[float] = field(default_factory=lambda: [0.4, 0.4, 0.2])  # 30-50K, 50-75K, 75-100K
    
    # Children distribution
    children_probs: List[float] = field(default_factory=lambda: [0.3, 0.35, 0.25, 0.1])  # 0, 1, 2, 3 children
    single_probability: float = 0.3  # Probability of being single
    
    # 2. SHOPPING BEHAVIOR
    # --------------------
    
    # Visit probability
    base_visit_probability: float = 0.15
    visit_prob_by_personality: Dict[str, float] = field(default_factory=lambda: {
        'price_anchor': 0.12,
        'convenience': 0.18,
        'planned': 0.15,
        'impulse': 0.20
    })
    
    # Basket size
    basket_size_lambda: float = 5.5  # Poisson mean for base basket size
    basket_size_by_trip: Dict[str, float] = field(default_factory=lambda: {
        'quick_trip': 3.0,
        'major_shop': 12.0,
        'fill_in': 6.0,
        'special_occasion': 8.0
    })
    
    # Trip purpose distribution
    trip_purpose_weights: Dict[str, float] = field(default_factory=lambda: {
        'quick_trip': 0.3,
        'major_shop': 0.4,
        'fill_in': 0.2,
        'special_occasion': 0.1
    })
    
    # 3. PRICE SENSITIVITY & UTILITY
    # -------------------------------
    
    # Price sensitivity by income bracket
    price_sensitivity_by_income: Dict[str, float] = field(default_factory=lambda: {
        '<30K': 0.8,
        '30-50K': 0.6,
        '50-75K': 0.4,
        '75-100K': 0.3,
        '>100K': 0.2
    })
    
    # Price sensitivity by personality
    price_sensitivity_by_personality: Dict[str, float] = field(default_factory=lambda: {
        'price_anchor': 0.9,
        'convenience': 0.2,
        'planned': 0.5,
        'impulse': 0.4
    })
    
    # Brand loyalty
    brand_loyalty_mean: float = 0.6
    brand_loyalty_std: float = 0.2
    brand_loyalty_by_personality: Dict[str, float] = field(default_factory=lambda: {
        'price_anchor': 0.3,
        'convenience': 0.7,
        'planned': 0.6,
        'impulse': 0.4
    })
    
    # Promotion response
    promotion_sensitivity_mean: float = 0.5
    promotion_sensitivity_std: float = 0.2
    promotion_quantity_boost: float = 1.5  # Multiplier for quantity when on promotion
    
    # 4. PURCHASE HISTORY WEIGHTS
    # ----------------------------
    
    loyalty_weight: float = 0.3
    habit_weight: float = 0.4
    inventory_weight: float = 0.5
    variety_weight: float = 0.2
    price_memory_weight: float = 0.1  # Weight for reference price effect
    
    # 5. QUANTITY DISTRIBUTION
    # ------------------------
    
    quantity_mean: float = 1.5
    quantity_std: float = 0.8
    quantity_max: int = 10  # Maximum quantity per item
    
    # 6. STORE LOYALTY
    # ----------------
    
    store_loyalty_weight: float = 0.6
    store_switching_probability: float = 0.15
    distance_weight: float = 0.4
    satisfaction_weight: float = 0.6
    
    # 7. TEMPORAL DYNAMICS
    # --------------------
    
    # Customer drift
    drift_rate: float = 0.05  # Weekly drift magnitude
    drift_probability: float = 0.1  # Probability of drift per week
    
    # Inventory depletion
    inventory_depletion_rate: float = 0.1  # Daily depletion rate
    replenishment_threshold: float = 0.3  # Inventory level to trigger repurchase
    
    # 8. BASKET COMPOSITION
    # ---------------------
    
    complement_probability: float = 0.4  # Probability of adding complementary items
    substitute_avoidance: float = 0.8  # Probability of avoiding substitutes in same basket
    category_diversity_weight: float = 0.3  # Preference for diverse categories
    
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
        
        # Check age probabilities sum to 1.0
        if not np.isclose(sum(self.age_probabilities), 1.0, atol=0.01):
            raise ValueError(
                f"Age probabilities must sum to 1.0, got {sum(self.age_probabilities):.3f}"
            )
        
        # Check household size probabilities sum to 1.0
        if not np.isclose(sum(self.household_size_probs), 1.0, atol=0.01):
            raise ValueError(
                f"Household size probabilities must sum to 1.0, got {sum(self.household_size_probs):.3f}"
            )
        
        # Check trip purpose weights sum to 1.0
        if not np.isclose(sum(self.trip_purpose_weights.values()), 1.0, atol=0.01):
            raise ValueError(
                f"Trip purpose weights must sum to 1.0, got {sum(self.trip_purpose_weights.values()):.3f}"
            )
        
        # Check income probabilities
        if not np.isclose(sum(self.young_income_probs), 1.0, atol=0.01):
            raise ValueError(
                f"Young income probabilities must sum to 1.0, got {sum(self.young_income_probs):.3f}"
            )
        if not np.isclose(sum(self.middle_income_probs), 1.0, atol=0.01):
            raise ValueError(
                f"Middle income probabilities must sum to 1.0, got {sum(self.middle_income_probs):.3f}"
            )
        if not np.isclose(sum(self.senior_income_probs), 1.0, atol=0.01):
            raise ValueError(
                f"Senior income probabilities must sum to 1.0, got {sum(self.senior_income_probs):.3f}"
            )
        
        # Check children probabilities
        if not np.isclose(sum(self.children_probs), 1.0, atol=0.01):
            raise ValueError(
                f"Children probabilities must sum to 1.0, got {sum(self.children_probs):.3f}"
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
