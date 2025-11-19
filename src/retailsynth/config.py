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
    
    # Phase 2.6: Non-linear utilities (Sprint 2)
    enable_nonlinear_utilities: bool = True  # Use behavioral economics utilities
    use_log_price: bool = True  # Log-price utility (diminishing marginal disutility)
    use_reference_prices: bool = True  # EWMA reference prices with loss aversion
    use_psychological_thresholds: bool = True  # Charm pricing effects
    use_quadratic_quality: bool = True  # Diminishing returns to quality
    loss_aversion_lambda: float = 2.5  # Loss aversion coefficient (Kahneman & Tversky)
    ewma_alpha: float = 0.3  # Reference price smoothing (30% new, 70% old)
    
    # Phase 2.7: Seasonality Learning (Sprint 2)
    enable_seasonality_learning: bool = True  # Use learned seasonal patterns instead of hard-coded
    seasonal_patterns_path: Optional[str] = 'data/processed/seasonal_patterns/seasonal_patterns.pkl'
    seasonality_min_confidence: float = 0.3  # Minimum confidence to use a seasonal pattern
    seasonality_fallback_category: bool = True  # Use category patterns when product data sparse
    seasonality_smoothing: float = 0.2  # Smoothing factor for seasonal indices (reduces noise)
    
    # Phase 2.1 & 2.2: Promotional Frequency (NEW)
    promo_frequency_min: float = 0.05  # 5% minimum promotional penetration
    promo_frequency_max: float = 0.20  # 20% maximum promotional penetration
    
    # Phase 2.2: Realistic Promotional Discount Levels (NEW)
    psychological_discounts_light: List[float] = field(default_factory=lambda: [0.05, 0.08, 0.10])
    psychological_discounts_moderate: List[float] = field(default_factory=lambda: [0.10, 0.12, 0.15, 0.18])
    psychological_discounts_deep: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30])
    
    # Phase 2.1 & 2.2: Promotional Engine (Sprint 2)
    display_end_cap_capacity: int = 10  # Number of end caps per store
    display_feature_capacity: int = 3  # Number of feature displays per store
    
    # Phase 2.3: Marketing Signal Weights (Sprint 2)
    marketing_discount_weight: float = 0.4  # Weight for discount depth in signal
    marketing_display_weight: float = 0.3  # Weight for display prominence in signal
    marketing_advertising_weight: float = 0.3  # Weight for advertising reach in signal
    

    # Phase 2.4: Heterogeneity Distribution Parameters (Sprint 2)
    hetero_promo_alpha: float = 3.0  # Beta distribution alpha for promo responsiveness
    hetero_promo_beta: float = 2.0  # Beta distribution beta for promo responsiveness
    hetero_display_alpha: float = 3.0  # Beta distribution alpha for display sensitivity
    hetero_display_beta: float = 3.0  # Beta distribution beta for display sensitivity
    
    # Phase 2.5: Marketing Signal Noise (Sprint 2)
    marketing_signal_noise: float = 0.1  # Standard deviation of noise in marketing signals
    
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
    base_visit_probability: float = 0.55  # Weekly visit probability (was 0.15 - too low!)
    visit_prob_by_personality: Dict[str, float] = field(default_factory=lambda: {
        'price_anchor': 0.40,     # was 0.12 - now realistic 1-2 visits/week
        'convenience': 0.60,      # was 0.18 - convenience shoppers visit more
        'planned': 0.50,          # was 0.15 - planned shoppers ~1 visit/week
        'impulse': 0.65           # was 0.20 - impulse shoppers visit frequently
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
    # DEPRECATED: promotion_sensitivity_mean/std are not used by Phase 2.4 heterogeneity engine.
    # Phase 2.4 uses Beta distributions in customer_heterogeneity.py instead (more flexible).
    # Kept for backward compatibility. Will be removed in v2.0.
    promotion_sensitivity_mean: float = 0.5  # DEPRECATED - not used
    promotion_sensitivity_std: float = 0.2   # DEPRECATED - not used
    promotion_quantity_boost: float = 1.5    # ✅ ACTIVE - used in basket_composer.py (Priority 2B)
    
    # 4. PURCHASE HISTORY WEIGHTS
    # ----------------------------
    
    loyalty_weight: float = 0.3
    habit_weight: float = 0.4
    inventory_weight: float = 0.5
    variety_weight: float = 0.2
    price_memory_weight: float = 0.1  # Weight for reference price effect
    
    # 5. QUANTITY DISTRIBUTION
    # ------------------------
    # Target from Dunnhumby: mean=2.47, std varies by product/customer
    
    quantity_mean: float = 2.5  # Increased from 1.5 to match Dunnhumby (Priority 2A fix)
    quantity_std: float = 1.2   # Increased for more variation
    quantity_max: int = 20  # Maximum quantity per item (was 10 - increased for bulk purchases)
    
    # 5B. TRIP PURPOSE BASKET SIZES (NEW - replaces hardcoded TRIP_CHARACTERISTICS)
    # ------------------------------------------------------------------------------
    # These parameters control basket sizes for different trip types
    # Target realistic retail ranges (not "industry standards")
    
    trip_stock_up_basket_mean: float = 10.0  # Was hardcoded 28.0 - too high!
    trip_stock_up_basket_std: float = 3.0
    trip_fill_in_basket_mean: float = 5.0  # Was hardcoded 5.5 - reasonable
    trip_fill_in_basket_std: float = 2.0
    trip_meal_prep_basket_mean: float = 8.0  # Was hardcoded 12.0 - too high
    trip_meal_prep_basket_std: float = 3.0
    trip_convenience_basket_mean: float = 3.0  # Was hardcoded 3.0 - good
    trip_convenience_basket_std: float = 1.5
    trip_special_basket_mean: float = 12.0  # Was hardcoded 22.0 - way too high!
    trip_special_basket_std: float = 4.0
    
    # 5C. TRIP PURPOSE PROBABILITIES (NEW - replaces hardcoded TRIP_PURPOSE_PROBABILITIES)
    # -----------------------------------------------------------------------------------
    # These control the mix of trip types for different customer personalities
    # Target: More small trips (fill-in, convenience), fewer large trips (stock-up)
    
    # For price_anchor customers (25% of population)
    trip_prob_price_anchor_stock_up: float = 0.25  # Was 0.45 - too high!
    trip_prob_price_anchor_fill_in: float = 0.40  # Was 0.30 - increase fill-ins
    trip_prob_price_anchor_convenience: float = 0.15  # Was 0.05 - too low!
    
    # For convenience customers (25% of population)  
    trip_prob_convenience_convenience: float = 0.35  # Was 0.35 - good
    trip_prob_convenience_fill_in: float = 0.35  # Was 0.35 - good
    trip_prob_convenience_stock_up: float = 0.15  # Was 0.15 - good
    
    # For planned customers (25% of population)
    trip_prob_planned_stock_up: float = 0.30  # Was 0.40 - reduce large trips
    trip_prob_planned_meal_prep: float = 0.35  # Was 0.30 - increase
    trip_prob_planned_fill_in: float = 0.25  # Was 0.20 - increase
    
    # For impulse customers (25% of population)
    trip_prob_impulse_convenience: float = 0.40  # Was 0.40 - good
    trip_prob_impulse_fill_in: float = 0.30  # Was 0.25 - increase slightly
    trip_prob_impulse_special: float = 0.15  # Was 0.20 - reduce large trips
    
    # 6. STORE LOYALTY
    # ----------------
    
    store_loyalty_weight: float = 0.6
    store_switching_probability: float = 0.15
    distance_weight: float = 0.4
    satisfaction_weight: float = 0.6
    
    # 7. TEMPORAL DYNAMICS
    # --------------------
    
    # Customer drift (mixture model: small gradual drift + occasional life events)
    drift_rate: float = 0.05  # Weekly drift magnitude (small changes)
    drift_probability: float = 0.1  # Probability of drift per week
    drift_life_event_probability: float = 0.1  # Probability that drift is a life event (10% of drifts)
    drift_life_event_multiplier: float = 5.0  # Life events are 5x larger than normal drift
    
    # Days since last visit (Gamma distribution for habit formation)
    days_since_last_visit_shape: float = 2.0  # Higher = more consistent habits
    days_since_last_visit_scale: float = 3.5  # Mean = shape × scale = 7 days
    
    # Inventory depletion
    inventory_depletion_rate: float = 0.1  # Daily depletion rate
    replenishment_threshold: float = 0.3  # Inventory level to trigger repurchase
    
    # Store Value → Visit Probability (Bain recursive mechanism)
    # These control the recursive visit probability calculation
    store_base_utility: float = 0.5        # γ₀: Base store utility
    store_value_weight: float = 0.6        # γ₁: How much SV affects visit decision
    marketing_visit_weight: float = 0.4    # β: How much marketing affects visit decision
    visit_memory_weight: float = 0.3       # θ: Memory parameter (0=no memory, 1=full persistence)
    
    # Multiple visits per week (Poisson model)
    enable_multiple_visits_per_week: bool = False  # Use Poisson instead of binary visits
    max_visits_per_week: float = 5.0              # Cap on visit rate (lambda)
    
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
