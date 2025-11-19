import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from retailsynth.engines.marketing_signal import MarketingSignalCalculator

# ============================================================================
# PROMOTIONAL ENGINE (v1.2 - Sprint 2.3)
# Comprehensive promotional system: mechanics, displays, features
# Phase 2.2: HMM integration, product tendencies, multi-store support
# Phase 2.3: Marketing signal calculation
# ============================================================================

logger = logging.getLogger(__name__)


@dataclass
class StorePromoContext:
    """
    Complete promotional state for a store in a given week
    """
    store_id: int
    week_number: int
    
    # Products on promotion
    promoted_products: List[int] = field(default_factory=list)
    
    # Promo mechanics
    promo_depths: Dict[int, float] = field(default_factory=dict)  # product_id → discount %
    promo_states: Dict[int, int] = field(default_factory=dict)    # product_id → HMM state
    promo_durations: Dict[int, int] = field(default_factory=dict) # product_id → weeks
    
    # Promo displays
    display_types: Dict[int, str] = field(default_factory=dict)   # product_id → display type
    end_cap_products: List[int] = field(default_factory=list)
    feature_display_products: List[int] = field(default_factory=list)
    
    # Promo features (advertising)
    in_ad_products: List[int] = field(default_factory=list)
    mailer_products: List[int] = field(default_factory=list)
    
    # Computed metrics
    avg_discount_depth: float = 0.0
    n_deep_discounts: int = 0
    marketing_signal_strength: float = 0.0
    
    def compute_metrics(self):
        """Compute summary metrics"""
        if self.promo_depths:
            self.avg_discount_depth = np.mean(list(self.promo_depths.values()))
            self.n_deep_discounts = sum(1 for d in self.promo_depths.values() if d > 0.30)


class PromotionalEngine:
    """
    Manages promotional activities at store level
    
    Components:
    1. Promo Mechanics - Discount depth, frequency, duration
    2. Promo Displays - End cap, shelf, feature placement
    3. Promo Features - In-ad, mailer advertising
    
    Integrates with HMM price states for realistic promotional patterns
    
    NEW in Phase 2.2:
    - Uses real HMM states for promotional selection
    - Product-specific promotional tendencies
    - Multi-store promotional contexts
    - Learned promotional patterns
    """
    
    def __init__(self, hmm_model=None, products_df=None, stores_df=None, config=None):
        """
        Initialize promotional engine
        
        Args:
            hmm_model: Optional HMM price dynamics model (Phase 2.2)
            products_df: DataFrame of products
            stores_df: DataFrame of stores
            config: Optional configuration dict
        """
        self.hmm_model = hmm_model
        self.products = products_df
        self.stores = stores_df
        self.config = config or {}
        
        # Number of HMM states (default: 4)
        self.n_states = 4
        
        # Promo mechanics configuration
        self._init_promo_mechanics()
        
        # Display configuration
        self._init_display_system()
        
        # Feature advertising configuration
        self._init_feature_system()
        
        # Phase 2.2: Product-specific promotional tendencies
        self.product_promo_tendencies = {}
        self._init_product_tendencies()

        # Phase 2.3: Marketing signal calculator
        self.signal_calculator = MarketingSignalCalculator(config=self.config)
        
        # Active promotions tracking
        self.active_promotions = {}  # store_id → {product_id: {start_week, duration}}
        
        # Phase 2.2: Multi-store promotional contexts cache
        self.store_contexts = {}  # (store_id, week) → StorePromoContext
        
        logger.info("PromotionalEngine initialized")
    
    def _init_promo_mechanics(self):
        """Initialize promotional mechanics parameters"""
        # Discount depth ranges by HMM state (REALISTIC RETAIL LEVELS)
        self.depth_ranges = {
            0: (0.00, 0.03),   # Regular: no/tiny discount
            1: (0.05, 0.12),   # Feature: light discount
            2: (0.12, 0.20),   # Deep: moderate discount
            3: (0.20, 0.35)    # Clearance: heavy discount (rare)
        }
        
        # Promotion frequency (% of products on promo)
        self.promo_frequency = self.config.get('promo_frequency', {
            'min': 0.10,  # 10% minimum
            'max': 0.30   # 30% maximum
        })
        
        # Promotion duration probabilities
        self.duration_probs = {
            1: 0.60,  # 60% are 1-week promos
            2: 0.25,  # 25% are 2-week
            3: 0.10,  # 10% are 3-week
            4: 0.05   # 5% are 4+ week
        }
        
        # Phase 2.2: State transition preferences (more realistic patterns)
        self.state_promo_weights = {
            0: 0.10,  # Regular: low promo probability
            1: 0.40,  # Feature: high promo probability
            2: 0.40,  # Deep: high promo probability
            3: 0.10   # Clearance: moderate (liquidation)
        }
    
    def _init_display_system(self):
        """Initialize display allocation system"""
        # Store capacity constraints
        self.display_capacity = {
            'end_cap': 10,         # 10 end caps per store
            'feature_display': 3   # 3 feature displays per store
        }
        
        # Display effectiveness (utility boost multiplier)
        self.display_effectiveness = {
            'none': 0.0,
            'shelf_tag': 0.3,       # Small tag on shelf
            'end_cap': 0.8,         # End of aisle display
            'feature_display': 1.2  # Large feature display
        }
        
        # Display probability by discount depth
        self.display_prob_by_depth = {
            'small': {      # <15% discount
                'shelf_tag': 0.8,
                'end_cap': 0.15,
                'feature_display': 0.05
            },
            'medium': {     # 15-30% discount
                'shelf_tag': 0.5,
                'end_cap': 0.35,
                'feature_display': 0.15
            },
            'deep': {       # >30% discount
                'shelf_tag': 0.2,
                'end_cap': 0.45,
                'feature_display': 0.35
            }
        }
    
    def _init_feature_system(self):
        """Initialize feature advertising system"""
        # Feature probability by display type
        self.feature_prob = {
            'feature_display': {'in_ad': 0.9, 'mailer': 0.6},
            'end_cap': {'in_ad': 0.5, 'mailer': 0.3},
            'shelf_tag': {'in_ad': 0.1, 'mailer': 0.05},
            'none': {'in_ad': 0.0, 'mailer': 0.0}
        }
    
    def _init_product_tendencies(self):
        """
        Initialize product-specific promotional tendencies (Phase 2.2)
        
        Products have different promotional patterns:
        - High promo tendency: Frequently promoted (e.g., soft drinks, snacks)
        - Low promo tendency: Rarely promoted (e.g., basics like milk, eggs)
        """
        if self.products is None:
            return
        
        # Default: assign random tendencies
        # In production, these would be learned from data
        np.random.seed(42)  # Consistent tendencies
        
        for product_id in self.products['product_id'].values:
            # Convert to int to ensure it's hashable (handles numpy scalars/arrays)            
            # Most products have moderate tendency (0.8-1.2)
            # Some have high tendency (1.2-1.5) - promotional items
            # Some have low tendency (0.5-0.8) - staples
            
            rand_val = np.random.random()
            if rand_val < 0.15:  # 15% high promo items
                tendency = np.random.uniform(1.2, 1.5)
            elif rand_val < 0.25:  # 10% low promo items
                tendency = np.random.uniform(0.5, 0.8)
            else:  # 75% moderate
                tendency = np.random.uniform(0.8, 1.2)
            
            self.product_promo_tendencies[product_id] = tendency
        
        logger.info(f"Initialized promotional tendencies for {len(self.product_promo_tendencies):,} products")
    
    def learn_promo_tendencies_from_data(self, transactions_df: pd.DataFrame, causal_df: pd.DataFrame = None):
        """
        Learn product-specific promotional tendencies from historical data (Phase 2.2)
        
        Args:
            transactions_df: Historical transaction data
            causal_df: Optional causal (promotional) data
        
        Learns:
            - Promotion frequency per product
            - Average discount depth per product
            - Display type preferences
        """
        logger.info("Learning promotional tendencies from data...")
        
        # Calculate promotion frequency per product
        product_weeks = transactions_df.groupby(['PRODUCT_ID', 'WEEK_NO']).size().reset_index(name='weeks_active')
        product_promos = transactions_df[transactions_df['RETAIL_DISC'] > 0].groupby('PRODUCT_ID').size().reset_index(name='weeks_promoted')
        
        promo_stats = product_weeks.groupby('PRODUCT_ID')['weeks_active'].sum().reset_index()
        promo_stats = promo_stats.merge(product_promos, on='PRODUCT_ID', how='left')
        promo_stats['weeks_promoted'] = promo_stats['weeks_promoted'].fillna(0)
        promo_stats['promo_frequency'] = promo_stats['weeks_promoted'] / promo_stats['weeks_active']
        
        # Normalize to tendency (mean=1.0)
        overall_mean = promo_stats['promo_frequency'].mean()
        promo_stats['tendency'] = promo_stats['promo_frequency'] / overall_mean
        
        # Update product tendencies
        for _, row in promo_stats.iterrows():
            product_id = int(row['PRODUCT_ID'])
            self.product_promo_tendencies[product_id] = float(row['tendency'])
        
        logger.info(f"✅ Learned tendencies for {len(self.product_promo_tendencies):,} products")
        logger.info(f"   Mean tendency: {np.mean(list(self.product_promo_tendencies.values())):.2f}")
        logger.info(f"   Std tendency: {np.std(list(self.product_promo_tendencies.values())):.2f}")
    
    def _select_promoted_products(self, product_ids: np.ndarray, 
                                  week_number: int, 
                                  n_products: int) -> Tuple[List[int], Dict[int, int]]:
        """
        Select which products to promote this week
        
        Phase 2.2: Now uses real HMM states and product tendencies
        """
        # Determine number of promotions
        promo_pct = np.random.uniform(
            self.promo_frequency['min'],
            self.promo_frequency['max']
        )
        n_promotions = int(n_products * promo_pct)
        n_promotions = max(1, min(n_promotions, n_products))
        
        # LOGGING: Track promo frequency
        if week_number % 5 == 0:
            logger.info(f"Week {week_number}: Promo frequency = {promo_pct:.1%}, selecting {n_promotions}/{n_products} products")
        
        if self.hmm_model is not None and hasattr(self.hmm_model, 'transition_matrices'):
            # Phase 2.2: Use real HMM model to determine promotional states
            promoted_products = []
            promo_states = {}
            
            # Get HMM states for each product
            for product_id in product_ids:
                if product_id in self.hmm_model.transition_matrices:
                    # Sample state based on transition matrix
                    # For week t, we use the stationary distribution
                    if product_id in self.hmm_model.initial_state_probs:
                        state_probs = self.hmm_model.initial_state_probs[product_id]
                    else:
                        # Use uniform if no learned probs
                        state_probs = np.ones(self.n_states) / self.n_states
                    
                    # Weight by promotional tendency
                    tendency = self.product_promo_tendencies.get(product_id, 1.0)
                    
                    # High tendency products get slightly deeper discounts
                    if tendency > 1.2:
                        discount *= 1.1  # 10% deeper
                    elif tendency < 0.8:
                        discount *= 0.9  # 10% shallower
                    
                    weighted_probs = state_probs * np.array([self.state_promo_weights.get(i, 0.25) for i in range(self.n_states)])
                    weighted_probs /= weighted_probs.sum()
                    
                    state = np.random.choice(self.n_states, p=weighted_probs)
                    
                    # Only promote if not in regular state
                    if state > 0:
                        promo_states[product_id] = int(state)
                        promoted_products.append(product_id)
            
            # If we got fewer promotions than desired, add more
            if len(promoted_products) < n_promotions:
                remaining = set(product_ids) - set(promoted_products)
                additional = np.random.choice(
                    list(remaining),
                    size=min(n_promotions - len(promoted_products), len(remaining)),
                    replace=False
                )
                for pid in additional:
                    promoted_products.append(pid)
                    promo_states[pid] = 1  # Default to feature state
            
            # If we got too many, trim based on tendency
            elif len(promoted_products) > n_promotions:
                # Keep products with highest tendency
                tendencies = {pid: self.product_promo_tendencies.get(pid, 1.0) for pid in promoted_products}
                sorted_products = sorted(tendencies.items(), key=lambda x: x[1], reverse=True)
                promoted_products = [pid for pid, _ in sorted_products[:n_promotions]]
                promo_states = {pid: promo_states[pid] for pid in promoted_products}
            
            logger.debug(f"HMM-based selection: {len(promoted_products)} products")
        else:
            # Fallback: Random selection with tendency weighting
            tendencies = np.array([
                self.product_promo_tendencies.get(pid, 1.0) 
                for pid in product_ids
            ])
            
            # Normalize to probabilities
            promo_probs = tendencies / tendencies.sum()
            
            # Sample products based on tendencies
            promo_indices = np.random.choice(
                n_products,
                size=n_promotions,
                replace=False,
                p=promo_probs
            )
            promoted_products = product_ids[promo_indices].tolist()
            
            # Assign HMM states (weighted toward feature/deep discount)
            states = np.random.choice(
                [0, 1, 2, 3],
                size=n_promotions,
                p=[0.1, 0.4, 0.4, 0.1]
            )
            promo_states = {pid: int(state) for pid, state in zip(promoted_products, states)}
            
            logger.debug(f"Tendency-weighted selection: {len(promoted_products)} products")
        
        return promoted_products, promo_states
    
    def _calculate_discount_depths(self, promoted_products: List[int], 
                                   promo_states: Dict[int, int], 
                                   store_id: int) -> Dict[int, float]:
        """
        Calculate discount percentage for each promoted product
        
        Phase 2.2: Now uses product-specific tendencies
        **NEW**: Uses psychological price points instead of uniform distribution
        Real retailers use discrete appealing points: 10%, 15%, 20%, 25%, 30%, 33%, 40%, 50%
        """
        promo_depths = {}
        
        # Get psychological price points from config (REALISTIC RETAIL LEVELS)
        light_points = getattr(self.config, 'psychological_discounts_light', [0.05, 0.08, 0.10])
        moderate_points = getattr(self.config, 'psychological_discounts_moderate', [0.10, 0.12, 0.15, 0.18])
        deep_points = getattr(self.config, 'psychological_discounts_deep', [0.15, 0.20, 0.25, 0.30])
        
        for product_id in promoted_products:
            state = promo_states.get(product_id, 1)
            
            # Sample from psychological price points based on state
            if state == 0:  # Regular: light discount
                discount = np.random.choice(light_points)
            elif state == 1:  # Feature: moderate discount
                discount = np.random.choice(moderate_points)
            else:  # state == 2: Deep discount
                discount = np.random.choice(deep_points)
            
            # Phase 2.2: Apply product-specific variation
            tendency = self.product_promo_tendencies.get(product_id, 1.0)
            
            # High tendency products get slightly deeper discounts
            # But keep at psychological price points
            if tendency > 1.2:
                # Shift to next higher discount tier if available
                if state == 0:
                    discount = np.random.choice(moderate_points)
                elif state == 1:
                    discount = np.random.choice(deep_points)
            elif tendency < 0.8:
                # Shift to next lower discount tier if available
                if state == 2:
                    discount = np.random.choice(moderate_points)
                elif state == 1:
                    discount = np.random.choice(light_points)
            
            promo_depths[product_id] = np.clip(discount, 0.0, 0.70)
        
        return promo_depths
    
    def _determine_durations(self, promoted_products: List[int], 
                            promo_depths: Dict[int, float]) -> Dict[int, int]:
        """
        Determine promotion duration for each product
        
        Logic: Deep discounts are shorter, moderate discounts are longer
        """
        promo_durations = {}
        
        for product_id in promoted_products:
            discount = promo_depths[product_id]
            
            if discount > 0.35:
                # Deep discount: shorter duration
                duration = np.random.choice([1, 2], p=[0.7, 0.3])
            elif discount > 0.15:
                # Medium discount: medium duration
                duration = np.random.choice([2, 3], p=[0.6, 0.4])
            else:
                # Light discount: longer duration
                duration = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
            
            promo_durations[product_id] = duration
        
        return promo_durations
    
    def _allocate_displays(self, promoted_products: List[int], 
                          promo_depths: Dict[int, float], 
                          store_id: int) -> Dict[int, str]:
        """
        Allocate display types to promoted products
        
        Rules:
        1. Deepest discounts get best displays
        2. Subject to capacity constraints
        3. Shelf tags unlimited
        """
        # Sort products by discount depth (descending)
        sorted_products = sorted(
            promoted_products,
            key=lambda p: promo_depths[p],
            reverse=True
        )
        
        allocations = {}
        endcaps_used = 0
        features_used = 0
        
        for product_id in sorted_products:
            discount = promo_depths[product_id]
            
            # Allocate based on availability and discount depth
            if discount > 0.35 and features_used < self.display_capacity['feature_display']:
                allocations[product_id] = 'feature_display'
                features_used += 1
            elif discount > 0.20 and endcaps_used < self.display_capacity['end_cap']:
                allocations[product_id] = 'end_cap'
                endcaps_used += 1
            elif discount > 0.10:
                allocations[product_id] = 'shelf_tag'
            else:
                allocations[product_id] = 'none'
        
        return allocations
    
    def _select_featured_products(self, promoted_products: List[int], 
                                  display_allocations: Dict[int, str], 
                                  store_id: int) -> Tuple[List[int], List[int]]:
        """
        Select products to feature in advertising (in-ad and mailer)
        
        Higher display prominence → more likely in ad
        """
        in_ad = []
        mailer = []
        
        for product_id in promoted_products:
            display_type = display_allocations.get(product_id, 'none')
            probs = self.feature_prob.get(display_type, {'in_ad': 0, 'mailer': 0})
            
            # Decide if in ad
            if np.random.random() < probs['in_ad']:
                in_ad.append(product_id)
            
            # Decide if in mailer
            if np.random.random() < probs['mailer']:
                mailer.append(product_id)
        
        return in_ad, mailer
    
    def _update_active_promotions(self, store_id: int, context: StorePromoContext):
        """Track active promotions for continuity across weeks"""
        if store_id not in self.active_promotions:
            self.active_promotions[store_id] = {}
        
        # Add new promotions
        for product_id in context.promoted_products:
            if product_id not in self.active_promotions[store_id]:
                self.active_promotions[store_id][product_id] = {
                    'start_week': context.week_number,
                    'duration': context.promo_durations[product_id]
                }
    
    def get_promotional_price(self, product_id: int, base_price: float, 
                             promo_context: StorePromoContext) -> float:
        """
        Calculate final promotional price
        
        Args:
            product_id: Product identifier
            base_price: Base price before promotion
            promo_context: Store promotional context
        
        Returns:
            float: Final price (base price * (1 - discount))
        """
        if product_id in promo_context.promo_depths:
            discount = promo_context.promo_depths[product_id]
            return base_price * (1.0 - discount)
        else:
            return base_price
    
    def get_promo_summary(self, promo_context: StorePromoContext) -> Dict:
        """Get summary statistics for promotional context"""
        return {
            'store_id': promo_context.store_id,
            'week_number': promo_context.week_number,
            'n_promotions': len(promo_context.promoted_products),
            'avg_discount': promo_context.avg_discount_depth,
            'n_deep_discounts': promo_context.n_deep_discounts,
            'n_end_caps': len(promo_context.end_cap_products),
            'n_features': len(promo_context.feature_display_products),
            'n_in_ad': len(promo_context.in_ad_products),
            'n_mailer': len(promo_context.mailer_products)
        }
    
    def generate_store_promotions(self, store_id: int, week_number: int, 
                                  base_prices: np.ndarray, 
                                  product_ids: np.ndarray) -> StorePromoContext:
        """
        Generate complete promotional context for a store in a specific week
        
        Args:
            store_id: Store identifier
            week_number: Week number
            base_prices: Array of base prices
            product_ids: Array of product IDs
        
        Returns:
            StorePromoContext: Complete promotional state
        """
        n_products = len(product_ids)
        
        # 1. Select products to promote
        promoted_products, promo_states = self._select_promoted_products(
            product_ids, week_number, n_products
        )
        
        # 2. Calculate discount depths
        promo_depths = self._calculate_discount_depths(
            promoted_products, promo_states, store_id
        )
        
        # 3. Determine promotion durations
        promo_durations = self._determine_durations(promoted_products, promo_depths)
        
        # 4. Allocate display types
        display_allocations = self._allocate_displays(
            promoted_products, promo_depths, store_id
        )
        
        # 5. Select products for feature advertising
        in_ad, mailer = self._select_featured_products(
            promoted_products, display_allocations, store_id
        )
        
        # 6. Create promotional context
        context = StorePromoContext(
            store_id=store_id,
            week_number=week_number,
            promoted_products=promoted_products,
            promo_depths=promo_depths,
            promo_states=promo_states,
            promo_durations=promo_durations,
            display_types=display_allocations,
            end_cap_products=[p for p, d in display_allocations.items() if d == 'end_cap'],
            feature_display_products=[p for p, d in display_allocations.items() if d == 'feature_display'],
            in_ad_products=in_ad,
            mailer_products=mailer
        )
        
        # Compute summary metrics
        context.compute_metrics()

        # Phase 2.3: Calculate marketing signal strength
        context.marketing_signal_strength = self.signal_calculator.calculate_signal_strength(context)
        
        # Update active promotions tracking
        self._update_active_promotions(store_id, context)
        
        return context
    
    def generate_store_promo_context(self, store_id: int, week_number: int, 
                                     promoted_products: Optional[List[int]] = None,
                                     base_prices: Optional[np.ndarray] = None,
                                     product_ids: Optional[np.ndarray] = None) -> StorePromoContext:
        """
        Convenience method for generating store promotional context (Phase 2.5)
        
        This is a wrapper around generate_store_promotions() that allows flexible calling
        with different parameter sets.
        
        Args:
            store_id: Store identifier
            week_number: Week number
            promoted_products: Optional pre-selected promoted products (if None, will select automatically)
            base_prices: Optional base prices array (required if promoted_products is None)
            product_ids: Optional product IDs array (required if promoted_products is None)
        
        Returns:
            StorePromoContext: Complete promotional state
        """
        # If base_prices and product_ids provided, use full generation
        if base_prices is not None and product_ids is not None:
            return self.generate_store_promotions(store_id, week_number, base_prices, product_ids)
        
        # Otherwise, if we have products_df and promoted_products, create simplified context
        if promoted_products is not None and self.products is not None:
            # Get product IDs from products_df
            if product_ids is None:
                product_ids = self.products['product_id'].values
            if base_prices is None:
                base_prices = self.products['base_price'].values
            
            # Create promo states (assume moderate promotion - state 1)
            promo_states = {pid: 1 for pid in promoted_products}
            
            # Calculate discount depths
            promo_depths = self._calculate_discount_depths(
                promoted_products, promo_states, store_id
            )
            
            # Determine durations
            promo_durations = self._determine_durations(promoted_products, promo_depths)
            
            # Allocate displays
            display_allocations = self._allocate_displays(
                promoted_products, promo_depths, store_id
            )
            
            # Select featured products
            in_ad, mailer = self._select_featured_products(
                promoted_products, display_allocations, store_id
            )
            
            # Create promotional context
            context = StorePromoContext(
                store_id=store_id,
                week_number=week_number,
                promoted_products=promoted_products,
                promo_depths=promo_depths,
                promo_states=promo_states,
                promo_durations=promo_durations,
                display_types=display_allocations,
                end_cap_products=[p for p, d in display_allocations.items() if d == 'end_cap'],
                feature_display_products=[p for p, d in display_allocations.items() if d == 'feature_display'],
                in_ad_products=in_ad,
                mailer_products=mailer
            )
            
            # Compute summary metrics
            context.compute_metrics()
            
            # Phase 2.3: Calculate marketing signal strength
            if hasattr(self, 'signal_calculator'):
                context.marketing_signal_strength = self.signal_calculator.calculate_signal_strength(context)
            
            return context
        
        # If neither option available, raise error
        raise ValueError(
            "Must provide either (base_prices + product_ids) or (promoted_products with products_df initialized)"
        )