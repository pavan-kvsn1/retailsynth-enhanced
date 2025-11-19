# Sprint 2: Technical Specifications

**Version:** 1.0  
**Date:** November 10, 2025

---

## 1. Promotional Engine Architecture

### 1.1 Core Components

#### PromoMechanics Class

```python
class PromoMechanics:
    """
    Handles promotional discount mechanics
    """
    
    def __init__(self, hmm_model, products_df):
        self.hmm = hmm_model
        self.products = products_df
        
        # Discount depth by HMM state
        self.depth_ranges = {
            0: (0.00, 0.05),   # Regular
            1: (0.10, 0.25),   # Feature  
            2: (0.25, 0.50),   # Deep discount
            3: (0.50, 0.70)    # Clearance
        }
    
    def get_promotion_depth(self, product_id, hmm_state, store_id):
        """
        Calculate discount percentage
        
        Inputs:
            - product_id: int
            - hmm_state: int (0-3)
            - store_id: int
        
        Returns:
            - discount: float (0.0-0.7)
        """
        min_disc, max_disc = self.depth_ranges[hmm_state]
        
        # Sample from range
        discount = np.random.uniform(min_disc, max_disc)
        
        # Add product-specific variation
        if product_id in self.product_promo_tendencies:
            tendency = self.product_promo_tendencies[product_id]
            discount *= tendency
        
        return np.clip(discount, 0.0, 0.7)
    
    def get_promotion_duration(self, product_id, discount_depth):
        """
        Determine promotion duration in weeks
        
        Logic:
        - Deep discounts: shorter (1-2 weeks)
        - Moderate discounts: longer (2-3 weeks)
        """
        if discount_depth > 0.35:
            return np.random.choice([1, 2], p=[0.7, 0.3])
        elif discount_depth > 0.15:
            return np.random.choice([2, 3], p=[0.6, 0.4])
        else:
            return np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
```

#### PromoDisplays Class

```python
class PromoDisplays:
    """
    Manages display type allocation
    """
    
    def __init__(self, n_stores):
        self.n_stores = n_stores
        
        # Capacity constraints per store
        self.capacity = {
            'end_cap': 10,
            'feature_display': 3
        }
        
        # Display effectiveness (utility boost multiplier)
        self.effectiveness = {
            'none': 0.0,
            'shelf_tag': 0.3,
            'end_cap': 0.8,
            'feature_display': 1.2
        }
    
    def allocate_displays(self, promoted_products, discount_depths, store_id):
        """
        Allocate display types to promoted products
        
        Rules:
        1. Deepest discounts get best displays
        2. Subject to capacity constraints
        3. Shelf tags unlimited
        
        Returns:
            Dict[product_id, display_type]
        """
        # Sort by discount depth (descending)
        sorted_promos = sorted(
            zip(promoted_products, discount_depths),
            key=lambda x: x[1],
            reverse=True
        )
        
        allocations = {}
        endcaps_used = 0
        features_used = 0
        
        for product_id, depth in sorted_promos:
            if depth > 0.35 and features_used < self.capacity['feature_display']:
                allocations[product_id] = 'feature_display'
                features_used += 1
            elif depth > 0.20 and endcaps_used < self.capacity['end_cap']:
                allocations[product_id] = 'end_cap'
                endcaps_used += 1
            elif depth > 0.10:
                allocations[product_id] = 'shelf_tag'
            else:
                allocations[product_id] = 'none'
        
        return allocations
```

#### PromoFeatures Class

```python
class PromoFeatures:
    """
    Manages promotional advertising (in-ad, mailer)
    """
    
    def __init__(self, products_df):
        self.products = products_df
        
        # Feature selection probabilities by display type
        self.feature_prob = {
            'feature_display': {'in_ad': 0.9, 'mailer': 0.6},
            'end_cap': {'in_ad': 0.5, 'mailer': 0.3},
            'shelf_tag': {'in_ad': 0.1, 'mailer': 0.05}
        }
    
    def select_featured_products(self, promo_products, display_allocations, 
                                 store_id, week):
        """
        Decide which products appear in ad/mailer
        
        Logic:
        - Higher display prominence → more likely in ad
        - Category balancing (not all same category)
        - Store-specific ad strategy
        
        Returns:
            {
                'in_ad': [product_ids],
                'mailer': [product_ids]
            }
        """
        in_ad = []
        mailer = []
        
        for product_id in promo_products:
            display_type = display_allocations.get(product_id, 'none')
            probs = self.feature_prob.get(display_type, {'in_ad': 0, 'mailer': 0})
            
            if np.random.random() < probs['in_ad']:
                in_ad.append(product_id)
            
            if np.random.random() < probs['mailer']:
                mailer.append(product_id)
        
        return {'in_ad': in_ad, 'mailer': mailer}
```

### 1.2 StorePromoContext Data Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict

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
    promo_depths: Dict[int, float] = field(default_factory=dict)  # product_id → discount
    promo_states: Dict[int, int] = field(default_factory=dict)    # product_id → HMM state
    promo_durations: Dict[int, int] = field(default_factory=dict) # product_id → weeks
    
    # Promo displays
    display_types: Dict[int, str] = field(default_factory=dict)   # product_id → display
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
        
        # Marketing signal computed by MarketingSignalEngine
```

---

## 2. Marketing Signal Engine

### 2.1 Signal Calculation

```python
class MarketingSignalEngine:
    """
    Calculates promotional attractiveness and impacts store visits
    """
    
    def __init__(self):
        # Component weights (sum to 1.0)
        self.weights = {
            'discount_depth': 0.30,
            'promo_breadth': 0.20,
            'display_quality': 0.25,
            'advertising': 0.25
        }
        
        # Calibration parameters
        self.max_lift = 1.5  # Max 50% increase in visit prob
    
    def calculate_signal(self, promo_context: StorePromoContext) -> float:
        """
        Calculate marketing signal strength (0-1)
        
        Components:
        1. Discount depth - how deep are discounts
        2. Promo breadth - how many items on promo
        3. Display quality - end caps, features
        4. Advertising - in-ad, mailer presence
        """
        
        # 1. Discount depth component
        if promo_context.avg_discount_depth > 0:
            depth_score = min(promo_context.avg_discount_depth / 0.40, 1.0)
        else:
            depth_score = 0.0
        
        # 2. Breadth component (assume 1000 products in assortment)
        n_promoted = len(promo_context.promoted_products)
        breadth_score = min(n_promoted / 150, 1.0)  # 15% promoted = max
        
        # 3. Display quality component
        n_endcaps = len(promo_context.end_cap_products)
        n_features = len(promo_context.feature_display_products)
        display_score = (
            (n_endcaps / 10) * 0.7 +  # 10 endcaps = max
            (n_features / 3) * 1.0     # 3 features = max
        ) / 1.7  # Normalize
        display_score = min(display_score, 1.0)
        
        # 4. Advertising component
        n_ad = len(promo_context.in_ad_products)
        n_mailer = len(promo_context.mailer_products)
        ad_score = min((n_ad / 30) * 0.8 + (n_mailer / 20) * 1.0, 1.0)
        
        # Weighted sum
        signal = (
            self.weights['discount_depth'] * depth_score +
            self.weights['promo_breadth'] * breadth_score +
            self.weights['display_quality'] * display_score +
            self.weights['advertising'] * ad_score
        )
        
        return np.clip(signal, 0.0, 1.0)
    
    def adjust_visit_probability(self, base_prob, signal, customer):
        """
        Adjust store visit probability based on marketing signal
        
        Lift = 1.0 + (customer_sensitivity * signal_strength * max_lift_factor)
        """
        promo_sensitivity = getattr(customer, 'beta_promo_sensitivity', 0.5)
        
        lift_factor = 1.0 + (promo_sensitivity * signal * (self.max_lift - 1.0))
        
        adjusted_prob = base_prob * lift_factor
        
        return min(adjusted_prob, 0.95)  # Cap at 95%
```

### 2.2 Integration with Store Loyalty

```python
# In loyalty_engine.py - modify existing method

def update_store_preference(self, customer_id, store_id, experience_score, 
                            week_number, promo_context=None):
    """
    Update store preference with promotional signal
    
    NEW: promo_context parameter
    """
    # Existing logic...
    
    # NEW: Add marketing signal boost
    if promo_context is not None:
        marketing_signal = self.marketing_engine.calculate_signal(promo_context)
        
        # Boost experience if strong promotions
        promo_boost = marketing_signal * 0.3  # Up to +0.3
        experience_score += promo_boost
    
    # Continue with existing logic...
```

---

## 3. Individual Heterogeneity System

### 3.1 Parameter Distributions

```python
class CustomerHeterogeneityEngine:
    """
    Generates individual-level utility parameters
    """
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        
        # Population-level parameter distributions
        self.parameter_specs = {
            # Price sensitivity (negative)
            'beta_price': {
                'distribution': 'normal',
                'mean': -2.5,
                'std': 0.8,
                'bounds': (-5.0, -0.5)
            },
            
            # Promotional discount sensitivity (positive)
            'beta_promo': {
                'distribution': 'normal',
                'mean': 1.8,
                'std': 0.6,
                'bounds': (0.2, 4.0)
            },
            
            # Quality/brand preference (positive)
            'beta_quality': {
                'distribution': 'lognormal',
                'mean': 1.2,
                'std': 0.5,
                'bounds': (0.3, 5.0)
            },
            
            # Display sensitivity (0-1)
            'beta_display': {
                'distribution': 'beta',
                'alpha': 2.0,
                'beta': 5.0,
                'bounds': (0.0, 1.0)
            },
            
            # Feature/ad sensitivity (positive)
            'beta_feature': {
                'distribution': 'normal',
                'mean': 0.8,
                'std': 0.3,
                'bounds': (0.1, 2.0)
            },
            
            # Convenience preference (positive)
            'beta_convenience': {
                'distribution': 'gamma',
                'alpha': 3.0,
                'scale': 0.5,
                'bounds': (0.5, 4.0)
            },
            
            # Promo sensitivity for store visits (0-1)
            'beta_promo_sensitivity': {
                'distribution': 'beta',
                'alpha': 3.0,
                'beta': 3.0,
                'bounds': (0.0, 1.0)
            }
        }
    
    def generate_parameters(self, n_customers):
        """
        Generate individual parameters for all customers
        
        Returns:
            pd.DataFrame with columns: customer_id, beta_price, beta_promo, ...
        """
        params = {'customer_id': np.arange(n_customers)}
        
        for param_name, spec in self.parameter_specs.items():
            # Generate from distribution
            if spec['distribution'] == 'normal':
                values = self.rng.normal(spec['mean'], spec['std'], n_customers)
            elif spec['distribution'] == 'lognormal':
                values = self.rng.lognormal(spec['mean'], spec['std'], n_customers)
            elif spec['distribution'] == 'beta':
                values = self.rng.beta(spec['alpha'], spec['beta'], n_customers)
            elif spec['distribution'] == 'gamma':
                values = self.rng.gamma(spec['alpha'], spec['scale'], n_customers)
            
            # Apply bounds
            values = np.clip(values, spec['bounds'][0], spec['bounds'][1])
            
            params[param_name] = values
        
        return pd.DataFrame(params)
```

### 3.2 Integration with Customer Generator

```python
# In customer_generator.py

def generate_customers(self, n_customers):
    """
    Generate customers with individual heterogeneity
    """
    # Existing demographic generation...
    
    # NEW: Generate individual utility parameters
    heterogeneity_engine = CustomerHeterogeneityEngine(seed=self.config.random_seed)
    utility_params = heterogeneity_engine.generate_parameters(n_customers)
    
    # Merge with demographics
    customers_df = pd.merge(
        demographics_df,
        utility_params,
        on='customer_id'
    )
    
    return customers_df
```

---

## 4. Promotional Response Model

### 4.1 Utility Adjustment

```python
class PromoResponseEngine:
    """
    Calculates customer-specific promotional utility adjustments
    """
    
    def __init__(self):
        # Scaling factors
        self.discount_scale = 5.0
        self.display_scale = 2.0
        self.feature_scale = 1.5
    
    def calculate_promo_utility(self, customer, product_id, promo_context):
        """
        Calculate total promotional utility boost
        
        Components:
        1. Discount response: β_promo * discount * scale
        2. Display response: β_display * effectiveness
        3. Feature response: β_feature * (ad + mailer)
        """
        total_utility = 0.0
        
        # 1. Discount response
        if product_id in promo_context.promo_depths:
            discount = promo_context.promo_depths[product_id]
            discount_utility = (
                customer.beta_promo * discount * self.discount_scale
            )
            total_utility += discount_utility
        
        # 2. Display response
        if product_id in promo_context.display_types:
            display_type = promo_context.display_types[product_id]
            
            # Get display effectiveness
            display_effect = {
                'none': 0.0,
                'shelf_tag': 0.3,
                'end_cap': 0.8,
                'feature_display': 1.2
            }[display_type]
            
            display_utility = (
                customer.beta_display * display_effect * self.display_scale
            )
            total_utility += display_utility
        
        # 3. Feature/advertising response
        feature_utility = 0.0
        if product_id in promo_context.in_ad_products:
            feature_utility += 0.7
        if product_id in promo_context.mailer_products:
            feature_utility += 0.8
        
        if feature_utility > 0:
            total_utility += customer.beta_feature * feature_utility * self.feature_scale
        
        return total_utility
    
    def calculate_expected_quantity(self, customer, product_id, promo_context, 
                                    base_quantity=1):
        """
        Calculate expected purchase quantity with stockpiling
        Uses arc elasticity for forward-looking behavior
        """
        # If deep discount, consider stockpiling
        if product_id in promo_context.promo_depths:
            discount = promo_context.promo_depths[product_id]
            
            if discount > 0.30:  # Deep discount threshold
                # Use arc elasticity
                state = promo_context.promo_states.get(product_id, 0)
                stockpile_multiplier = self.calculate_stockpile_multiplier(
                    discount, state, customer
                )
                return int(base_quantity * stockpile_multiplier)
        
        return base_quantity
```

### 4.2 Arc Elasticity Integration

```python
def calculate_stockpile_multiplier(self, discount, hmm_state, customer):
    """
    Calculate quantity multiplier for stockpiling
    
    Logic:
    - Deep discount + expectation of price increase → buy extra
    - Customer-specific forward-looking behavior
    """
    # HMM state indicates likelihood of returning to regular price
    if hmm_state == 2:  # Deep discount state
        # Expect transition back to regular
        expected_price_increase = 0.35  # Average
        
        # Customer forward-looking tendency
        foresight = getattr(customer, 'foresight_factor', 0.5)
        
        # Stockpile multiplier
        multiplier = 1.0 + (discount * expected_price_increase * foresight * 4.0)
        
        return min(multiplier, 5.0)  # Cap at 5x
    
    return 1.0
```

---

## 5. Integration Points

### 5.1 Main Generator Orchestration

```python
# In main_generator.py

def load_promotional_system(self):
    """Initialize promotional components"""
    
    # Load HMM (already exists)
    self.hmm_model = self.load_hmm_model()
    
    # NEW: Initialize promotional components
    self.promo_mechanics = PromoMechanics(self.hmm_model, self.products)
    self.promo_displays = PromoDisplays(len(self.stores))
    self.promo_features = PromoFeatures(self.products)
    self.marketing_signal = MarketingSignalEngine()
    
    # NEW: Initialize heterogeneity
    self.heterogeneity_engine = CustomerHeterogeneityEngine()
    
    # NEW: Initialize promo response
    self.promo_response = PromoResponseEngine()

def generate_week_promotions(self, week_number):
    """
    Generate promotional context for all stores for this week
    
    Returns:
        Dict[store_id, StorePromoContext]
    """
    store_promos = {}
    
    for store_id in self.stores['store_id']:
        # Get HMM states for products
        product_states = self.hmm_model.get_states_for_week(week_number)
        
        # Determine which products to promote
        promoted_products = self.select_promoted_products(product_states, store_id)
        
        # Get discount depths
        promo_depths = {
            pid: self.promo_mechanics.get_promotion_depth(pid, product_states[pid], store_id)
            for pid in promoted_products
        }
        
        # Allocate displays
        display_allocations = self.promo_displays.allocate_displays(
            promoted_products, 
            list(promo_depths.values()), 
            store_id
        )
        
        # Select featured products
        features = self.promo_features.select_featured_products(
            promoted_products, 
            display_allocations, 
            store_id, 
            week_number
        )
        
        # Create context
        context = StorePromoContext(
            store_id=store_id,
            week_number=week_number,
            promoted_products=promoted_products,
            promo_depths=promo_depths,
            promo_states=product_states,
            display_types=display_allocations,
            end_cap_products=[p for p, d in display_allocations.items() if d == 'end_cap'],
            feature_display_products=[p for p, d in display_allocations.items() if d == 'feature_display'],
            in_ad_products=features['in_ad'],
            mailer_products=features['mailer']
        )
        
        # Calculate marketing signal
        context.marketing_signal_strength = self.marketing_signal.calculate_signal(context)
        context.compute_metrics()
        
        store_promos[store_id] = context
    
    return store_promos
```

### 5.2 Transaction Generator Integration

```python
# In transaction_generator.py

def generate_week_transactions_vectorized(self, week_number, ..., promo_contexts):
    """
    NEW: promo_contexts parameter - Dict[store_id, StorePromoContext]
    """
    
    # Existing store visit logic...
    
    # NEW: Adjust visit probabilities with marketing signal
    for i, customer_id in enumerate(visiting_customers):
        store_id = customer_stores[i]
        promo_context = promo_contexts[store_id]
        
        visit_probs[i] = self.marketing_signal.adjust_visit_probability(
            visit_probs[i], 
            promo_context.marketing_signal_strength,
            customers.iloc[customer_id]
        )
    
    # Existing product choice logic...
    
    # NEW: Add promotional utility adjustments
    for i, customer_id in enumerate(shopping_customers):
        store_id = customer_stores[i]
        promo_context = promo_contexts[store_id]
        
        # Calculate promo utility for each product
        promo_utilities = np.array([
            self.promo_response.calculate_promo_utility(
                customers.iloc[customer_id],
                product_id,
                promo_context
            )
            for product_id in available_products
        ])
        
        # Add to base utilities
        all_utilities[i, :] += promo_utilities
    
    # Continue with existing sampling logic...
```

---

## 6. Data Structures Summary

### Store Promo Context
- `store_id`, `week_number`
- `promoted_products`, `promo_depths`, `promo_states`
- `display_types`, `end_cap_products`, `feature_display_products`
- `in_ad_products`, `mailer_products`
- `marketing_signal_strength`

### Customer Heterogeneity Parameters
- `beta_price`, `beta_promo`, `beta_quality`
- `beta_display`, `beta_feature`, `beta_convenience`
- `beta_promo_sensitivity`

### Promotional Tracking
- Weekly promo context per store
- Customer response logged in transactions
- Display effectiveness metrics

---

## 7. Non-Linear Utility System

### 7.1 Core Components

#### NonLinearUtilityEngine Class

```python
class NonLinearUtilityEngine:
    """
    Non-linear utility transformations for realistic price response
    
    Features:
    1. Log-price utility (diminishing sensitivity)
    2. Reference price effects (loss aversion)
    3. Threshold effects (psychological pricing)
    4. Quadratic quality (saturation)
    """
    
    def __init__(self, config):
        # Reference price tracking
        self.reference_prices = {}  # product_id → reference price
        self.price_history = {}     # product_id → deque of recent prices
        self.reference_window = 10  # Number of weeks for reference
        
        # Loss aversion parameters (Kahneman & Tversky, 1979)
        self.lambda_gain = 1.0      # Utility from gains
        self.lambda_loss = 2.5      # Utility from losses (2.5x stronger)
        
        # Threshold effects
        self.price_thresholds = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        self.threshold_sensitivity = 0.10  # Window around threshold
        
        # Quality saturation
        self.saturation_coef = 0.1
    
    def calculate_price_utility(self, product_id, current_price, customer):
        """
        Calculate non-linear price utility
        
        Components:
        1. Log-price: β_price * log(price)
        2. Reference price: β_ref * f(price - ref_price)
        3. Threshold: bonus/penalty near psychological thresholds
        
        Returns:
            float: Total price utility
        """
        total_utility = 0.0
        
        # 1. Log-price utility (diminishing sensitivity)
        # As price increases, marginal disutility decreases
        log_price_utility = customer.beta_price * np.log(current_price + 0.01)
        total_utility += log_price_utility
        
        # 2. Reference price effect (if reference exists)
        ref_price = self.reference_prices.get(product_id)
        if ref_price is not None:
            ref_utility = self._calculate_reference_utility(
                current_price, ref_price, customer
            )
            total_utility += ref_utility
        
        # 3. Threshold effects
        threshold_utility = self._calculate_threshold_effect(current_price)
        total_utility += threshold_utility
        
        return total_utility
    
    def _calculate_reference_utility(self, current_price, ref_price, customer):
        """
        Calculate utility based on deviation from reference price
        
        Loss aversion: Losses loom larger than gains
        - Price above reference: loss (strong negative utility)
        - Price below reference: gain (moderate positive utility)
        """
        price_diff = ref_price - current_price
        
        # Get customer reference sensitivity (new parameter)
        beta_ref = getattr(customer, 'beta_reference', 1.0)
        
        if price_diff >= 0:
            # Gain: current price below reference (good deal)
            utility = beta_ref * price_diff * self.lambda_gain
        else:
            # Loss: current price above reference (bad deal)
            utility = beta_ref * price_diff * self.lambda_loss
        
        return utility
    
    def _calculate_threshold_effect(self, price):
        """
        Psychological price thresholds (left-digit effect)
        
        $0.99 feels significantly cheaper than $1.00
        $4.95 feels cheaper than $5.00
        
        Returns small utility bonus if just below threshold
        """
        for threshold in self.price_thresholds:
            # Check if price is just below threshold
            if threshold - self.threshold_sensitivity < price < threshold:
                # Bonus for being below threshold
                return 0.2
            
            # Small penalty for being just above
            if threshold < price < threshold + self.threshold_sensitivity:
                return -0.1
        
        return 0.0
    
    def calculate_quality_utility(self, quality, customer):
        """
        Quadratic quality utility (diminishing returns)
        
        U_quality = β_quality * quality - β_saturation * quality²
        
        Models saturation: at high quality levels, additional
        quality improvements have less impact
        """
        beta_quality = customer.beta_quality
        
        linear_term = beta_quality * quality
        saturation_term = self.saturation_coef * (quality ** 2)
        
        return linear_term - saturation_term
    
    def update_reference_price(self, product_id, observed_price):
        """
        Update reference price with new observation
        
        Uses exponentially weighted moving average
        More recent prices have higher weight
        """
        if product_id not in self.price_history:
            self.price_history[product_id] = []
        
        # Add new price
        self.price_history[product_id].append(observed_price)
        
        # Keep only recent history
        if len(self.price_history[product_id]) > self.reference_window:
            self.price_history[product_id] = self.price_history[product_id][-self.reference_window:]
        
        # Calculate EWMA reference price
        prices = np.array(self.price_history[product_id])
        n = len(prices)
        
        # Exponential weights (more weight to recent prices)
        decay = 0.3
        weights = np.exp(-decay * np.arange(n, 0, -1))
        weights /= weights.sum()
        
        ref_price = np.average(prices, weights=weights)
        
        self.reference_prices[product_id] = ref_price
    
    def get_price_response_curve(self, product_id, customer, price_range=None):
        """
        Generate price-utility curve for visualization
        
        Used for validation and debugging
        """
        if price_range is None:
            ref_price = self.reference_prices.get(product_id, 5.0)
            price_range = np.linspace(ref_price * 0.5, ref_price * 2.0, 100)
        
        utilities = [
            self.calculate_price_utility(product_id, p, customer)
            for p in price_range
        ]
        
        return price_range, utilities
```

### 7.2 Reference Price Theory

**Theoretical Foundation:**

Reference price models are based on **Adaptation-Level Theory** (Helson, 1964) and **Prospect Theory** (Kahneman & Tversky, 1979):

1. **Adaptation Level**: Customers form internal reference prices based on past observations
2. **Loss Aversion**: Losses (prices above reference) have 2-3x stronger impact than equivalent gains
3. **Value Function**: Concave for gains, convex for losses (S-shaped)

**Implementation:**

```
Reference Price = EWMA(past prices)
                = Σ w_t * price_t, where w_t decreases with time

Utility from price = α * log(price)  [base sensitivity]
                   + β * (ref - price) * λ  [reference effect]
                   
where λ = 1.0 if price < ref (gain)
        = 2.5 if price > ref (loss)
```

### 7.3 Threshold Effects

**Left-Digit Effect** (Thomas & Morwitz, 2005):

Consumers overweight the leftmost digit:
- $2.99 vs $3.00: Perceived as significantly different
- $4.95 vs $5.05: Larger gap than actual 10 cents

**Implementation:**

```python
# Identify thresholds: [1, 5, 10, 20, 50, 100]
# If price in (threshold - 0.10, threshold):
#     utility += 0.2  # "Good deal" bonus
# If price in (threshold, threshold + 0.10):
#     utility -= 0.1  # "Overpriced" penalty
```

### 7.4 Integration with Utility Engine

```python
# In utility_engine.py

class EnhancedUtilityEngine:
    def __init__(self, ..., nonlinear_engine):
        self.nonlinear_engine = nonlinear_engine
        # Other components...
    
    def compute_utility(self, customer, product, context):
        """
        Compute total utility with non-linear effects
        """
        # Use non-linear price utility (instead of linear)
        price_utility = self.nonlinear_engine.calculate_price_utility(
            product['product_id'],
            context['current_price'],
            customer
        )
        
        # Use quadratic quality utility
        quality_utility = self.nonlinear_engine.calculate_quality_utility(
            product['quality'],
            customer
        )
        
        # Other components (brand, promo, etc.)
        # ...
        
        total_utility = price_utility + quality_utility + other_utilities
        
        return total_utility
```

### 7.5 Validation Tests

```python
def test_price_response_curves():
    """
    Validate non-linear price response
    """
    # Split data: first 2 years for learning, last 6 months for validation
    train_transactions = transactions[transactions['week_no'] <= 100]
    test_transactions = transactions[transactions['week_no'] > 100]
    
    # Learn patterns from training data
    learner = NonLinearUtilityEngine(config)
    learner.learn_seasonal_patterns()
    
    # Calculate actual patterns in test data
    test_patterns = calculate_actual_patterns(test_transactions)
    
    # Compare correlations
    correlations = []
    
    for product_id in learner.seasonal_indices.keys():
        if product_id not in test_patterns:
            continue
        
        learned = np.array([learner.seasonal_indices[product_id][w] for w in range(1, 53)])
        actual = np.array([test_patterns[product_id][w] for w in range(1, 53)])
        
        corr = np.corrcoef(learned, actual)[0, 1]
        correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    
    print(f"Average correlation: {avg_corr:.3f}")
    assert avg_corr > 0.65, f"Correlation too low: {avg_corr:.3f}"

def test_reference_price_updating():
    """Test reference price adapts to price changes"""
    learner = NonLinearUtilityEngine(config)
    product_id = 123
    
    # Feed initial prices
    for _ in range(5):
        learner.update_reference_price(product_id, 5.0)
    
    assert abs(learner.reference_prices[product_id] - 5.0) < 0.1
    
    # Feed higher prices
    for _ in range(5):
        learner.update_reference_price(product_id, 7.0)
    
    # Reference should increase (but not to 7.0 due to weighting)
    assert 5.5 < learner.reference_prices[product_id] < 6.5
```

---

## 8. Seasonality Learning System

### 8.1 Core Components

#### SeasonalityLearningEngine Class

```python
class SeasonalityLearningEngine:
    """
    Learn seasonal patterns from Dunnhumby transaction data
    
    Extracts product-specific and category-level seasonal indices
    Replaces hard-coded seasonality with data-driven patterns
    """
    
    def __init__(self, transactions_df, products_df, config):
        self.transactions = transactions_df
        self.products = products_df
        self.config = config
        
        # Learned patterns
        self.seasonal_indices = {}  # product_id → {week_of_year: multiplier}
        self.category_patterns = {}  # category → {week_of_year: multiplier}
        self.seasonal_products = {}  # highly seasonal products
        
        # Seasonality metrics
        self.seasonality_scores = {}  # product_id → seasonality strength
    
    def learn_seasonal_patterns(self, min_observations=26):
        """
        Learn seasonal patterns from transaction data
        
        Method:
        1. Aggregate sales by product-week-of-year
        2. Calculate seasonal indices (week_avg / overall_avg)
        3. Smooth indices with moving average
        4. Learn category-level patterns for coverage
        
        Args:
            min_observations: Minimum weeks of data required per product
        """
        print("Learning seasonal patterns from Dunnhumby transactions...")
        
        # Add week-of-year column
        self.transactions['week_of_year'] = (
            (self.transactions['week_no'] - 1) % 52
        ) + 1
        
        # Aggregate sales by product-week
        product_week_sales = self.transactions.groupby(
            ['product_id', 'week_of_year']
        ).agg({
            'quantity': 'sum',
            'sales_value': 'sum'
        }).reset_index()
        
        # Learn patterns for each product
        products_with_patterns = 0
        
        for product_id in tqdm(self.products['product_id'].unique(), 
                               desc="Learning product patterns"):
            product_sales = product_week_sales[
                product_week_sales['product_id'] == product_id
            ]
            
            # Need sufficient data
            if len(product_sales) < min_observations:
                continue
            
            # Calculate seasonal indices
            indices = self._calculate_seasonal_indices(
                product_sales['week_of_year'].values,
                product_sales['quantity'].values
            )
            
            # Calculate seasonality strength
            seasonality = self._calculate_seasonality_strength(indices)
            
            self.seasonal_indices[product_id] = indices
            self.seasonality_scores[product_id] = seasonality
            products_with_patterns += 1
        
        print(f"✅ Learned patterns for {products_with_patterns:,} products")
        
        # Learn category-level patterns for fallback
        self._learn_category_patterns()
        
        # Identify highly seasonal products
        self._identify_seasonal_products()
    
    def _calculate_seasonal_indices(self, weeks, quantities):
        """
        Calculate seasonal index for each week of year
        
        Index = (average sales in week X) / (average sales across all weeks)
        
        Index = 1.0: Average week
        Index > 1.0: Above-average week (high season)
        Index < 1.0: Below-average week (low season)
        """
        overall_avg = quantities.mean()
        
        if overall_avg == 0:
            return {week: 1.0 for week in range(1, 53)}
        
        indices = {}
        
        for week in range(1, 53):
            week_mask = (weeks == week)
            week_count = week_mask.sum()
            
            if week_count > 0:
                week_avg = quantities[week_mask].mean()
                indices[week] = week_avg / overall_avg
            else:
                # No observations for this week - use 1.0
                indices[week] = 1.0
        
        # Smooth indices to reduce noise
        indices = self._smooth_seasonal_indices(indices)
        
        return indices
    
    def _smooth_seasonal_indices(self, indices, window=3):
        """
        Smooth seasonal indices using circular moving average
        
        Window=3: Average of [week-3, ..., week, ..., week+3]
        Circular: Week 52 neighbors include Week 1
        """
        weeks = sorted(indices.keys())
        values = np.array([indices[w] for w in weeks])
        
        smoothed = np.zeros_like(values)
        
        for i in range(len(values)):
            # Circular window
            window_indices = []
            for j in range(-window, window + 1):
                idx = (i + j) % len(values)
                window_indices.append(idx)
            
            window_values = values[window_indices]
            smoothed[i] = window_values.mean()
        
        return {w: s for w, s in zip(weeks, smoothed)}
    
    def _calculate_seasonality_strength(self, indices):
        """
        Calculate how seasonal a product is
        
        Seasonality = max(index) / min(index)
        
        1.0: No seasonality (flat across year)
        2.0: Moderate seasonality (2x peak vs trough)
        5.0+: Highly seasonal (5x+ peak vs trough)
        """
        values = list(indices.values())
        
        if len(values) == 0 or min(values) == 0:
            return 1.0
        
        return max(values) / min(values)
    
    def _learn_category_patterns(self):
        """
        Learn seasonal patterns at category level
        
        Used as fallback when product-specific pattern unavailable
        """
        print("Learning category-level patterns...")
        
        # Merge transactions with product categories
        trans_with_category = self.transactions.merge(
            self.products[['product_id', 'commodity_desc']],
            on='product_id',
            how='left'
        )
        
        # Aggregate by category-week
        category_week_sales = trans_with_category.groupby(
            ['commodity_desc', 'week_of_year']
        )['quantity'].sum().reset_index()
        
        categories_learned = 0
        
        for category in self.products['commodity_desc'].dropna().unique():
            category_sales = category_week_sales[
                category_week_sales['commodity_desc'] == category
            ]
            
            if len(category_sales) < 26:
                continue
            
            indices = self._calculate_seasonal_indices(
                category_sales['week_of_year'].values,
                category_sales['quantity'].values
            )
            
            self.category_patterns[category] = indices
            categories_learned += 1
        
        print(f"✅ Learned patterns for {categories_learned} categories")
    
    def _identify_seasonal_products(self, threshold=1.5):
        """
        Identify highly seasonal products for special handling
        
        Threshold = 1.5: Products with 50%+ variation across year
        """
        self.seasonal_products = {}
        
        for product_id, score in self.seasonality_scores.items():
            if score >= threshold:
                indices = self.seasonal_indices[product_id]
                
                self.seasonal_products[product_id] = {
                    'seasonality_score': score,
                    'peak_week': max(indices, key=indices.get),
                    'peak_index': max(indices.values()),
                    'trough_week': min(indices, key=indices.get),
                    'trough_index': min(indices.values())
                }
        
        print(f"✅ Identified {len(self.seasonal_products):,} highly seasonal products")
    
    def get_seasonal_multiplier(self, product_id, week_of_year, category=None):
        """
        Get seasonal multiplier for a product in a specific week
        
        Fallback hierarchy:
        1. Product-specific pattern (if available)
        2. Category pattern (if product not available)
        3. No seasonality (return 1.0)
        
        Args:
            product_id: int
            week_of_year: int (1-52)
            category: str (optional, for fallback)
        
        Returns:
            float: Seasonal multiplier (typically 0.5 to 2.0)
        """
        # Normalize week_of_year
        week_of_year = ((week_of_year - 1) % 52) + 1
        
        # Try product-specific
        if product_id in self.seasonal_indices:
            return self.seasonal_indices[product_id].get(week_of_year, 1.0)
        
        # Try category
        if category and category in self.category_patterns:
            return self.category_patterns[category].get(week_of_year, 1.0)
        
        # No pattern available
        return 1.0
    
    def save_learned_patterns(self, output_path):
        """Save learned patterns to disk"""
        import pickle
        
        patterns = {
            'seasonal_indices': self.seasonal_indices,
            'category_patterns': self.category_patterns,
            'seasonal_products': self.seasonal_products,
            'seasonality_scores': self.seasonality_scores
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(patterns, f)
        
        print(f"✅ Saved seasonal patterns to {output_path}")
    
    @staticmethod
    def load_learned_patterns(input_path):
        """Load previously learned patterns"""
        import pickle
        
        with open(input_path, 'rb') as f:
            patterns = pickle.load(f)
        
        return patterns
    
    def generate_seasonality_report(self):
        """Generate summary report of learned patterns"""
        report = {
            'total_products': len(self.seasonal_indices),
            'total_categories': len(self.category_patterns),
            'highly_seasonal': len(self.seasonal_products),
            'avg_seasonality_score': np.mean(list(self.seasonality_scores.values())),
            'top_seasonal_products': sorted(
                self.seasonal_products.items(),
                key=lambda x: x[1]['seasonality_score'],
                reverse=True
            )[:10]
        }
        
        return report
```

### 8.2 Integration with Seasonality Engine

```python
# In seasonality_engine.py - update existing engine

class SeasonalityEngine:
    """
    Manages seasonal effects on product demand
    
    Now supports both:
    - Learned patterns from data
    - Hard-coded patterns (legacy/fallback)
    """
    
    def __init__(self, learned_patterns_path=None):
        self.use_learned = False
        self.seasonal_indices = {}
        self.category_patterns = {}
        
        if learned_patterns_path and os.path.exists(learned_patterns_path):
            self._load_learned_patterns(learned_patterns_path)
            self.use_learned = True
            print(f"✅ Using learned seasonal patterns from {learned_patterns_path}")
        else:
            print("⚠️  Using hard-coded seasonal patterns (fallback)")
            self._init_hardcoded_patterns()
    
    def _load_learned_patterns(self, path):
        """Load learned patterns from file"""
        patterns = SeasonalityLearningEngine.load_learned_patterns(path)
        
        self.seasonal_indices = patterns['seasonal_indices']
        self.category_patterns = patterns['category_patterns']
        self.seasonal_products = patterns.get('seasonal_products', {})
    
    def get_seasonal_effect(self, product_id, week_number, category=None):
        """
        Get seasonal multiplier for utility calculation
        
        Returns:
            float: Multiplier (1.0 = no effect)
        """
        if self.use_learned:
            week_of_year = ((week_number - 1) % 52) + 1
            
            # Try product-specific
            if product_id in self.seasonal_indices:
                return self.seasonal_indices[product_id].get(week_of_year, 1.0)
            
            # Try category
            if category and category in self.category_patterns:
                return self.category_patterns[category].get(week_of_year, 1.0)
            
            return 1.0
        else:
            # Use hard-coded patterns
            return self._get_hardcoded_effect(week_number)
```

### 8.3 Standalone Learning Script

```python
# scripts/learn_seasonal_patterns.py

"""
Standalone script to learn seasonal patterns from Dunnhumby
"""

import argparse
import pandas as pd
from retailsynth.engines.seasonality_learning import SeasonalityLearningEngine

def main():
    parser = argparse.ArgumentParser(description='Learn seasonal patterns from Dunnhumby')
    parser.add_argument('--transactions', required=True, help='Path to transaction_data.csv')
    parser.add_argument('--products', required=True, help='Path to product.csv')
    parser.add_argument('--output', required=True, help='Output path for learned patterns')
    parser.add_argument('--min-obs', type=int, default=26, help='Min weeks required')
    
    args = parser.parse_args()
    
    print("Loading Dunnhumby data...")
    transactions = pd.read_csv(args.transactions)
    products = pd.read_csv(args.products)
    
    print(f"Loaded {len(transactions):,} transactions, {len(products):,} products")
    
    # Learn patterns
    learner = SeasonalityLearningEngine(transactions, products, config=None)
    learner.learn_seasonal_patterns(min_observations=args.min_obs)
    
    # Save patterns
    learner.save_learned_patterns(args.output)
    
    # Generate report
    report = learner.generate_seasonality_report()
    print("\n" + "="*70)
    print("SEASONALITY LEARNING REPORT")
    print("="*70)
    print(f"Total products with patterns: {report['total_products']:,}")
    print(f"Total categories with patterns: {report['total_categories']}")
    print(f"Highly seasonal products: {report['highly_seasonal']:,}")
    print(f"Avg seasonality score: {report['avg_seasonality_score']:.2f}")
    print("\nTop 10 most seasonal products:")
    for product_id, info in report['top_seasonal_products']:
        print(f"  Product {product_id}: {info['seasonality_score']:.2f}x "
              f"(peak week {info['peak_week']}, trough week {info['trough_week']})")

if __name__ == "__main__":
    main()
```

### 8.4 Validation Tests

```python
def test_seasonal_pattern_correlation():
    """
    Validate learned patterns correlate with holdout data
    """
    # Split data: first 2 years for learning, last 6 months for validation
    train_transactions = transactions[transactions['week_no'] <= 100]
    test_transactions = transactions[transactions['week_no'] > 100]
    
    # Learn patterns from training data
    learner = SeasonalityLearningEngine(train_transactions, products, config)
    learner.learn_seasonal_patterns()
    
    # Calculate actual patterns in test data
    test_patterns = calculate_actual_patterns(test_transactions)
    
    # Compare correlations
    correlations = []
    
    for product_id in learner.seasonal_indices.keys():
        if product_id not in test_patterns:
            continue
        
        learned = np.array([learner.seasonal_indices[product_id][w] for w in range(1, 53)])
        actual = np.array([test_patterns[product_id][w] for w in range(1, 53)])
        
        corr = np.corrcoef(learned, actual)[0, 1]
        correlations.append(corr)
    
    avg_corr = np.mean(correlations)
    
    print(f"Average correlation: {avg_corr:.3f}")
    assert avg_corr > 0.65, f"Correlation too low: {avg_corr:.3f}"

def test_seasonal_multiplier_ranges():
    """Test that seasonal multipliers are reasonable"""
    learner = SeasonalityLearningEngine(transactions, products, config)
    learner.learn_seasonal_patterns()
    
    for product_id, indices in learner.seasonal_indices.items():
        values = list(indices.values())
        
        # Should be roughly centered around 1.0
        assert 0.9 < np.mean(values) < 1.1, f"Product {product_id} not centered"
        
        # Should not have extreme outliers
        assert min(values) > 0.1, f"Product {product_id} has extreme minimum"
        assert max(values) < 10.0, f"Product {product_id} has extreme maximum"
```

---

**This spec provides the complete technical foundation for Sprint 2 implementation, including non-linear utilities and seasonality learning.**
