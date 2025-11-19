# RetailSynth Sprint 2: Action Plan

**Date:** November 10, 2025  
**Duration:** 3-4 weeks  
**Status:** Ready to start

---

## 🎯 Sprint 2 Goals

Based on Sprint 1 completion and your requirements:

1. **Split Price & Promo** → Separate engines (pricing for base, promo for discounts)
2. **Promo Organization** → Mechanics (depth), Displays (location), Features (ads)
3. **Marketing Signal** → Promo attractiveness impacts store visit probability
4. **Individual Heterogeneity** → Replace archetypes with distributions
5. **Promo Response** → Customer-specific promotional sensitivity
6. **Non-Linear Utilities** → Log/quadratic transformations, reference prices
7. **Seasonality Learning** → Extract patterns from Dunnhumby data

---

## 📊 Current State (Sprint 1 Complete)

| Component | Status | Location |
|-----------|--------|----------|
| Product Catalog | ✅ Complete | 19,671 products |
| HMM Price Dynamics | ✅ Complete | `price_hmm.py` |
| Cross-Price Elasticity | ✅ Complete | `cross_price_elasticity.py` |
| Arc Elasticity | ⚠️ Learned, not integrated | `arc_elasticity.py` |
| Purchase History | ✅ Complete | `purchase_history_engine.py` |
| Basket Composition | ✅ Complete | `basket_composer.py` |
| **Promotional Engine** | ❌ Empty | `promotional_engine.py` (0 bytes) |
| **Individual Heterogeneity** | ❌ Archetype-based | Needs implementation |

---

## 🏗️ Sprint 2 Architecture

```
Sprint 2 Flow:
┌─────────────────────┐
│  Pricing Engine     │ → Base prices (inflation, cost, competition)
└─────────────────────┘
          ↓
┌─────────────────────┐
│ Promotional Engine  │ → Promo overlay (mechanics, displays, features)
│  • Depth (discount) │
│  • Display (endcap) │
│  • Feature (in-ad)  │
└─────────────────────┘
          ↓
┌─────────────────────┐
│ Store Promo Context │ → Store-week specific promo state
└─────────────────────┘
          ↓
┌─────────────────────┐
│ Marketing Signal    │ → Impacts store visit probability
└─────────────────────┘
          ↓
┌─────────────────────┐
│  Heterogeneity      │ → Individual customer parameters
│  β_promo ~ N(μ,σ)   │
└─────────────────────┘
          ↓
┌─────────────────────┐
│  Promo Response     │ → Customer-specific utility boost
└─────────────────────┘
```

---

## 📋 Task Breakdown

### **Phase 2.1: Pricing-Promo Separation** (3 days)

**Tasks:**
1. Refactor `pricing_engine.py` - Remove promo logic, keep base prices only
2. Build `promotional_engine.py` - Full promo system
3. Integrate with transaction generator

**Deliverables:**
- Clean pricing engine (base prices)
- Comprehensive promotional engine
- Unit tests

---

### **Phase 2.2: Promo Organization** (3 days)

**Tasks:**
1. **Promo Mechanics** - Discount depth, frequency, duration
2. **Promo Displays** - End cap, shelf tag, feature display
3. **Store Promo Context** - Complete state per store-week

**Implementation:**
```python
@dataclass
class StorePromoContext:
    store_id: int
    week_number: int
    
    # Promo mechanics
    promoted_products: List[int]
    promo_depths: Dict[int, float]  # product_id → discount %
    
    # Promo displays
    display_types: Dict[int, str]  # product_id → display type
    end_cap_products: List[int]
    feature_display_products: List[int]
    
    # Promo features (advertising)
    in_ad_products: List[int]
    mailer_products: List[int]
    
    # Marketing signal
    marketing_strength: float  # 0-1 scale
```

**Deliverables:**
- PromoMechanics class
- PromoDisplays class
- StorePromoContext dataclass

---

### **Phase 2.3: Marketing Signal** (3 days)

**Tasks:**
1. Calculate marketing signal from promo context
2. Integrate with store visit probability
3. Validate lift from promotions

**Implementation:**
```python
class MarketingSignalEngine:
    def calculate_signal(self, promo_context):
        """
        Signal = f(discount_depth, n_promos, display_quality, advertising)
        Returns: float 0-1
        """
        pass
    
    def adjust_visit_probability(self, base_prob, signal, customer):
        """
        Strong promotions → Higher visit probability
        """
        lift = 1.0 + (customer.promo_sensitivity * signal)
        return base_prob * lift
```

**Deliverables:**
- MarketingSignalEngine class
- Visit probability adjustment
- Validation tests

---

### **Phase 2.4: Individual Heterogeneity** (4 days)

**Tasks:**
1. Design parameter distributions
2. Generate individual parameters
3. Update utility engine
4. Validate distributions

**Implementation:**
```python
class CustomerHeterogeneityEngine:
    distributions = {
        'beta_price': Normal(μ=-2.5, σ=0.8),
        'beta_promo': Normal(μ=1.8, σ=0.6),
        'beta_quality': LogNormal(μ=1.2, σ=0.5),
        'beta_display': Beta(α=2, β=5),
        'beta_feature': Normal(μ=0.8, σ=0.3),
    }
    
    def generate_customer_parameters(self, n_customers):
        """Generate individual params for all customers"""
        pass
```

**Deliverables:**
- CustomerHeterogeneityEngine class
- Individual parameter generation
- Updated utility calculation

---

### **Phase 2.5: Promo Response** (3 days)

**Tasks:**
1. Build promo response model
2. Integrate with utility engine
3. **Integrate arc elasticity** (stockpiling on deep discounts)

**Implementation:**
```python
class PromoResponseEngine:
    def calculate_promo_utility(self, customer, product, promo_context):
        """
        Utility boost = β_promo * discount
                      + β_display * display_effect
                      + β_feature * ad_presence
        """
        discount_utility = customer.beta_promo * discount * 5.0
        display_utility = customer.beta_display * display_effect
        feature_utility = customer.beta_feature * (in_ad + mailer)
        
        return discount_utility + display_utility + feature_utility
```

**Deliverables:**
- PromoResponseEngine class
- Customer-specific response
- Arc elasticity integration

### **Phase 2.5: Promo Response** (3 days)

**Tasks:**
1. Build promo response model
2. Integrate with utility engine
3. **Integrate arc elasticity** (stockpiling on deep discounts)

**Implementation:**
```python
class PromoResponseEngine:
    def calculate_promo_utility(self, customer, product, promo_context):
        """
        Utility boost = β_promo * discount
                      + β_display * display_effect
                      + β_feature * ad_presence
        """
        discount_utility = customer.beta_promo * discount * 5.0
        display_utility = customer.beta_display * display_effect
        feature_utility = customer.beta_feature * (in_ad + mailer)
        
        return discount_utility + display_utility + feature_utility
```

**Deliverables:**
- PromoResponseEngine class
- Customer-specific response
- Arc elasticity integration

---

### **Phase 2.6: Non-Linear Utilities** (3 days)

**Goal:** Replace linear price effects with realistic non-linear transformations

**Tasks:**
1. **Log/Quadratic Transformations** (Day 1)
   - Implement log-price utility (diminishing sensitivity)
   - Add quadratic quality effects (saturation)
   - Test different functional forms

2. **Reference Prices** (Day 2)
   - Calculate reference price per product (moving average)
   - Implement gain/loss asymmetry (loss aversion)
   - Model price expectations

3. **Threshold Effects** (Day 3)
   - Implement price thresholds ($0.99 vs $1.00)
   - Add discount perception thresholds (20% feels different than 19%)
   - Test behavioral discontinuities

**Implementation:**
```python
class NonLinearUtilityEngine:
    """
    Non-linear utility transformations for realistic price response
    """
    
    def __init__(self):
        # Reference price tracking
        self.reference_prices = {}  # product_id → reference price
        self.price_history = {}     # product_id → [recent prices]
        
        # Loss aversion parameters
        self.lambda_gain = 1.0      # Gain sensitivity
        self.lambda_loss = 2.5      # Loss sensitivity (2.5x stronger)
    
    def calculate_price_utility(self, product_id, current_price, customer):
        """
        Non-linear price utility with reference prices
        
        U_price = β_price * log(price)  [Log transformation]
                + β_ref * (ref_price - price) * lambda  [Reference effect]
        """
        # 1. Log-price utility (diminishing sensitivity)
        log_price_utility = customer.beta_price * np.log(current_price + 0.01)
        
        # 2. Reference price effect
        ref_price = self.get_reference_price(product_id)
        price_diff = ref_price - current_price
        
        # Loss aversion (asymmetric response)
        if price_diff >= 0:  # Gain (price below reference)
            ref_utility = customer.beta_ref * price_diff * self.lambda_gain
        else:  # Loss (price above reference)
            ref_utility = customer.beta_ref * price_diff * self.lambda_loss
        
        # 3. Threshold effects
        threshold_utility = self.calculate_threshold_effect(current_price)
        
        return log_price_utility + ref_utility + threshold_utility
    
    def calculate_quality_utility(self, quality, customer):
        """
        Quadratic quality utility (diminishing returns)
        
        U_quality = β_quality * quality - β_saturation * quality²
        """
        linear_term = customer.beta_quality * quality
        saturation_term = 0.1 * (quality ** 2)  # Diminishing returns
        
        return linear_term - saturation_term
    
    def get_reference_price(self, product_id):
        """
        Calculate reference price as exponentially weighted moving average
        
        Reference price = adaptation-level theory (Helson, 1964)
        """
        if product_id not in self.price_history:
            return None
        
        prices = self.price_history[product_id]
        
        # Exponential weighting (recent prices matter more)
        weights = np.exp(-0.3 * np.arange(len(prices), 0, -1))
        weights /= weights.sum()
        
        ref_price = np.average(prices, weights=weights)
        
        return ref_price
    
    def calculate_threshold_effect(self, price):
        """
        Psychological price thresholds
        
        $0.99 feels cheaper than $1.00 (left-digit effect)
        """
        # Check if just below threshold
        thresholds = [1.0, 5.0, 10.0, 20.0, 50.0]
        
        for threshold in thresholds:
            if threshold - 0.10 < price < threshold:
                return 0.2  # Small positive utility for "good deal"
        
        return 0.0
    
    def update_price_history(self, product_id, price):
        """Update price history for reference price calculation"""
        if product_id not in self.price_history:
            self.price_history[product_id] = []
        
        self.price_history[product_id].append(price)
        
        # Keep last 10 observations
        if len(self.price_history[product_id]) > 10:
            self.price_history[product_id] = self.price_history[product_id][-10:]
        
        # Update reference price
        self.reference_prices[product_id] = self.get_reference_price(product_id)
```

**Validation:**
```python
# Price response curves
def test_price_response_curves():
    """
    Test that price response shows:
    1. Diminishing sensitivity (log transformation)
    2. Loss aversion (asymmetric around reference)
    3. Threshold effects (discontinuities)
    """
    prices = np.linspace(1.0, 10.0, 100)
    utilities = [engine.calculate_price_utility(pid, p, customer) for p in prices]
    
    # Plot and verify non-linearity
    assert is_concave(utilities)  # Log effect
    assert shows_loss_aversion(utilities, ref_price)
    assert has_threshold_discontinuities(utilities)
```

**Deliverables:**
- NonLinearUtilityEngine class
- Reference price tracking system
- Threshold effect implementation
- Validation plots showing non-linear response curves

---

### **Phase 2.7: Seasonality Learning** (4 days)

**Goal:** Replace hard-coded seasonality with learned patterns from Dunnhumby

**Tasks:**
1. **Extract Seasonal Patterns from Dunnhumby** (Days 1-2)
   - Analyze transaction data by week-of-year
   - Calculate product-specific seasonal indices
   - Identify category-level seasonal patterns
   - Learn holiday effects

2. **Replace Hard-Coded Seasonality** (Day 3)
   - Remove manual seasonality definitions
   - Load learned patterns
   - Apply product-specific multipliers

3. **Product-Specific Patterns** (Day 4)
   - High-variance products (ice cream, soup)
   - Low-variance products (milk, bread)
   - Category interactions (turkey + cranberries)

**Implementation:**
```python
class SeasonalityLearningEngine:
    """
    Learn seasonal patterns from Dunnhumby transaction data
    """
    
    def __init__(self, transactions_df, products_df):
        self.transactions = transactions_df
        self.products = products_df
        self.seasonal_indices = {}  # product_id → week_of_year → multiplier
        self.category_patterns = {}  # category → week_of_year → multiplier
    
    def learn_seasonal_patterns(self):
        """
        Extract seasonal patterns from historical data
        
        Method: Seasonal decomposition
        1. Aggregate sales by product-week
        2. Detrend (remove growth)
        3. Extract seasonal component
        4. Normalize to multipliers
        """
        print("Learning seasonal patterns from Dunnhumby data...")
        
        # Add week-of-year to transactions
        self.transactions['week_of_year'] = (
            self.transactions['week_no'] % 52
        ).replace(0, 52)
        
        # Aggregate by product-week
        product_week_sales = self.transactions.groupby(
            ['product_id', 'week_of_year']
        )['quantity'].sum().reset_index()
        
        # Learn patterns for each product
        for product_id in tqdm(self.products['product_id'].unique()):
            product_sales = product_week_sales[
                product_week_sales['product_id'] == product_id
            ]
            
            if len(product_sales) < 26:  # Need at least 6 months
                continue
            
            # Calculate seasonal indices
            indices = self._calculate_seasonal_indices(
                product_sales['week_of_year'].values,
                product_sales['quantity'].values
            )
            
            self.seasonal_indices[product_id] = indices
        
        # Learn category-level patterns
        self._learn_category_patterns()
        
        print(f"✅ Learned patterns for {len(self.seasonal_indices):,} products")
    
    def _calculate_seasonal_indices(self, weeks, quantities):
        """
        Calculate seasonal index for each week of year
        
        Index = (week_avg / overall_avg)
        """
        overall_avg = quantities.mean()
        
        indices = {}
        for week in range(1, 53):
            week_mask = (weeks == week)
            if week_mask.sum() > 0:
                week_avg = quantities[week_mask].mean()
                indices[week] = week_avg / overall_avg
            else:
                indices[week] = 1.0  # Neutral
        
        # Smooth indices (moving average)
        indices = self._smooth_indices(indices)
        
        return indices
    
    def _smooth_indices(self, indices, window=3):
        """Smooth seasonal indices to reduce noise"""
        weeks = sorted(indices.keys())
        values = [indices[w] for w in weeks]
        
        # Circular moving average
        smoothed = []
        for i in range(len(values)):
            window_vals = []
            for j in range(-window, window + 1):
                idx = (i + j) % len(values)
                window_vals.append(values[idx])
            smoothed.append(np.mean(window_vals))
        
        return {w: s for w, s in zip(weeks, smoothed)}
    
    def _learn_category_patterns(self):
        """Learn seasonal patterns at category level"""
        # Merge with product categories
        trans_with_cat = self.transactions.merge(
            self.products[['product_id', 'commodity_desc']],
            on='product_id'
        )
        
        # Aggregate by category-week
        cat_week_sales = trans_with_cat.groupby(
            ['commodity_desc', 'week_of_year']
        )['quantity'].sum().reset_index()
        
        # Calculate indices for each category
        for category in self.products['commodity_desc'].unique():
            cat_sales = cat_week_sales[
                cat_week_sales['commodity_desc'] == category
            ]
            
            if len(cat_sales) < 26:
                continue
            
            indices = self._calculate_seasonal_indices(
                cat_sales['week_of_year'].values,
                cat_sales['quantity'].values
            )
            
            self.category_patterns[category] = indices
    
    def get_seasonal_multiplier(self, product_id, week_of_year, category=None):
        """
        Get seasonal multiplier for a product in a given week
        
        Fallback hierarchy:
        1. Product-specific pattern
        2. Category pattern
        3. No seasonality (1.0)
        """
        # Try product-specific
        if product_id in self.seasonal_indices:
            return self.seasonal_indices[product_id].get(week_of_year, 1.0)
        
        # Try category
        if category and category in self.category_patterns:
            return self.category_patterns[category].get(week_of_year, 1.0)
        
        # Default
        return 1.0
    
    def identify_seasonal_products(self, threshold=1.5):
        """
        Identify highly seasonal products
        
        Seasonality = max(indices) / min(indices)
        """
        seasonal_products = {}
        
        for product_id, indices in self.seasonal_indices.items():
            values = list(indices.values())
            seasonality = max(values) / min(values)
            
            if seasonality >= threshold:
                seasonal_products[product_id] = {
                    'seasonality': seasonality,
                    'peak_week': max(indices, key=indices.get),
                    'trough_week': min(indices, key=indices.get)
                }
        
        return seasonal_products
    
    def save_patterns(self, output_path):
        """Save learned patterns to file"""
        import pickle
        
        patterns = {
            'seasonal_indices': self.seasonal_indices,
            'category_patterns': self.category_patterns
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(patterns, f)
        
        print(f"✅ Saved seasonal patterns to {output_path}")
    
    def visualize_seasonal_patterns(self, product_ids):
        """Create visualization of seasonal patterns"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(product_ids), 1, figsize=(12, 4*len(product_ids)))
        
        for i, product_id in enumerate(product_ids):
            if product_id in self.seasonal_indices:
                indices = self.seasonal_indices[product_id]
                weeks = sorted(indices.keys())
                values = [indices[w] for w in weeks]
                
                ax = axes[i] if len(product_ids) > 1 else axes
                ax.plot(weeks, values, marker='o')
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Week of Year')
                ax.set_ylabel('Seasonal Index')
                ax.set_title(f'Product {product_id} Seasonal Pattern')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

**Integration:**
```python
# In seasonality_engine.py - replace hard-coded logic

class SeasonalityEngine:
    def __init__(self, learned_patterns_path=None):
        if learned_patterns_path:
            # Load learned patterns
            with open(learned_patterns_path, 'rb') as f:
                patterns = pickle.load(f)
            self.seasonal_indices = patterns['seasonal_indices']
            self.category_patterns = patterns['category_patterns']
            self.use_learned = True
        else:
            # Fallback to hard-coded (legacy)
            self.use_learned = False
    
    def get_seasonal_effect(self, product_id, week_number, category=None):
        """Get seasonal multiplier for utility calculation"""
        if self.use_learned:
            week_of_year = (week_number % 52) or 52
            return self._get_learned_multiplier(product_id, week_of_year, category)
        else:
            return self._get_hardcoded_multiplier(week_number)
```

**Validation:**
```python
def validate_seasonal_patterns():
    """
    Validate learned patterns match real data
    """
    # Load learned patterns
    learner = SeasonalityLearningEngine(transactions, products)
    learner.learn_seasonal_patterns()
    
    # Test correlation with held-out data
    for product_id in sample_products:
        predicted_pattern = learner.seasonal_indices[product_id]
        actual_pattern = get_actual_pattern(product_id, holdout_data)
        
        correlation = np.corrcoef(
            list(predicted_pattern.values()),
            list(actual_pattern.values())
        )[0, 1]
        
        assert correlation > 0.7, f"Low correlation for product {product_id}"
    
    print("✅ Seasonal patterns validated")
```

**Deliverables:**
- SeasonalityLearningEngine class
- Learned patterns saved to file
- Product-specific seasonal indices
- Category-level patterns
- Validation showing high correlation with real data

---

## 📁 Files to Create/Modify

### **New Files:**
1. `src/retailsynth/engines/promotional_engine.py` (main promo system)
2. `src/retailsynth/engines/promo_mechanics.py` (discount mechanics)
3. `src/retailsynth/engines/promo_displays.py` (display allocation)
4. `src/retailsynth/engines/marketing_signal.py` (visit impact)
5. `src/retailsynth/engines/customer_heterogeneity.py` (parameter distributions)
6. `src/retailsynth/engines/promo_response.py` (utility adjustments)
7. `src/retailsynth/models/store_promo_context.py` (dataclass)
8. **`src/retailsynth/engines/nonlinear_utility.py` (non-linear transformations)**
9. **`src/retailsynth/engines/seasonality_learning.py` (pattern extraction)**
10. **`scripts/learn_seasonal_patterns.py` (standalone script)**

### **Files to Modify:**
1. `src/retailsynth/engines/pricing_engine.py` (remove promo logic)
2. `src/retailsynth/generators/customer_generator.py` (add heterogeneity)
3. `src/retailsynth/generators/transaction_generator.py` (integrate promos)
4. `src/retailsynth/engines/loyalty_engine.py` (add marketing signal to visits)
5. `src/retailsynth/engines/utility_engine.py` (add promo response, non-linear utilities)
6. `src/retailsynth/generators/main_generator.py` (orchestration)
7. **`src/retailsynth/engines/seasonality_engine.py` (use learned patterns)**


---

## 🎯 Success Metrics

### **Technical Validation:**
- ✅ Promotions increase store visits (measurable lift)
- ✅ Deep discounts trigger stockpiling (arc elasticity)
- ✅ Customer response heterogeneous (not uniform)
- ✅ Display types have different effectiveness
- ✅ In-ad products show higher lift

### **Data Quality:**
- ✅ Promotion frequency matches Dunnhumby (10-30% by category)
- ✅ Discount depths realistic (10-50% range)
- ✅ Promotional lift: 20-100% (category dependent)
- ✅ Parameter distributions look reasonable

### **Performance:**
- ✅ Generation speed maintained (<400s/week)
- ✅ Memory usage acceptable

---

## 📈 Expected Improvements

| Metric | Before Sprint 2 | After Sprint 2 | Improvement |
|--------|----------------|----------------|-------------|
| **Promo Realism** | Random discounts | HMM + mechanics | ✅ +60% |
| **Customer Diversity** | 4 archetypes | Continuous params | ✅ +80% |
| **Promo Response** | Uniform | Heterogeneous | ✅ +70% |
| **Store Visit Model** | Static | Promo-influenced | ✅ New feature |
| **Validation Score** | ~65% | ~80% target | ✅ +15% |

---

## 🚀 Implementation Order

**Week 1:**
- Day 1-3: Pricing-Promo Separation
- Day 4-6: Promo Organization (mechanics, displays)
- Day 7: Marketing Signal basics

**Week 2:**
- Day 8-9: Marketing Signal integration
- Day 10-13: Individual Heterogeneity

**Week 3:**
- Day 14-16: Promo Response + Arc Elasticity
- Day 17-19: Non-Linear Utilities
- Day 20: Testing & Integration

**Week 4:**
- Day 21-24: Seasonality Learning
- Day 25-27: Full System Testing

**Week 5 (if needed):**
- Day 28-30: Calibration & Fine-tuning
- Day 31-32: Documentation & Final validation

---

## 📝 Next Steps

### **Immediate Actions:**

1. **Review this plan** - Confirm approach aligns with your vision
2. **Prioritize phases** - Any phase more urgent than others?
3. **Start Phase 2.1** - Pricing-Promo separation (quick win)

### **Questions for You:**

1. Should we use Dunnhumby `causal_data.csv` to learn display/feature patterns?
2. Any specific promo mechanics from your domain knowledge?
3. Target metrics for promotional lift (20-100% reasonable)?
4. Keep arc elasticity simple or add sophistication?

---

**Ready to start Phase 2.1 when you are!** 🚀
