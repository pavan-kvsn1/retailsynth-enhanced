# Visit Probability Fix - Action Plan

**Created**: 2025-11-11  
**Priority**: 🔴 CRITICAL - Core mechanism missing  
**Expected Impact**: 0.561 → 0.75+ visit frequency KS score (+34%)

---

## 🔍 Root Cause Analysis

### Current Implementation (WRONG ❌)
```python
# utility_engine.py:142-155
def compute_store_visit_probabilities_gpu(...):
    base_prob = 0.5
    loyalty_boost = loyalty_levels * 0.2
    days_factor = jnp.minimum(0.3, days_since_visit / 30.0)
    visit_probs = base_prob + loyalty_boost + days_factor
    return jnp.clip(visit_probs, 0.0, 1.0)
```

**Issues**:
1. ❌ No store value (SV) calculation
2. ❌ No recursive probability with memory
3. ❌ No marketing signal → visit utility
4. ❌ No self-reinforcement loop
5. ❌ Simple additive formula (not utility-based)

### Bain Paper Approach (CORRECT ✅)

```python
# Step 1: Calculate store inclusive value (SV)
SV_ut = log(sum(exp(CV_ct)) for c in categories)

# Step 2: Calculate visit utility
X_store_ut = γ_0 + γ_1*SV_u,t-1 + β_marketing*Marketing_Signal_ut

# Step 3: Recursive probability with memory
P(Visit_ut) = θ_u * P(Visit_u,t-1) + (1-θ_u) * Logit(X_store_ut)

# Where:
# - SV_u,t-1: Inclusive value from LAST period (if visited)
# - Marketing_Signal_ut: Sum of product discounts this period
# - θ_u: Memory weight (0-1, how much last visit matters)
# - γ_0: Base store utility
# - γ_1: Store value weight (how much SV affects visits)
```

---

## 📋 Implementation Plan

### Phase 1: Immediate Tuning Fixes (1-2 hours) 🟡 QUICK WINS

**Goal**: Get maximum performance from current system before rebuilding

#### 1.1 Fix Optuna Ranges
**File**: `scripts/tune_parameters_optuna.py`

```python
# Line 170: Already fixed
config.basket_size_lambda = trial.suggest_float('basket_size_lambda', 3.0, 12.0)

# Line ~210: Raise loyalty weight minimums
config.store_loyalty_weight = trial.suggest_float('store_loyalty_weight', 0.5, 0.9)  # Was 0.4-0.8
config.habit_weight = trial.suggest_float('habit_weight', 0.4, 0.8)                  # Was 0.2-0.7
config.loyalty_weight = trial.suggest_float('loyalty_weight', 0.4, 0.8)              # Was 0.3-0.7

# Add price sensitivity tuning (NEW)
config.hetero_price_mean = trial.suggest_float('hetero_price_mean', 0.8, 1.8)
config.hetero_price_std = trial.suggest_float('hetero_price_std', 0.3, 0.6)
```

**Expected Impact**: +5-8% from better parameter ranges

#### 1.2 Run Quick Calibration
```bash
# Test with fixed ranges
python scripts/tune_parameters_optuna.py \
    --real-data data/processed/dunnhumby_calibration.csv \
    --tier 1 \
    --n-trials 20 \
    --objective combined
```

**Decision Point**: If this gets us to 0.70-0.72, we can proceed with Phase 2. If still < 0.68, skip to Phase 2 immediately.

---

### Phase 2: Core Visit Probability Mechanism (4-6 hours) 🔴 CRITICAL

**Goal**: Implement Bain's recursive visit probability with SV

#### 2.1 Create Store Value Calculator (NEW FILE)
**File**: `src/retailsynth/engines/store_value_engine.py`

```python
"""
Store Value (SV) Engine - Implements Bain's inclusive value calculation
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Dict, Tuple

class StoreValueEngine:
    """
    Calculates store inclusive value (SV) from product utilities
    
    SV_ut = log(sum(exp(CV_ct)) for c in categories)
    
    This represents the "value" a customer expects from visiting the store,
    based on available products and their utilities.
    """
    
    def __init__(self, config):
        self.config = config
        self.gamma_0 = config.store_base_utility           # Base store utility
        self.gamma_1 = config.store_value_weight            # SV → visit utility weight
        self.beta_marketing = config.marketing_visit_weight # Marketing → visit weight
        self.theta = config.visit_memory_weight             # Memory parameter
    
    @partial(jit, static_argnums=(0,))
    def compute_store_value_gpu(self,
                               product_utilities: jnp.ndarray,
                               product_categories: jnp.ndarray,
                               n_categories: int) -> jnp.ndarray:
        """
        Calculate store inclusive value (SV) for each customer
        
        Args:
            product_utilities: [n_customers, n_products] - utility of each product
            product_categories: [n_products] - category index for each product
            n_categories: Number of unique categories
        
        Returns:
            store_values: [n_customers] - SV for each customer
        """
        n_customers = product_utilities.shape[0]
        
        # Calculate category values (CV) by taking log-sum-exp per category
        category_values = jnp.zeros((n_customers, n_categories))
        
        for cat_idx in range(n_categories):
            # Get products in this category
            cat_mask = product_categories == cat_idx
            cat_utilities = jnp.where(cat_mask, product_utilities, -jnp.inf)
            
            # Log-sum-exp: CV_c = log(sum(exp(utility_p)) for p in category c)
            max_util = jnp.max(cat_utilities, axis=1, keepdims=True)
            max_util = jnp.where(jnp.isinf(max_util), 0, max_util)
            
            category_values = category_values.at[:, cat_idx].set(
                max_util.squeeze() + jnp.log(jnp.sum(jnp.exp(cat_utilities - max_util), axis=1))
            )
        
        # Store value: SV = log(sum(exp(CV_c)) for c in categories)
        max_cv = jnp.max(category_values, axis=1, keepdims=True)
        store_values = max_cv.squeeze() + jnp.log(jnp.sum(jnp.exp(category_values - max_cv), axis=1))
        
        return store_values
    
    @partial(jit, static_argnums=(0,))
    def compute_visit_utilities_gpu(self,
                                   store_values: jnp.ndarray,
                                   prev_store_values: jnp.ndarray,
                                   marketing_signals: jnp.ndarray,
                                   visited_last_period: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate visit utility from store value and marketing signals
        
        X_store_ut = γ_0 + γ_1*SV_u,t-1 + β_marketing*Marketing_ut
        
        Args:
            store_values: [n_customers] - Current period SV (for reference)
            prev_store_values: [n_customers] - Previous period SV (if visited)
            marketing_signals: [n_customers] - Marketing signal strength
            visited_last_period: [n_customers] - Binary flag if visited last period
        
        Returns:
            visit_utilities: [n_customers] - Utility of visiting
        """
        # Only use previous SV if customer actually visited last period
        sv_component = jnp.where(visited_last_period, prev_store_values, 0.0)
        
        visit_utilities = (
            self.gamma_0 +                           # Base utility
            self.gamma_1 * sv_component +            # Previous store value
            self.beta_marketing * marketing_signals  # Marketing signal
        )
        
        return visit_utilities
    
    @partial(jit, static_argnums=(0,))
    def compute_visit_probabilities_recursive_gpu(self,
                                                 visit_utilities: jnp.ndarray,
                                                 prev_visit_probs: jnp.ndarray,
                                                 customer_theta: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate recursive visit probabilities with memory
        
        P(Visit_ut) = θ_u * P(Visit_u,t-1) + (1-θ_u) * Logit(X_store_ut)
        
        Args:
            visit_utilities: [n_customers] - Visit utility X_store_ut
            prev_visit_probs: [n_customers] - Previous period visit probability
            customer_theta: [n_customers] - Memory weight per customer
        
        Returns:
            visit_probs: [n_customers] - Probability of visiting
        """
        # Logit transform: P = exp(X) / (1 + exp(X))
        current_prob = jax.nn.sigmoid(visit_utilities)
        
        # Recursive mixture: memory × last_prob + (1-memory) × current_prob
        visit_probs = customer_theta * prev_visit_probs + (1 - customer_theta) * current_prob
        
        return jnp.clip(visit_probs, 0.0, 1.0)
```

**Config additions** (`config.py`):
```python
# Store value → visit probability parameters
store_base_utility: float = 0.5        # γ_0: Base store utility
store_value_weight: float = 0.6        # γ_1: SV → visit utility weight
marketing_visit_weight: float = 0.4    # β: Marketing → visit weight
visit_memory_weight: float = 0.3       # θ: How much last visit matters (0-1)
```

#### 2.2 Update Utility Engine to Track SV
**File**: `src/retailsynth/engines/utility_engine.py`

Add state tracking:
```python
class UtilityEngine:
    def __init__(self, config):
        self.config = config
        self.store_value_engine = StoreValueEngine(config)
        
        # State tracking for recursive probability
        self.prev_visit_probs = None
        self.prev_store_values = None
        self.visited_last_period = None
    
    def compute_visit_probabilities_with_sv(self,
                                           product_utilities: jnp.ndarray,
                                           product_categories: jnp.ndarray,
                                           marketing_signals: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        New method: Compute visit probabilities using full Bain mechanism
        
        Returns:
            visit_probs: [n_customers] - Probability of visiting
            store_values: [n_customers] - Current period SV (for next period)
        """
        n_customers = product_utilities.shape[0]
        
        # Step 1: Calculate current period store values
        store_values = self.store_value_engine.compute_store_value_gpu(
            product_utilities, product_categories, self.config.n_categories
        )
        
        # Step 2: Initialize state if first period
        if self.prev_store_values is None:
            self.prev_store_values = jnp.zeros(n_customers)
            self.prev_visit_probs = jnp.full(n_customers, self.config.base_visit_probability)
            self.visited_last_period = jnp.zeros(n_customers, dtype=bool)
        
        # Step 3: Calculate visit utilities
        visit_utilities = self.store_value_engine.compute_visit_utilities_gpu(
            store_values,
            self.prev_store_values,
            marketing_signals,
            self.visited_last_period
        )
        
        # Step 4: Recursive probabilities
        customer_theta = jnp.full(n_customers, self.config.visit_memory_weight)
        visit_probs = self.store_value_engine.compute_visit_probabilities_recursive_gpu(
            visit_utilities,
            self.prev_visit_probs,
            customer_theta
        )
        
        return visit_probs, store_values
    
    def update_visit_state(self, visited_customers: jnp.ndarray, store_values: jnp.ndarray, visit_probs: jnp.ndarray):
        """Update state for next period"""
        self.visited_last_period = visited_customers
        self.prev_store_values = store_values
        self.prev_visit_probs = visit_probs
```

#### 2.3 Wire Marketing Signal into Visits
**File**: `src/retailsynth/generators/transaction_generator.py`

```python
# Before generating transactions each week:

# Calculate marketing signal strength for each customer
marketing_signals = self.compute_marketing_signals(
    promo_context,
    customer_store_assignments
)

# Use new visit probability method
visit_probs, store_values = self.utility_engine.compute_visit_probabilities_with_sv(
    product_utilities,
    product_categories,
    marketing_signals
)

# After transactions generated, update state
self.utility_engine.update_visit_state(visiting_customers, store_values, visit_probs)
```

**Expected Impact**: +15-20% visit frequency matching from proper mechanism

---

### Phase 3: Add to Optuna Tuning (1 hour) 🟢 CALIBRATION

#### 3.1 Add New Parameters to Tier 2
**File**: `scripts/tune_parameters_optuna.py`

```python
if 2 in self.tiers:
    # ... existing parameters ...
    
    # 12. Store Value → Visit Probability (4 params) - NEW Bain Mechanism
    config.store_base_utility = trial.suggest_float('store_base_utility', 0.2, 0.8)
    config.store_value_weight = trial.suggest_float('store_value_weight', 0.4, 0.9)
    config.marketing_visit_weight = trial.suggest_float('marketing_visit_weight', 0.2, 0.6)
    config.visit_memory_weight = trial.suggest_float('visit_memory_weight', 0.2, 0.5)
```

Update tier counts:
```python
tier_counts = {1: 15, 2: 31, 3: 29}  # Tier 2: 27 + 4 SV params
```

**Expected Impact**: +5-8% from optimal calibration

---

### Phase 4: Testing & Validation (2 hours) 🔵 QUALITY

#### 4.1 Unit Tests
**File**: `tests/test_store_value_engine.py`

```python
def test_store_value_calculation():
    """Test that SV increases with better product utilities"""
    engine = StoreValueEngine(config)
    
    # Scenario 1: Low utilities
    utilities_low = jnp.array([[1.0, 1.5, 2.0, 1.2]])
    sv_low = engine.compute_store_value_gpu(utilities_low, categories, n_cat)
    
    # Scenario 2: High utilities (promotions active)
    utilities_high = jnp.array([[3.0, 3.5, 4.0, 3.2]])
    sv_high = engine.compute_store_value_gpu(utilities_high, categories, n_cat)
    
    # SV should increase with better utilities
    assert sv_high[0] > sv_low[0]

def test_recursive_probability():
    """Test that visiting increases future visit probability"""
    engine = StoreValueEngine(config)
    
    # Period 1: Visit with high SV
    visit_util_1 = jnp.array([2.0])
    prev_prob_1 = jnp.array([0.3])
    theta = jnp.array([0.3])
    
    prob_1 = engine.compute_visit_probabilities_recursive_gpu(visit_util_1, prev_prob_1, theta)
    
    # Period 2: Same utility, but now we visited before
    prob_2 = engine.compute_visit_probabilities_recursive_gpu(visit_util_1, prob_1, theta)
    
    # Self-reinforcement: visiting → higher prob next time
    assert prob_2[0] > prob_1[0]

def test_marketing_signal_boosts_visits():
    """Test that marketing signals increase visit probability"""
    engine = StoreValueEngine(config)
    
    # No marketing
    visit_util_no_marketing = engine.compute_visit_utilities_gpu(
        sv_current=jnp.array([5.0]),
        sv_prev=jnp.array([5.0]),
        marketing=jnp.array([0.0]),
        visited=jnp.array([True])
    )
    
    # With marketing (promotions)
    visit_util_with_marketing = engine.compute_visit_utilities_gpu(
        sv_current=jnp.array([5.0]),
        sv_prev=jnp.array([5.0]),
        marketing=jnp.array([2.0]),  # Strong promotions
        visited=jnp.array([True])
    )
    
    assert visit_util_with_marketing[0] > visit_util_no_marketing[0]
```

#### 4.2 Integration Tests
```python
def test_end_to_end_visit_dynamics():
    """Test full visit probability system over multiple periods"""
    # Run 10 weeks
    # Week 1-3: No promotions → baseline visits
    # Week 4-6: Heavy promotions → visits should increase
    # Week 7-10: Promotions end → visits should gradually decrease (memory)
    
    # Verify:
    # 1. Promotions increase visits
    # 2. Memory keeps visits elevated after promotions end
    # 3. Self-reinforcement loop works
```

---

## 📊 Expected Outcomes

### Performance Timeline

| Phase | Time | Visit Freq KS | Combined KS | Key Improvement |
|-------|------|---------------|-------------|-----------------|
| **Baseline** | - | 0.561 | 0.664 | Current state |
| **Phase 1** | 2h | 0.60-0.62 | 0.68-0.70 | Better parameter ranges |
| **Phase 2** | 6h | 0.72-0.78 | 0.78-0.82 | Bain mechanism implemented |
| **Phase 3** | 1h | 0.75-0.80 | 0.80-0.85 | Optimal calibration |
| **Phase 4** | 2h | - | - | Quality assurance |
| **TOTAL** | **11h** | **0.75-0.80** | **0.80-0.85** | **+34-43% improvement** |

### Validation Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Visit Frequency | 0.561 | 0.75+ | Recursive SV probability |
| Basket Size | 0.645 | 0.75+ | Fixed lambda range (3-12) |
| Revenue | 0.660 | 0.70+ | Follows from basket fix |
| Quantity | 0.934 | 0.93+ | Already excellent |
| **Combined** | **0.664** | **0.80+** | **All fixes together** |

---

## 🗓️ Implementation Schedule

### Day 1 (Today)
- ✅ **Morning (2h)**: Phase 1 - Quick tuning fixes
  - Fix Optuna ranges
  - Run quick calibration
  - Assess if Phase 2 needed (likely yes)

- **Afternoon (4h)**: Phase 2.1 - Store Value Engine
  - Create `store_value_engine.py`
  - Implement SV calculation
  - Unit test SV computation

### Day 2
- **Morning (2h)**: Phase 2.2-2.3 - Integration
  - Update utility engine
  - Wire into transaction generator
  - Test end-to-end

- **Afternoon (3h)**: Phase 3 & 4 - Calibration & Testing
  - Add to Optuna
  - Run full calibration
  - Validate results

---

## 🎯 Success Criteria

### Must Have ✅
1. Store value (SV) calculated from product utilities
2. Recursive visit probability with memory
3. Marketing signal feeds into visit utility
4. Self-reinforcement: visiting → higher SV → more visits
5. Visit frequency KS > 0.75

### Nice to Have 🌟
1. Customer-specific θ (memory weight)
2. Store-specific γ_0 (base utility)
3. Time-varying marketing signals
4. Adaptive memory (θ changes with experience)

---

## 📚 References

**Bain et al. (2023) - Key Equations**:
- Equation 3: Store inclusive value (SV)
- Equation 4: Visit utility function
- Equation 5: Recursive probability
- Section 3.2: Self-reinforcement mechanism

**Implementation Priority**:
1. 🔴 Core mechanism (Phase 2)
2. 🟡 Parameter tuning (Phase 1 & 3)
3. 🔵 Quality assurance (Phase 4)

---

## 🚀 Next Steps

**Immediate** (Start now):
1. Apply Phase 1 fixes to `tune_parameters_optuna.py`
2. Run quick calibration to baseline current system
3. Start Phase 2.1: Create `store_value_engine.py`

**This Week**:
- Complete Phases 1-4
- Achieve 0.80+ combined KS score
- Document results

**Next Week**:
- Sprint 3 planning based on results
- Advanced features (customer-specific θ, etc.)

---

**Status**: 📋 PLAN READY - Ready to implement  
**Owner**: Development team  
**Timeline**: 11 hours (1.5 days)  
**Expected Result**: 0.664 → 0.80+ KS score (+34-43%)
