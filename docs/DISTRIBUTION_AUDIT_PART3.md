# Comprehensive Distribution Audit - Part 3

## 6. Product & Pricing (10 Parameters)

### 6.1 Base Product Prices ⚠️ **MODERATE**

**Location**: `generators/product_generator.py:47-53`

```python
# CURRENT
if dept == 'Fresh':
    prices = np.random.uniform(1.0, 15.0, dept_mask.sum())
elif dept == 'Pantry':
    prices = np.random.uniform(1.5, 20.0, dept_mask.sum())
```

| Aspect | Current | Dunnhumby | Issue | Priority |
|--------|---------|-----------|-------|----------|
| Distribution | **Uniform** | Log-Normal | Wrong shape | 🟡 MODERATE |
| Shape | Flat | Right-skewed | Too many mid-priced | 🟡 |
| Price clustering | No | Yes ($0.99, $1.99, $2.49) | Missing realism | 🟡 |

**Why Uniform Fails**:
- Real grocery: Many low-priced items ($1-3), fewer high-priced ($10+)
- Uniform gives equal probability to all prices in range
- Missing psychological price points

**Fix**:
```python
# Option 1: Log-Normal (matches retail distributions)
if dept == 'Fresh':
    mean_log = np.log(4.0)  # Median price ~$4
    sigma_log = 0.6
    prices = np.random.lognormal(mean_log, sigma_log, n)
    prices = np.clip(prices, 1.0, 15.0)

# Option 2: Psychological price points (best realism)
price_points = [0.99, 1.49, 1.99, 2.49, 2.99, 3.49, 3.99, 4.99, 5.99, 7.99, 9.99, 12.99]
point_probs = [0.15, 0.15, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.05, 0.03, 0.01, 0.01]
prices = np.random.choice(price_points, size=n, p=point_probs)
```

**Impact**: Better price distribution matching, improved revenue KS (+5-8%)

---

### 6.2 Product Lifecycle Multipliers ⚠️ **MODERATE**

**Location**: `engines/lifecycle_engine.py:92-96`

```python
# CURRENT
multipliers = {
    'launch': np.random.uniform(0.6, 0.8),
    'growth': np.random.uniform(1.1, 1.3),
    'maturity': np.random.uniform(0.95, 1.05),
    'decline': np.random.uniform(0.5, 0.7)
}
```

| Aspect | Current | Ideal | Issue | Priority |
|--------|---------|-------|-------|----------|
| Distribution | Uniform | Beta | No clustering | 🟡 |
| Ranges | Stage-dependent | ✅ Good | Well-designed | ✅ |

**Why Uniform Suboptimal**:
- Demand multipliers should cluster around typical values
- Beta distribution better models bounded multipliers

**Fix**:
```python
# Beta distributions (cluster around modal values)
if stage == 'launch':
    # Beta(2, 5) rescaled to [0.6, 0.8] → mode at ~0.65
    mult = 0.6 + 0.2 * np.random.beta(2, 5)
elif stage == 'growth':
    # Beta(5, 2) rescaled to [1.1, 1.3] → mode at ~1.25
    mult = 1.1 + 0.2 * np.random.beta(5, 2)
```

**Impact**: Minor improvement (+2-3%)

---

### 6.3 Product Role Assignment

**Location**: `generators/product_generator.py:56-60`

```python
roles = np.random.choice(
    ['lpg_line', 'front_basket', 'mid_basket', 'back_basket'],
    size=n_products,
    p=[0.15, 0.25, 0.40, 0.20]
)
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Distribution | Categorical | Categorical | ✅ Correct | - |
| Mix | [0.15, 0.25, 0.40, 0.20] | Reasonable | ✅ OK | - |

**Status**: ✅ Good - Appropriate retail assortment structure

---

### 6.4 Brand Assignment

**Location**: `generators/product_generator.py:63`

```python
brands = np.random.choice(all_brands, size=n_products)
```

| Aspect | Current | Ideal | Issue | Priority |
|--------|---------|-------|-------|----------|
| Distribution | Uniform | Power law (Zipf) | No concentration | 🟢 LOW |

**Why Uniform Suboptimal**:
- Real retail: Few dominant brands (Coke, Pepsi), many small brands
- Zipf's law: Brand popularity follows power law distribution

**Fix** (low priority):
```python
# Power law brand distribution
brand_ranks = np.arange(1, len(all_brands) + 1)
brand_probs = 1 / brand_ranks**1.2  # Zipf exponent ~1.2
brand_probs = brand_probs / brand_probs.sum()
brands = np.random.choice(all_brands, size=n_products, p=brand_probs)
```

**Impact**: Minor (+1-2%)

---

## 7. Promotional Mechanics (12 Parameters - Sprint 2)

### 7.1 Promotional Frequency ✅ **GOOD**

**Location**: `config.py:58-59` (Phase 2.1 & 2.2)

```python
promo_frequency_min: float = 0.10  # 10% of products on promo
promo_frequency_max: float = 0.30  # 30% of products on promo
```

| Aspect | Current | Dunnhumby | Status | Priority |
|--------|---------|-----------|--------|----------|
| Range | 10-30% | 15-25% typical | ✅ Reasonable | - |
| Implementation | Uniform sampling | ✅ Correct | Good | ✅ |

**Config Parameters** (Phase 2.1 - NEW):
- `promo_frequency_min: 0.10` → Tunable (Tier 2) ✅
- `promo_frequency_max: 0.30` → Tunable (Tier 2) ✅

**Status**: ✅ Well-implemented

---

### 7.2 Discount Depth Distribution ⚠️ **MODERATE**

**Location**: `engines/promotional_engine.py:118-123`

```python
# CURRENT
depth_ranges = {
    0: (0.00, 0.05),   # Regular
    1: (0.10, 0.25),   # Feature
    2: (0.25, 0.50),   # Deep
    3: (0.50, 0.70)    # Clearance
}
# Sampled uniformly within range
discount = np.random.uniform(low, high)
```

| Aspect | Current | Dunnhumby | Issue | Priority |
|--------|---------|-----------|-------|----------|
| Distribution | Uniform within bands | Clustered at points | No clustering | 🟡 MODERATE |
| Common discounts | Equal probability | 10%, 20%, 25%, 50% | Missing patterns | 🟡 |
| Clearance frequency | Too high | Rare (1-2%) | Over-represented | 🟡 |

**Why Uniform Fails**:
- Real promotions cluster at psychological points (10%, 20%, 25%, 33%, 50%)
- Not all discounts in 10-25% range are equally common
- 15% and 20% are much more common than 13% or 22%

**Fix**:
```python
# Option 1: Psychological discount points
def sample_discount(state):
    discount_clusters = {
        0: [0.00, 0.05],
        1: [0.10, 0.15, 0.20, 0.25],  # Feature: cluster at common points
        2: [0.30, 0.33, 0.40, 0.50],  # Deep
        3: [0.50, 0.60, 0.70, 0.75]   # Clearance
    }
    points = discount_clusters[state]
    return np.random.choice(points)

# Option 2: Beta distribution peaked around common values
if state == 1:  # Feature: 10-25%, peak at 20%
    # Beta(3, 2) peaks left of center
    discount = 0.10 + 0.15 * np.random.beta(3, 2)
```

**Impact**: Better promotional pattern matching (+3-5%)

---

### 7.3 Display Capacity

**Location**: `config.py:60-61` (Phase 2.2)

```python
display_end_cap_capacity: int = 10
display_feature_capacity: int = 3
```

| Aspect | Current | Reality | Status | Priority |
|--------|---------|---------|--------|----------|
| End caps | 10 per store | 8-12 typical | ✅ Reasonable | - |
| Features | 3 per store | 2-4 typical | ✅ Reasonable | - |

**Status**: ✅ Good defaults

---

### 7.4 Marketing Signal Weights ✅ **GOOD**

**Location**: `config.py:64-66` (Phase 2.3)

```python
marketing_discount_weight: float = 0.4
marketing_display_weight: float = 0.3
marketing_advertising_weight: float = 0.3
```

| Param | Range | Status | Priority |
|-------|-------|--------|----------|
| `marketing_discount_weight` | 0.2-0.6 tunable | ✅ OK | Tier 2 |
| `marketing_display_weight` | 0.2-0.5 tunable | ✅ OK | Tier 2 |
| `marketing_advertising_weight` | 0.2-0.5 tunable | ✅ OK | Tier 2 |

**Status**: ✅ Well-designed Phase 2.3 implementation

---

### 7.5 Marketing Signal Noise

**Location**: `config.py:75` (Phase 2.5)

```python
marketing_signal_noise: float = 0.1
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Distribution | Gaussian noise (σ=0.1) | Gaussian | ✅ Correct | - |
| Magnitude | 10% std dev | Reasonable | ✅ OK | - |

**Status**: ✅ Appropriate noise model

---

### 7.6 Promotion Response Distribution

**Location**: `config.py:183-185`

```python
promotion_sensitivity_mean: float = 0.5
promotion_sensitivity_std: float = 0.2
promotion_quantity_boost: float = 1.5
```

| Param | Range | Status | Priority |
|-------|-------|--------|----------|
| `promotion_sensitivity_mean` | 0.3-0.7 tunable | ✅ OK | Tier 2 |
| `promotion_sensitivity_std` | 0.1-0.3 tunable | ✅ OK | Tier 2 |
| `promotion_quantity_boost` | 1.2-2.0 tunable | ✅ OK | Tier 2 |

**Status**: ✅ All tunable, well-designed

---

## 8. Advanced Features - Phase 2.6 & 2.7 (8 Parameters)

### 8.1 Non-Linear Utilities ✅ **EXCELLENT** (Phase 2.6)

**Location**: `config.py:42-48`

```python
enable_nonlinear_utilities: bool = True
use_log_price: bool = True
use_reference_prices: bool = True
use_psychological_thresholds: bool = True
use_quadratic_quality: bool = True
loss_aversion_lambda: float = 2.5
ewma_alpha: float = 0.3
```

| Feature | Current | Research | Status | Priority |
|---------|---------|----------|--------|----------|
| Log-price utility | ✅ Enabled | Diminishing marginal disutility | ✅ CORRECT | - |
| Reference prices (EWMA) | ✅ Enabled | Kahneman & Tversky | ✅ CORRECT | - |
| Loss aversion λ | 2.5 | 2.0-2.5 literature | ✅ PERFECT | - |
| EWMA α | 0.3 (0.1-0.5 tunable) | 0.2-0.4 typical | ✅ GOOD | Tier 2 |
| Psychological thresholds | ✅ Enabled | Charm pricing | ✅ GOOD | - |

**Status**: ✅ **STATE-OF-THE-ART** - Behavioral economics implementation

**Config Parameters** (Phase 2.6 - NEW):
- `loss_aversion_lambda: 2.5` → Tunable (Tier 2) ✅
- `ewma_alpha: 0.3` → Tunable (Tier 2) ✅

---

### 8.2 Seasonality Learning ✅ **EXCELLENT** (Phase 2.7)

**Location**: `config.py:51-55`

```python
enable_seasonality_learning: bool = True
seasonal_patterns_path: str = 'data/processed/seasonal_patterns/seasonal_patterns.pkl'
seasonality_min_confidence: float = 0.3
seasonality_fallback_category: bool = True
seasonality_smoothing: float = 0.2
```

| Feature | Current | Status | Priority |
|---------|---------|--------|----------|
| Learned patterns | ✅ From Dunnhumby data | ✅ EXCELLENT | - |
| Product-specific | ✅ Yes | ✅ BEST PRACTICE | - |
| Category fallback | ✅ Yes | ✅ SMART | - |
| Min confidence | 0.3 (0.2-0.5 tunable) | ✅ OK | Tier 2 |
| Smoothing | 0.2 | ✅ Reasonable | - |

**Status**: ✅ **SUPERIOR TO HARD-CODED** - Data-driven approach

**Config Parameters** (Phase 2.7 - NEW):
- `seasonality_min_confidence: 0.3` → Tunable (Tier 2) ✅
- `seasonality_smoothing: 0.2` → Fixed ✅

---

## 9. Summary Matrix - All 91 Parameters

### Priority Breakdown

| Priority | Count | Action | Expected Gain |
|----------|-------|--------|---------------|
| 🔴 **CRITICAL** | 15 | Fix immediately | +27% (0.628 → 0.80) |
| 🟡 **MODERATE** | 22 | Next sprint | +6-10% (0.80 → 0.85-0.88) |
| 🟢 **LOW** | 8 | Future enhancement | +1-3% |
| ✅ **GOOD** | 46 | No changes needed | - |
| **TOTAL** | **91** | - | **+40% total** |

---

### Complete Parameter Classification

#### 🔴 CRITICAL Issues (15 parameters)

| # | Parameter | Current | Fix | Impact |
|---|-----------|---------|-----|--------|
| 1 | Quantity distribution | Normal | Log-Normal | +21% |
| 2 | Basket size range | 1-30 | 3-15 | +36% |
| 3 | Visit prob range | 0.15-0.50 | 0.30-0.75 | +60% |
| 4 | `quantity_mean` range | 1.2-2.5 | 1.2-1.8 | Indirect |
| 5 | `quantity_max` | 5-15 | 10-20 | Indirect |
| 6 | `base_visit_probability` default | 0.15 | 0.50-0.60 | Direct |
| 7-10 | `visit_prob_by_personality` (4) | 0.12-0.20 | 0.30-0.75 | Direct |

**Files to Fix**:
1. `transaction_generator.py` (quantity distribution)
2. `tune_parameters_optuna.py` (all search ranges)
3. `config.py` (default values)

---

#### 🟡 MODERATE Issues (22 parameters)

| # | Parameter | Current | Improvement | Impact |
|---|-----------|---------|-------------|--------|
| 1 | Basket size type | Poisson | Negative Binomial | +4% |
| 2 | Product prices | Uniform | Log-Normal | +5-8% |
| 3 | Discount depth | Uniform | Psychological points | +3-5% |
| 4 | Trip basket size | Normal | Gamma | +2-3% |
| 5 | Brand loyalty strength | Normal (clipped) | Beta | +2-3% |
| 6 | # preferred brands | Categorical | Poisson | +1-2% |
| 7 | Customer drift | Constant | Mixture | +3-5% |
| 8 | Income brackets | Categorical | Log-Normal (future) | +1-2% |
| 9 | Lifecycle multipliers | Uniform | Beta | +2-3% |
| 10 | Days since visit | Exponential | Gamma | +2-3% |
| 11-22 | Various minor improvements | - | - | +<1% each |

---

#### ✅ GOOD Implementations (46 parameters)

**Phase 2.4 Heterogeneity (6 params)** - ✅ EXCELLENT
- Price sensitivity (Log-Normal) - Textbook correct
- Promo responsiveness (Beta) - Appropriate
- Quality preference (Beta) - Well-designed
- Display sensitivity (Beta) - Good
- Brand loyalty (from heterogeneity) - Well-integrated
- Store loyalty (from heterogeneity) - Good

**Phase 2.6 Non-Linear Utilities (7 params)** - ✅ STATE-OF-THE-ART
- Log-price utility - Correct
- Reference prices (EWMA) - Perfect
- Loss aversion (λ=2.5) - Literature-backed
- Psychological thresholds - Good
- Quadratic quality - Appropriate

**Phase 2.7 Seasonality (5 params)** - ✅ SUPERIOR
- Learned patterns - Data-driven
- Product-specific - Best practice
- Category fallback - Smart design
- Min confidence tunable - Flexible
- Smoothing - Appropriate

**Demographics (15 params)** - ✅ GOOD
- Age distribution - Correct categorical
- Household size - Census-aligned
- Income (age-dependent) - Well-designed
- Marital status - Good logic
- Children - Appropriate
- Personality mix - Standard segmentation

**Basket Composition (3 params)** - ✅ GOOD
- Complement probability - Tunable
- Substitute avoidance - Tunable
- Category diversity - Tunable

**Store Loyalty (4 params)** - ✅ EXCELLENT
- Dirichlet preference weights - Textbook correct
- Number of preferred stores - Reasonable
- Switching probability - Tunable
- Weight parameters - Well-designed

**Purchase History Weights (5 params)** - ✅ GOOD
- All tunable (Tier 1)
- Reasonable ranges
- Well-validated

**Marketing Signals (4 params)** - ✅ GOOD
- Discount weight - Tunable
- Display weight - Tunable
- Advertising weight - Tunable
- Noise model - Appropriate

---

### Distribution Type Summary

| Distribution | Count | Usage | Status |
|--------------|-------|-------|--------|
| **Log-Normal** | 3 | Price sensitivity, prices (future), revenue (emergent) | ✅ EXCELLENT |
| **Beta** | 6 | Promo response, quality, display, loyalty (future) | ✅ GOOD |
| **Categorical** | 18 | Demographics, segmentation, discrete choices | ✅ APPROPRIATE |
| **Normal** | 5 | Quantity ❌, drift, trip basket ⚠️, brand loyalty ⚠️ | 🔴 FIX 1, 🟡 IMPROVE 3 |
| **Poisson** | 2 | Basket size ⚠️, brand count (future) | 🟡 UPGRADE 1 |
| **Negative Binomial** | 1 | Quantity (future) | ✅ RECOMMENDED |
| **Uniform** | 8 | Prices ⚠️, discounts ⚠️, sizes, years | 🟡 UPGRADE 2 |
| **Exponential** | 1 | Days since visit ⚠️ | 🟡 UPGRADE TO GAMMA |
| **Gamma** | 0 | None (recommended for several) | 🟡 ADD FOR 3 PARAMS |
| **Dirichlet** | 1 | Store preferences | ✅ PERFECT |
| **Bernoulli** | 8 | Binary features, visit decisions | ✅ CORRECT |
| **Power Law/Zipf** | 0 | None (recommended for brands) | 🟢 FUTURE |

---

## 10. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1) 🔴

**Day 1-2**: Fix distributions
1. Update `transaction_generator.py`:
   - Change quantity: Normal → Log-Normal
   - Add support for Negative Binomial basket size

**Day 3**: Fix tuning ranges
2. Update `tune_parameters_optuna.py`:
   - Basket size λ: 1-30 → 3-15
   - Visit probability: 0.15-0.50 → 0.30-0.75
   - Quantity mean: 1.2-2.5 → 1.2-1.8
   - Quantity max: 5-15 → 10-20

**Day 4**: Fix defaults
3. Update `config.py`:
   - `base_visit_probability`: 0.15 → 0.55
   - `visit_prob_by_personality`: All increase by +0.25-0.35
   - `quantity_max`: 10 → 20

**Day 5**: Test & Validate
4. Re-run tuning with fixed ranges
5. Verify KS scores improve: 0.628 → 0.80+

---

### Phase 2: Moderate Improvements (Week 2-3) 🟡

**Week 2**: Distribution upgrades
1. Upgrade basket size: Poisson → Negative Binomial
2. Upgrade product prices: Uniform → Log-Normal
3. Upgrade discount depth: Uniform → Psychological points
4. Upgrade trip basket size: Normal → Gamma

**Week 3**: Behavioral refinements
5. Upgrade brand loyalty: Normal → Beta
6. Upgrade # brands: Categorical → Poisson
7. Upgrade customer drift: Constant → Mixture
8. Upgrade days since visit: Exponential → Gamma

**Validation**: Target KS score 0.80 → 0.85-0.88

---

### Phase 3: Future Enhancements (Later) 🟢

1. Make income continuous (Log-Normal)
2. Add brand power law distribution
3. Make shopping hours store-type dependent
4. Add product-specific inventory depletion
5. Store sizes: Uniform → Log-Normal

---

## 11. Files Requiring Changes

### High Priority (Phase 1)
```
src/retailsynth/generators/transaction_generator.py
scripts/tune_parameters_optuna.py
src/retailsynth/config.py
```

### Medium Priority (Phase 2)
```
src/retailsynth/generators/product_generator.py
src/retailsynth/engines/promotional_engine.py
src/retailsynth/engines/trip_purpose.py
src/retailsynth/generators/customer_generator.py
src/retailsynth/engines/customer_state.py
```

### Low Priority (Phase 3)
```
src/retailsynth/generators/store_generator.py
src/retailsynth/engines/lifecycle_engine.py
```

---

## 12. Validation Metrics

After each phase, measure:

### Distribution Matching (KS Complement)
- Quantity distribution: Target 0.85+ (from 0.70)
- Basket size: Target 0.78+ (from 0.55)
- Visit frequency: Target 0.72+ (from 0.45)
- Revenue: Target 0.75+ (from 0.60)
- **Overall combined**: Target 0.88 (from 0.628)

### Business Metrics
- Average basket size: 5-12 items (not 29!)
- Average transaction value: $20-40
- Visit frequency: 0.5-0.7 per week (not 0.15!)
- Quantity at 1: 70-80% (not 40%)

---

## 13. References

- **Retail Analytics**: Talluri & Van Ryzin (basket size distributions)
- **Behavioral Economics**: Kahneman & Tversky (loss aversion, reference prices)
- **Demand Forecasting**: Hyndman & Athanasopoulos (seasonal patterns)
- **Grocery Research**: Dunnhumby "Complete Journey" dataset
- **Distribution Theory**: Johnson, Kotz & Balakrishnan (statistical distributions)

---

**Document Version**: 1.0 Comprehensive  
**Last Updated**: November 2024  
**Coverage**: All 91 parameters across RetailSynth v4.1  
**Next Review**: After Phase 1 implementation
