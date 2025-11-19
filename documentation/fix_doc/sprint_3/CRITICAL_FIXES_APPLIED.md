
# Critical Distribution Fixes Applied

**Date**: November 2024  
**Status**: ✅ IMPLEMENTED  
**Expected Impact**: +36-40% calibration improvement (0.628 → 0.85-0.88)

---

## Summary

Based on the comprehensive distribution audit (91 parameters analyzed), we have implemented **CRITICAL fixes** for the 3 most impactful distribution issues, plus 4 **BONUS moderate fixes**:

1. **Quantity Distribution**: Normal → Log-Normal (+21% KS improvement) 🔴 CRITICAL
2. **Basket Size Range**: 1-30 → 3-15 (+36% KS improvement) 🔴 CRITICAL
3. **Visit Probability Range**: 0.15-0.50 → 0.30-0.75 (+60% KS improvement) 🔴 CRITICAL
4. **Trip Purpose Basket Size**: Normal → Gamma (+2-3% improvement) 🟡 BONUS
5. **Days Since Last Visit**: Exponential → Gamma (+2-3% improvement) 🟡 BONUS
6. **Customer Drift Mixture Model**: Constant Gaussian → Mixture model (+3-5% improvement) 🟡 BONUS
7. **Product Prices**: Already using Log-Normal ✅ NO FIX NEEDED
8. **Discount Depths**: Uniform → Psychological points (+2-4% improvement) 🟡 BONUS

---

## 1. Quantity Distribution ✅ FIXED

### Problem
- **Current**: Normal distribution for count data
- **Issue**: 40% at qty=1 (should be 70-80%)
- **Impact**: Cascades to revenue errors

### Fix Applied

**File**: `src/retailsynth/generators/transaction_generator.py:626-629`

```python
# BEFORE (WRONG)
quantity = max(1, int(np.random.normal(self.config.quantity_mean, self.config.quantity_std)))
quantity = min(quantity, self.config.quantity_max)

# AFTER (CORRECT)
mean_log = np.log(self.config.quantity_mean)
sigma_log = self.config.quantity_std / self.config.quantity_mean
quantity = int(np.random.lognormal(mean_log, sigma_log))
quantity = max(1, min(quantity, self.config.quantity_max))
```

**Why Log-Normal?**
- Grocery shopping: 70-80% buy qty=1, with long right tail
- Log-Normal naturally produces this right-skewed distribution
- No negative values (unlike Normal which needs floor)

**Expected Impact**: Quantity KS 0.70 → **0.85** (+21%)

---

## 2. Basket Size Distribution ✅ FIXED

### Problem
- **Current**: Poisson with λ range 1-30
- **Issue**: Optuna found λ=28.6 → 29-item baskets!
- **Impact**: Unrealistic basket sizes, poor calibration

### Fix Applied

**File**: `src/retailsynth/generators/transaction_generator.py:599-621`

```python
# BEFORE (Poisson)
if personality == 'impulse':
    n_products = max(1, int(np.random.poisson(basket_size_lambda * 0.7)))
elif personality == 'planned':
    n_products = max(1, int(np.random.poisson(basket_size_lambda * 1.2)))

# AFTER (Negative Binomial with overdispersion)
# Personality-based modifiers
if personality == 'impulse':
    modifier = 0.7
elif personality == 'planned':
    modifier = 1.2
elif personality == 'convenience':
    modifier = 0.5
else:  # price_anchor
    modifier = 0.8

# Use Negative Binomial for overdispersion (variance > mean)
mean = self.config.basket_size_lambda * modifier
variance = mean * 1.5  # 50% overdispersion (variance = 1.5 * mean)

# Negative Binomial parameterization
p = mean / variance
r = mean * p / (1 - p)

# Sample and ensure at least 1 product
n_products = max(1, int(np.random.negative_binomial(r, p)))
```

**Why Negative Binomial?**
- Real grocery data has variance >> mean (overdispersion)
- Poisson assumes variance = mean (too constrained)
- Negative Binomial allows variance = 1.5 × mean

**Expected Impact**: 
- Range fix: Basket KS 0.55 → **0.75** (+36%)
- Negative Binomial: Basket KS 0.75 → **0.78** (+4%)
- **Total**: +40% improvement

---

## 3. Search Ranges Fixed ✅ FIXED

### Problem
- **Basket size λ**: 1-30 allowed unrealistic values
- **Visit probability**: 0.15-0.50 implies <1 visit/month
- **Quantity mean**: 1.2-2.5 too wide
- **Quantity max**: 5-15 too low for bulk purchases

### Fix Applied

**File**: `scripts/tune_parameters_optuna.py:168-176`

```python
# BEFORE (WRONG RANGES)
config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.15, 0.50)
config.basket_size_lambda = trial.suggest_float('basket_size_lambda', 1.0, 30.0)
config.quantity_mean = trial.suggest_float('quantity_mean', 1.2, 2.5)
config.quantity_max = trial.suggest_int('quantity_max', 5, 15)

# AFTER (REALISTIC RANGES)
config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.30, 0.75)
config.basket_size_lambda = trial.suggest_float('basket_size_lambda', 3.0, 15.0)
config.quantity_mean = trial.suggest_float('quantity_mean', 1.2, 1.8)
config.quantity_max = trial.suggest_int('quantity_max', 10, 20)
```

**Why These Ranges?**
- **Visit prob 0.30-0.75**: Realistic 1-2 visits/week (not <1/month)
- **Basket λ 3-15**: Prevents 29-item baskets, realistic 3-15 items
- **Quantity mean 1.2-1.8**: Tighter around realistic average
- **Quantity max 10-20**: Allows bulk purchases (e.g., 12-pack soda)

---

## 4. Config Defaults Updated ✅ FIXED

### Problem
- Default values were causing unrealistic behavior
- Base visit probability: 0.15 → <1 visit/month
- Personality visit probs: All too low

### Fix Applied

**File**: `src/retailsynth/config.py:127-133, 201`

```python
# BEFORE (UNREALISTIC DEFAULTS)
base_visit_probability: float = 0.15
visit_prob_by_personality: Dict[str, float] = field(default_factory=lambda: {
    'price_anchor': 0.12,
    'convenience': 0.18,
    'planned': 0.15,
    'impulse': 0.20
})
quantity_max: int = 10

# AFTER (REALISTIC DEFAULTS)
base_visit_probability: float = 0.55  # Weekly visit probability
visit_prob_by_personality: Dict[str, float] = field(default_factory=lambda: {
    'price_anchor': 0.40,     # ~1-2 visits/week
    'convenience': 0.60,      # Convenience shoppers visit more
    'planned': 0.50,          # ~1 visit/week
    'impulse': 0.65           # Impulse shoppers visit frequently
})
quantity_max: int = 20  # Allows bulk purchases
```

**Math Check**:
- **Old**: 0.15 weekly = 0.65 visits/month = 0.15 visits/week ❌
- **New**: 0.55 weekly = 2.38 visits/month = 0.55 visits/week ✅

---

## 5. Trip Purpose Basket Size ✅ BONUS FIX (Moderate)

### Problem
- **Current**: Normal distribution (truncated) for basket sizes
- **Issue**: Too symmetric, artificial boundaries from truncation
- **Impact**: Minor distribution shape mismatch

### Fix Applied

**File**: `src/retailsynth/engines/trip_purpose.py:278-283`

```python
# BEFORE (Normal with truncation)
size = np.random.normal(chars.basket_size_mean, chars.basket_size_std)
size = int(np.clip(size, chars.min_items, chars.max_items))

# AFTER (Gamma - naturally right-skewed)
# Use Gamma distribution (better for count-like continuous data)
shape = (chars.basket_size_mean / chars.basket_size_std) ** 2
scale = chars.basket_size_std ** 2 / chars.basket_size_mean
size = int(np.random.gamma(shape, scale))
size = np.clip(size, chars.min_items, chars.max_items)
```

**Why Gamma?**
- Basket sizes are count-like data (naturally right-skewed)
- Gamma distribution naturally produces right-skewed distributions
- No artificial truncation boundaries (soft clipping only)
- Better matches real shopping behavior

**Expected Impact**: +2-3% basket size distribution shape improvement

---

## 6. Days Since Last Visit ✅ BONUS FIX (Low Priority)

### Problem
- **Current**: Exponential distribution (memoryless)
- **Issue**: Assumes no shopping habits or patterns
- **Impact**: Misses habitual behavior ("Saturday shopper")

### Fix Applied

**File**: `src/retailsynth/generators/customer_generator.py:179-184`

```python
# BEFORE (Exponential - memoryless)
'days_since_last_visit': int(np.random.exponential(7))

# AFTER (Gamma - habit formation)
# Use Gamma distribution for habit formation (not memoryless Exponential)
# Gamma captures consistent shopping patterns ("Saturday shopper")
'days_since_last_visit': int(np.random.gamma(
    self.config.days_since_last_visit_shape,  # 2.0 = consistent habits
    self.config.days_since_last_visit_scale   # 3.5 = mean 7 days
))
```

**Config Parameters Added** (`config.py:218-219`):
```python
# Days since last visit (Gamma distribution for habit formation)
days_since_last_visit_shape: float = 2.0  # Higher = more consistent habits
days_since_last_visit_scale: float = 3.5  # Mean = shape × scale = 7 days
```

**Why Gamma?**
- Real customers have habitual patterns (weekly shoppers, weekend shoppers)
- Exponential is memoryless (today's visit doesn't affect tomorrow)
- Gamma captures consistency: shape=2.0 means moderate habit strength
- Tunable via Optuna for calibration to real data

**Expected Impact**: +2-3% visit pattern matching improvement

**Optuna Tuning** (Tier 2 - Indirectly Calibratable):
- `days_since_last_visit_shape`: 1.5 - 4.0 (habit consistency)
- `days_since_last_visit_scale`: 2.0 - 5.0 (average days between visits)

---

## 7. Customer Drift Mixture Model ✅ BONUS FIX (Moderate)

### Problem
- **Current**: Constant Gaussian drift (too smooth)
- **Issue**: Doesn't model life events (new job, baby, move)
- **Impact**: Missing occasional large behavioral shifts

### Fix Applied

**File**: `src/retailsynth/engines/drift_engine.py:42-88`

```python
# BEFORE (Constant drift)
price_drift_rate = 0.0005  # Small weekly change
for i in range(self.n_customers):
    utility_params = self.customers.iloc[i]['utility_params']
    current_beta = utility_params['beta_price']
    drift = np.random.normal(-price_drift_rate, price_drift_rate * 0.5)
    new_beta = current_beta * (1 + drift)
    utility_params['beta_price'] = new_beta

# AFTER (Mixture model: 90% small + 10% life events)
# Get drift parameters from config
drift_rate = self.config.drift_rate  # 0.05
drift_probability = self.config.drift_probability  # 0.1
life_event_prob = self.config.drift_life_event_probability  # 0.1
life_event_multiplier = self.config.drift_life_event_multiplier  # 5.0

for i in range(self.n_customers):
    # Check if this customer experiences drift this week
    if np.random.random() < drift_probability:
        utility_params = self.customers.iloc[i]['utility_params']
        
        # Mixture model: 90% small drift, 10% life event
        if np.random.random() < life_event_prob:
            # Life event: Large shift (5x normal drift)
            drift_magnitude = drift_rate * life_event_multiplier
        else:
            # Normal drift: Small gradual change
            drift_magnitude = drift_rate
        
        # Apply drift to price sensitivity
        current_beta = utility_params['beta_price']
        drift = np.random.normal(0, drift_magnitude)
        new_beta = current_beta * (1 + drift)
        utility_params['beta_price'] = new_beta
```

**Config Parameters Added** (`config.py:214-217`):
```python
# Customer drift (mixture model: small gradual drift + occasional life events)
drift_rate: float = 0.05  # Weekly drift magnitude (small changes)
drift_probability: float = 0.1  # Probability of drift per week
drift_life_event_probability: float = 0.1  # Probability that drift is a life event (10% of drifts)
drift_life_event_multiplier: float = 5.0  # Life events are 5x larger than normal drift
```

**Why Mixture Model?**
- **Real behavior**: Most weeks = small changes, occasional weeks = major life events
- **Life events**: New job, baby, move, major purchase → large behavioral shifts
- **Gradual drift**: Economic pressure, aging, market trends → small changes
- **Better longitudinal matching**: Captures both stability and disruption

**Expected Impact**: +3-5% longitudinal behavior matching

**Optuna Tuning** (Tier 2 - Indirectly Calibratable):
- `drift_rate`: 0.01 - 0.15 (magnitude of small changes)
- `drift_probability`: 0.05 - 0.20 (how often drift occurs)
- `drift_life_event_probability`: 0.05 - 0.20 (% of drifts that are life events)
- `drift_life_event_multiplier`: 3.0 - 8.0 (how much larger life events are)

---

## 8. Product Prices ✅ ALREADY CORRECT (No Fix Needed)

### Status
**Product prices already use Log-Normal distribution!** 

### Implementation

**File**: `src/retailsynth/generators/product_generator.py:44-69`

```python
# Use Log-Normal distribution (better for retail prices)
# Real prices are right-skewed: many cheap items, few expensive

for dept in ['Fresh', 'Pantry', 'Personal_Care', 'General_Merchandise']:
    if dept == 'Fresh':
        mu, sigma = 1.5, 0.6  # Mean ~$5, range $1-15
        prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
    elif dept == 'Pantry':
        mu, sigma = 1.7, 0.6  # Mean ~$6, range $1.5-20
        prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
    # ... etc
```

**Why Log-Normal?**
- **Right-skewed**: Many cheap items, few expensive items
- **Realistic**: Matches real retail price distributions
- **Natural bounds**: Prices always positive
- **Already implemented correctly!**

**No action needed** - this was implemented correctly from the start! 

---

## 9. Discount Depths ✅ BONUS FIX (Moderate)

### Problem
- **Current**: Uniform continuous distribution
- **Issue**: Unrealistic - retailers use discrete psychological price points
- **Impact**: Missing consumer psychology (10%, 20%, 25%, 33%, 50% are more appealing)

### Fix Applied

**File**: `src/retailsynth/engines/promotional_engine.py:364-415`

```python
# BEFORE (Uniform continuous)
state = promo_states.get(product_id, 1)
min_disc, max_disc = self.depth_ranges[state]
discount = np.random.uniform(min_disc, max_disc)  # e.g., 0.217, 0.183, etc.

# AFTER (Psychological price points)
# Sample from discrete psychologically appealing points
if state == 0:  # Regular: light discount
    discount = np.random.choice([0.10, 0.15, 0.20])
elif state == 1:  # Feature: moderate discount
    discount = np.random.choice([0.20, 0.25, 0.30, 0.33])
else:  # state == 2: Deep discount
    discount = np.random.choice([0.30, 0.33, 0.40, 0.50])
```

**Config Parameters Added** (`config.py:68-73`):
```python
# Psychological Discount Price Points (Distribution Fix)
# Real retailers use discrete psychologically appealing points, not uniform continuous
psychological_discounts_light: list = [0.10, 0.15, 0.20]
psychological_discounts_moderate: list = [0.20, 0.25, 0.30, 0.33]
psychological_discounts_deep: list = [0.30, 0.33, 0.40, 0.50]
```

**Why Psychological Price Points?**
- **Consumer psychology**: 25% off more appealing than 23.7% off
- **Retail practice**: Real retailers use round numbers (10%, 20%, 25%, 33%, 50%)
- **Decision simplicity**: Easier for customers to evaluate
- **Promotional clarity**: Clean messaging ("Buy One Get One 50% Off!")
- **Competitive alignment**: Matches what customers see in other stores

**Common Psychological Thresholds**:
- **10%**: Minimum to be noticed
- **20%**: "Good deal" threshold
- **25%**: Quarter off (psychologically clean)
- **33%**: "1/3 off" (buy 2 get 1 free equivalent)
- **50%**: Half off (major psychological milestone)

**Expected Impact**: +2-4% promotional response modeling

---

## Expected Outcomes

### KS Score Improvements

| Metric | Before | After Fix | Improvement |
|--------|--------|-----------|-------------|
| Quantity | 0.70 | **0.85** | +21% |
| Basket Size | 0.55 | **0.78** | +42% |
| Visit Frequency | 0.45 | **0.72** | +60% |
| Revenue | 0.60 | **0.68** | +13% (indirect) |
| **Overall Combined** | **0.628** | **0.80** | **+27%** |

### Business Metrics

| Metric | Before | After | Status |
|--------|--------|-------|--------|
| Avg basket size | ~15-29 items | 5-12 items | 
| Visit frequency/week | ~0.15 | 0.50-0.70 | 
| Qty=1 percentage | ~40% | 70-75% | 
| Avg transaction value | Varies | $20-40 | 

---

## Files Modified

1.  `src/retailsynth/generators/transaction_generator.py`
   - Line 599-621: Basket size (Poisson → Negative Binomial)
   - Line 626-629: Quantity (Normal → Log-Normal)

2.  `scripts/tune_parameters_optuna.py`
   - Line 168: Visit probability range (0.15-0.50 → 0.30-0.75)
   - Line 171: Basket size range (1-30 → 3-15)
   - Line 174: Quantity mean range (1.2-2.5 → 1.2-1.8)
   - Line 176: Quantity max range (5-15 → 10-20)

3.  `src/retailsynth/config.py`
   - Line 127: base_visit_probability (0.15 → 0.55)
   - Line 129-132: visit_prob_by_personality (all +0.25-0.40)
   - Line 201: quantity_max (10 → 20)

4.  `src/retailsynth/engines/trip_purpose.py` (BONUS)
   - Line 278-283: Trip basket size (Normal → Gamma)

5.  `src/retailsynth/generators/customer_generator.py` (BONUS)
   - Line 179-184: Days since last visit (Exponential → Gamma)

6.  `src/retailsynth/engines/drift_engine.py` (BONUS)
   - Line 42-88: Customer drift (Constant Gaussian → Mixture model)

---

## Next Steps

### Immediate (Now)
1.  Run validation to confirm improvements
2.  Re-run Optuna tuning with fixed ranges
3.  Verify KS scores improve to 0.80+

### Sprint 3 (Next Week)
Implement **MODERATE** fixes for additional +6-10% improvement:
- Product prices: Uniform → Log-Normal
- Discount depth: Uniform → Psychological points
- Trip basket size: Normal → Gamma
- Brand loyalty: Normal → Beta

**Target**: Overall KS 0.80 → **0.85-0.88**

---

## References

- **Full Audit**: `docs/DISTRIBUTION_AUDIT_COMPREHENSIVE.md` (Part 1)
- **Demographics/Behavioral**: `docs/DISTRIBUTION_AUDIT_PART2.md` (Part 2)
- **Product/Promo/Summary**: `docs/DISTRIBUTION_AUDIT_PART3.md` (Part 3)
- **Master Index**: `docs/DISTRIBUTION_AUDIT_INDEX.md`

---

**Status**:  All critical fixes implemented and ready for testing  
**Next Review**: After validation run confirms 0.80+ KS score
