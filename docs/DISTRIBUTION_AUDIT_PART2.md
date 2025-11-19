# Comprehensive Distribution Audit - Part 2

## 3. Customer Demographics (15 Parameters)

### 3.1 Age Distribution ✅ **GOOD**

**Location**: `generators/customer_generator.py:43`

```python
ages = np.random.choice(age_values, size=n, p=age_probabilities)
```

| Aspect | Current | Dunnhumby | Status | Priority |
|--------|---------|-----------|--------|----------|
| Distribution | Categorical | Categorical | ✅ Correct | - |
| Bins | 5 age groups | Standard | ✅ Appropriate | - |
| Probabilities | Configurable | From census | ✅ Flexible | - |

**Config Parameters** (all ✅ GOOD):
- `age_values: [25, 35, 45, 55, 65]` ✅
- `age_probabilities: [0.2, 0.25, 0.25, 0.2, 0.1]` ✅

**Status**: Well-implemented categorical distribution

---

### 3.2 Household Size Distribution ✅ **GOOD**

**Location**: `generators/customer_generator.py:44`

```python
household_sizes = np.random.choice(household_sizes, size=n, p=household_size_probs)
```

| Aspect | Current | Census Data | Status | Priority |
|--------|---------|-------------|--------|----------|
| Distribution | Categorical | Categorical | ✅ Correct | - |
| Range | 1-5 people | Standard | ✅ Appropriate | - |
| Probabilities | [0.28, 0.35, 0.16, 0.15, 0.06] | Close to US census | ✅ Realistic | - |

**Config Parameters** (all ✅ GOOD):
- `household_sizes: [1, 2, 3, 4, 5]` ✅
- `household_size_probs: [0.28, 0.35, 0.16, 0.15, 0.06]` ✅

**Status**: Matches US Census Bureau data well

---

### 3.3 Income Distribution ⚠️ **MODERATE**

**Location**: `generators/customer_generator.py:58-72`

```python
# Age-dependent categorical sampling
income_brackets[young_mask] = np.random.choice(
    ['<30K', '30-50K', '50-75K'], 
    p=config.young_income_probs
)
```

| Aspect | Current | Ideal | Issue | Priority |
|--------|---------|-------|-------|----------|
| Distribution | Categorical (age-dependent) | Log-Normal (continuous) | Too coarse | 🟡 |
| Age dependency | ✅ Implemented | ✅ Good | Well-designed | ✅ |
| Granularity | 5 brackets | Continuous $20K-$200K | Low resolution | 🟡 |

**Why Categorical Suboptimal**:
- Income is fundamentally continuous
- Brackets lose information ($30K vs $49K both "30-50K")
- Log-normal is standard for income distributions

**Current Fix**: Acceptable for v1.0, consider continuous in v2.0

**Config Parameters**:
- `young_income_probs: [0.35, 0.4, 0.25]` ✅ Reasonable
- `middle_income_probs: [0.25, 0.35, 0.25, 0.15]` ✅ Reasonable
- `senior_income_probs: [0.4, 0.4, 0.2]` ✅ Reasonable

---

### 3.4 Marital Status & Children

**Location**: `generators/customer_generator.py:87-98`

```python
# Marital status
marital_status = np.where(
    (ages < 30) | (np.random.random(n) < config.single_probability),
    'Single', 'Married'
)

# Children count
children_count = np.where(
    (ages < 25) | (marital_status == 'Single'),
    0,
    np.random.choice([0, 1, 2, 3], size=n, p=config.children_probs)
)
```

| Aspect | Current | Reality | Status | Priority |
|--------|---------|---------|--------|----------|
| Marital (distribution) | Bernoulli (age-dependent) | Bernoulli | ✅ Correct | - |
| Children (distribution) | Categorical | Poisson/Categorical | ✅ OK | - |
| Age dependency | ✅ Implemented | ✅ Realistic | Good logic | ✅ |

**Config Parameters** (all ✅ GOOD):
- `single_probability: 0.3` ✅
- `children_probs: [0.3, 0.35, 0.25, 0.1]` ✅

**Status**: Well-designed demographic dependencies

---

### 3.5 Shopping Personality Mix

**Location**: `generators/customer_generator.py:45-50`

```python
personalities = np.random.choice(
    ['price_anchor', 'convenience', 'planned', 'impulse'],
    size=n,
    p=[config.price_anchor_customers, config.convenience_customers,
       config.planned_customers, config.impulse_customers]
)
```

| Aspect | Current | Literature | Status | Priority |
|--------|---------|------------|--------|----------|
| Distribution | Categorical | Latent Class | ✅ Reasonable | - |
| Categories | 4 types | 3-5 typical | ✅ Standard | - |
| Mix | [0.25, 0.25, 0.30, 0.20] | Tunable | ✅ Flexible | - |

**Config Parameters** (all ✅ GOOD):
- `price_anchor_customers: 0.25` ✅
- `convenience_customers: 0.25` ✅
- `planned_customers: 0.30` ✅
- `impulse_customers: 0.20` ✅

**Status**: Standard retail segmentation approach

---

## 4. Behavioral Parameters (18 Parameters)

### 4.1 Price Sensitivity Distribution ✅ **EXCELLENT**

**Location**: `engines/customer_heterogeneity.py:95-101` (Phase 2.4)

```python
price_sensitivity_dist = {
    'type': 'lognormal',
    'params': {'mean': 0.15, 'sigma': 0.4},
    'bounds': (0.5, 2.5)
}
```

| Aspect | Current | Research | Status | Priority |
|--------|---------|----------|--------|----------|
| Distribution | **Log-Normal** | Log-Normal | ✅ PERFECT | - |
| Mean | ~1.2 | 0.8-1.5 | ✅ Correct | - |
| Shape | Right-skewed | Right-skewed | ✅ Ideal | - |

**Why Log-Normal is Perfect**:
- Price sensitivity is multiplicative (2x sensitive → 2x response)
- Most customers ~1.0, some very price-sensitive (2.0+)
- Textbook-correct distribution for elasticity parameters

**Status**: ✅ **BEST PRACTICE** - No changes needed

---

### 4.2 Promotional Responsiveness ✅ **GOOD**

**Location**: `engines/customer_heterogeneity.py:111-117` (Phase 2.4)

```python
promo_responsiveness_dist = {
    'type': 'beta',
    'params': {'alpha': 3, 'beta': 2},
    'bounds': (0.5, 2.0)
}
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Distribution | Beta (rescaled) | Beta/Gamma | ✅ Good | - |
| Shape | Left-skewed (α>β) | Flexible | ✅ Reasonable | - |
| Bounds | (0.5, 2.0) | Appropriate | ✅ OK | - |

**Config Parameters** (Phase 2.4 - NEW):
- `hetero_promo_alpha: 3.0` → Tunable (Tier 2) ✅
- `hetero_promo_beta: 2.0` → Tunable (Tier 2) ✅

**Status**: ✅ Good - Beta appropriate for bounded behavioral parameters

---

### 4.3 Quality Preference ✅ **GOOD**

**Location**: `engines/customer_heterogeneity.py:103-109` (Phase 2.4)

```python
quality_preference_dist = {
    'type': 'beta',
    'params': {'alpha': 5, 'beta': 3},
    'bounds': (0.3, 1.5)
}
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Distribution | Beta (rescaled) | Beta | ✅ Correct | - |
| Shape | Left-skewed (α>β) | Most prefer quality | ✅ Realistic | - |
| Mean | ~0.9 | 0.8-1.0 | ✅ OK | - |

**Status**: ✅ Good - Most customers quality-focused, some indifferent

---

### 4.4 Display Sensitivity ✅ **GOOD**

**Location**: `engines/customer_heterogeneity.py` (Phase 2.4)

```python
display_sensitivity_dist = {
    'type': 'beta',
    'params': {'alpha': 3, 'beta': 3},
    'bounds': (0.3, 1.5)
}
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Distribution | Beta (symmetric) | Beta | ✅ Good | - |
| Shape | Symmetric (α=β) | Balanced | ✅ Reasonable | - |

**Config Parameters** (Phase 2.4):
- `hetero_display_alpha: 3.0` → Tunable (Tier 2) ✅
- `hetero_display_beta: 3.0` → Tunable (Tier 2) ✅

**Status**: ✅ Good implementation

---

### 4.5 Brand Loyalty ⚠️ **MODERATE**

**Location**: `generators/customer_generator.py:135-146`

```python
# Number of preferred brands
n_preferred_brands = np.random.choice([3, 4, 5, 6, 7], p=[0.15, 0.25, 0.30, 0.20, 0.10])
preferred_brands = np.random.choice(all_brands, size=n_preferred_brands, replace=False)

# Loyalty strength
brand_loyalty_param = customer_params_df.loc[i, 'brand_loyalty']
loyalty_std = 0.15
pref = np.random.normal(brand_loyalty_param * 0.5, loyalty_std)
pref = np.clip(pref, 0.1, 0.95)
```

| Aspect | Current | Ideal | Issue | Priority |
|--------|---------|-------|-------|----------|
| Number of brands | Categorical [3-7] | Poisson/Negative Binomial | Too rigid | 🟡 |
| Loyalty strength | Normal (clipped) | Beta | Bounded param needs Beta | 🟡 |
| Heterogeneity | ✅ From Phase 2.4 | ✅ Good | Well-integrated | ✅ |

**Why Current Approach Suboptimal**:
- Number of preferred brands: categorical is arbitrary, should be Poisson
- Loyalty strength: Normal needs clipping (0.1, 0.95), Beta naturally bounded

**Fix**:
```python
# Number of preferred brands (Poisson)
n_preferred_brands = np.random.poisson(4.5) + 1  # Mean ~5 brands
n_preferred_brands = min(n_preferred_brands, len(all_brands))

# Loyalty strength (Beta, rescaled)
alpha, beta = 5, 3  # Left-skewed (most are loyal)
pref = np.random.beta(alpha, beta) * 0.85 + 0.1  # Rescale to (0.1, 0.95)
```

**Impact**: Minor improvement (+2-3%)

**Config Parameters**:
- `brand_loyalty_mean: 0.6` ✅
- `brand_loyalty_std: 0.2` ✅
- `brand_loyalty_by_personality` → Good concept ✅

---

### 4.6 Purchase History Weights (5 params) ✅ **GOOD**

**Location**: `config.py:190-194`

```python
loyalty_weight: float = 0.3
habit_weight: float = 0.4
inventory_weight: float = 0.5
variety_weight: float = 0.2
price_memory_weight: float = 0.1
```

| Param | Range | Status | Priority |
|-------|-------|--------|----------|
| `loyalty_weight` | 0.1-0.6 tunable | ✅ OK | Tier 1 |
| `habit_weight` | 0.2-0.7 tunable | ✅ OK | Tier 1 |
| `inventory_weight` | 0.3-0.8 tunable | ✅ OK | Tier 1 |
| `variety_weight` | 0.1-0.5 tunable | ✅ OK | Tier 1 |
| `price_memory_weight` | 0.05-0.3 tunable | ✅ OK | Tier 1 |

**Status**: All tunable weights are well-designed ✅

---

### 4.7 Basket Composition Probabilities (3 params) ✅ **GOOD**

**Location**: `config.py:225-227`

```python
complement_probability: float = 0.4
substitute_avoidance: float = 0.8
category_diversity_weight: float = 0.3
```

| Param | Range | Status | Priority |
|-------|-------|--------|----------|
| `complement_probability` | 0.2-0.7 tunable | ✅ OK | Tier 1 |
| `substitute_avoidance` | 0.6-0.95 tunable | ✅ OK | Tier 1 |
| `category_diversity_weight` | 0.1-0.6 tunable | ✅ OK | Tier 1 |

**Status**: Sprint 1.4 basket composition well-implemented ✅

---

### 4.8 Store Loyalty Parameters (4 params)

**Location**: `config.py:206-209`

```python
store_loyalty_weight: float = 0.6
store_switching_probability: float = 0.15
distance_weight: float = 0.4
satisfaction_weight: float = 0.6
```

| Param | Range | Status | Priority |
|-------|-------|--------|----------|
| `store_loyalty_weight` | 0.4-0.8 tunable | ✅ OK | Tier 2 |
| `store_switching_probability` | 0.05-0.30 tunable | ✅ OK | Tier 2 |
| `distance_weight` | 0.2-0.6 tunable | ✅ OK | Tier 2 |
| `satisfaction_weight` | 0.4-0.8 tunable | ✅ OK | Tier 2 |

**Store Preference Assignment**:

**Location**: `engines/loyalty_engine.py:38-46`

```python
# Number of preferred stores
n_preferred = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])

# Assign preference weights using Dirichlet
weights = np.random.dirichlet(np.ones(n_preferred))
weights = weights * 0.8  # 80% to preferred stores
```

| Aspect | Current | Ideal | Status | Priority |
|--------|---------|-------|--------|----------|
| Number of stores | Categorical [2-4] | Categorical | ✅ OK | - |
| Weight distribution | Dirichlet | Dirichlet | ✅ EXCELLENT | - |
| Exploration | 20% | Tunable | ✅ Good | - |

**Status**: ✅ Dirichlet is textbook-correct for preference distributions

---

## 5. Store & Location (8 Parameters)

### 5.1 Store Type Distribution

**Location**: `generators/store_generator.py:28`

```python
store_type = np.random.choice(store_types, p=[0.4, 0.3, 0.2, 0.1])
# Types: Hypermarket, Supermarket, Convenience, Specialty
```

| Aspect | Current | Reality | Status | Priority |
|--------|---------|---------|--------|----------|
| Distribution | Categorical | Categorical | ✅ Correct | - |
| Mix | [0.4, 0.3, 0.2, 0.1] | Region-dependent | ⚠️ Fixed | 🟢 |

**Status**: Reasonable defaults, could make region-specific

---

### 5.2 Store Attributes

**Location**: `generators/store_generator.py:45-50`

```python
'region': np.random.choice(regions),
'established_year': np.random.randint(1995, 2023),
'has_pharmacy': np.random.choice([True, False], p=[0.6, 0.4]),
'has_deli': np.random.choice([True, False], p=[0.7, 0.3]),
'has_bakery': np.random.choice([True, False], p=[0.8, 0.2]),
```

| Feature | Distribution | Probability | Status |
|---------|--------------|-------------|--------|
| Region | Uniform categorical | Equal | ✅ OK |
| Established year | Uniform discrete | 1995-2023 | ✅ OK |
| Has pharmacy | Bernoulli | 60% | ✅ Realistic |
| Has deli | Bernoulli | 70% | ✅ Realistic |
| Has bakery | Bernoulli | 80% | ✅ Realistic |

**Status**: All reasonable ✅

---

### 5.3 Store Size

**Location**: `generators/store_generator.py:31-37`

```python
if store_type == 'Hypermarket':
    square_feet = np.random.randint(120000, 200000)
elif store_type == 'Supermarket':
    square_feet = np.random.randint(40000, 80000)
```

| Aspect | Current | Ideal | Issue | Priority |
|--------|---------|-------|-------|----------|
| Distribution | Uniform discrete | Log-Normal | Too flat | 🟢 LOW |
| Ranges | Type-dependent | ✅ Good | Well-designed | ✅ |

**Why Uniform Suboptimal**:
- Store sizes within type are log-normally distributed (power law)
- Uniform gives equal probability to all sizes

**Fix** (low priority):
```python
# Log-normal within type ranges
mean_log = np.log(80000)  # For supermarket
sigma_log = 0.3
square_feet = int(np.random.lognormal(mean_log, sigma_log))
square_feet = np.clip(square_feet, 40000, 80000)
```

**Impact**: Very minor (+<1%)

---

### 5.4 Shopping Time Distribution ✅ **GOOD**

**Location**: `generators/transaction_generator.py:636-639`

```python
hour = np.random.choice(
    range(8, 20),  # 8am - 8pm
    p=[0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.05, 0.05, 0.08, 0.12, 0.05]
)
```

| Aspect | Current | Reality | Status | Priority |
|--------|---------|---------|--------|----------|
| Distribution | Categorical | Categorical | ✅ Correct | - |
| Peak hours | 11am-12pm, 6pm-7pm | Matches reality | ✅ Realistic | - |
| Range | 8am-8pm | Store-dependent | ⚠️ Fixed | 🟢 |

**Status**: Good default, could make store-type dependent (24hr vs regular)

---

