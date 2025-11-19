# Pricing & Promotional Engine Separation Design

## Sprint 2 Goal 1: Clean Separation of Pricing and Promotions

**Status**: Design Phase  
**Date**: 2025-11-11  
**Priority**: High (Sprint 2 Foundation)

---

## Problem Statement

### Current Architecture (MIXED):
```
┌─────────────────────────────────────────┐
│         PriceStateHMM (MIXED)           │
│                                         │
│  States = Base Price + Promo Combined   │
│  - State 0: Regular (0-5% discount)     │
│  - State 1: Feature (10-25% discount)   │
│  - State 2: Deep (25-50% discount)      │
│  - State 3: Clearance (50-100%)         │
│                                         │
│  ❌ Cannot distinguish:                 │
│     - Base price change vs promo        │
│     - Permanent vs temporary discounts  │
└─────────────────────────────────────────┘
```

### Issues:
1. **No separation** between base price dynamics and promotional discounts
2. **Cannot model** temporary promotions on top of base prices
3. **Cannot capture** marketing signal from promotional intensity
4. **Violates retail reality**: Base prices are strategic, promos are tactical

---

## New Architecture (SEPARATED):

```
┌──────────────────────────────────────────────────────────────────┐
│                    RETAIL PRICING SYSTEM                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ├─────────────────────────────────────┐
                              ▼                                     ▼
        ┌─────────────────────────────────┐    ┌──────────────────────────────────┐
        │   BASE PRICE HMM (Strategic)    │    │  PROMOTIONAL HMM (Tactical)      │
        │                                 │    │                                  │
        │  States = Price Tiers           │    │  States = Promo Intensity        │
        │  - State 0: Low Price           │    │  - State 0: No Promo             │
        │  - State 1: Mid Price           │    │  - State 1: Light Promo          │
        │  - State 2: High Price          │    │  - State 2: Moderate Promo       │
        │  - State 3: Premium Price       │    │  - State 3: Heavy Promo          │
        │                                 │    │                                  │
        │  Transitions: Slow (monthly)    │    │  Transitions: Fast (weekly)      │
        │  Learned from: Regular prices   │    │  Learned from: Promo flags       │
        │  Output: base_price_t           │    │  Output: promo_state_t           │
        └─────────────────────────────────┘    └──────────────────────────────────┘
                              │                                     │
                              │                                     │
                              └──────────────┬──────────────────────┘
                                             ▼
                              ┌──────────────────────────────┐
                              │   PROMOTIONAL ENGINE         │
                              │                              │
                              │  Combines:                   │
                              │  1. Base Price (from HMM)    │
                              │  2. Promo State (from HMM)   │
                              │  3. Promo Mechanics          │
                              │  4. Display/Feature flags    │
                              │                              │
                              │  Output:                     │
                              │  - final_price_t             │
                              │  - promo_depth_t             │
                              │  - marketing_signal_t        │
                              └──────────────────────────────┘
```

---

## Component Design

### 1. Base Price HMM (New)

**Purpose**: Model strategic base price dynamics (slow-moving)

**File**: `src/retailsynth/engines/base_price_hmm.py`

**States** (4 states based on price percentiles):
```python
State 0: Low Price      (P0-P25 of product's price distribution)
State 1: Mid-Low Price  (P25-P50)
State 2: Mid-High Price (P50-P75)
State 3: High Price     (P75-P100)
```

**Transition Dynamics**:
- **Slow transitions**: Base prices change monthly/quarterly
- **Sticky states**: High diagonal probabilities (0.85-0.95)
- **Gradual changes**: Adjacent state transitions more likely

**Learning From**:
```python
# Filter to NON-promotional weeks only
regular_prices = transactions[
    (transactions['RETAIL_DISC'] == 0) &
    (transactions['DISPLAY'].isna()) &
    (transactions['MAILER'].isna())
]

# Calculate base price per product-week
base_prices = regular_prices.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
    'SALES_VALUE': 'sum',
    'QUANTITY': 'sum'
})
base_prices['base_price'] = base_prices['SALES_VALUE'] / base_prices['QUANTITY']
```

**Transition Matrix Example**:
```
         Low   Mid-L  Mid-H  High
Low    [ 0.90  0.08   0.02   0.00 ]
Mid-L  [ 0.05  0.88   0.06   0.01 ]
Mid-H  [ 0.01  0.06   0.88   0.05 ]
High   [ 0.00  0.02   0.08   0.90 ]
```

**Emission Distribution**:
- Mean price per state (product-specific)
- Std dev per state (captures price volatility)

---

### 2. Promotional HMM (New)

**Purpose**: Model tactical promotional state transitions (fast-moving)

**File**: `src/retailsynth/engines/promo_hmm.py`

**States** (4 states based on promotional intensity):
```python
State 0: No Promo       (0% discount, no display/ad)
State 1: Light Promo    (5-15% discount, shelf promo)
State 2: Moderate Promo (15-30% discount, display OR ad)
State 3: Heavy Promo    (30-50% discount, display AND ad)
```

**Transition Dynamics**:
- **Fast transitions**: Promotions change weekly
- **Cyclical patterns**: Promo → No Promo → Promo cycles
- **Seasonal effects**: Higher promo frequency in holidays

**Learning From**:
```python
# Identify promotional weeks
promo_weeks = transactions[
    (transactions['RETAIL_DISC'] > 0) |
    (transactions['DISPLAY'].notna()) |
    (transactions['MAILER'].notna())
]

# Calculate promo intensity per product-week
promo_intensity = promo_weeks.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
    'RETAIL_DISC': 'sum',
    'SALES_VALUE': 'sum',
    'DISPLAY': lambda x: x.notna().any(),
    'MAILER': lambda x: x.notna().any()
})

promo_intensity['discount_depth'] = (
    promo_intensity['RETAIL_DISC'] / promo_intensity['SALES_VALUE']
)
promo_intensity['has_display'] = promo_intensity['DISPLAY']
promo_intensity['has_ad'] = promo_intensity['MAILER']
```

**Transition Matrix Example**:
```
         None   Light  Mod    Heavy
None   [ 0.70  0.20   0.08   0.02 ]
Light  [ 0.40  0.35   0.20   0.05 ]
Mod    [ 0.30  0.25   0.30   0.15 ]
Heavy  [ 0.50  0.20   0.20   0.10 ]
```

**Emission Distribution**:
- Discount depth per state (mean, std)
- Display probability per state
- Ad probability per state

---

### 3. Promotional Engine (Updated)

**Purpose**: Combine base prices + promo states → final prices + marketing signal

**File**: `src/retailsynth/engines/promotional_engine.py` (refactored)

**Process Flow**:
```python
def generate_store_week_prices(store_id, week):
    # Step 1: Get base prices from Base Price HMM
    base_prices = base_price_hmm.sample_prices(week)
    
    # Step 2: Get promo states from Promotional HMM
    promo_states = promo_hmm.sample_states(week)
    
    # Step 3: For each product, apply promo mechanics
    final_prices = {}
    promo_context = StorePromoContext(store_id, week)
    
    for product_id in products:
        base_price = base_prices[product_id]
        promo_state = promo_states[product_id]
        
        # Apply promotional discount based on state
        if promo_state == 0:  # No promo
            final_price = base_price
            discount_depth = 0.0
            has_display = False
            has_ad = False
        
        elif promo_state == 1:  # Light promo
            discount_depth = sample_from_emission(promo_state, 'discount')
            final_price = base_price * (1 - discount_depth)
            has_display = sample_from_emission(promo_state, 'display')
            has_ad = False
        
        elif promo_state == 2:  # Moderate promo
            discount_depth = sample_from_emission(promo_state, 'discount')
            final_price = base_price * (1 - discount_depth)
            has_display = sample_from_emission(promo_state, 'display')
            has_ad = sample_from_emission(promo_state, 'ad')
        
        elif promo_state == 3:  # Heavy promo
            discount_depth = sample_from_emission(promo_state, 'discount')
            final_price = base_price * (1 - discount_depth)
            has_display = True
            has_ad = True
        
        # Store in context
        promo_context.add_promotion(
            product_id, final_price, discount_depth, has_display, has_ad
        )
    
    # Step 4: Calculate marketing signal
    marketing_signal = marketing_calculator.calculate_signal(promo_context)
    
    return promo_context, marketing_signal
```

---

## Data Separation Strategy

### How to Learn from Dunnhumby Data:

**Challenge**: Dunnhumby doesn't explicitly separate base price vs promo

**Solution**: Infer separation using heuristics

#### Method 1: Discount-Based Separation
```python
# Base Price Weeks: No discount activity
base_price_weeks = transactions[
    (transactions['RETAIL_DISC'] == 0) &
    (transactions['COUPON_DISC'] == 0)
]

# Promotional Weeks: Has discount activity
promo_weeks = transactions[
    (transactions['RETAIL_DISC'] > 0) |
    (transactions['COUPON_DISC'] > 0)
]
```

#### Method 2: Causal Data Integration
```python
# Use causal_data.csv for explicit promo flags
# Merge with transactions
merged = transactions.merge(
    causal_data,
    on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
    how='left'
)

# Base Price: No causal activity
base_price_weeks = merged[
    merged['DISPLAY'].isna() &
    merged['MAILER'].isna()
]

# Promotional: Has causal activity
promo_weeks = merged[
    merged['DISPLAY'].notna() |
    merged['MAILER'].notna()
]
```

#### Method 3: Statistical Smoothing
```python
# Estimate base price as rolling median of non-promo weeks
for product_id in products:
    product_data = transactions[transactions['PRODUCT_ID'] == product_id]
    
    # Calculate effective price per week
    weekly_prices = product_data.groupby('WEEK_NO').apply(
        lambda x: x['SALES_VALUE'].sum() / x['QUANTITY'].sum()
    )
    
    # Identify outliers (likely promos)
    median_price = weekly_prices.median()
    std_price = weekly_prices.std()
    
    # Base price weeks: within 1 std of median
    base_weeks = weekly_prices[
        abs(weekly_prices - median_price) < std_price
    ]
    
    # Promo weeks: below median - 1 std
    promo_weeks = weekly_prices[
        weekly_prices < median_price - std_price
    ]
```

---

## Implementation Plan

### Phase 1: Create Base Price HMM ✅
**File**: `src/retailsynth/engines/base_price_hmm.py`

**Key Methods**:
- `learn_from_data(transactions_df, causal_df)` - Learn from non-promo weeks
- `sample_base_prices(week, products)` - Generate base prices
- `get_transition_matrix(product_id)` - Get product-specific transitions
- `save_parameters(path)` / `load_parameters(path)` - Persistence

### Phase 2: Create Promotional HMM ✅
**File**: `src/retailsynth/engines/promo_hmm.py`

**Key Methods**:
- `learn_from_data(transactions_df, causal_df)` - Learn from promo weeks
- `sample_promo_states(week, products)` - Generate promo states
- `sample_promo_mechanics(state)` - Sample discount/display/ad from state
- `save_parameters(path)` / `load_parameters(path)` - Persistence

### Phase 3: Refactor Promotional Engine ✅
**File**: `src/retailsynth/engines/promotional_engine.py`

**Changes**:
- Remove old mixed HMM logic
- Integrate `base_price_hmm` and `promo_hmm`
- Update `generate_store_promo_context()` to use both HMMs
- Ensure `StorePromoContext` captures all promo details

### Phase 4: Create Learning Script ✅
**File**: `scripts/learn_separated_pricing.py`

**Process**:
1. Load Dunnhumby data
2. Separate base price vs promo weeks
3. Learn Base Price HMM
4. Learn Promotional HMM
5. Save both models
6. Generate validation report

### Phase 5: Update Main Generator ✅
**File**: `src/retailsynth/generators/main_generator.py`

**Changes**:
- Load both HMMs instead of single mixed HMM
- Pass both to PromotionalEngine
- Update transaction generation to use separated pricing

---

## Validation Metrics

### Base Price HMM Validation:
1. **Transition Stability**: Diagonal elements > 0.85 (slow changes)
2. **Price Coverage**: >80% of products have learned parameters
3. **Price Realism**: Mean prices within 10% of observed regular prices
4. **State Distribution**: Roughly balanced across 4 states

### Promotional HMM Validation:
1. **Promo Frequency**: Matches observed promo frequency (15-25% of weeks)
2. **Discount Depth**: Mean discount per state matches observed
3. **Display/Ad Correlation**: Display+Ad more common in heavy promo states
4. **Cyclical Patterns**: Captures weekly/seasonal promo cycles

### Combined System Validation:
1. **Price Distribution**: Final prices match observed distribution
2. **Discount Distribution**: Discount depths match observed
3. **Marketing Signal**: Correlates with observed store traffic
4. **Temporal Patterns**: Captures promotional calendars (holidays, etc.)

---

## Benefits of Separation

### 1. **Realism**
- ✅ Matches retail practice (base prices are strategic, promos are tactical)
- ✅ Captures different time scales (monthly base price changes, weekly promos)

### 2. **Flexibility**
- ✅ Can model promotions independently of base prices
- ✅ Can test "what if" scenarios (e.g., no promos, double promos)

### 3. **Marketing Signal**
- ✅ Promotional intensity can influence store visit probability
- ✅ Separates price-based utility from marketing-based attraction

### 4. **Customer Heterogeneity**
- ✅ Different customers respond differently to base prices vs promos
- ✅ "Cherry pickers" vs "loyal" customers

### 5. **Validation**
- ✅ Easier to validate each component separately
- ✅ Can compare base price dynamics vs promo dynamics independently

---

## Next Steps

1. ✅ Review and approve this design
2. ⏳ Implement Base Price HMM
3. ⏳ Implement Promotional HMM
4. ⏳ Refactor Promotional Engine
5. ⏳ Create learning script
6. ⏳ Run validation tests
7. ⏳ Integrate into main generator

---

**Status**: Ready for Implementation  
**Estimated Time**: 2-3 days  
**Sprint**: Sprint 2.1 (Pricing & Promo Separation)
