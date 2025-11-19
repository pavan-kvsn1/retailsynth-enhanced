# PRIORITY 2 & 3 DIAGNOSTIC REPORT
**Date:** November 12, 2025  
**Status:** 🔴 Issues Identified - Ready for Implementation

---

## 📋 PRIORITY 2: QUANTITY LOGIC IN BASKET COMPOSER 🔴

### **Problem:**
`quantity_mean`, `quantity_std`, and `quantity_max` config parameters are **PARTIALLY BYPASSED** in basket composition.

### **Root Cause Analysis:**

**TWO DIFFERENT CODE PATHS for quantity assignment:**

#### **Path 1: Transaction Generator (Simple Basket Mode)** ✅ WORKING
```python
# File: src/retailsynth/generators/transaction_generator.py:411
quantity = max(1, int(np.random.normal(self.config.quantity_mean, self.config.quantity_std)))
quantity = min(quantity, self.config.quantity_max)  # Cap at max
```
✅ **Uses config parameters correctly**

#### **Path 2: Basket Composer (Realistic Basket Mode)** ❌ BROKEN
```python
# File: src/retailsynth/engines/basket_composer.py:446-456
if customer_state and product_id in customer_state.purchase_count:
    # Repeat customers buy typical quantity
    quantity = constraint.typical_quantity  # ❌ HARDCODED from category constraints!
else:
    # New products: usually buy 1
    quantity = 1  # ❌ HARDCODED!

# Occasionally buy more (stockpiling)
if trip_purpose == TripPurpose.STOCK_UP and np.random.random() < 0.3:
    quantity = min(quantity + 1, constraint.max_quantity_per_product)  # ❌ HARDCODED +1!
```
❌ **Ignores `quantity_mean`, `quantity_std`, and `quantity_max` entirely!**

### **Impact:**

**When Basket Composer is Enabled (Current Default):**
- `quantity_mean` = IGNORED ❌
- `quantity_std` = IGNORED ❌
- `quantity_max` = PARTIALLY USED (only in `_enforce_quantity_constraints`)
- Quantities are determined by:
  1. Category constraint `typical_quantity` (e.g., DAIRY=2, MEAT=1, SNACKS=1)
  2. Hardcoded values (1 for new products)
  3. Hardcoded increments (+1 for stock-up trips)

**Result:**
- Quantity distribution is FIXED and cannot be tuned
- Optuna tuning of `quantity_mean` and `quantity_std` has NO EFFECT
- Quantities are too rigid and not responsive to customer heterogeneity

### **Observed Behavior:**
```
Current synthetic: avg quantity = 1.13 (target: 2.47)
KS score = 0.927 (good)  # But this is misleading - it's good by accident!
```

The quantity is WAY TOO LOW because hardcoded logic always assigns 1 for new products.

---

## 📋 PRIORITY 3: PROMOTIONAL RESPONSE INTEGRATION 🟡

### **Problem:**
Promotional response parameters (`promotion_sensitivity_mean`, `promotion_sensitivity_std`, `promotion_quantity_boost`) are **DEFINED BUT NOT CONNECTED TO PHASE 2.5**.

### **Root Cause Analysis:**

**Config Parameters (DEFINED):**
```python
# File: src/retailsynth/config.py:191-193
promotion_sensitivity_mean: float = 0.5
promotion_sensitivity_std: float = 0.2
promotion_quantity_boost: float = 1.5  # Multiplier for quantity when on promotion
```

**Phase 2.5 Implementation (WORKING BUT DISCONNECTED):**
```python
# File: src/retailsynth/engines/promo_response.py
class PromoResponseCalculator:
    def __init__(self, config: Optional[Dict] = None):
        # ❌ Does NOT use config.promotion_sensitivity_mean/std
        # ❌ Does NOT use config.promotion_quantity_boost
        
    def calculate_promo_response(self, customer_params, ...):
        # Uses customer_params['promo_responsiveness_param']  # From Phase 2.4
        # ✅ This works for utility boost
        # ❌ But does NOT boost QUANTITY when on promo
```

**Phase 2.4 Heterogeneity (WORKING):**
```python
# Customer heterogeneity DOES generate individual promo_responsiveness_param
# But it's NOT derived from config.promotion_sensitivity_mean/std
# It's derived from archetype distributions instead
```

### **Impact:**

**Promotional Sensitivity Parameters:**
- `promotion_sensitivity_mean` = UNUSED ❌
- `promotion_sensitivity_std` = UNUSED ❌
- Phase 2.4 generates `promo_responsiveness_param` independently
- Config parameters are dead code

**Promotional Quantity Boost:**
- `promotion_quantity_boost` = UNUSED ❌
- Customers do NOT buy more when products are on promotion
- Utility boost exists, but quantity stays the same
- **This is UNREALISTIC** - people buy more of sale items!

### **Expected Behavior (Missing):**
```python
# When product is on promotion and customer decides to buy it:
if is_promoted:
    base_quantity = sample_from_config_distribution()
    final_quantity = base_quantity * config.promotion_quantity_boost  # MISSING!
```

### **Observed Impact:**
```
Current: Quantity is FIXED regardless of promotion
Expected: Quantity should INCREASE by 1.5x when on promo
Result: Missing ~33% of promotional volume effect
```

---

## 🎯 PROPOSED FIXES

### **Fix 2A: Integrate Config Quantities into Basket Composer** 🔴

**Change:** Modify `basket_composer.py:_sample_products_from_category()` to use config-based quantity distribution.

**Before:**
```python
quantity = constraint.typical_quantity  # Hardcoded
```

**After:**
```python
# Use config-based distribution with category constraints as bounds
base_quantity = max(1, int(np.random.normal(self.config.quantity_mean, self.config.quantity_std)))
quantity = min(base_quantity, constraint.max_quantity_per_product)
```

**Files to Change:**
- `src/retailsynth/engines/basket_composer.py` (lines 446-456)

**Expected Impact:**
- `quantity_mean` and `quantity_std` now tunable ✅
- Average quantity should increase from 1.13 to ~2.47
- More variation in quantities (not everyone buys 1 or 2)

---

### **Fix 2B: Add Promotional Quantity Boost** 🟡

**Change:** Apply `promotion_quantity_boost` when customer buys promoted products.

**Approach:**
1. Pass promo flags to basket composer
2. When sampling quantity, check if product is on promotion
3. Apply multiplier: `final_quantity = base_quantity * boost`

**Where to Apply:**
- Option 1: In `basket_composer.py` when sampling products
- Option 2: In `transaction_generator.py` when finalizing basket
- **Recommended:** Option 1 (basket composer knows product context)

**Implementation:**
```python
def _sample_products_from_category(self, ..., promo_flags: dict = None):
    # ... existing code ...
    
    base_quantity = max(1, int(np.random.normal(...)))
    
    # Apply promotional boost if product is on promotion
    if promo_flags and product_id in promo_flags and promo_flags[product_id] > 0:
        boost = self.config.promotion_quantity_boost
        # Apply boost with some probability (not all customers stockpile)
        if np.random.random() < 0.6:  # 60% of customers stockpile
            base_quantity = int(base_quantity * boost)
    
    quantity = min(base_quantity, constraint.max_quantity_per_product)
```

**Files to Change:**
- `src/retailsynth/engines/basket_composer.py`
- Update method signatures to accept `promo_flags`
- Update callers to pass promo flags

**Expected Impact:**
- Promotional products have higher quantities ✅
- More realistic promotional response
- Revenue increases during promotion weeks
- `promotion_quantity_boost` now tunable

---

### **Fix 3: Connect Promotion Sensitivity Config to Phase 2.4** 🟡 OPTIONAL

**Change:** Use `promotion_sensitivity_mean` and `promotion_sensitivity_std` when generating heterogeneity parameters.

**Current (Phase 2.4):**
```python
# Uses archetype distributions
promo_responsiveness = archetype['promo_responsiveness']
```

**After:**
```python
# Use config parameters as base, vary by archetype
base_mean = config.promotion_sensitivity_mean
base_std = config.promotion_sensitivity_std
promo_responsiveness = np.random.normal(base_mean * archetype_multiplier, base_std)
```

**Priority:** LOWER - Phase 2.4 heterogeneity is already working well. This just adds another tuning knob.

---

## 📊 EXPECTED IMPROVEMENTS

### **Current State:**
```
Quantity: 1.13 (target: 2.47)  ❌ -54% too low
Quantity KS: 0.927             ✅ Good (but misleading)
Promotional volume: FLAT       ❌ No boost during promos
```

### **After Fix 2A (Config Quantities):**
```
Quantity: 2.2-2.6 (target: 2.47)  ✅ Within ±10%
Quantity KS: 0.85-0.95            ✅ Good
quantity_mean/std tunable         ✅ Working
```

### **After Fix 2B (Promo Boost):**
```
Promo week quantity: +30-50%      ✅ Realistic
promotion_quantity_boost tunable  ✅ Working
Revenue during promos: +20-30%    ✅ More realistic
```

---

## 🔬 TESTING PLAN

### **Test 1: Quantity Distribution (Fix 2A)**
```bash
# Before fix
python scripts/generate_with_elasticity.py --n-customers 100 --n-weeks 5
# Check: avg quantity should be ~1.1

# After fix
python scripts/generate_with_elasticity.py --n-customers 100 --n-weeks 5
# Check: avg quantity should be ~2.4
```

### **Test 2: Quantity Tuning (Fix 2A)**
```bash
# Run Optuna with different quantity_mean values
python scripts/tune_parameters_optuna.py --objective all --tier 1 --n-trials 10
# Verify: quantity_mean changes affect output
```

### **Test 3: Promotional Boost (Fix 2B)**
```python
# Compare promo vs non-promo weeks
promo_weeks_qty = synth_df[synth_df['is_promo'] == 1]['quantity'].mean()
non_promo_weeks_qty = synth_df[synth_df['is_promo'] == 0]['quantity'].mean()
boost_ratio = promo_weeks_qty / non_promo_weeks_qty
assert 1.3 <= boost_ratio <= 1.6  # Should be 30-60% higher
```

---

## ✅ IMPLEMENTATION ORDER

### **Phase 1: Fix 2A (High Priority)** 🔴
1. ✅ Modify `basket_composer.py` to use config quantities
2. ✅ Test with simple generation
3. ✅ Verify quantity tuning works in Optuna

### **Phase 2: Fix 2B (Medium Priority)** 🟡
1. Add `promo_flags` parameter to basket composer methods
2. Apply promotional quantity boost logic
3. Test promotional response
4. Verify boost is tunable

### **Phase 3: Fix 3 (Low Priority)** 🟢
1. Connect promotion_sensitivity config to Phase 2.4
2. Optional - only if time permits

---

## 📝 RELATED FILES

**To Modify:**
- `src/retailsynth/engines/basket_composer.py` (quantity logic)
- `src/retailsynth/config.py` (verify parameters)
- `scripts/tune_parameters_optuna.py` (add promo boost to tuning)

**To Test:**
- `scripts/generate_with_elasticity.py`
- `scripts/tune_parameters_optuna.py`

**To Update:**
- `PRIORITY_1_FIXES_IMPLEMENTED.md` (add Priority 2 & 3)
- `COMPREHENSIVE_PARAMETER_AUDIT.md` (mark as fixed)

---

## 🎯 SUCCESS CRITERIA

### **Fix 2A Success:**
- ✅ Average quantity: 2.2-2.6 (target: 2.47)
- ✅ `quantity_mean` tuning affects output
- ✅ `quantity_std` tuning affects output
- ✅ Quantity distribution matches target

### **Fix 2B Success:**
- ✅ Promo week quantity > non-promo week quantity
- ✅ Boost ratio: 1.3-1.6x
- ✅ `promotion_quantity_boost` tuning affects output
- ✅ Revenue increases during promo weeks

---

**Diagnostic Complete:** November 12, 2025, 1:45pm IST  
**Ready for Implementation:** YES ✅  
**Estimated Time:** 30-45 minutes  
**Expected Score Improvement:** +0.05 to +0.10 (from 0.54 to 0.59-0.64)
