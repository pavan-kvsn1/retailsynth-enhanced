# 🔴 CRITICAL: Visit Frequency Diagnostic Report
**Date:** November 12, 2025  
**Issue:** Visit frequency remains high (~3-5 visits/week) despite parameter range reductions  
**Target:** ~0.5-1.0 visits/week (realistic retail behavior)

---

## 🔍 Executive Summary

**ROOT CAUSE IDENTIFIED:** The transaction generator is using **LEGACY visit probability code** with hardcoded parameters instead of the calibrated Phase 2 recursive mechanism. This completely bypasses all parameter tuning efforts.

**Impact:** CRITICAL - All visit frequency tuning is ineffective until this is fixed.

---

## 📊 Diagnostic Findings

### **1. 🚨 CRITICAL: Wrong Visit Probability Method Called**

**Location:** `src/retailsynth/generators/transaction_generator.py:94`

```python
# CURRENT (WRONG):
visit_probs = self.utility_engine.compute_store_visit_probabilities_gpu(
    self.precomp.days_since_visit_jax,
    self.precomp.loyalty_levels_jax,
    subkey
)
```

This calls the **LEGACY** method which has:

**Location:** `src/retailsynth/engines/utility_engine.py:175-179`

```python
def compute_store_visit_probabilities_gpu(self, ...):
    """
    LEGACY METHOD: Simple additive formula (pre-Phase 2)
    Kept for backward compatibility. New code should use compute_visit_probabilities_with_sv().
    """
    base_prob = 0.5  # ⚠️ HARDCODED!
    loyalty_boost = loyalty_levels * 0.2
    days_factor = jnp.minimum(0.3, days_since_visit / 30.0)
    visit_probs = base_prob + loyalty_boost + days_factor
    return jnp.clip(visit_probs, 0.0, 1.0)
```

**Problems:**
- ✅ Hardcoded `base_prob = 0.5` (50% base visit probability!)
- ✅ Fixed `loyalty_boost = 0.2` multiplier
- ✅ Ignores `config.base_visit_probability` completely
- ✅ No store value (SV) calculation
- ✅ No marketing signal integration
- ✅ No recursive memory mechanism
- ✅ Probabilities can reach 100% (0.5 + 0.2 + 0.3 = 1.0)

**Result:** Visit probabilities are consistently 70-100%, leading to 3-5 visits/week.

---

### **2. 📉 Phase 2 Code Exists But Is Never Called**

The proper implementation exists in `utility_engine.py:181-238`:

```python
def compute_visit_probabilities_with_sv(self,
                                       product_utilities,
                                       product_categories,
                                       promo_context,
                                       n_categories):
    """
    Compute visit probabilities using Bain's recursive mechanism with store value (Phase 2)
    
    This implements the full Bain model:
    1. Calculate store inclusive value (SV) from product utilities
    2. Get previous state (prev_visit_probs, prev_store_values, visited_last_period)
    3. Compute visit utilities from SV + marketing signals
    4. Calculate recursive probabilities with memory
    """
```

This method:
- ✅ Uses `config.base_visit_probability`
- ✅ Implements Store Value (SV) calculation
- ✅ Integrates marketing signals properly
- ✅ Has recursive memory with dampening
- ✅ Includes safety caps to prevent saturation

**But it's NEVER CALLED anywhere in the codebase!**

---

### **3. 🔧 Parameter Tuning Has Been Ineffective**

**Tuning Script Changes (11/12/2025):**
```python
# base_visit_probability: Reduced from 0.15-0.45 to 0.15-0.3
config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.15, 0.3)

# loyalty_weight: Reduced from 0.1-0.6 to 0.1-0.4
config.loyalty_weight = trial.suggest_float('loyalty_weight', 0.1, 0.4)

# marketing_visit_weight: Reduced from 0.05-0.3 to 0.05-0.2
config.marketing_visit_weight = trial.suggest_float('marketing_visit_weight', 0.05, 0.2)

# visit_memory_weight: Already minimal at 0.01-0.15
config.visit_memory_weight = trial.suggest_float('visit_memory_weight', 0.01, 0.15)
```

**Impact:** ZERO - These parameters are never used because the legacy code path is hardcoded.

---

### **4. 🔁 Feedback Loop Analysis**

**Current System (Legacy Path):**
```
Hardcoded base_prob (0.5)
   ↓
+ Loyalty boost (up to 0.2)
   ↓
+ Days since visit (up to 0.3)
   ↓
= Visit probability (0.7-1.0)
   ↓
Customer visits frequently (3-5 times/week)
   ↓
More loyalty built
   ↓
POSITIVE FEEDBACK LOOP → Saturation
```

**Intended System (Phase 2 - Not Active):**
```
config.base_visit_probability (tunable: 0.15-0.3)
   ↓
+ Store Value calculation
   ↓
+ Marketing signal (weighted: 0.05-0.2)
   ↓
Recursive probability with memory decay
   ↓
Dampening (×0.3) + safety cap (0.85 max)
   ↓
Realistic visit probability (0.2-0.5)
   ↓
= 0.5-1.5 visits/week (REALISTIC)
```

---

## 🎯 Impact Assessment

### **Metrics Affected:**

1. **Visit Frequency** 🔴 CRITICAL
   - Current: 3-5 visits/week (300-500%)
   - Target: 0.5-1.0 visits/week (100%)
   - Error: **+400%**

2. **Transaction Volume** 🔴 HIGH
   - More visits → More transactions
   - Inflates dataset size artificially
   - Biases all downstream metrics

3. **Revenue** 🟡 MEDIUM
   - Indirectly inflated by visit count
   - More visits → More purchases → Higher total revenue
   - But the root cause is visit frequency, not basket size

4. **Model Realism** 🔴 CRITICAL
   - Current behavior: Customers shop 3-5 times per week
   - Real behavior: Customers shop 0.5-1.0 times per week
   - **Model is generating unrealistic behavior patterns**

---

## 🔧 Recommended Fixes

### **Priority 1: Switch to Phase 2 Visit Probability (IMMEDIATE)**

**File:** `src/retailsynth/generators/transaction_generator.py`

**Change Line 94-98 from:**
```python
# REMOVE THIS:
visit_probs = self.utility_engine.compute_store_visit_probabilities_gpu(
    self.precomp.days_since_visit_jax,
    self.precomp.loyalty_levels_jax,
    subkey
)
```

**To:**
```python
# ADD THIS:
# Step 1: Compute utilities first (needed for SV calculation)
self.rng_key, subkey = jax.random.split(self.rng_key)
all_utilities = self.utility_engine.compute_all_utilities_gpu(
    current_prices_jax,
    self.precomp.beta_price_jax,
    self.precomp.brand_pref_matrix_jax,
    self.precomp.beta_brand_jax,
    promo_flags_jax,
    self.precomp.beta_promo_jax,
    self.precomp.role_pref_matrix_jax,
    self.precomp.beta_role_jax,
    subkey
)

# Step 2: Use Phase 2 visit probability with Store Value
if self.products is not None:
    product_categories = jnp.array(self.products['category_id'].values)
    n_categories = len(self.products['category_id'].unique())
    
    visit_probs, store_values = self.utility_engine.compute_visit_probabilities_with_sv(
        all_utilities,
        product_categories,
        self.store_promo_contexts,
        n_categories
    )
else:
    # Fallback to legacy if products not available
    visit_probs = self.utility_engine.compute_store_visit_probabilities_gpu(
        self.precomp.days_since_visit_jax,
        self.precomp.loyalty_levels_jax,
        subkey
    )
```

**Impact:** Visit frequency should immediately drop to realistic levels (0.5-1.5 visits/week).

---

### **Priority 2: Verify Parameter Flow (VALIDATION)**

After Priority 1 fix, verify that:

1. ✅ `config.base_visit_probability` is actually used
2. ✅ `config.marketing_visit_weight` affects behavior
3. ✅ `config.visit_memory_weight` creates temporal dependence
4. ✅ Dampening and safety caps are active

**Test:**
```python
# Run with extreme parameters to verify they're being used
config.base_visit_probability = 0.05  # Very low
config.marketing_visit_weight = 0.01  # Minimal marketing

# Expected: Visit frequency should drop dramatically
# If it doesn't change, parameters still not flowing through
```

---

### **Priority 3: Remove or Deprecate Legacy Code (CLEANUP)**

**File:** `src/retailsynth/engines/utility_engine.py`

Either:
1. **Remove** `compute_store_visit_probabilities_gpu()` entirely, OR
2. **Redirect** it to call the Phase 2 method, OR
3. **Add warning** that it's deprecated and shouldn't be used

```python
@deprecated("Use compute_visit_probabilities_with_sv() instead")
def compute_store_visit_probabilities_gpu(self, ...):
    raise NotImplementedError(
        "This legacy method is deprecated. "
        "Use compute_visit_probabilities_with_sv() for Phase 2 behavior."
    )
```

---

### **Priority 4: Add Integration Test (PREVENTION)**

Create test to ensure calibrated parameters affect output:

```python
def test_base_visit_probability_affects_frequency():
    """Verify that config.base_visit_probability actually changes visit frequency"""
    
    # Test 1: Low base probability
    config_low = EnhancedRetailConfig(base_visit_probability=0.1)
    synth_low = generate(config_low)
    freq_low = calculate_visit_frequency(synth_low)
    
    # Test 2: High base probability
    config_high = EnhancedRetailConfig(base_visit_probability=0.4)
    synth_high = generate(config_high)
    freq_high = calculate_visit_frequency(synth_high)
    
    # Assert: High config should produce higher frequency
    assert freq_high > freq_low * 1.5, "base_visit_probability not affecting output!"
```

---

## 📋 Implementation Checklist

- [ ] **IMMEDIATE:** Fix transaction_generator.py to use Phase 2 visit probability
- [ ] **VALIDATE:** Run Optuna tuning and verify visit frequency responds to parameters
- [ ] **TEST:** Confirm `base_visit_probability = 0.15` produces ~0.5 visits/week
- [ ] **CLEANUP:** Deprecate or remove legacy visit probability method
- [ ] **DOCUMENT:** Update code comments to indicate Phase 2 is the default
- [ ] **MONITOR:** Add logging to track which visit probability method is active

---

## 🎓 Lessons Learned

1. **Code Path Verification:** Always verify that tuned parameters are actually used in the execution path
2. **Legacy Code Risk:** Keeping old code paths can cause silent failures when new code isn't integrated
3. **Integration Testing:** Need tests that verify parameter → output causality
4. **Documentation:** Code comments should indicate which methods are current vs deprecated

---

## 📊 Expected Results After Fix

**Before (Current - Legacy):**
- Visit frequency: 3-5 visits/week
- Visit probability: 0.7-1.0 (saturated)
- Parameter sensitivity: NONE

**After (Phase 2):**
- Visit frequency: 0.5-1.5 visits/week ✅
- Visit probability: 0.2-0.5 (realistic) ✅
- Parameter sensitivity: HIGH ✅

**Optuna Tuning:**
- Currently: All trials score 0.0 (nothing works)
- After fix: Should see score > 0.3 with proper parameters
- Optimization will actually converge

---

## 🚀 Next Steps

1. **Implement Priority 1 fix** (switch to Phase 2 method)
2. **Run quick test** with 100 customers, 10 weeks
3. **Check visit frequency** - should be ~0.5-1.0
4. **If still high:** Debug Phase 2 implementation
5. **If fixed:** Re-run Optuna tuning with full parameter space
6. **Monitor:** Track visit frequency distribution in logs

---

**Report Compiled By:** Cascade AI  
**Status:** READY FOR IMPLEMENTATION  
**Priority:** 🔴 CRITICAL - Blocks all calibration work
