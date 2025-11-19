# Phase 2.5 Integration Guide

**Purpose:** Integrate promotional response calculator into transaction generation flow

---

## 🎯 Integration Overview

Phase 2.5 adds **customer-specific promotional response** to the transaction generator. This means:
- ✅ Each customer responds differently to the same promotion
- ✅ Individual parameters (Phase 2.4) drive promotional utility boosts
- ✅ Marketing signals (Phase 2.3) amplify responses
- ✅ Display and advertising effects are customer-specific

---

## 📋 Required Changes

### **1. Transaction Generator** ✅ DONE

**File:** `transaction_generator.py`

**Changes:**
- Added `PromoResponseCalculator` parameter to `__init__`
- Added `store_promo_contexts` parameter to `generate_week_transactions_vectorized()`
- Added `_apply_promotional_response()` method
- Added helper methods:
  - `_get_customer_hetero_params()`
  - `_calculate_discount_depth()`
  - `_get_advertising_type()`

**Key Integration Point:**
```python
# After computing base utilities and history adjustments
# Phase 2.5: Apply promotional response
if self.enable_promo_response:
    all_utilities_np = self._apply_promotional_response(
        all_utilities_np,
        visiting_indices,
        current_prices,
        promo_flags,
        store_promo_contexts,
        week_number
    )
```

---

### **2. Precomputation Engine** ✅ DONE

**File:** `precomputation_engine.py`

**Changes:**
- Added `hetero_params_dict` attribute
- Added `_extract_heterogeneity_params()` method
- Extracts Phase 2.4 parameters from `customers_df['hetero_params']`

**Storage:**
```python
self.hetero_params_dict = {
    customer_id: {
        'customer_id': customer_id,
        'promo_responsiveness_param': ...,
        'price_sensitivity_param': ...,
        'display_sensitivity_param': ...,
        'advertising_receptivity_param': ...,
        # ... 6 more parameters
    },
    ...
}
```

---

### **3. Main Generator** 🔨 NEEDS UPDATE

**File:** `main_generator.py`

**Required Changes:**

#### **3a. Initialize PromoResponseCalculator**

```python
from retailsynth.engines.promo_response import PromoResponseCalculator

# In __init__ or generate() method:
self.promo_response_calc = PromoResponseCalculator()
```

#### **3b. Pass calculator to TransactionGenerator**

```python
transaction_generator = ComprehensiveTransactionGenerator(
    precomp=self.precomp,
    utility_engine=self.utility_engine,
    store_loyalty=self.store_loyalty,
    config=self.config,
    state_manager=self.state_manager,
    history_engine=self.history_engine,
    basket_composer=self.basket_composer,
    promo_response_calc=self.promo_response_calc  # ADD THIS
)
```

#### **3c. Generate store promo contexts**

```python
# For each week, generate store promo contexts
store_promo_contexts = {}
for store_id in self.datasets['stores']['store_id']:
    store_promo_contexts[store_id] = self.promotional_engine.generate_store_promo_context(
        store_id=store_id,
        week_number=week,
        promoted_products=promoted_products_this_week
    )
```

#### **3d. Pass contexts to transaction generator**

```python
transactions, items = transaction_generator.generate_week_transactions_vectorized(
    week_number=week,
    current_prices=week_prices,
    promo_flags=promo_flags,
    week_date=week_date,
    store_promo_contexts=store_promo_contexts  # ADD THIS
)
```

---

## 🔌 Data Flow

```
1. PromotionalEngine generates StorePromoContext (Phase 2.2/2.3)
        ↓
2. MarketingSignalCalculator computes signal strength (Phase 2.3)
        ↓
3. TransactionGenerator receives:
   - Base utilities (from GPU utility engine)
   - Store promo contexts (with marketing signals)
   - Customer hetero params (from Phase 2.4)
        ↓
4. For each visiting customer & promoted product:
   - Get customer's individual parameters
   - Get promotion details (discount, display, ad type)
   - Get marketing signal strength
        ↓
5. PromoResponseCalculator calculates:
   - Discount boost (individual sensitivity)
   - Display boost (individual sensitivity)
   - Advertising boost (individual receptivity)
   - Signal multiplier (amplification)
        ↓
6. Apply promo_boost to utility:
   final_utility = base_utility + promo_boost
        ↓
7. Customer makes purchase decision with boosted utilities
```

---

## 🧪 Testing Integration

### **Test 1: Verify heterogeneity parameters loaded**

```python
# In precomputation engine
print(f"Hetero params loaded: {len(precomp.hetero_params_dict)}")

# Should print: "Hetero params loaded: <n_customers>"
```

### **Test 2: Verify promo response calculator initialized**

```python
# In transaction generator
print(f"Promo response enabled: {transaction_gen.enable_promo_response}")
print(f"Promo calculator: {transaction_gen.promo_response_calc}")

# Should print: "Promo response enabled: True"
```

### **Test 3: Verify store promo contexts passed**

```python
# In transaction generation
print(f"Store promo contexts: {len(store_promo_contexts)}")
print(f"Sample context: {store_promo_contexts[1]}")

# Should print context with marketing_signal_strength
```

### **Test 4: Verify promotional response applied**

```python
# In _apply_promotional_response
print(f"Applying promo response for {len(promoted_product_indices)} promoted products")
print(f"Customer {customer_id} response to product {product_id}:")
print(f"  Base utility: {base_utility:.3f}")
print(f"  Promo boost: {promo_response.promo_boost:.3f}")
print(f"  Final utility: {promo_response.final_utility:.3f}")
```

---

## ⚠️ Potential Issues

### **Issue 1: Missing hetero_params**

**Symptom:** `hetero_params_dict` is empty

**Cause:** Customer generator (Phase 2.4) not integrated

**Solution:** Ensure `customer_generator.py` creates `hetero_params` field

### **Issue 2: Store promo contexts not passed**

**Symptom:** Promotional response skipped

**Cause:** `store_promo_contexts=None` in transaction generation

**Solution:** Generate contexts in main_generator before calling transaction generator

### **Issue 3: Performance degradation**

**Symptom:** Slower transaction generation

**Cause:** Individual promotional response calculation per customer-product pair

**Optimization:** 
- Only apply for promoted products (already implemented)
- Consider vectorizing response calculation for multiple products
- Cache marketing signal per store

---

## 📊 Expected Behavior

### **Before Phase 2.5:**
```python
# Same promotion, same utility boost for all customers
Promo: 20% off
Customer 1: utility += 0.3
Customer 2: utility += 0.3
Customer 3: utility += 0.3
```

### **After Phase 2.5:**
```python
# Same promotion, different response per customer
Promo: 20% off, end_cap display, in_ad_only
Customer 1 (high promo_responsiveness=1.8): utility += 0.52
Customer 2 (low promo_responsiveness=0.7): utility += 0.19
Customer 3 (medium promo_responsiveness=1.2): utility += 0.34
```

---

## ✅ Integration Checklist

- [x] **transaction_generator.py** - Add promotional response calculation
- [x] **precomputation_engine.py** - Store heterogeneity parameters
- [ ] **main_generator.py** - Initialize PromoResponseCalculator
- [ ] **main_generator.py** - Pass calculator to transaction generator
- [ ] **main_generator.py** - Generate store promo contexts per week
- [ ] **main_generator.py** - Pass contexts to transaction generator
- [ ] **Test** - Run quick integration test
- [ ] **Test** - Validate promotional utility boosts
- [ ] **Test** - Verify heterogeneity in responses

---

## 🚀 Next Steps After Integration

1. **Run tests:**
   ```bash
   python scripts/test_phase_2_5_quick.py
   ```

2. **Generate small dataset** - Validate integration with 100 customers, 1 week

3. **Check promotional impact:**
   - Compare utilities before/after promo response
   - Verify different customers respond differently
   - Validate marketing signal amplification

4. **Performance profiling** - Ensure acceptable speed with large datasets

5. **Move to Phase 2.6** - Non-linear utilities (loss aversion, reference prices)

---

## 📝 Summary

**Phase 2.5 integration requires:**
1. ✅ Transaction generator updated (apply promo response)
2. ✅ Precomputation engine updated (store hetero params)
3. 🔨 Main generator updates (3-4 lines of code)

**Result:** Every customer responds uniquely to promotions based on their individual parameters!

**Impact:** More realistic promotional effects, better validation against real data!
