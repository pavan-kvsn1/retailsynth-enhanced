# Phase 2.5 Integration Plan

## 🎯 Goal
Integrate customer-specific promotional response (Phase 2.5) into the utility calculation flow

---

## 📍 Current State

### **Where Promotions Are Used:**
`utility_engine.py` line 44-45:
```python
# Current: Simple binary flag
promo_utility = jnp.outer(beta_promo, promo_flags.astype(jnp.float32))
```

This is **too simple** - it treats all customers the same!

---

## 🔄 Required Changes

### **Option A: Pre-Compute Promotional Boosts (RECOMMENDED)**

**Advantages:**
- ✅ Keeps GPU-accelerated utility engine fast
- ✅ Uses Phase 2.5 sophisticated response model
- ✅ Clean separation of concerns

**Implementation:**

1. **In `transaction_generator.py` (before utility calculation):**

```python
# NEW: Calculate customer-specific promotional boosts
from retailsynth.engines.promo_response import PromoResponseCalculator

promo_calc = PromoResponseCalculator()

# For each visiting customer and promoted product:
promo_boosts = np.zeros((n_visiting_customers, n_products))

for i, customer_idx in enumerate(visiting_indices):
    customer_hetero_params = customers_df.loc[customer_idx, 'hetero_params']
    
    for j, product_id in enumerate(product_ids):
        if product_is_promoted[j]:
            # Get promotion details
            discount_depth = promo_context.promo_depths.get(product_id, 0.0)
            display_type = promo_context.display_types.get(product_id, 'none')
            advertising_type = get_advertising_type(product_id, promo_context)
            marketing_signal = promo_context.marketing_signal_strength
            
            # Calculate individual response
            response = promo_calc.calculate_promo_response(
                customer_params=customer_hetero_params,
                base_utility=0.0,  # Will be added later
                discount_depth=discount_depth,
                marketing_signal=marketing_signal,
                display_type=display_type,
                advertising_type=advertising_type
            )
            
            promo_boosts[i, j] = response.promo_boost

# Pass to utility engine
all_utilities = self.utility_engine.compute_all_utilities_gpu(
    ...,
    promo_boosts_jax=jnp.array(promo_boosts)  # NEW parameter
)
```

2. **In `utility_engine.py` (modify `compute_all_utilities_gpu`):**

```python
@partial(jit, static_argnums=(0,))
def compute_all_utilities_gpu(self,
                              prices: jnp.ndarray,
                              beta_price: jnp.ndarray,
                              brand_prefs: jnp.ndarray,
                              beta_brand: jnp.ndarray,
                              promo_boosts: jnp.ndarray,  # NEW: Pre-computed boosts
                              role_prefs: jnp.ndarray,
                              beta_role: jnp.ndarray,
                              random_key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Compute utilities with Phase 2.5 promotional response
    """
    # Price utility
    log_prices = jnp.log(prices + 1e-6)
    price_utility = jnp.outer(beta_price, log_prices)
    
    # Brand utility
    brand_utility = beta_brand[:, None] * brand_prefs
    
    # Phase 2.5: Customer-specific promotional boost (pre-computed)
    promo_utility = promo_boosts  # Already customer × product matrix
    
    # Role utility
    role_utility = beta_role[:, None] * role_prefs
    
    # Total utility
    base_utility = -1.0
    total_utility = base_utility + price_utility + brand_utility + promo_utility + role_utility
    
    # Add Gumbel noise
    noise = jax.random.gumbel(random_key, shape=total_utility.shape) * 0.6
    
    return total_utility + noise
```

---

### **Option B: Vectorized Promotional Response (ADVANCED)**

Create a JAX-compatible version of promotional response for GPU acceleration.

**Challenges:**
- Complex logic in PromoResponseCalculator
- Non-linear functions may not JIT well
- More development time

**Defer to later if needed**

---

## 📋 Integration Steps

### **Step 1: Add Promotional Context to Transaction Generator**

Store promotional context from PromotionalEngine:

```python
# In main_generator.py
promo_context = self.promotional_engine.generate_week_promotions(week_number)

# Pass to transaction generator
transactions = transaction_gen.generate_week_transactions(
    ...,
    promo_context=promo_context  # NEW
)
```

### **Step 2: Pre-Compute Promotional Boosts**

In `transaction_generator.py`, add new method:

```python
def _compute_promotional_boosts(
    self,
    visiting_indices: np.ndarray,
    promo_context: StorePromoContext,
    customers_df: pd.DataFrame
) -> np.ndarray:
    """
    Pre-compute Phase 2.5 promotional response boosts
    
    Returns:
        np.ndarray: (n_visiting_customers, n_products) promotional boosts
    """
    # Implementation from above
```

### **Step 3: Modify Utility Engine**

Update `compute_all_utilities_gpu` signature and logic.

### **Step 4: Update Call Sites**

Fix all places that call `compute_all_utilities_gpu`.

---

## 🧪 Testing

1. **Unit Test:** Verify promotional boosts are calculated correctly
2. **Integration Test:** Check boosts are passed to utility engine
3. **End-to-End Test:** Verify customer purchases respond to promotions
4. **Validation:** Compare with/without Phase 2.5

---

## ⚠️ Important Notes

### **Performance Consideration:**
Pre-computing promotional boosts for all customers × products could be expensive.

**Optimization:**
Only compute for:
- Visiting customers (already filtered)
- Promoted products only (sparse matrix)

### **Backward Compatibility:**
Keep simple promo flag as fallback for non-Phase 2.5 mode.

---

## 🚀 Implementation Priority

1. **HIGH:** Pre-compute promotional boosts in transaction generator
2. **HIGH:** Modify utility engine to accept pre-computed boosts
3. **HIGH:** Update main generator to pass promo context
4. **MEDIUM:** Add integration tests
5. **LOW:** Optimize for sparse promotional matrices

---

## 📊 Expected Impact

### **Before (Current):**
```python
promo_utility = beta_promo × promo_flag
# Same for all customers!
```

### **After (Phase 2.5):**
```python
promo_boost[customer_i, product_j] = f(
    customer_hetero_params,
    discount_depth,
    marketing_signal,
    display_type,
    advertising_type
)
# Unique for each customer!
```

### **Result:**
- ✅ Individual promotional response
- ✅ Customers respond differently to same promotion
- ✅ Realistic discount sensitivity curves
- ✅ Display and advertising effects
- ✅ Marketing signal amplification

---

**Status:** 📋 Ready to implement
**Estimated effort:** 2-3 hours
**Risk:** Low (pre-computation keeps GPU engine fast)
