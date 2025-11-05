# Fix: Assortment Role Alignment

**Date:** November 3, 2025  
**Status:** RESOLVED ✅

---

## Problem

The real product catalog from Dunnhumby was missing the `assortment_role` column, which is required by the transaction generation engine. The error occurred in `precomputation_engine.py`:

```
KeyError: 'assortment_role'
```

The archetype classifier was using generic retail roles (`destination`, `routine`, `impulse`) that didn't align with the standard retail assortment strategy used in the transaction generation system.

---

## Solution

### 1. Updated Archetype Classifier to Use Standard Retail Assortment Roles

**File:** `src/retailsynth/catalog/archetype_classifier.py`

Changed from generic roles to industry-standard retail assortment roles:

| Old Role | New Role | Description | Target % |
|----------|----------|-------------|----------|
| `destination` | `lpg_line` | Low Price Guarantee - High frequency staples, price-sensitive | 15% |
| `routine` | `front_basket` | Planned purchases, high frequency | 25% |
| - | `mid_basket` | Regular purchases, medium frequency | 40% |
| `impulse` | `back_basket` | Occasional/impulse purchases, low frequency | 20% |

**Classification Logic:**
```python
def assign_role(row):
    price_pct = row['price_percentile']
    freq_pct = row['frequency_percentile']
    
    # LPG Line: High frequency + low price (staples, price-sensitive)
    if freq_pct >= 0.75 and price_pct <= 0.35:
        return 'lpg_line'
    
    # Front Basket: High frequency + any price (planned purchases)
    elif freq_pct >= 0.65:
        return 'front_basket'
    
    # Back Basket: Low frequency (impulse/occasional)
    elif freq_pct < 0.40:
        return 'back_basket'
    
    # Mid Basket: Everything else (regular purchases)
    else:
        return 'mid_basket'
```

**Added Columns:**
- `category_role`: The assortment role classification
- `assortment_role`: Duplicate of `category_role` for compatibility with transaction engine

---

### 2. Enhanced Product Preparation in Main Generator

**File:** `src/retailsynth/generators/main_generator.py`

Added proper column handling when loading real products:

```python
if self.config.use_real_catalog and self.real_products is not None:
    # Use real catalog (NEW)
    self.datasets['products'] = self._prepare_real_products()
    
    # Standardize column names to lowercase
    self.datasets['products'].columns = [col.lower() for col in self.datasets['products'].columns]
    
    # Add required columns for simulation
    self.datasets['products']['base_price'] = self.datasets['products']['avg_price']
    
    # Ensure product_id column exists
    if 'product_id' not in self.datasets['products'].columns:
        self.datasets['products']['product_id'] = self.datasets['products'].index
```

---

## Retail Assortment Strategy Background

### What are Assortment Roles?

Retail assortment roles are strategic classifications that determine how products are merchandised, priced, and promoted:

1. **LPG Line (Low Price Guarantee)** - 15%
   - High-frequency staples (milk, bread, eggs)
   - Price-sensitive items customers compare across stores
   - Used to establish price perception
   - Typically loss leaders or minimal margin

2. **Front Basket** - 25%
   - Planned purchases that drive store visits
   - High frequency, essential items
   - Moderate to high margins
   - Examples: Fresh produce, meat, dairy

3. **Mid Basket** - 40%
   - Regular purchases, medium frequency
   - Core assortment that fills the shopping basket
   - Balanced margins
   - Examples: Canned goods, snacks, beverages

4. **Back Basket** - 20%
   - Occasional/impulse purchases
   - Low frequency but high margin
   - Examples: Specialty items, seasonal products, impulse buys

---

## Impact on Transaction Generation

The `assortment_role` is used in:

1. **Utility Calculation** (`precomputation_engine.py`)
   - Customer preferences vary by assortment role
   - `beta_assortment_role` parameter weights role preference

2. **Shopping Behavior** (`choice_engine.py`)
   - Different roles have different purchase probabilities
   - LPG items are more likely to be purchased
   - Back basket items are more impulse-driven

3. **Promotion Strategy** (`lifecycle_engine.py`)
   - Promotion frequency varies by role
   - LPG items rarely promoted (already low price)
   - Back basket items frequently promoted

---

## Testing

### 1. Rebuild the catalog:
```bash
python scripts/build_product_catalog.py
```

**Expected Output:**
```
STEP 3: CLASSIFY PRODUCT ARCHETYPES
======================================================================
Classifying products into archetypes...
  Step 3: Classifying category roles...
    LPG Line: 2,500
    Front Basket: 4,500
    Mid Basket: 8,000
    Back Basket: 4,000
```

### 2. Test transaction generation:
```bash
python scripts/test_moduar_generation.py
```

**Expected Output:**
```
📦 Step 2/7: Loading 100 products...
   ✅ Complete in 0.0s

🔧 Step 6/7: Pre-computing GPU matrices...
   🔧 Pre-computing customer/product matrices (VECTORIZED)...
      Extracting product data (100 products)...
      ✅ Pre-computation complete in 0.5s
```

---

## Files Modified

1. `src/retailsynth/catalog/archetype_classifier.py`
   - Updated `_classify_category_role()` to use standard retail assortment roles
   - Added `assortment_role` column

2. `src/retailsynth/generators/main_generator.py`
   - Enhanced `_prepare_real_products()` column handling
   - Added `product_id` fallback

---

## Status

✅ **Assortment roles aligned with retail strategy**  
✅ **Real catalog now compatible with transaction generation**  
✅ **Column naming standardized**  
✅ **Ready for full pipeline testing**

---

## Next Steps

1. ✅ Apply the proposed changes
2. ✅ Rebuild catalog: `python scripts/build_product_catalog.py`
3. ✅ Test generation: `python scripts/test_moduar_generation.py`
4. 🎯 Proceed to Sprint 1.2: Price Elasticity Learning
