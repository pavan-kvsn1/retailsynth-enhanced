# Parameter Impact Matrix

**Question**: Which parameters affect which metrics? How much overlap is there?

---

## 📊 **Impact Matrix**

| Parameter | Basket Size | Revenue | Visit Freq | Quantity | Primary Effect |
|-----------|-------------|---------|------------|----------|----------------|
| **Visit Behavior** |
| `base_visit_probability` | ❌ | ❌ | ✅✅✅ | ❌ | Visit freq only |
| **Basket Size** |
| `basket_size_lambda` | ✅✅✅ | ✅✅ | ❌ | ❌ | Basket size, revenue |
| **Quantity** |
| `quantity_mean` | ❌ | ✅ | ❌ | ✅✅✅ | Quantity, revenue |
| `quantity_std` | ❌ | ✅ | ❌ | ✅✅ | Quantity variance |
| `quantity_max` | ❌ | ✅ | ❌ | ✅✅ | Quantity cap |
| **Temporal Dynamics** |
| `inventory_depletion_rate` | ✅ | ✅ | ✅ | ❌ | All (repeat purchases) |
| `replenishment_threshold` | ✅ | ✅ | ✅ | ❌ | All (repeat purchases) |
| **Basket Composition** |
| `complement_probability` | ✅✅ | ✅✅ | ❌ | ❌ | Basket size, revenue |
| `substitute_avoidance` | ✅ | ✅ | ❌ | ❌ | Basket diversity |
| `category_diversity_weight` | ✅ | ✅ | ❌ | ❌ | Basket diversity |
| **Purchase History** |
| `loyalty_weight` | ✅ | ✅ | ✅ | ❌ | All (repeat behavior) |
| `habit_weight` | ✅ | ✅ | ✅ | ❌ | All (repeat behavior) |
| `inventory_weight` | ✅ | ✅ | ✅ | ❌ | All (repeat behavior) |
| `variety_weight` | ✅ | ✅ | ❌ | ❌ | Basket diversity |
| `price_memory_weight` | ✅ | ✅✅ | ❌ | ❌ | Revenue (price sensitivity) |

**Legend**:
- ✅✅✅ = Primary effect (90%+ impact)
- ✅✅ = Strong effect (50-90% impact)
- ✅ = Moderate effect (10-50% impact)
- ❌ = Minimal/no effect (<10% impact)

---

## 🎯 **Parameter Grouping by Primary Metric**

### **Group 1: Visit Frequency Only (1 param)**
```python
# These ONLY affect visit frequency
base_visit_probability  # ✅✅✅ Visit freq
```

**Overlap**: **NONE** - This is the cleanest parameter!

---

### **Group 2: Basket Size Primary (4 params)**
```python
# These primarily affect basket size
basket_size_lambda           # ✅✅✅ Basket, ✅✅ Revenue
complement_probability       # ✅✅ Basket, ✅✅ Revenue
substitute_avoidance         # ✅ Basket, ✅ Revenue
category_diversity_weight    # ✅ Basket, ✅ Revenue
```

**Overlap**: **HIGH with Revenue** (70-80%)
- All basket size params also affect revenue (bigger baskets = more revenue)
- But basket size is the PRIMARY effect

---

### **Group 3: Quantity Primary (3 params)**
```python
# These primarily affect quantity distribution
quantity_mean    # ✅✅✅ Quantity, ✅ Revenue
quantity_std     # ✅✅ Quantity, ✅ Revenue
quantity_max     # ✅✅ Quantity, ✅ Revenue
```

**Overlap**: **MODERATE with Revenue** (30-40%)
- Higher quantities → more revenue
- But quantity distribution is the PRIMARY effect

---

### **Group 4: Multi-Metric (7 params)**
```python
# These affect MULTIPLE metrics significantly
inventory_depletion_rate   # ✅ Basket, ✅ Revenue, ✅ Visit freq
replenishment_threshold    # ✅ Basket, ✅ Revenue, ✅ Visit freq
loyalty_weight             # ✅ Basket, ✅ Revenue, ✅ Visit freq
habit_weight               # ✅ Basket, ✅ Revenue, ✅ Visit freq
inventory_weight           # ✅ Basket, ✅ Revenue, ✅ Visit freq
variety_weight             # ✅ Basket, ✅ Revenue
price_memory_weight        # ✅ Basket, ✅✅ Revenue
```

**Overlap**: **VERY HIGH** (affects 2-3 metrics each)
- These are the "coupling" parameters
- Hardest to tune because they create trade-offs

---

## 📊 **Overlap Analysis**

### **Metric Overlap Percentages**

| Metric Pair | Shared Parameters | Overlap % | Conflict Risk |
|-------------|-------------------|-----------|---------------|
| **Basket ↔ Revenue** | 11/15 params | **73%** | 🔴 HIGH |
| **Basket ↔ Visit Freq** | 5/15 params | **33%** | 🟡 MEDIUM |
| **Basket ↔ Quantity** | 0/15 params | **0%** | 🟢 NONE |
| **Revenue ↔ Visit Freq** | 5/15 params | **33%** | 🟡 MEDIUM |
| **Revenue ↔ Quantity** | 3/15 params | **20%** | 🟢 LOW |
| **Visit Freq ↔ Quantity** | 0/15 params | **0%** | 🟢 NONE |

---

## 🎯 **Optimization Strategy by Overlap**

### **Strategy 1: Sequential Tuning (Recommended)**

Tune in order of independence:

#### **Stage 1: Independent Parameters (4 params, 20 trials)**
```python
# No conflicts - tune first
base_visit_probability   # Visit freq only
quantity_mean           # Quantity only
quantity_std            # Quantity only
quantity_max            # Quantity only
```
**Time**: 40 minutes  
**Conflicts**: NONE

#### **Stage 2: Basket-Revenue Parameters (4 params, 20 trials)**
```python
# High overlap but same direction (bigger basket = more revenue)
basket_size_lambda
complement_probability
substitute_avoidance
category_diversity_weight
```
**Time**: 40 minutes  
**Conflicts**: LOW (aligned objectives)

#### **Stage 3: Multi-Metric Parameters (7 params, 30 trials)**
```python
# Affects multiple metrics - tune last with Stage 1 & 2 fixed
inventory_depletion_rate
replenishment_threshold
loyalty_weight
habit_weight
inventory_weight
variety_weight
price_memory_weight
```
**Time**: 60 minutes  
**Conflicts**: HIGH (requires balancing)

**Total**: 140 minutes (~2.3 hours)

---

### **Strategy 2: Parallel Tuning by Metric**

Run 4 separate optimizations:

#### **Optimization 1: Visit Frequency (1 param, 15 trials)**
```bash
python scripts/tune_parameters_optuna.py \
    --objective visit_frequency \
    --n-trials 15
```
**Tunes**: `base_visit_probability`

#### **Optimization 2: Quantity (3 params, 20 trials)**
```bash
python scripts/tune_parameters_optuna.py \
    --objective quantity \
    --n-trials 20
```
**Tunes**: `quantity_mean`, `quantity_std`, `quantity_max`

#### **Optimization 3: Basket Size (4 params, 25 trials)**
```bash
python scripts/tune_parameters_optuna.py \
    --objective basket_size \
    --n-trials 25
```
**Tunes**: `basket_size_lambda`, `complement_probability`, `substitute_avoidance`, `category_diversity_weight`

#### **Optimization 4: Combined (7 params, 30 trials)**
```bash
python scripts/tune_parameters_optuna.py \
    --objective combined \
    --n-trials 30
```
**Tunes**: Multi-metric parameters with Opt 1-3 results fixed

**Total**: 90 trials × 2 min = 180 minutes (~3 hours)

---

## ⚠️ **Conflict Analysis**

### **High Conflict: Basket Size ↔ Revenue**

**Problem**: Parameters that increase basket size also increase revenue

**Example**:
```python
basket_size_lambda = 15.0  # Large baskets
→ Basket size: ✅ Matches real (5.2 items)
→ Revenue: ❌ Too high ($80 vs $45 real)
```

**Solution**: Use combined objective (balances both)

---

### **Medium Conflict: Visit Freq ↔ Basket/Revenue**

**Problem**: Purchase history params affect all three

**Example**:
```python
loyalty_weight = 0.8  # High loyalty
→ Visit freq: ✅ Customers visit more often
→ Basket size: ❌ Smaller baskets (buying same items)
→ Revenue: ❌ Lower revenue per visit
```

**Solution**: Tune visit freq first, then basket/revenue

---

### **No Conflict: Quantity ↔ Visit Freq**

**Problem**: NONE - completely independent!

**Example**:
```python
quantity_mean = 2.0
base_visit_probability = 0.35
→ Both can be optimized independently ✅
```

**Solution**: Tune in parallel

---

## 🎯 **Recommended Approach**

### **Option A: Fast & Simple (2 hours)**
```bash
# Tune all 15 params together with combined objective
python scripts/tune_parameters_optuna.py \
    --objective combined \
    --tier 1 \
    --n-trials 50
```

**Pros**: Simple, one command  
**Cons**: May not find global optimum due to conflicts

---

### **Option B: Sequential & Optimal (2.5 hours)**
```bash
# Stage 1: Independent params (no conflicts)
python scripts/tune_parameters_optuna.py \
    --objective combined \
    --n-trials 20 \
    --params "base_visit_prob,quantity_mean,quantity_std,quantity_max"

# Stage 2: Basket-revenue params (aligned)
python scripts/tune_parameters_optuna.py \
    --objective combined \
    --n-trials 20 \
    --params "basket_size_lambda,complement_probability,substitute_avoidance,category_diversity_weight"

# Stage 3: Multi-metric params (conflicts)
python scripts/tune_parameters_optuna.py \
    --objective combined \
    --n-trials 30 \
    --params "inventory_depletion_rate,replenishment_threshold,loyalty_weight,habit_weight,inventory_weight,variety_weight,price_memory_weight"
```

**Pros**: Better optimization, fewer conflicts  
**Cons**: More complex, 3 commands

---

## 📊 **Summary**

| Metric | Dedicated Params | Shared Params | Independence |
|--------|------------------|---------------|--------------|
| **Visit Frequency** | 1 | 5 | 🟢 HIGH (1 dedicated) |
| **Quantity** | 3 | 0 | 🟢 HIGH (3 dedicated) |
| **Basket Size** | 0 | 11 | 🔴 LOW (all shared) |
| **Revenue** | 0 | 14 | 🔴 LOW (all shared) |

**Key Insight**: 
- **Visit freq & Quantity** are easiest to tune (dedicated params)
- **Basket & Revenue** are hardest (all params shared)
- **73% overlap** between basket and revenue (biggest conflict)

---

**Bottom Line**: Use **combined objective** for simplicity, or **sequential tuning** for optimality!
