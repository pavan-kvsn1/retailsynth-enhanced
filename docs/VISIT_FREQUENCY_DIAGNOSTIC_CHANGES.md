# Visit Frequency Diagnostic Changes

## 🎯 Problem: High Visit Probability (90%+) but Low Visit Frequency

**Date**: 2025-11-11  
**Status**: Diagnostic logging added + parameter ranges adjusted

---

## Changes Made

### **1. Visit Probability Percentile Logging** ✅

**File**: `src/retailsynth/generators/transaction_generator.py`  
**Location**: After line 205

**What it shows:**
```
Visit prob percentiles:
  P10: 0.156    ← 10% of customers have prob < 15.6%
  P50 (median): 0.423  ← Half have prob < 42.3%
  P90: 0.912    ← 10% have prob > 91.2%
```

**Purpose**: Reveals if "90% visit probability" is just a few customers or everyone.

---

### **2. Expected vs Actual Visits Logging** ✅

**File**: `src/retailsynth/generators/transaction_generator.py`  
**Location**: After line 224

**What it shows:**
```
Expected visits: 450.3 (45.0%)
Actual visits: 447 (44.7%)
```

**Purpose**: Confirms if sampling is working correctly (actual ≈ expected).

---

### **3. Visit Frequency Debug Logging** ✅

**File**: `scripts/tune_parameters_optuna.py`  
**Location**: After line 360 (every 5 trials)

**What it shows:**
```
Visit Frequency Debug:
  Customer 1: 45 visits / 52.0 weeks = 0.87
  Customer 2: 48 visits / 52.0 weeks = 0.92
  Customer 3: 3 visits / 8.0 weeks = 0.38   ← Churned early!
  Customer 4: 2 visits / 45.0 weeks = 0.04  ← Sporadic!

Visit Frequency Distribution:
  Mean: 0.285
  Median: 0.312
  P10: 0.045   ← 10% have very low frequency
  P90: 0.654   ← 10% have high frequency
```

**Purpose**: Identifies if issue is churn, heterogeneity, or calculation error.

---

### **4. Increased Base Visit Probability Range** ✅

**File**: `scripts/tune_parameters_optuna.py`  
**Location**: Line 168

**Change:**
```python
# BEFORE
config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.30, 0.75)

# AFTER
config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.45, 0.75)
```

**Reason**: Need higher baseline to achieve good frequency.

---

### **5. Increased Marketing Visit Weight** ✅

**File**: `scripts/tune_parameters_optuna.py`  
**Location**: Line 228

**Change:**
```python
# BEFORE
config.marketing_visit_weight = trial.suggest_float('marketing_visit_weight', 0.5, 2.0)

# AFTER
config.marketing_visit_weight = trial.suggest_float('marketing_visit_weight', 1.0, 3.0)
```

**Reason**: Stronger promotional effect needed for visit frequency impact.

---

### **6. Decreased Visit Memory Weight** ✅

**File**: `scripts/tune_parameters_optuna.py`  
**Location**: Line 230

**Change:**
```python
# BEFORE
config.visit_memory_weight = trial.suggest_float('visit_memory_weight', 0.0, 0.3)

# AFTER
config.visit_memory_weight = trial.suggest_float('visit_memory_weight', 0.0, 0.15)
```

**Reason**: Less memory = more responsive to current week's promotions.

---

## What to Look For

### **Scenario A: Heterogeneity Problem**

**If you see:**
```
Visit prob percentiles:
  P10: 0.05
  P50: 0.15
  P90: 0.95   ← Only 10% have high probability!
```

**Diagnosis**: Most customers have low probability, a few have very high.  
**Fix**: Adjust heterogeneity parameters or increase base probability.

---

### **Scenario B: Churn Problem**

**If you see:**
```
Customer 1: 45 visits / 52.0 weeks = 0.87  ✅
Customer 2: 3 visits / 8.0 weeks = 0.38    ⚠️ Churned!
Customer 3: 2 visits / 45.0 weeks = 0.04   ❌ Sporadic!
```

**Diagnosis**: Customers churning early or visiting sporadically.  
**Fix**: Reduce churn rate, increase loyalty.

---

### **Scenario C: Sampling Problem**

**If you see:**
```
Expected visits: 900.0 (90.0%)
Actual visits: 450 (45.0%)   ← Half of expected!
```

**Diagnosis**: Sampling is broken or probability not being used.  
**Fix**: Check if `enable_sv` is True, verify sampling logic.

---

### **Scenario D: Calculation Problem**

**If you see:**
```
Customer 1: 48 visits / 8.0 weeks = 6.00   ← Wrong denominator!
```

**Diagnosis**: Active weeks calculation is wrong.  
**Fix**: Use actual week count, not date range.

---

## Expected Impact

### **Before Changes:**

```
Base visit prob: 0.30-0.75 (often ~0.40)
Marketing weight: 0.5-2.0 (often ~0.8)
Memory weight: 0.0-0.3 (often ~0.2)

Result:
- Visit prob: 0.30-0.50
- Visit frequency: 0.15-0.25
```

### **After Changes:**

```
Base visit prob: 0.45-0.75 (often ~0.60)
Marketing weight: 1.0-3.0 (often ~1.8)
Memory weight: 0.0-0.15 (often ~0.05)

Result:
- Visit prob: 0.50-0.75
- Visit frequency: 0.35-0.55
```

---

## How to Use

### **Step 1: Run Simulation**

```bash
python scripts/run_simulation.py --weeks 20 --customers 1000
```

Watch for the new logging output every 10 weeks.

---

### **Step 2: Run Optuna**

```bash
python scripts/tune_parameters_optuna.py \
    --objective visit_frequency \
    --n-trials 50 \
    --tiers 1,2
```

Check the visit frequency debug output every 5 trials.

---

### **Step 3: Diagnose**

Based on the logging, identify which scenario (A, B, C, or D) you're seeing.

---

### **Step 4: Iterate**

If needed, adjust parameter ranges further based on diagnostics.

---

## Files Modified

| File | Changes | Purpose |
|------|---------|---------|
| `transaction_generator.py` | +13 lines | Percentile & visit count logging |
| `tune_parameters_optuna.py` | +17 lines, 3 params | Debug logging + adjusted ranges |

**Total**: 30 lines of diagnostic code + 3 parameter adjustments

---

## Next Steps

1. ✅ **Run with new logging** - See what the actual issue is
2. ⏳ **Analyze output** - Identify scenario A, B, C, or D
3. ⏳ **Adjust accordingly** - Fine-tune based on findings
4. ⏳ **Run Optuna** - Let it find optimal parameters

---

**Status**: ✅ **Diagnostic Changes Complete**  
**Action**: Run simulation and check the logging output!
