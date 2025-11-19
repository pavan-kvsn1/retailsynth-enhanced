# 🎯 PROPOSAL: Connect Config to Heterogeneity Engine
**Date:** November 12, 2025, 3:10pm IST  
**Status:** 🟡 PROPOSED - Awaiting Decision  
**Complexity:** MEDIUM (2-3 hours)

---

## 📊 MOTIVATION

### **Current Optuna Results:**
```
Before all fixes: 0.0000 (complete failure)
After Priority 1: 0.5406 (54% quality)
After Optuna (20 trials): 0.5447 (54.5% quality)

Improvement from Optuna: +0.8% only
```

**Observation:** We're hitting a **ceiling**. Tier 1 parameters are optimized, but heterogeneity is still hardcoded.

### **Hypothesis:**
If we make heterogeneity distributions tunable, Optuna could find better customer parameter distributions and push score to **0.60-0.65** (60-65% quality).

---

## 🎯 PROPOSED SOLUTION

### **Keep Beta Distributions, Make Them Configurable**

Instead of converting mean/std → Beta (complex), we expose **Beta parameters directly** to config.

### **Implementation:**

**Step 1: Add Config Parameters (Tier 3)**
```python
# config.py - NEW TIER 3 PARAMETERS

# Heterogeneity: Promo Responsiveness (Beta distribution)
promo_responsiveness_alpha: float = 3.0  # Shape parameter
promo_responsiveness_beta: float = 2.0   # Shape parameter
promo_responsiveness_min: float = 0.5    # Lower bound
promo_responsiveness_max: float = 2.0    # Upper bound

# Heterogeneity: Price Sensitivity (Log-normal)
price_sensitivity_mu: float = 0.15       # Log-normal mean
price_sensitivity_sigma: float = 0.4     # Log-normal std
price_sensitivity_min: float = 0.5
price_sensitivity_max: float = 2.5

# Heterogeneity: Display Sensitivity (Beta)
display_sensitivity_alpha: float = 3.0
display_sensitivity_beta: float = 3.0
display_sensitivity_min: float = 0.3
display_sensitivity_max: float = 1.2

# ... (similar for other 3 parameters)
```

**Step 2: Update Heterogeneity Engine**
```python
# customer_heterogeneity.py

def __init__(self, random_seed: Optional[int] = None, config = None):
    self.random_seed = random_seed
    self.config = config  # NEW
    if random_seed is not None:
        np.random.seed(random_seed)
    
    self._init_distributions()

def _init_distributions(self):
    if self.config:
        # Use config parameters
        self.promo_responsiveness_dist = {
            'type': 'beta',
            'params': {
                'alpha': self.config.promo_responsiveness_alpha,
                'beta': self.config.promo_responsiveness_beta
            },
            'bounds': (self.config.promo_responsiveness_min, 
                      self.config.promo_responsiveness_max)
        }
        # ... similar for other parameters
    else:
        # Fallback to hardcoded (backward compatible)
        self.promo_responsiveness_dist = {
            'type': 'beta',
            'params': {'alpha': 3, 'beta': 2},
            'bounds': (0.5, 2.0)
        }
```

**Step 3: Add to Optuna Tier 3**
```python
# tune_parameters_optuna.py

if 3 in self.tiers:
    print(f"   🔧 Tuning Tier 3 parameters (Heterogeneity)...")
    
    # Promo Responsiveness
    config.promo_responsiveness_alpha = trial.suggest_float('promo_resp_alpha', 1.5, 5.0)
    config.promo_responsiveness_beta = trial.suggest_float('promo_resp_beta', 1.5, 5.0)
    
    # Price Sensitivity
    config.price_sensitivity_mu = trial.suggest_float('price_sens_mu', 0.0, 0.3)
    config.price_sensitivity_sigma = trial.suggest_float('price_sens_sigma', 0.2, 0.6)
    
    # Display Sensitivity
    config.display_sensitivity_alpha = trial.suggest_float('display_alpha', 2.0, 5.0)
    config.display_sensitivity_beta = trial.suggest_float('display_beta', 2.0, 5.0)
    
    # ... (12 more parameters for 6 distributions)
```

---

## 📊 IMPACT ANALYSIS

### **Pros:** ✅

1. **Optuna Can Tune Customer Behavior**
   - Currently: Customer heterogeneity is FIXED
   - After: Optuna can optimize how customers vary
   - Potential: +5-10% calibration improvement

2. **More Realistic Distributions**
   - Current Beta(3,2) might not match Dunnhumby
   - Optuna could find Beta(4,1.5) or Beta(2,3) works better
   - Data-driven instead of assumed

3. **Unlock Next Level of Quality**
   - Tier 1: 0.54 (done)
   - Tier 2: 0.56-0.58 (next)
   - **Tier 3: 0.60-0.65 (with heterogeneity tuning)**

4. **Clean Implementation**
   - No complex conversions
   - Backward compatible
   - Beta parameters are intuitive

5. **Addresses Your Concern**
   - Config parameters no longer orphaned
   - Everything is tunable
   - Complete system

### **Cons:** ❌

1. **More Parameters to Tune**
   - Adds ~18 new parameters (3 per distribution × 6 distributions)
   - Optuna search space grows
   - Need more trials (50-100 instead of 20)

2. **Longer Tuning Time**
   - Tier 3 would take 1-2 hours
   - But only run once

3. **Complexity**
   - Need to understand Beta distributions
   - Alpha/beta less intuitive than mean/std
   - Documentation burden

4. **Diminishing Returns?**
   - Might only improve by 5-10%
   - Is it worth 2-3 hours of work?

5. **Current Distributions Might Be Good**
   - Beta(3,2) is reasonable
   - Not broken, just not optimized

---

## 🎯 RECOMMENDATION

### **Option A: Implement Now** 🟡 MAYBE

**When:**
- If you want to push for 60-65% quality
- If you have 2-3 hours to implement
- If you're curious about optimal heterogeneity

**Expected Outcome:**
- Score: 0.60-0.65 (vs current 0.5447)
- Improvement: +10-20%
- Time: 2-3 hours implementation + 1-2 hours Tier 3 tuning

### **Option B: Defer to Later** ✅ RECOMMENDED

**When:**
- After running Tier 2 optimization first
- After seeing if 0.56-0.58 is "good enough"
- If Tier 2 plateaus, THEN add Tier 3

**Rationale:**
1. Tier 2 parameters (promo frequency, marketing weights, loyalty) might get you to 0.58-0.60
2. If that's sufficient, save the effort
3. If not, THEN invest in Tier 3 heterogeneity
4. **Incremental approach:** Tier 1 → Tier 2 → (assess) → Tier 3 if needed

### **Option C: Simple Version** 🟢 COMPROMISE

**Implementation:**
Only expose **mean-shifting parameters**, not full Beta distributions:

```python
# config.py - SIMPLE VERSION
promo_responsiveness_multiplier: float = 1.0  # Scales Beta(3,2) mean
price_sensitivity_multiplier: float = 1.0
display_sensitivity_multiplier: float = 1.0
```

Then in heterogeneity engine:
```python
# Scale the distribution mean
base_alpha, base_beta = 3, 2
scaled_alpha = base_alpha * config.promo_responsiveness_multiplier
# Keep beta same to preserve shape
```

**Pros:**
- ✅ Much simpler (3 params instead of 18)
- ✅ Easier to understand
- ✅ Still tunable
- ✅ 30 min implementation

**Cons:**
- ❌ Less flexible than full Beta tuning
- ❌ Can't change distribution shape, only mean

---

## 🎲 MY RECOMMENDATION

### **Do Option B: Defer to After Tier 2**

**Reasoning:**
1. You just ran Tier 1, got 0.5447
2. **Next logical step:** Run Tier 2 (promo, loyalty, marketing)
3. Tier 2 likely gets you to 0.56-0.60
4. **Then assess:** Is 0.60 good enough?
   - If YES: Stop, you're done! ✅
   - If NO: Implement Tier 3 heterogeneity tuning

**Timeline:**
```
Now: Run Tier 2 optimization (30-40 trials, 30 min)
Expected: Score 0.56-0.60

If score < 0.60:
  Then: Implement Tier 3 (2-3 hours)
  Then: Run Tier 3 optimization (50 trials, 1 hour)
  Expected: Score 0.60-0.65

If score >= 0.60:
  Done! No need for Tier 3
```

---

## 🎯 DECISION MATRIX

| Scenario | Score After Tier 2 | Action | Effort | Final Score |
|----------|-------------------|--------|--------|-------------|
| **Best Case** | 0.60-0.62 | Stop | 0 hours | 0.60-0.62 ✅ |
| **Good Case** | 0.56-0.59 | Add Tier 3 | 3-4 hours | 0.62-0.65 ✅ |
| **Worst Case** | 0.54-0.55 | Add Tier 3 + debug | 5-6 hours | 0.58-0.62 🟡 |

**Expected:** Good Case (0.56-0.59 after Tier 2)

---

## 🚀 PROPOSED NEXT STEPS

### **Immediate (Now):**
```bash
# Run Tier 2 optimization
python scripts/tune_parameters_optuna.py \
  --objective all \
  --tier 2 \
  --n-trials 40
```

**Expected:** 30-40 minutes, score 0.56-0.60

### **After Tier 2:**
1. Check score
2. If < 0.60: Implement Tier 3 heterogeneity
3. If >= 0.60: Celebrate and move on! 🎉

---

## ❓ YOUR DECISION

**What would you like to do?**

**A)** Implement Tier 3 heterogeneity tuning NOW (2-3 hours)
**B)** Run Tier 2 first, then decide (30 min now, maybe 3 hours later)
**C)** Simple version - just add multipliers (30 min)
**D)** Keep as-is - heterogeneity is good enough

**My vote:** **B** - Run Tier 2 first, it's the logical next step and might be sufficient!

What do you think? 🤔
