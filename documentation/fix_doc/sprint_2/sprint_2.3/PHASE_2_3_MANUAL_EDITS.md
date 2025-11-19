# Phase 2.3: Manual Integration Edits

To complete Phase 2.3 integration, please make these 3 small edits to `promotional_engine.py`:

---

## Edit 1: Update Version Header (Lines 7-11)

**Change from:**
```python
# ============================================================================
# PROMOTIONAL ENGINE (v1.1 - Sprint 2.2)
# Comprehensive promotional system: mechanics, displays, features
# Phase 2.2: HMM integration, product tendencies, multi-store support
# ============================================================================
```

**Change to:**
```python
# ============================================================================
# PROMOTIONAL ENGINE (v1.2 - Sprint 2.3)
# Comprehensive promotional system: mechanics, displays, features
# Phase 2.2: HMM integration, product tendencies, multi-store support
# Phase 2.3: Marketing signal calculation
# ============================================================================
```

---

## Edit 2: Initialize Marketing Signal Calculator (After Line 100)

**Add these lines after line 100 (`self._init_product_tendencies()`):**

```python
        # Phase 2.3: Marketing signal calculator
        from retailsynth.engines.marketing_signal import MarketingSignalCalculator
        self.signal_calculator = MarketingSignalCalculator(config=self.config)
```

**Result should look like:**
```python
        # Phase 2.2: Product-specific promotional tendencies
        self.product_promo_tendencies = {}
        self._init_product_tendencies()
        
        # Phase 2.3: Marketing signal calculator
        from retailsynth.engines.marketing_signal import MarketingSignalCalculator
        self.signal_calculator = MarketingSignalCalculator(config=self.config)
        
        # Active promotions tracking
```

---

## Edit 3: Calculate Marketing Signal (After Line 574)

**Add these lines after `context.compute_metrics()` (around line 574):**

```python
        # Phase 2.3: Calculate marketing signal strength
        context.marketing_signal_strength = self.signal_calculator.calculate_signal_strength(context)
```

**Result should look like:**
```python
        # Compute summary metrics
        context.compute_metrics()
        
        # Phase 2.3: Calculate marketing signal strength
        context.marketing_signal_strength = self.signal_calculator.calculate_signal_strength(context)
        
        # Update active promotions tracking
        self._update_active_promotions(store_id, context)
        
        return context
```

---

## Verification

After making these edits:

1. **No syntax errors** - File should still be valid Python
2. **Marketing signal calculator initialized** - Check `__init__` method
3. **Signal strength calculated** - Check `generate_store_promotions` method
4. **Version updated** - Header shows v1.2

**That's it!** These 3 edits complete the Phase 2.3 integration.
