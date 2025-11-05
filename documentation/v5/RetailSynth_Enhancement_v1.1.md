# RetailSynth Enhancement Project v1.1
## Comprehensive Documentation for Claude Code Collaboration

**Project Goal:** Transform enhanced_retailsynth_v4_0 from an unvalidated prototype into a research-grade synthetic retail data generator with 80%+ validation pass rate against Dunnhumby Complete Journey dataset.

**Timeline:** 6-8 weeks intensive development

**Collaboration Model:** Human + Claude Code (AI coding agent)

**Version:** 1.1
**Last Updated:** October 31, 2025
**Changes from v1.0:**
- ✅ Reordered Phase 1 sprints - Product catalog now Sprint 1.1 (first priority)
- ✅ Expanded product catalog specification - 20K SKUs from Dunnhumby
- ✅ Added HMM specification for price/cross-price elasticity
- ✅ Added arc price elasticity requirements
- ✅ Updated dependencies and validation metrics

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current State Assessment](#current-state-assessment)
3. [Critical Issues Documentation](#critical-issues-documentation)
4. [Goals & Success Metrics](#goals--success-metrics)
5. [Product Roadmap](#product-roadmap)
6. [Development Workflow](#development-workflow)
7. [Repository Structure](#repository-structure)
8. [Technical Architecture](#technical-architecture)
9. [Testing & Validation Strategy](#testing--validation-strategy)
10. [Collaboration Guidelines](#collaboration-guidelines)
11. [Release Plan](#release-plan)

---

## 1. Project Overview

### What We're Building

A **statistically validated synthetic retail data generator** that:
- Generates realistic shopping transactions matching real-world patterns
- Calibrated against Dunnhumby Complete Journey (84M transactions, 2,500 households, **92K products**)
- Uses **20K representative SKUs** extracted from real product catalog
- Implements **Hidden Markov Models for price elasticity** (own-price, cross-price, and arc elasticity)
- Suitable for academic research, algorithm benchmarking, and ML training
- Implements econometric foundations (utility theory, discrete choice models)
- Achieves 80%+ validation pass rate across 4 validation levels

### Why This Matters

**Current Problem:**
- Enhanced v4.0 code uses fictional product catalog (incompatible with real data)
- No price elasticity modeling (products chosen independently)
- No empirical accuracy guarantees
- Cannot be validated against Dunnhumby

**Solution:**
- **Extract 20K real SKUs** from Dunnhumby (representative sample of 92K)
- **Implement HMM for price dynamics** (state-dependent pricing and elasticity)
- **Model cross-price and arc elasticity** (substitution and intertemporal effects)
- Fix remaining 10 structural limitations
- Validate against real grocery data
- Create publication-ready tool

---

## 2. Current State Assessment

### What Works Well ✅

1. **Performance Optimization**
   - JAX GPU acceleration
   - Vectorized operations (no Python loops)
   - Fast generation: 100K customers × 5K products × 52 weeks in 12-20 min
   - Memory efficient

2. **Rich Feature Set**
   - Comprehensive temporal dynamics (customer drift, lifecycle, seasonality)
   - Store loyalty modeling
   - 3-level category hierarchy
   - Brand positioning framework
   - 15+ output datasets

3. **Code Quality** (Partial)
   - Well-structured classes
   - Good documentation headers
   - Clear variable names

### What's Broken ❌

1. **Product Catalog Mismatch** ⚠️ **BLOCKING ISSUE**
   - Synthetic products vs. real Dunnhumby products
   - No mapping possible for validation
   - Fictional brands vs. real manufacturers
   - Wrong category structure

2. **No Price Elasticity Modeling** ⚠️ **BLOCKING ISSUE**
   - Products chosen independently
   - No substitution effects
   - No complementarity
   - No price state dynamics
   - Missing arc elasticity (intertemporal)

3. **Other Fundamental Gaps** (See Section 3)
   - No purchase history
   - Archetype-based not individual-level
   - No basket composition logic
   - And 8 more critical issues

---

## 3. Critical Issues Documentation

### Issue #0: Product Catalog Alignment ⚠️ BLOCKING - MUST FIX FIRST
**Priority:** P0 (Sprint 1.1 - Days 1-5)
**Effort:** 5-7 days
**Impact:** **CRITICAL - BLOCKS ALL VALIDATION**

**Current Behavior:**
```python
# Synthetic 3-level hierarchy with made-up products
hierarchy = {
    'Fresh': {
        'Produce': ['Apples', 'Bananas', ...],  # Generic names
        'Dairy': ['Milk', 'Yogurt', ...]
    }
}

# Fictional brands
brands = ['Premium Brand A', 'Value Brand B', ...]
```

**Problem:**
- Cannot map synthetic products to real Dunnhumby products
- Different category structures (4 departments vs. Dunnhumby's structure)
- Made-up brands vs. real manufacturers
- No validation possible without 1:1 or archetype mapping
- 5K products insufficient (Dunnhumby has 92K)

**Solution Required:**

#### Step 1: Extract Dunnhumby Product Master (Days 1-2)

```python
# Load Dunnhumby product.csv
"""
PRODUCT_ID    MANUFACTURER  DEPARTMENT  BRAND           COMMODITY_DESC        SUB_COMMODITY_DESC       CURR_SIZE_OF_PRODUCT
1                101       GROCERY     PRIVATE LABEL   SOFT DRINKS           CARBONATED SOFT DRINKS   12 PK 12 OZ CAN
2                102       GROCERY     COCA COLA       SOFT DRINKS           CARBONATED SOFT DRINKS   2 LITER
...
92000            500       DRUG GM     TYLENOL         PAIN REMEDIES         PAIN RELIEF MEDICATION   100 CT
"""

# Analysis required:
# 1. Unique products: 92,004
# 2. Departments: ~35 (GROCERY, DRUG GM, MEAT, PRODUCE, DAIRY, etc.)
# 3. Commodity categories: ~300
# 4. Sub-commodities: ~1,500
# 5. Manufacturers: ~5,000
# 6. Brands: ~8,000
```

#### Step 2: Create Representative 20K SKU Sample (Days 2-3)

**Sampling Strategy:**
```python
class ProductCatalogBuilder:
    """
    Build representative 20K SKU catalog from Dunnhumby's 92K products
    """
    
    def __init__(self, dunnhumby_products_df: pd.DataFrame):
        self.full_catalog = dunnhumby_products_df
        self.n_target_skus = 20000
        
    def create_representative_sample(self) -> pd.DataFrame:
        """
        Create 20K SKU sample that preserves:
        1. Category structure (proportional sampling)
        2. Brand mix (major brands + private label)
        3. Price distribution (low/mid/high)
        4. Purchase frequency distribution (movers vs. long-tail)
        """
        
        # Step 1: Load transaction data to get purchase frequencies
        transactions = pd.read_csv('transaction_data.csv')
        product_popularity = transactions.groupby('PRODUCT_ID').size()
        
        # Step 2: Classify products by popularity
        self.full_catalog['purchase_frequency'] = self.full_catalog['PRODUCT_ID'].map(
            product_popularity
        ).fillna(0)
        
        # Create tiers (using Pareto distribution)
        # - Tier A: Top 20% of products (80% of volume) → Sample 40% (8,000 SKUs)
        # - Tier B: Next 30% (15% of volume) → Sample 35% (7,000 SKUs)
        # - Tier C: Remaining 50% (5% of volume) → Sample 25% (5,000 SKUs)
        
        tier_a_cutoff = self.full_catalog['purchase_frequency'].quantile(0.80)
        tier_b_cutoff = self.full_catalog['purchase_frequency'].quantile(0.50)
        
        self.full_catalog['tier'] = pd.cut(
            self.full_catalog['purchase_frequency'],
            bins=[-np.inf, tier_b_cutoff, tier_a_cutoff, np.inf],
            labels=['C', 'B', 'A']
        )
        
        # Step 3: Stratified sampling by department + tier
        sample_skus = []
        
        for department in self.full_catalog['DEPARTMENT'].unique():
            dept_products = self.full_catalog[
                self.full_catalog['DEPARTMENT'] == department
            ]
            
            # Proportional allocation
            dept_share = len(dept_products) / len(self.full_catalog)
            dept_target = int(self.n_target_skus * dept_share)
            
            # Sample by tier within department
            for tier, tier_weight in [('A', 0.4), ('B', 0.35), ('C', 0.25)]:
                tier_products = dept_products[dept_products['tier'] == tier]
                tier_sample_size = int(dept_target * tier_weight)
                
                if len(tier_products) > 0:
                    sample = tier_products.sample(
                        n=min(tier_sample_size, len(tier_products)),
                        weights='purchase_frequency',  # Weighted sampling
                        random_state=42
                    )
                    sample_skus.append(sample)
        
        representative_catalog = pd.concat(sample_skus).reset_index(drop=True)
        
        # Step 4: Ensure major brands are included
        representative_catalog = self._ensure_major_brands(representative_catalog)
        
        # Step 5: Ensure category diversity
        representative_catalog = self._ensure_category_coverage(representative_catalog)
        
        return representative_catalog
    
    def _ensure_major_brands(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Ensure top brands by revenue are represented"""
        
        # Get top 100 brands by transaction volume
        transactions = pd.read_csv('transaction_data.csv')
        brand_volumes = (
            transactions
            .merge(self.full_catalog[['PRODUCT_ID', 'BRAND']], on='PRODUCT_ID')
            .groupby('BRAND')['SALES_VALUE']
            .sum()
            .sort_values(ascending=False)
            .head(100)
        )
        
        major_brands = brand_volumes.index
        
        # Check coverage
        covered_brands = catalog['BRAND'].unique()
        missing_brands = set(major_brands) - set(covered_brands)
        
        # Add representative products for missing major brands
        if missing_brands:
            for brand in missing_brands:
                brand_products = self.full_catalog[self.full_catalog['BRAND'] == brand]
                if len(brand_products) > 0:
                    # Add most popular product from this brand
                    top_product = brand_products.nlargest(1, 'purchase_frequency')
                    catalog = pd.concat([catalog, top_product])
        
        return catalog
    
    def _ensure_category_coverage(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Ensure all major categories have minimum representation"""
        
        # Minimum products per commodity category
        min_per_commodity = 3
        
        for commodity in self.full_catalog['COMMODITY_DESC'].unique():
            commodity_count = (catalog['COMMODITY_DESC'] == commodity).sum()
            
            if commodity_count < min_per_commodity:
                commodity_products = self.full_catalog[
                    self.full_catalog['COMMODITY_DESC'] == commodity
                ]
                
                additional_needed = min_per_commodity - commodity_count
                additional = commodity_products.nlargest(
                    additional_needed, 
                    'purchase_frequency'
                )
                catalog = pd.concat([catalog, additional])
        
        return catalog.drop_duplicates(subset='PRODUCT_ID').reset_index(drop=True)
    
    def create_product_archetypes(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Create product archetypes for behavioral modeling
        
        Archetypes defined by:
        - Price tier (economy/mid/premium)
        - Purchase frequency (staple/regular/occasional)
        - Category role (destination/routine/impulse)
        """
        
        # Calculate price percentiles within category
        catalog['price_percentile'] = catalog.groupby('COMMODITY_DESC')['avg_price'].rank(pct=True)
        
        # Classify price tier
        catalog['price_tier'] = pd.cut(
            catalog['price_percentile'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['economy', 'mid_tier', 'premium']
        )
        
        # Classify purchase frequency
        catalog['frequency_tier'] = pd.cut(
            catalog['purchase_frequency'],
            bins=[0, catalog['purchase_frequency'].quantile(0.5), 
                  catalog['purchase_frequency'].quantile(0.85), np.inf],
            labels=['occasional', 'regular', 'staple']
        )
        
        # Create archetype ID
        catalog['archetype'] = (
            catalog['DEPARTMENT'] + '_' +
            catalog['price_tier'] + '_' +
            catalog['frequency_tier']
        )
        
        return catalog
    
    def save_product_catalog(self, catalog: pd.DataFrame, output_dir: str):
        """Save processed product catalog"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save full catalog
        catalog.to_parquet(output_path / 'product_catalog_20k.parquet')
        
        # Save category hierarchy mapping
        hierarchy = self._build_hierarchy_mapping(catalog)
        with open(output_path / 'category_hierarchy.json', 'w') as f:
            json.dump(hierarchy, f, indent=2)
        
        # Save archetype definitions
        archetypes = catalog.groupby('archetype').agg({
            'PRODUCT_ID': 'count',
            'avg_price': 'mean',
            'purchase_frequency': 'mean'
        }).reset_index()
        archetypes.to_csv(output_path / 'product_archetypes.csv', index=False)
        
        print(f"✅ Saved 20K product catalog to {output_dir}")
        print(f"   - Products: {len(catalog):,}")
        print(f"   - Departments: {catalog['DEPARTMENT'].nunique()}")
        print(f"   - Brands: {catalog['BRAND'].nunique()}")
        print(f"   - Archetypes: {catalog['archetype'].nunique()}")
    
    def _build_hierarchy_mapping(self, catalog: pd.DataFrame) -> Dict:
        """Build hierarchical category structure"""
        hierarchy = {}
        
        for _, row in catalog.iterrows():
            dept = row['DEPARTMENT']
            commodity = row['COMMODITY_DESC']
            sub_commodity = row['SUB_COMMODITY_DESC']
            
            if dept not in hierarchy:
                hierarchy[dept] = {}
            if commodity not in hierarchy[dept]:
                hierarchy[dept][commodity] = {}
            if sub_commodity not in hierarchy[dept][commodity]:
                hierarchy[dept][commodity][sub_commodity] = []
            
            hierarchy[dept][commodity][sub_commodity].append({
                'product_id': int(row['PRODUCT_ID']),
                'brand': row['BRAND'],
                'manufacturer': row['MANUFACTURER'],
                'size': row['CURR_SIZE_OF_PRODUCT']
            })
        
        return hierarchy


# Usage
if __name__ == "__main__":
    # Load Dunnhumby data
    products_df = pd.read_csv('dunnhumby_data/product.csv')
    transactions_df = pd.read_csv('dunnhumby_data/transaction_data.csv')
    
    # Enrich with transaction statistics
    product_stats = transactions_df.groupby('PRODUCT_ID').agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'BASKET_ID': 'nunique',
        'household_key': 'nunique'
    }).reset_index()
    
    product_stats.columns = ['PRODUCT_ID', 'total_revenue', 'total_quantity', 
                             'total_baskets', 'total_customers']
    
    products_enriched = products_df.merge(product_stats, on='PRODUCT_ID', how='left')
    products_enriched['avg_price'] = (
        products_enriched['total_revenue'] / products_enriched['total_quantity']
    ).fillna(products_enriched['total_revenue'].mean())
    
    # Build 20K catalog
    builder = ProductCatalogBuilder(products_enriched)
    catalog_20k = builder.create_representative_sample()
    catalog_20k = builder.create_product_archetypes(catalog_20k)
    builder.save_product_catalog(catalog_20k, './data/processed/product_catalog')
    
    # Validation
    print("\n" + "="*70)
    print("PRODUCT CATALOG VALIDATION")
    print("="*70)
    
    print("\nOriginal Dunnhumby catalog:")
    print(f"  Total SKUs: {len(products_df):,}")
    print(f"  Departments: {products_df['DEPARTMENT'].nunique()}")
    print(f"  Brands: {products_df['BRAND'].nunique()}")
    
    print("\nRepresentative 20K catalog:")
    print(f"  Total SKUs: {len(catalog_20k):,}")
    print(f"  Departments: {catalog_20k['DEPARTMENT'].nunique()}")
    print(f"  Brands: {catalog_20k['BRAND'].nunique()}")
    print(f"  Archetypes: {catalog_20k['archetype'].nunique()}")
    
    # Coverage check
    original_dept_dist = products_df['DEPARTMENT'].value_counts(normalize=True)
    sample_dept_dist = catalog_20k['DEPARTMENT'].value_counts(normalize=True)
    
    print("\nDepartment distribution match:")
    for dept in original_dept_dist.index:
        orig_pct = original_dept_dist.get(dept, 0) * 100
        sample_pct = sample_dept_dist.get(dept, 0) * 100
        diff = abs(orig_pct - sample_pct)
        print(f"  {dept}: {orig_pct:.1f}% → {sample_pct:.1f}% (Δ{diff:.1f}%)")
```

#### Step 3: Integration with Generator (Days 4-5)

```python
# Replace synthetic product generation with real catalog loading

class EnhancedRetailSynthV4_1:
    
    def __init__(self, config: EnhancedRetailConfig):
        self.config = config
        
        # Load real 20K product catalog
        self.product_catalog = pd.read_parquet(
            config.product_catalog_path  # './data/processed/product_catalog/product_catalog_20k.parquet'
        )
        
        # Load category hierarchy
        with open(config.category_hierarchy_path, 'r') as f:
            self.category_hierarchy = json.load(f)
        
        # Use real products instead of generating synthetic
        self.products = self.product_catalog.copy()
        
        print(f"✅ Loaded {len(self.products):,} real products from Dunnhumby")
        print(f"   Departments: {self.products['DEPARTMENT'].nunique()}")
        print(f"   Brands: {self.products['BRAND'].nunique()}")
```

**Files to Create:**
- `src/retailsynth/catalog/product_catalog_builder.py` - 20K sampling logic
- `src/retailsynth/catalog/hierarchy_mapper.py` - Category mapping
- `src/retailsynth/catalog/archetype_classifier.py` - Product archetypes
- `scripts/build_product_catalog.py` - Standalone script

**Files to Modify:**
- `src/retailsynth/generators/product_generator.py` - Load instead of generate
- `src/retailsynth/config.py` - Add catalog paths

**Dependencies:**
- Dunnhumby product.csv
- Dunnhumby transaction_data.csv (for frequency stats)
- Pandas for data manipulation

**Validation Tests:**
```python
def test_product_catalog_coverage():
    """Test that 20K catalog covers all major categories"""
    catalog = load_product_catalog()
    
    # Check department coverage
    assert catalog['DEPARTMENT'].nunique() >= 30, "Missing departments"
    
    # Check brand coverage (top 100 brands)
    top_brands = get_top_brands_from_dunnhumby(n=100)
    catalog_brands = set(catalog['BRAND'].unique())
    coverage = len(top_brands & catalog_brands) / len(top_brands)
    assert coverage >= 0.90, f"Only {coverage*100:.1f}% of top brands covered"
    
    # Check archetype distribution
    archetypes = catalog['archetype'].value_counts()
    assert len(archetypes) >= 50, "Too few archetypes"
    
    # Check price distribution preservation
    original_price_dist = get_original_price_distribution()
    sample_price_dist = catalog['avg_price'].describe()
    
    mean_error = abs(original_price_dist['mean'] - sample_price_dist['mean']) / original_price_dist['mean']
    assert mean_error < 0.10, f"Price mean error {mean_error*100:.1f}% too high"

def test_product_frequency_distribution():
    """Test that purchase frequency distribution is preserved"""
    from scipy.stats import ks_2samp
    
    original_freq = get_original_purchase_frequencies()
    sample_freq = catalog['purchase_frequency']
    
    ks_stat, p_value = ks_2samp(original_freq, sample_freq)
    assert p_value > 0.05, f"Purchase frequency distribution mismatch (p={p_value:.4f})"
```

**Success Criteria:**
- ✅ 20,000 representative SKUs extracted
- ✅ All 30+ departments covered
- ✅ Top 100 brands included (90%+ coverage)
- ✅ Purchase frequency distribution preserved (KS test p > 0.05)
- ✅ Price distribution preserved (mean error < 10%)
- ✅ Category hierarchy mapped to Dunnhumby structure
- ✅ Product archetypes defined for behavioral modeling

---

### Issue #1: Price and Cross-Price Elasticity with HMM ⚠️ CRITICAL
**Priority:** P0 (Sprint 1.2 - Days 6-12)
**Effort:** 7-9 days
**Impact:** **VERY HIGH - CORE ECONOMIC BEHAVIOR**

**Current Behavior:**
```python
# Products chosen independently based on simple utility
utility[product_i] = β_price * price[i] + β_brand * brand[i] + ...
# No substitution, no complementarity, no price dynamics
```

**Problem:**
- No cross-price effects (Coke price ↑ → Pepsi demand ↑)
- No arc price elasticity (future price expectations affect current purchase)
- No price state dynamics (prices follow patterns, not random)
- Linear price effects (unrealistic)
- Missing econometric foundations

**Solution Required:**

#### Part A: Hidden Markov Model for Price States (Days 6-8)

**Theoretical Foundation:**
```
Price states reflect unobservable market conditions:
- State 0: Regular pricing (baseline)
- State 1: Feature pricing (in-ad, moderate discount)
- State 2: Deep discount (TPR - Temporary Price Reduction)

State transitions follow Markov property:
P(State_t | State_{t-1}, State_{t-2}, ...) = P(State_t | State_{t-1})
```

**Implementation:**
```python
class PriceStateHMM:
    """
    Hidden Markov Model for product price states
    
    Follows RetailSynth paper methodology with extensions for:
    - Product-specific transition matrices
    - Category-level state correlation
    - Seasonal state probability modulation
    """
    
    def __init__(self, products_df: pd.DataFrame):
        self.products = products_df
        self.n_states = 3  # Regular, Feature, Deep Discount
        
        # State definitions
        self.states = {
            0: {'name': 'regular', 'discount_range': (0.0, 0.05)},
            1: {'name': 'feature', 'discount_range': (0.10, 0.25)},
            2: {'name': 'deep_discount', 'discount_range': (0.25, 0.50)}
        }
        
        # Initialize from data
        self.transition_matrices = {}
        self.emission_distributions = {}
        self.initial_state_probs = {}
    
    def learn_from_data(self, transactions_df: pd.DataFrame, causal_df: pd.DataFrame):
        """
        Learn HMM parameters from Dunnhumby data
        
        Args:
            transactions_df: Transaction data with prices
            causal_df: Promotional data (DISPLAY, MAILER flags)
        """
        print("Learning HMM parameters from Dunnhumby data...")
        
        # Merge transaction and promotional data
        data = transactions_df.merge(
            causal_df,
            on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
            how='left'
        )
        
        # Calculate discount percentage per product-week
        product_week_prices = data.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
            'SALES_VALUE': 'sum',
            'QUANTITY': 'sum',
            'RETAIL_DISC': 'sum',
            'DISPLAY': lambda x: (x == 'A').any(),  # Any display activity
            'MAILER': lambda x: (x == 'A').any()     # Any mailer activity
        }).reset_index()
        
        product_week_prices['avg_price'] = (
            product_week_prices['SALES_VALUE'] / product_week_prices['QUANTITY']
        )
        product_week_prices['discount_pct'] = (
            product_week_prices['RETAIL_DISC'] / product_week_prices['SALES_VALUE']
        )
        
        # Infer hidden states using discount + promotional flags
        def classify_state(row):
            discount = row['discount_pct']
            has_display = row['DISPLAY']
            has_mailer = row['MAILER']
            
            if discount >= 0.25 or (has_display and has_mailer):
                return 2  # Deep discount
            elif discount >= 0.10 or has_display or has_mailer:
                return 1  # Feature
            else:
                return 0  # Regular
        
        product_week_prices['state'] = product_week_prices.apply(classify_state, axis=1)
        
        # Learn parameters for each product
        for product_id in tqdm(self.products['PRODUCT_ID'].unique()):
            product_data = product_week_prices[
                product_week_prices['PRODUCT_ID'] == product_id
            ].sort_values('WEEK_NO')
            
            if len(product_data) < 10:  # Need sufficient history
                continue
            
            # Learn transition matrix
            transition_matrix = self._estimate_transition_matrix(product_data['state'].values)
            self.transition_matrices[product_id] = transition_matrix
            
            # Learn emission distributions (price given state)
            emission_dists = self._estimate_emission_distributions(
                product_data[['state', 'avg_price', 'discount_pct']]
            )
            self.emission_distributions[product_id] = emission_dists
            
            # Initial state probabilities
            initial_probs = product_data['state'].value_counts(normalize=True).to_dict()
            self.initial_state_probs[product_id] = initial_probs
        
        print(f"✅ Learned HMM parameters for {len(self.transition_matrices):,} products")
    
    def _estimate_transition_matrix(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Estimate state transition matrix from observed sequence
        
        Returns:
            3x3 matrix where element [i,j] = P(state_j | state_i)
        """
        n_states = 3
        transitions = np.zeros((n_states, n_states))
        
        for t in range(len(state_sequence) - 1):
            current_state = int(state_sequence[t])
            next_state = int(state_sequence[t + 1])
            transitions[current_state, next_state] += 1
        
        # Normalize rows (add Laplace smoothing)
        transitions += 1  # Smoothing
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / row_sums
        
        return transition_matrix
    
    def _estimate_emission_distributions(self, data: pd.DataFrame) -> Dict:
        """
        Estimate price distribution for each state
        
        Returns:
            Dictionary mapping state -> {mean, std, discount_depth}
        """
        emissions = {}
        
        for state in [0, 1, 2]:
            state_data = data[data['state'] == state]
            
            if len(state_data) > 0:
                emissions[state] = {
                    'price_mean': state_data['avg_price'].mean(),
                    'price_std': state_data['avg_price'].std(),
                    'discount_mean': state_data['discount_pct'].mean(),
                    'discount_std': state_data['discount_pct'].std()
                }
            else:
                # Default values if no observations
                emissions[state] = {
                    'price_mean': data['avg_price'].mean(),
                    'price_std': data['avg_price'].std(),
                    'discount_mean': 0.0,
                    'discount_std': 0.1
                }
        
        return emissions
    
    def generate_price_sequence(self, 
                                product_id: int, 
                                n_weeks: int,
                                base_price: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price sequence for a product using HMM
        
        Returns:
            prices: Array of prices for each week
            states: Array of hidden states for each week
        """
        if product_id not in self.transition_matrices:
            # Use average transition matrix for unseen products
            product_id = self._get_similar_product(product_id)
        
        transition_matrix = self.transition_matrices[product_id]
        emission_dists = self.emission_distributions[product_id]
        initial_probs = self.initial_state_probs.get(product_id, {0: 0.7, 1: 0.2, 2: 0.1})
        
        # Generate state sequence
        states = np.zeros(n_weeks, dtype=int)
        
        # Initial state
        states[0] = np.random.choice(
            [0, 1, 2], 
            p=[initial_probs.get(s, 0.33) for s in [0, 1, 2]]
        )
        
        # State transitions
        for t in range(1, n_weeks):
            prev_state = states[t-1]
            transition_probs = transition_matrix[prev_state]
            states[t] = np.random.choice([0, 1, 2], p=transition_probs)
        
        # Generate prices from states
        prices = np.zeros(n_weeks)
        
        for t in range(n_weeks):
            state = states[t]
            emission = emission_dists[state]
            
            # Apply discount based on state
            discount = max(0, np.random.normal(
                emission['discount_mean'],
                emission['discount_std']
            ))
            
            prices[t] = base_price * (1 - discount)
            
            # Add small noise
            prices[t] *= np.random.normal(1.0, 0.02)
            prices[t] = max(0.50, prices[t])  # Floor price
        
        return prices, states
    
    def _get_similar_product(self, product_id: int) -> int:
        """Find product with similar characteristics for fallback"""
        product = self.products[self.products['PRODUCT_ID'] == product_id].iloc[0]
        
        # Find products in same commodity
        similar = self.products[
            self.products['COMMODITY_DESC'] == product['COMMODITY_DESC']
        ]
        
        # Return one with learned parameters
        for similar_id in similar['PRODUCT_ID']:
            if similar_id in self.transition_matrices:
                return similar_id
        
        # Fallback to any product with parameters
        return list(self.transition_matrices.keys())[0]
    
    def save_hmm_parameters(self, output_path: str):
        """Save learned HMM parameters"""
        import pickle
        
        hmm_data = {
            'transition_matrices': self.transition_matrices,
            'emission_distributions': self.emission_distributions,
            'initial_state_probs': self.initial_state_probs,
            'states': self.states
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        print(f"✅ Saved HMM parameters to {output_path}")
```

#### Part B: Cross-Price Elasticity Matrix (Days 9-10)

**Theoretical Foundation:**
```
Cross-price elasticity measures substitution/complementarity:

ε_ij = (∂Q_i / ∂P_j) * (P_j / Q_i)

Where:
- ε_ij > 0: Substitutes (Coke ↔ Pepsi)
- ε_ij < 0: Complements (Chips ↔ Dip)
- ε_ij ≈ 0: Independent

We estimate this empirically from Dunnhumby data.
```

**Implementation:**
```python
class CrossPriceElasticityEngine:
    """
    Estimate and apply cross-price elasticity effects
    
    Uses log-log regression on Dunnhumby data:
    log(Q_i,t) = α_i + β_i * log(P_i,t) + Σ_j γ_ij * log(P_j,t) + ε_t
    
    Where γ_ij is the cross-price elasticity between products i and j
    """
    
    def __init__(self, products_df: pd.DataFrame):
        self.products = products_df
        self.cross_elasticity_matrix = None
        self.substitute_groups = None
        self.complement_pairs = None
    
    def estimate_from_data(self, transactions_df: pd.DataFrame):
        """
        Estimate cross-price elasticity from Dunnhumby transactions
        
        Approach:
        1. Aggregate to product-week level
        2. For each product, identify potential substitutes/complements
        3. Run regression to estimate elasticities
        4. Build sparse elasticity matrix
        """
        print("Estimating cross-price elasticity matrix...")
        
        # Step 1: Product-week aggregation
        product_week = transactions_df.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
            'QUANTITY': 'sum',
            'SALES_VALUE': 'sum',
            'household_key': 'nunique'
        }).reset_index()
        
        product_week['price'] = product_week['SALES_VALUE'] / product_week['QUANTITY']
        product_week['quantity_per_customer'] = product_week['QUANTITY'] / product_week['household_key']
        
        # Step 2: Identify substitute candidates (same category)
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import lil_matrix
        
        n_products = len(self.products)
        product_id_to_idx = {pid: idx for idx, pid in enumerate(self.products['PRODUCT_ID'])}
        
        # Sparse matrix for elasticities
        elasticity_matrix = lil_matrix((n_products, n_products))
        
        # Step 3: Estimate elasticities by commodity category
        for commodity in tqdm(self.products['COMMODITY_DESC'].unique()):
            commodity_products = self.products[
                self.products['COMMODITY_DESC'] == commodity
            ]['PRODUCT_ID'].values
            
            if len(commodity_products) < 2:
                continue
            
            # Get data for these products
            commodity_data = product_week[
                product_week['PRODUCT_ID'].isin(commodity_products)
            ]
            
            # Pivot to wide format (products as columns)
            quantity_wide = commodity_data.pivot(
                index='WEEK_NO',
                columns='PRODUCT_ID',
                values='quantity_per_customer'
            ).fillna(0)
            
            price_wide = commodity_data.pivot(
                index='WEEK_NO',
                columns='PRODUCT_ID',
                values='price'
            ).fillna(method='ffill').fillna(method='bfill')
            
            # Log transformation
            log_quantity = np.log(quantity_wide + 1)
            log_price = np.log(price_wide + 0.01)
            
            # Estimate elasticities for each product
            for focal_product in commodity_products:
                if focal_product not in log_quantity.columns:
                    continue
                
                y = log_quantity[focal_product].values
                
                # Build feature matrix (own price + competitor prices)
                X_cols = []
                feature_products = []
                
                # Own price
                if focal_product in log_price.columns:
                    X_cols.append(log_price[focal_product].values)
                    feature_products.append(focal_product)
                
                # Competitor prices (top 5 by volume)
                competitors = [p for p in commodity_products if p != focal_product]
                competitor_volumes = quantity_wide[competitors].sum().sort_values(ascending=False)
                top_competitors = competitor_volumes.head(5).index
                
                for comp_product in top_competitors:
                    if comp_product in log_price.columns:
                        X_cols.append(log_price[comp_product].values)
                        feature_products.append(comp_product)
                
                if len(X_cols) < 2:  # Need at least own price + 1 competitor
                    continue
                
                X = np.column_stack(X_cols)
                
                # Regression with regularization
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                
                # Handle missing values
                valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                if valid_mask.sum() < 10:
                    continue
                
                model.fit(X[valid_mask], y[valid_mask])
                
                # Extract elasticities
                focal_idx = product_id_to_idx[focal_product]
                
                for i, feature_product in enumerate(feature_products):
                    elasticity = model.coef_[i]
                    feature_idx = product_id_to_idx[feature_product]
                    
                    # Only store if significant
                    if abs(elasticity) > 0.1:
                        elasticity_matrix[focal_idx, feature_idx] = elasticity
        
        # Convert to CSR for efficient computation
        self.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        print(f"✅ Estimated cross-price elasticity matrix")
        print(f"   Products: {n_products:,}")
        print(f"   Non-zero elasticities: {self.cross_elasticity_matrix.nnz:,}")
        print(f"   Sparsity: {1 - self.cross_elasticity_matrix.nnz / (n_products**2):.4f}")
        
        # Identify substitute/complement groups
        self._identify_product_relationships()
    
    def _identify_product_relationships(self):
        """Classify product pairs as substitutes or complements"""
        from scipy.sparse import find
        
        # Extract non-zero elasticities
        rows, cols, elasticities = find(self.cross_elasticity_matrix)
        
        # Classify
        substitute_pairs = []
        complement_pairs = []
        
        for i, j, elasticity in zip(rows, cols, elasticities):
            if i == j:  # Skip own-price
                continue
            
            if elasticity > 0.2:  # Substitutes
                substitute_pairs.append((
                    self.products.iloc[i]['PRODUCT_ID'],
                    self.products.iloc[j]['PRODUCT_ID'],
                    elasticity
                ))
            elif elasticity < -0.2:  # Complements
                complement_pairs.append((
                    self.products.iloc[i]['PRODUCT_ID'],
                    self.products.iloc[j]['PRODUCT_ID'],
                    elasticity
                ))
        
        self.substitute_groups = pd.DataFrame(
            substitute_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        )
        
        self.complement_pairs = pd.DataFrame(
            complement_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        )
        
        print(f"\nProduct relationships identified:")
        print(f"  Substitute pairs: {len(self.substitute_groups):,}")
        print(f"  Complement pairs: {len(self.complement_pairs):,}")
    
    def apply_cross_price_effects(self,
                                  focal_product_id: int,
                                  base_utility: float,
                                  current_prices: Dict[int, float],
                                  reference_prices: Dict[int, float]) -> float:
        """
        Adjust utility based on cross-price effects
        
        Args:
            focal_product_id: Product being evaluated
            base_utility: Base utility before cross-price adjustment
            current_prices: Current prices for all products
            reference_prices: Reference prices (e.g., average price)
        
        Returns:
            Adjusted utility
        """
        focal_idx = self.products[
            self.products['PRODUCT_ID'] == focal_product_id
        ].index[0]
        
        # Get cross-price elasticities for this product
        elasticities = self.cross_elasticity_matrix[focal_idx].toarray().flatten()
        
        # Calculate price change effects
        cross_price_adjustment = 0.0
        
        for j, elasticity in enumerate(elasticities):
            if abs(elasticity) < 0.01:  # Skip near-zero
                continue
            
            other_product_id = self.products.iloc[j]['PRODUCT_ID']
            
            if other_product_id in current_prices and other_product_id in reference_prices:
                # Price change percentage
                price_change_pct = (
                    current_prices[other_product_id] - reference_prices[other_product_id]
                ) / reference_prices[other_product_id]
                
                # Utility adjustment = elasticity * price_change
                cross_price_adjustment += elasticity * price_change_pct
        
        # Apply adjustment
        adjusted_utility = base_utility + cross_price_adjustment
        
        return adjusted_utility
```

#### Part C: Arc Price Elasticity (Intertemporal) (Days 11-12)

**Theoretical Foundation:**
```
Arc elasticity captures intertemporal substitution:
- Customers stock up when prices low (anticipate future high prices)
- Customers defer when prices high (anticipate future promotions)

Arc elasticity = Expected utility from buying now vs. later

Modeled as:
U(buy_now) = immediate_utility + β * E[future_utility | current_price]

Where β is discount factor and E[future_utility] depends on price expectations.
```

**Implementation:**
```python
class ArcPriceElasticityEngine:
    """
    Model intertemporal price effects (arc elasticity)
    
    Customers form expectations about future prices based on:
    1. HMM state (if in discount state, expect return to regular)
    2. Purchase history (inventory depletion rate)
    3. Seasonality (holiday promotions expected)
    """
    
    def __init__(self, price_hmm: PriceStateHMM):
        self.price_hmm = price_hmm
        self.inventory_decay_rate = 0.25  # 25% per week
        self.future_discount_factor = 0.95  # Weekly discount
    
    def calculate_arc_effect(self,
                            product_id: int,
                            current_price: float,
                            current_state: int,
                            customer_inventory: float,
                            weeks_since_purchase: int) -> float:
        """
        Calculate utility adjustment from arc elasticity
        
        Args:
            product_id: Product being evaluated
            current_price: Current price
            current_state: Current HMM price state (0, 1, 2)
            customer_inventory: Estimated inventory level (0-1)
            weeks_since_purchase: Weeks since last purchase
        
        Returns:
            Utility adjustment (positive = buy now, negative = defer)
        """
        
        # Component 1: Inventory urgency
        # As inventory depletes, urgency increases
        inventory_urgency = -customer_inventory  # Low inventory = high urgency
        
        # Component 2: Price expectation
        # Predict future price based on HMM
        expected_future_price = self._predict_future_price(
            product_id,
            current_price,
            current_state
        )
        
        # Price difference (current vs. expected future)
        price_advantage = (expected_future_price - current_price) / current_price
        
        # Component 3: Stockpiling incentive
        # If current price much lower than expected, incentive to buy extra
        stockpile_bonus = 0.0
        if price_advantage > 0.15:  # Future price expected 15%+ higher
            stockpile_bonus = 0.5 * price_advantage
        
        # Component 4: Deferral if high price and low urgency
        deferral_penalty = 0.0
        if price_advantage < -0.10 and customer_inventory > 0.3:
            # Current price high, inventory OK → defer
            deferral_penalty = -0.3
        
        # Total arc effect
        arc_adjustment = (
            inventory_urgency * 2.0 +           # Inventory most important
            price_advantage * 1.5 +              # Price expectations
            stockpile_bonus +                    # Stockpiling behavior
            deferral_penalty                     # Deferral behavior
        )
        
        return arc_adjustment
    
    def _predict_future_price(self,
                             product_id: int,
                             current_price: float,
                             current_state: int,
                             horizon: int = 4) -> float:
        """
        Predict expected price in 'horizon' weeks using HMM
        
        Uses transition matrix to compute expected future state,
        then emission distribution for that state.
        """
        if product_id not in self.price_hmm.transition_matrices:
            # Default expectation: return to regular price
            return current_price * 1.10
        
        transition_matrix = self.price_hmm.transition_matrices[product_id]
        emission_dists = self.price_hmm.emission_distributions[product_id]
        
        # Compute state distribution after 'horizon' steps
        # P(state_t+horizon | state_t) = transition_matrix ^ horizon
        state_probs = np.zeros(3)
        state_probs[current_state] = 1.0
        
        for _ in range(horizon):
            state_probs = state_probs @ transition_matrix
        
        # Expected price = weighted average across states
        expected_price = 0.0
        for state in [0, 1, 2]:
            state_price = emission_dists[state]['price_mean']
            expected_price += state_probs[state] * state_price
        
        return expected_price
    
    def update_customer_inventory(self,
                                  customer_id: int,
                                  product_id: int,
                                  quantity_purchased: int,
                                  current_inventory: float) -> float:
        """
        Update customer's estimated inventory after purchase
        
        Simple model:
        inventory_new = min(1.0, current_inventory * decay + quantity_normalized)
        """
        # Normalize quantity (assume 1 unit = 1 week consumption)
        quantity_normalized = quantity_purchased / 1.0
        
        # Decay current inventory
        inventory_after_decay = current_inventory * (1 - self.inventory_decay_rate)
        
        # Add purchase
        new_inventory = min(1.0, inventory_after_decay + quantity_normalized)
        
        return new_inventory
```

#### Part D: Integration into Choice Model (Day 12)

```python
class EnhancedUtilityEngine:
    """
    Utility calculation with HMM price states, cross-price, and arc elasticity
    """
    
    def __init__(self,
                 price_hmm: PriceStateHMM,
                 cross_price_engine: CrossPriceElasticityEngine,
                 arc_elasticity_engine: ArcPriceElasticityEngine):
        self.price_hmm = price_hmm
        self.cross_price = cross_price_engine
        self.arc_elasticity = arc_elasticity_engine
    
    def compute_product_utilities(self,
                                  customer: Customer,
                                  products: pd.DataFrame,
                                  current_prices: Dict[int, float],
                                  price_states: Dict[int, int],
                                  customer_inventories: Dict[int, float],
                                  weeks_since_purchase: Dict[int, int]) -> np.ndarray:
        """
        Compute utilities for all products with full price elasticity
        
        Returns:
            Array of utilities (one per product)
        """
        n_products = len(products)
        utilities = np.zeros(n_products)
        
        # Reference prices (for cross-price calculations)
        reference_prices = {
            pid: products[products['PRODUCT_ID'] == pid]['avg_price'].values[0]
            for pid in products['PRODUCT_ID']
        }
        
        for i, product in products.iterrows():
            product_id = product['PRODUCT_ID']
            price = current_prices[product_id]
            
            # Base utility (without price effects)
            base_utility = (
                customer.β_quality * product.get('quality_score', 0.5) +
                customer.β_brand * customer.brand_preferences.get(product['BRAND'], 0.5) +
                customer.β_role * product.get('role_match', 0.5)
            )
            
            # Own-price effect (log-linear)
            own_price_effect = customer.β_price * np.log(price + 0.01)
            
            # Cross-price effects
            cross_price_adjustment = self.cross_price.apply_cross_price_effects(
                focal_product_id=product_id,
                base_utility=base_utility,
                current_prices=current_prices,
                reference_prices=reference_prices
            )
            
            # Arc elasticity (intertemporal)
            arc_adjustment = self.arc_elasticity.calculate_arc_effect(
                product_id=product_id,
                current_price=price,
                current_state=price_states.get(product_id, 0),
                customer_inventory=customer_inventories.get(product_id, 0.0),
                weeks_since_purchase=weeks_since_purchase.get(product_id, 999)
            )
            
            # Promotional bonus (if in feature/discount state)
            promo_bonus = 0.0
            if price_states.get(product_id, 0) > 0:
                promo_bonus = customer.β_promotion * 0.5
            
            # Total utility
            utilities[i] = (
                base_utility +
                own_price_effect +
                cross_price_adjustment +
                arc_adjustment +
                promo_bonus
            )
        
        return utilities
```

**Files to Create:**
- `src/retailsynth/engines/price_hmm.py` - HMM implementation
- `src/retailsynth/engines/cross_price_elasticity.py` - Cross-price effects
- `src/retailsynth/engines/arc_elasticity.py` - Intertemporal effects
- `src/retailsynth/engines/enhanced_utility_engine.py` - Integrated utility
- `scripts/learn_price_elasticity.py` - Learn from Dunnhumby

**Files to Modify:**
- `src/retailsynth/generators/transaction_generator.py` - Use enhanced utility
- `src/retailsynth/config.py` - Add elasticity parameters

**Dependencies:**
- Dunnhumby transaction_data.csv
- Dunnhumby causal_data.csv (promotional flags)
- scikit-learn (for regression)
- scipy (for sparse matrices)

**Validation Tests:**
```python
def test_price_hmm_states():
    """Test HMM generates realistic price sequences"""
    hmm = PriceStateHMM(products_df)
    hmm.learn_from_data(transactions_df, causal_df)
    
    # Generate sequence
    prices, states = hmm.generate_price_sequence(
        product_id=12345,
        n_weeks=52,
        base_price=5.99
    )
    
    # Check state frequencies match learned distribution
    state_freq = np.bincount(states, minlength=3) / len(states)
    expected_freq = hmm.initial_state_probs[12345]
    
    for s in [0, 1, 2]:
        assert abs(state_freq[s] - expected_freq.get(s, 0.33)) < 0.15

def test_cross_price_substitution():
    """Test that Coke price increase boosts Pepsi demand"""
    engine = CrossPriceElasticityEngine(products_df)
    engine.estimate_from_data(transactions_df)
    
    # Get Coke and Pepsi product IDs
    coke_id = get_product_id('Coca Cola')
    pepsi_id = get_product_id('Pepsi')
    
    # Check cross-elasticity is positive (substitutes)
    coke_idx = products_df[products_df['PRODUCT_ID'] == coke_id].index[0]
    pepsi_idx = products_df[products_df['PRODUCT_ID'] == pepsi_id].index[0]
    
    cross_elasticity = engine.cross_elasticity_matrix[pepsi_idx, coke_idx]
    assert cross_elasticity > 0.1, "Coke and Pepsi should be substitutes"

def test_arc_elasticity_stockpiling():
    """Test stockpiling behavior when prices low"""
    arc_engine = ArcPriceElasticityEngine(price_hmm)
    
    # Scenario: Deep discount state, low inventory
    arc_effect = arc_engine.calculate_arc_effect(
        product_id=12345,
        current_price=3.99,  # Low price
        current_state=2,     # Deep discount
        customer_inventory=0.1,  # Low inventory
        weeks_since_purchase=3
    )
    
    # Should encourage purchase (positive adjustment)
    assert arc_effect > 0.5, "Should encourage stockpiling at low prices"

def test_arc_elasticity_deferral():
    """Test deferral behavior when prices high"""
    arc_engine = ArcPriceElasticityEngine(price_hmm)
    
    # Scenario: High price, sufficient inventory
    arc_effect = arc_engine.calculate_arc_effect(
        product_id=12345,
        current_price=7.99,  # High price
        current_state=0,     # Regular state
        customer_inventory=0.6,  # Sufficient inventory
        weeks_since_purchase=1
    )
    
    # Should discourage purchase (negative adjustment)
    assert arc_effect < -0.2, "Should discourage purchase at high prices with inventory"
```

**Success Criteria:**
- ✅ HMM learned from Dunnhumby promotional data
- ✅ State transition matrices estimated per product/category
- ✅ Cross-price elasticity matrix computed (20K × 20K sparse)
- ✅ Substitute pairs identified (cross-elasticity > 0.2)
- ✅ Complement pairs identified (cross-elasticity < -0.2)
- ✅ Arc elasticity model captures stockpiling behavior
- ✅ Arc elasticity model captures deferral behavior
- ✅ Integrated utility function uses all three elasticity components
- ✅ Validation: Price increases for Coke boost Pepsi demand (test with real data patterns)

---

### Issue #2: No Purchase History / State Dependence ⚠️ CRITICAL
**Priority:** P0 (Sprint 1.3 - Days 13-17)
**Effort:** 5-7 days
**Impact:** Very High

**Current Behavior:**
```python
# Each week generated independently
for week in weeks:
    transactions = generate_week_transactions(week)  # No history used
```

**Problem:**
- No brand loyalty from past purchases
- No inventory depletion cycles
- No habit formation
- Unrealistic customer trajectories

**Solution Required:**
```python
# Customer state tracking
class CustomerState:
    last_purchases: Dict[int, int]  # product_id -> weeks_since
    brand_experience: Dict[str, float]  # brand -> cumulative satisfaction
    category_inventory: Dict[str, float]  # category -> estimated stock
    purchase_frequency: Dict[int, float]  # product -> expected frequency

# Utility incorporates history
utility[product_i] = base_utility[i] + 
                     α * loyalty_bonus[i] + 
                     β * inventory_need[category] +
                     γ * habit_strength[i]
```

**Files to Create:**
- `customer_state.py` - State tracking
- `history_engine.py` - Historical effects
- `loyalty_model.py` - Brand loyalty dynamics

**Files to Modify:**
- `transaction_generator.py` - Add state updates
- `main_generator.py` - Initialize/maintain state

**Dependencies:**
- Customer state persistence across weeks
- Brand-product mapping
- Category-product mapping

**Validation Test:**
- Repeat purchase rate
- Brand loyalty metrics
- Inter-purchase timing distributions

---

### Issue #3: No Basket Composition Logic ⚠️ CRITICAL
**Priority:** P0 (Sprint 1.4 - Days 18-22)
**Effort:** 4-5 days
**Impact:** Very High

**Current Behavior:**
```python
# 4 discrete customer types
price_anchor_customers: 0.25
convenience_customers: 0.25
planned_customers: 0.30
impulse_customers: 0.20
```

**Problem:**
- Only 4 discrete types
- Within-type homogeneity
- Cannot capture mixed behaviors
- Real customers are continuous spectrum

**Solution Required:**
```python
# Individual-level parameters from distributions
class Customer:
    β_price ~ Normal(μ=-2.5, σ=0.8)  # Each customer unique
    β_quality ~ Lognormal(μ=1.2, σ=0.5)
    β_convenience ~ Gamma(α=2, β=3)
    # etc. for all parameters

# Bayesian hierarchical model
μ_population ~ prior
σ_population ~ prior
β_individual ~ Normal(μ_population, σ_population)
```

**Files to Create:**
- `customer_heterogeneity.py` - Individual parameter generation
- `bayesian_priors.py` - Prior distributions

**Files to Modify:**
- `customer_generator.py` - Generate from distributions
- `config.py` - Add distribution parameters

**Dependencies:**
- NumPyro or PyMC for Bayesian sampling
- Calibration data for prior parameters

**Validation Test:**
- Parameter distribution shapes
- Within-group variance
- Behavioral diversity metrics

---

### Issue #4: No Basket Composition Logic ⚠️ CRITICAL
**Priority:** P0 (Must Fix Second - Depends on Product Catalog)
**Effort:** 2-3 days
**Impact:** Very High

**Current Behavior:**
```python
# Products sampled independently based on utility
chosen_products = sample_from_utilities(all_utilities, n_products)
```

**Problem:**
- Nonsensical baskets (5 milks, no other items)
- No meal planning
- No trip purpose structure
- Missing category constraints

**Solution Required:**
```python
# Trip purpose framework
class TripPurpose:
    STOCK_UP = "stock_up"      # Large basket, staples focus
    FILL_IN = "fill_in"        # Small basket, specific needs
    MEAL_PREP = "meal_prep"    # Recipe-based basket
    CONVENIENCE = "convenience" # Grab-and-go

# Basket generation with constraints
def generate_basket(customer, trip_purpose):
    # 1. Determine must-have categories for trip type
    required_categories = get_required_categories(trip_purpose)
    
    # 2. Sample products within category constraints
    basket = {}
    for category in required_categories:
        basket[category] = choose_products_in_category(customer, category)
    
    # 3. Add complementary items
    basket.update(add_complements(basket, customer))
    
    return basket
```

**Files to Create:**
- `trip_purpose.py` - Trip type taxonomy
- `basket_composer.py` - Basket generation logic
- `category_constraints.py` - Must-have rules

**Files to Modify:**
- `transaction_generator.py` - Use basket composer
- `customer_generator.py` - Add trip purpose preferences

**Dependencies:**
- Category hierarchy
- Product-category mapping
- Complement/substitute relationships

**Validation Test:**
- Basket coherence scores
- Category co-occurrence patterns
- Trip type distributions

---

### Issue #5: Archetype-Based Not Individual Heterogeneity ⚠️ HIGH
**Priority:** P1 (Week 2)
**Effort:** 2-3 days
**Impact:** Medium

**Current Behavior:**
```python
# Manual seasonality definitions
'Thanksgiving': {'start_week': 47, 'duration': 2, 'intensity': 1.8}
```

**Problem:**
- Generic not product-specific
- No data validation
- Missing category-season interactions

**Solution Required:**
```python
# Extract from Dunnhumby data
seasonal_patterns = learn_seasonality_from_data(
    transactions_df,
    groupby=['product_id', 'category', 'week_of_year']
)

# Apply learned patterns
def get_seasonal_multiplier(product_id, week_of_year):
    return seasonal_patterns[product_id][week_of_year]
```

**Files to Create:**
- `seasonality_learner.py` - Extract from data
- `seasonal_patterns.json` - Learned patterns

**Files to Modify:**
- `seasonality_engine.py` - Use learned patterns

**Validation Test:**
- Seasonal pattern correlation with real data
- Product-specific seasonality accuracy

---

### Issue #6: Random Promotional Response ⚠️ HIGH
**Priority:** P1 (Week 2)
**Effort:** 3-4 days
**Impact:** High

**Current Behavior:**
```python
# Random promotions
promo_indices = np.random.choice(n_products, size=n_promotions)
```

**Problem:**
- No strategic promotions
- No customer-specific response
- No temporal patterns

**Solution Required:**
```python
# HMM for promotion states (following RetailSynth paper)
class PromotionalEngine:
    def __init__(self):
        self.states = ['regular', 'featured', 'deep_discount']
        self.transition_matrix = learn_from_data()
    
    def generate_promotions(self, week, products):
        # State transition
        current_states = self.transition_states(week)
        
        # Customer-specific response
        for customer in customers:
            promo_sensitivity = customer.β_promotion
            response_prob = promotional_lift(promo_sensitivity, state)
```

**Files to Create:**
- `promotional_engine.py` - HMM implementation
- `promotion_response.py` - Customer response

**Files to Modify:**
- `pricing_engine.py` - Use promotional engine

**Validation Test:**
- Promotion frequency distribution
- Lift measurement accuracy
- Customer response heterogeneity

---

### Issue #7: No Geographic Clustering ⚠️ MEDIUM
**Priority:** P2 (Week 4)
**Effort:** 2-3 days
**Impact:** Medium

**Solution Required:**
```python
# Geographic structure
class CustomerGeography:
    latitude: float
    longitude: float
    neighborhood_type: str  # urban, suburban, rural
    income_cluster: str
    
# Store assortment by geography
class Store:
    location: Tuple[float, float]
    neighborhood_profile: NeighborhoodProfile
    assortment: Dict[str, List[int]]  # Varies by store
```

**Files to Create:**
- `geography.py`
- `store_assortment.py`

---

### Issue #8: Linear Utilities ⚠️ MEDIUM
**Priority:** P2 (Week 3)
**Effort:** 2 days
**Impact:** Medium

**Solution Required:**
```python
# Non-linear transformations
utility = (β_price * log(price) +  # Log instead of linear
           β_quality * quality**2 +  # Quadratic for saturation
           β_brand * indicator(preferred_brand))
```

---

### Issue #9: Fixed Basket Sizes ⚠️ LOW
**Priority:** P3 (Week 4)
**Effort:** 1 day
**Impact:** Low

**Solution Required:**
```python
# Trip-purpose based sizes
basket_size ~ NegativeBinomial(mu=trip_purpose.mean_size, dispersion=2.0)
```

---

### Issue #10: No Store Differentiation ⚠️ MEDIUM
**Priority:** P2 (Week 4)
**Effort:** 2 days
**Impact:** Medium

**Solution Required:**
- Store-specific assortments
- Store-specific pricing
- Store quality attributes

---

### Issue #11: No Quantity Model ⚠️ LOW
**Priority:** P3 (Week 5)
**Effort:** 1-2 days
**Impact:** Low

**Solution Required:**
```python
# Quantity depends on context
quantity = model(
    household_size=customer.household_size,
    promotion=is_promoted,
    price_per_unit=unit_price,
    product_type=product.package_type
)
```

---

### Issue #12: No Customer Entry/Exit ⚠️ LOW
**Priority:** P3 (Week 6)
**Effort:** 1 day
**Impact:** Low

**Solution Required:**
- Customer birth/death process
- Entry rate = f(market_growth)
- Exit rate = f(churn_model)

---

## 5. Product Roadmap

### Phase 0: Setup & Planning (Week 0) - CURRENT
**Duration:** 3-5 days
**Owner:** Human + Claude

**Deliverables:**
- [x] Current state assessment
- [x] Issue documentation
- [x] Roadmap creation
- [ ] Development environment setup
- [ ] Repository structure
- [ ] CI/CD pipeline
- [ ] Project management board
- [ ] Download Dunnhumby data

---

### Phase 1: Critical Fixes (Weeks 1-3) - REVISED
**Duration:** 15-21 days
**Owner:** Shared (Claude implements, Human tests)

**Goal:** Fix the 4 must-have issues that block validation

**Sprint 1.1: Product Catalog Alignment (Days 1-5)**
- Extract 20K SKUs from Dunnhumby (92K products)
- Create representative sample (stratified by frequency/category)
- Map category hierarchy to Dunnhumby structure
- Define product archetypes (price tier × frequency × role)
- Test: Category coverage, brand coverage, frequency distribution
- **Blocking issue - Must complete first**

**Sprint 1.2: Price & Cross-Price Elasticity with HMM (Days 6-12)**
- Learn HMM from Dunnhumby promotional data
- Estimate state transition matrices (per product)
- Estimate cross-price elasticity matrix (20K × 20K sparse)
- Implement arc price elasticity (intertemporal)
- Integrate into utility engine
- Test: State transitions, substitution patterns, stockpiling behavior

**Sprint 1.3: Purchase History (Days 13-17)**
- Implement customer state tracking
- Add history-dependent utilities
- Create loyalty model
- Test: Repeat purchase patterns

**Sprint 1.4: Basket Composition (Days 18-22)**
- Implement trip purpose framework
- Add basket composition rules
- Create category constraints
- Test: Basket coherence

**Deliverables:**
- ✅ 20K real product catalog from Dunnhumby
- ✅ HMM price dynamics engine
- ✅ Cross-price elasticity matrix
- ✅ Arc elasticity model
- ✅ Purchase history system
- ✅ Basket composition logic
- ✅ Unit tests for all 4 components
- ✅ Integration tests
- ⚠️ Expected validation: ~65% pass rate (up from 40%)

**Key Changes from v1.0:**
- Product catalog moved to Sprint 1.1 (FIRST priority)
- Price elasticity expanded to include HMM, cross-price, and arc elasticity
- Sprint 1.2 extended from 4 days to 7 days (more complex)
- Phase 1 extended from 2 weeks to 3 weeks

---

### Phase 2: Behavioral Improvements (Weeks 4-5)
**Duration:** 10-14 days
**Owner:** Shared

**Goal:** Add econometric sophistication

**Sprint 2.1: Individual Heterogeneity (Days 1-4)**
- Replace archetypes with distributions
- Implement Bayesian parameter generation
- Add hierarchical model
- Test: Parameter distributions

**Sprint 2.2: Promotional Response (Days 5-7)**
- Implement HMM for promotions
- Add customer-specific response
- Create promotional lift model
- Test: Promotion patterns

**Sprint 2.3: Non-Linear Utilities (Days 8-10)**
- Add log/quadratic transformations
- Implement reference prices
- Add threshold effects
- Test: Price response curves

**Sprint 2.4: Seasonality Learning (Days 11-14)**
- Extract patterns from Dunnhumby
- Replace hard-coded seasonality
- Add product-specific patterns
- Test: Seasonal correlation

**Deliverables:**
- ✅ Individual-level parameters
- ✅ Promotional engine (builds on HMM from Phase 1)
- ✅ Non-linear utilities
- ✅ Data-driven seasonality
- ⚠️ Expected validation: ~75% pass rate

---

### Phase 3: Calibration & Validation (Weeks 6-7)
**Duration:** 10-14 days
**Owner:** Shared

**Goal:** Achieve 80% validation pass rate

**Sprint 3.1: Data Preparation (Days 1-2)**
- Preprocess Dunnhumby data
- Extract calibration targets
- Create validation datasets

**Sprint 3.2: Manual Calibration (Days 3-6)**
- Grid search key parameters
- Visual validation
- Iterative refinement

**Sprint 3.3: Automated Optimization (Days 7-10)**
- Set up Optuna
- Define objective function
- Run 100+ trials
- Select best parameters

**Sprint 3.4: Full Validation (Days 11-14)**
- Run all 4 validation levels
- Generate validation report
- Iterative refinement
- Final parameter lock


**Updated Validation Metrics:**

**Level 1: Distribution Matching (Target: 90%)**
- Visit frequency: KS p-value > 0.05
- Basket size: KS p-value > 0.05
- Basket value: KS p-value > 0.05
- **Product popularity: KS p-value > 0.05** ← New with real catalog
- **Price distribution: KS p-value > 0.05** ← New with HMM

**Level 2: Aggregate Statistics (Target: 85%)**
- All existing metrics
- **Own-price elasticity error < 15%** ← New
- **Cross-price elasticity sign accuracy > 80%** ← New

**Level 3: Behavioral Patterns (Target: 75%)**
- All existing metrics
- **Substitution pattern overlap > 40%** ← New
- **Stockpiling behavior present** ← New with arc elasticity

**Level 4: Predictive Accuracy (Target: 80%)**
- All existing metrics
- **Price elasticity predictions within 20% of real** ← New

**Deliverables:**
- ✅ Calibrated parameters
- ✅ Validation report (80%+ pass rate)
- ✅ Comparison visualizations
- ✅ Limitations documentation

---

### Phase 4: Polish & Documentation (Weeks 7-8)
**Duration:** 10-14 days
**Owner:** Shared

**Goal:** Make production-ready

**Sprint 4.1: Code Quality (Days 1-4)**
- Refactor monolithic code
- Add comprehensive tests
- Improve documentation
- Code review

**Sprint 4.2: Usability (Days 5-8)**
- Create CLI interface
- Add configuration system
- Write example notebooks
- User testing

**Sprint 4.3: Documentation (Days 9-12)**
- API documentation
- User guide
- Technical paper draft
- GitHub README

**Sprint 4.4: Release (Days 13-14)**
- Final testing
- Create release package
- Publish to GitHub
- Write announcement

**Deliverables:**
- ✅ Clean, modular codebase
- ✅ Comprehensive documentation
- ✅ Example notebooks
- ✅ v1.0 release

---

## 6. Development Workflow

### Tools & Setup

**Required Tools:**
```bash
# Core development
- Python 3.10+
- Git
- VS Code or PyCharm

# Claude Code integration
- Claude Code CLI
- Project directory structure

# Python packages
- NumPy, SciPy, Pandas
- JAX (GPU support)
- NumPyro (Bayesian)
- Optuna (optimization)
- Pytest (testing)
- Sphinx (docs)
```

**Environment Setup:**
```bash
# 1. Clone repository
git clone https://github.com/[your-org]/retailsynth-enhanced.git
cd retailsynth-enhanced

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Configure Claude Code
claude-code init
# Follow prompts to set up project

# 5. Verify setup
pytest tests/
python -c "import jax; print(jax.devices())"
```

### Collaboration Workflow

**Daily Iteration Cycle:**

```
Human Morning:
1. Review Claude's overnight work
2. Test new implementations
3. Provide feedback via issues/comments
4. Define next tasks in project board

Claude Code Session:
5. Pick highest priority task from board
6. Implement solution
7. Write tests
8. Document changes
9. Create pull request

Human Evening:
10. Review PR
11. Run validation tests
12. Merge or request changes
13. Update roadmap if needed
```

**Communication Protocol:**

1. **Issue Creation**
   - Human creates issues for bugs/features
   - Use templates (Bug, Feature, Enhancement)
   - Assign priority labels (P0, P1, P2, P3)
   - Tag with phase (Phase1, Phase2, etc.)

2. **Claude Code Instructions**
   ```markdown
   # Clear Task Format
   
   **Task:** [Brief description]
   
   **Context:** 
   - Why this is needed
   - Related issues
   - Dependencies
   
   **Requirements:**
   - [ ] Specific requirement 1
   - [ ] Specific requirement 2
   
   **Files to Modify:**
   - `path/to/file.py` - What changes
   
   **Tests Required:**
   - Test scenario 1
   - Test scenario 2
   
   **Validation:**
   How to verify it works
   ```

3. **Code Review Checklist**
   - [ ] Functionality works as specified
   - [ ] Tests pass (pytest)
   - [ ] Documentation added/updated
   - [ ] Type hints present
   - [ ] No performance regression
   - [ ] Follows style guide (black, flake8)

### Version Control

**Branch Strategy:**
```
main
├── develop (integration branch)
├── feature/purchase-history
├── feature/cross-price-elasticity
├── feature/basket-composition
└── hotfix/critical-bug
```

**Commit Messages:**
```
feat: Add purchase history tracking to customer state
fix: Correct utility calculation in cross-price model
docs: Update API documentation for basket composer
test: Add unit tests for promotional engine
refactor: Split monolithic file into modules
```

**Pull Request Template:**
```markdown
## Description
Brief description of changes

## Related Issues
Closes #123, #456

## Changes Made
- Change 1
- Change 2

## Tests Added
- Test 1
- Test 2

## Validation Results
[Paste test output or validation metrics]

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes
```

---

## 7. Repository Structure

```
retailsynth-enhanced/
│
├── README.md                          # Project overview
├── LICENSE                            # MIT or Apache 2.0
├── pyproject.toml                     # Project metadata
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
├── .gitignore
├── .github/
│   ├── workflows/
│   │   ├── tests.yml                  # CI: Run tests on PR
│   │   ├── lint.yml                   # CI: Code quality checks
│   │   └── docs.yml                   # CI: Build documentation
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── docs/                              # Documentation
│   ├── conf.py                        # Sphinx config
│   ├── index.rst                      # Docs home
│   ├── api/                           # API reference
│   ├── guides/                        # User guides
│   │   ├── quickstart.md
│   │   ├── calibration.md
│   │   └── validation.md
│   └── research/                      # Research docs
│       ├── methodology.md
│       ├── validation_report.md
│       └── limitations.md
│
├── data/                              # Data (gitignored, except samples)
│   ├── raw/                           # Dunnhumby data
│   │   └── dunnhumby/                 # Dunnhumby Complete Journey
│   │       ├── product.csv            # 92K products ← NEW
│   │       ├── transaction_data.csv
│   │       ├── hh_demographic.csv
│   │       ├── causal_data.csv        # Promotional flags ← NEW
│   │       └── ...
│   ├── processed/
│   │   ├── product_catalog/           # Product Catalog
│   │   │   ├── product_catalog_20k.parquet
│   │   │   ├── category_hierarchy.json
│   │   │   ├── product_archetypes.csv
│   │   │   └── hmm_parameters.pkl     # Hidden Markov Model Parameters
│   │   ├── elasticity/                # Elasticity
│   │   │   ├── cross_price_matrix.npz
│   │   │   ├── substitute_pairs.csv
│   │   │   └── complement_pairs.csv
│   │   └── calibration/
│   │       └── targets.npy
│   └── samples/                       # Sample outputs (in repo)
│
├── src/
│   └── retailsynth/                   # Main package
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── catalog/                   # Product Catalog Builder
│       │   ├── __init__.py
│       │   ├── product_catalog_builder.py
│       │   ├── hierarchy_mapper.py
│       │   └── archetype_classifier.py
│       ├── models/                    # Core models
│       │   ├── __init__.py
│       │   ├── customer.py            # Customer class
│       │   ├── product.py             # Product class
│       │   ├── store.py               # Store class
│       │   └── transaction.py         # Transaction class
│       ├── generators/                # Data generation
│       │   ├── __init__.py
│       │   ├── customer_generator.py
│       │   ├── product_generator.py
│       │   ├── transaction_generator.py
│       │   └── basket_composer.py     # NEW: Basket logic
│       ├── engines/                   # Behavioral engines
│       │   ├── __init__.py
│       │   ├── utility_engine.py      # Utility calculations
│       │   ├── choice_engine.py       # Choice models
│       │   ├── loyalty_engine.py      # Loyalty/history
│       │   ├── promotional_engine.py  # Promotions
│       │   ├── seasonality_engine.py
│       │   └── lifecycle_engine.py
│       │   ├── price_hmm.py           # Pricing Engine (Hidden Markov Model)
│       │   ├── cross_price_elasticity.py  # Cross Price Elasticity
│       │   ├── arc_elasticity.py      # Arc Elasticity
│       │   ├── enhanced_utility_engine.py  # Enhanced Utility Engine
│       │   └── ...
│       ├── calibration/               # Calibration system
│       │   ├── __init__.py
│       │   ├── preprocessor.py        # Dunnhumby preprocessing
│       │   ├── calibrator.py          # Manual calibration
│       │   ├── optimizer.py           # Optuna optimization
│       │   └── targets.py             # Target distributions
│       ├── validation/                # Validation system
│       │   ├── __init__.py
│       │   ├── validator.py           # Main validator
│       │   ├── distributions.py       # Level 1 tests
│       │   ├── aggregates.py          # Level 2 tests
│       │   ├── behavior.py            # Level 3 tests
│       │   └── predictive.py          # Level 4 tests
│       ├── utils/                     # Utilities
│       │   ├── __init__.py
│       │   ├── distributions.py       # Distribution helpers
│       │   ├── metrics.py            # Metric calculations
│       │   └── visualization.py       # Plotting helpers
│       └── cli.py                     # Command-line interface
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── unit/                          # Unit tests
│   │   ├── test_customer.py
│   │   ├── test_product.py
│   │   ├── test_utility_engine.py
│   │   ├── test_choice_engine.py
│   │   └── test_basket_composer.py
│   │   ├── test_product_catalog_builder.py
│   │   ├── test_price_hmm.py
│   │   ├── test_cross_elasticity.py
│   │   └── test_arc_elasticity.py
│   ├── integration/                   # Integration tests
│   │   ├── test_generation_pipeline.py
│   │   ├── test_calibration_workflow.py
│   │   └── test_validation_suite.py
│   └── validation/                    # Validation tests
│       ├── test_level1_distributions.py
│       ├── test_level2_aggregates.py
│       ├── test_level3_behavior.py
│       └── test_level4_predictive.py
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_calibration_demo.ipynb
│   ├── 04_validation_analysis.ipynb
│   └── 05_use_cases.ipynb
│
├── scripts/                           # Utility scripts
│   ├── download_dunnhumby.sh
│   ├── preprocess_data.py
│   ├── run_calibration.py
│   ├── run_validation.py
│   └── generate_synthetic.py
│   ├── build_product_catalog.py      # Product Catalog Builder
│   ├── learn_price_elasticity.py     # Price Elasticity Learner
│
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default config
│   ├── calibrated.yaml                # Calibrated params
│   ├── small_test.yaml                # Small test config
│   └── production.yaml                # Production config
│
└── outputs/                           # Generated outputs (gitignored)
    ├── synthetic_data/
    ├── validation_reports/
    ├── visualizations/
    └── models/
```

---


## 8. Technical Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RetailSynth Enhanced                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Configuration Layer                     │  │
│  │  - YAML configs                                       │  │
│  │  - Calibrated parameters                              │  │
│  │  - Runtime settings                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Data Generation Pipeline                 │  │
│  │                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐   ┌─────────────┐ │  │
│  │  │  Customer    │  │   Product    │   │    Store    │ │  │
│  │  │  Generator   │  │   Generator  │   │  Generator  │ │  │
│  │  └──────────────┘  └──────────────┘   └─────────────┘ │  │
│  │         │                  │                │         │  │
│  │         └──────────────────┴────────────────┘         │  │
│  │                            │                          │  │
│  │                  ┌─────────▼─────────┐                │  │
│  │                  │  Precomputation   │                │  │
│  │                  │  Engine (JAX/GPU) │                │  │
│  │                  └─────────┬─────────┘                │  │
│  │                            │                          │  │
│  │         ┌──────────────────┴─────────────────┐        │  │
│  │         │   Transaction Generation Loop      │        │  │
│  │         │   (Weekly Iterations)              │        │  │
│  │         │                                    │        │  │
│  │         │  For each week:                    │        │  │
│  │         │  1. Customer state update          │        │  │
│  │         │  2. Store visit decisions          │        │  │
│  │         │  3. Trip purpose selection         │        │  │
│  │         │  4. Basket composition             │        │  │
│  │         │  5. Product choices (with history) │        │  │
│  │         │  6. Quantity decisions             │        │  │
│  │         │  7. State persistence              │        │  │
│  │         └────────────────────────────────────┘        │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│  ┌───────────────────────────────── ─────────────────────┐  │
│  │              Behavioral Engines                       │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │   Utility   │  │    Choice    │  │   Loyalty    │  │  │
│  │  │   Engine    │  │    Engine    │  │   Engine     │  │  │
│  │  └─────────────┘  └──────────────┘  └──────────────┘  │  │
│  │                                                       │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │Promotional  │  │  Seasonality │  │  Lifecycle   │  │  │
│  │  │   Engine    │  │    Engine    │  │   Engine     │  │  │
│  │  └─────────────┘  └──────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Output Data Products                     │  │
│  │                                                       │  │
│  │  - Transactions (parquet)                             │  │
│  │  - Customers (parquet)                                │  │
│  │  - Products (parquet)                                 │  │
│  │  - Business metrics (parquet)                         │  │
│  │  - Validation reports (json/html)                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│             Calibration & Validation System                 │
│                                                             │
│  ┌──────────────┐       ┌──────────────┐                    │
│  │  Dunnhumby   │──────>│ Preprocessor │                    │
│  │     Data     │       └─────┬────────┘                    │
│  └──────────────┘             │                             │
│                               │                             │
│                        ┌──────▼─────────┐                   │
│                        │  Calibration   │                   │
│                        │   Targets      │                   │
│                        └──────┬─────────┘                   │
│                               │                             │
│                   ┌───────────┴──────────────┐              │
│                   │                          │              │
│          ┌────────▼────────┐         ┌───────▼────────┐     │
│          │     Manual      │         │   Automated    │     │
│          │   Calibration   │         │  Optimization  │     │
│          │                 │         │   (Optuna)     │     │
│          └────────┬────────┘         └───────┬────────┘     │
│                   │                          │              │
│                   └────────────┬─────────────┘              │
│                                │                            │
│                        ┌───────▼────────┐                   │
│                        │   Validator    │                   │
│                        │                │                   │
│                        │  Level 1-4     │                   │
│                        │  Tests         │                   │
│                        └────────┬───────┘                   │
│                                 │                           │
│                        ┌────────▼───────┐                   │
│                        │   Validation   │                   │
│                        │     Report     │                   │
│                        └────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Modularity**
   - Each component is independently testable
   - Clear interfaces between components
   - Easy to swap implementations

2. **Performance**
   - JAX for GPU acceleration where beneficial
   - Vectorized operations
   - Efficient memory usage
   - Batch processing

3. **Extensibility**
   - Plugin architecture for new engines
   - Easy to add new validation tests
   - Configuration-driven behavior

4. **Reproducibility**
   - Fixed random seeds
   - Version-controlled configs
   - Deterministic outputs

5. **Validation-First**
   - Every feature has validation test
   - Continuous validation in CI
   - Regression detection

---

## 9. Testing & Validation Strategy

### Testing Pyramid

```
                    /\
                   /  \
                  /    \
                 / E2E  \          5% (Full pipeline)
                /--------\
               /          \
              / Integration\       15% (Component interaction)
             /--------------\
            /                \
           /   Unit Tests     \    80% (Individual functions)
          /--------------------\
```

### Unit Tests

**Coverage Target:** 80%+

**Key Test Files:**
- `test_customer.py` - Customer class and generation
- `test_utility_engine.py` - Utility calculations
- `test_choice_engine.py` - Choice models
- `test_basket_composer.py` - Basket composition
- `test_loyalty_engine.py` - Purchase history
- `test_promotional_engine.py` - Promotions
- `test_product_catalog_builder.py` - Product Catalog Builder
- `test_price_hmm.py` - Price Elasticity Learner
- `test_cross_elasticity.py` - Cross Price Elasticity
- `test_arc_elasticity.py` - Arc Elasticity

**Test Template:**
```python
# tests/unit/test_basket_composer.py

import pytest
from retailsynth.generators.basket_composer import BasketComposer
from retailsynth.models import Customer, Product

class TestBasketComposer:
    
    @pytest.fixture
    def composer(self):
        return BasketComposer(config=default_config)
    
    @pytest.fixture
    def mock_customer(self):
        return Customer(
            id=1,
            shopping_personality='planned',
            price_sensitivity=0.8
        )
    
    def test_trip_purpose_selection(self, composer, mock_customer):
        """Test that trip purpose is selected based on customer type"""
        trip_purpose = composer.select_trip_purpose(mock_customer)
        assert trip_purpose in ['stock_up', 'fill_in', 'meal_prep', 'convenience']
        
    def test_basket_has_required_categories(self, composer, mock_customer):
        """Test that basket contains must-have categories for trip type"""
        basket = composer.generate_basket(mock_customer, trip_purpose='meal_prep')
        
        # Meal prep should include main dish category
        category_names = [item.category for item in basket]
        assert any('Meat' in cat or 'Produce' in cat for cat in category_names)
    
    def test_basket_coherence(self, composer, mock_customer):
        """Test that basket items make sense together"""
        basket = composer.generate_basket(mock_customer, trip_purpose='meal_prep')
        
        # Should not have 5+ items from same subcategory
        subcategory_counts = {}
        for item in basket:
            subcategory_counts[item.subcategory] = subcategory_counts.get(item.subcategory, 0) + 1
        
        assert all(count <= 3 for count in subcategory_counts.values())
    
    @pytest.mark.parametrize("trip_purpose,expected_min,expected_max", [
        ('convenience', 1, 5),
        ('fill_in', 3, 10),
        ('meal_prep', 5, 15),
        ('stock_up', 10, 30)
    ])
    def test_basket_size_ranges(self, composer, mock_customer, trip_purpose, expected_min, expected_max):
        """Test that basket sizes match expected ranges for trip types"""
        basket = composer.generate_basket(mock_customer, trip_purpose=trip_purpose)
        
        assert expected_min <= len(basket) <= expected_max
```

### Integration Tests

**Coverage Target:** 15%

**Test Scenarios:**
- Full generation pipeline (customers → products → transactions)
- Calibration workflow
- Validation workflow
- State persistence across weeks

**Example:**
```python
# tests/integration/test_generation_pipeline.py

def test_full_generation_pipeline():
    """Test complete pipeline from config to output"""
    
    # 1. Load config
    config = load_config('configs/small_test.yaml')
    
    # 2. Generate data
    generator = RetailSynthGenerator(config)
    datasets = generator.generate_all_datasets()
    
    # 3. Verify outputs
    assert 'transactions' in datasets
    assert 'customers' in datasets
    assert 'products' in datasets
    
    # 4. Check data quality
    transactions = datasets['transactions']
    assert len(transactions) > 0
    assert transactions['customer_id'].nunique() == config.n_customers
    assert transactions['total_revenue'].sum() > 0
    
    # 5. Verify relationships
    customer_ids = set(datasets['customers']['customer_id'])
    transaction_customer_ids = set(transactions['customer_id'])
    assert transaction_customer_ids.issubset(customer_ids)
```

### Validation Tests

**These are the actual validation metrics we're targeting**

```python
# tests/validation/test_level1_distributions.py

class TestLevel1Distributions:
    
    @pytest.fixture(scope='class')
    def real_data(self):
        return load_dunnhumby_data()
    
    @pytest.fixture(scope='class')
    def synthetic_data(self):
        return generate_synthetic_data(config='calibrated')
    
    def test_visit_frequency_distribution(self, real_data, synthetic_data):
        """KS test for visit frequency distribution"""
        real_visits = calculate_visit_frequency(real_data)
        synth_visits = calculate_visit_frequency(synthetic_data)
        
        ks_stat, p_value = stats.ks_2samp(real_visits, synth_visits)
        
        assert p_value > 0.05, f"Visit frequency distribution mismatch (p={p_value:.4f})"
        assert ks_stat < 0.1, f"KS statistic too high ({ks_stat:.4f})"
    
    def test_basket_size_distribution(self, real_data, synthetic_data):
        """KS test for basket size distribution"""
        real_baskets = calculate_basket_sizes(real_data)
        synth_baskets = calculate_basket_sizes(synthetic_data)
        
        ks_stat, p_value = stats.ks_2samp(real_baskets, synth_baskets)
        
        assert p_value > 0.05, f"Basket size distribution mismatch (p={p_value:.4f})"
    
    # ... more distribution tests
```

### Continuous Integration

**GitHub Actions Workflow:**
```yaml
# .github/workflows/tests.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=src/retailsynth --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
    
    - name: Check code style
      run: |
        black --check src/
        flake8 src/
```

---

## 10. Collaboration Guidelines

### For Human Developer

**Daily Tasks:**
1. **Morning (15-30 min)**
   - Review Claude's overnight PRs
   - Run tests locally
   - Provide feedback on GitHub

2. **Midday (30-60 min)**
   - Create new issues for bugs/features found
   - Update project board
   - Define next priority tasks

3. **Evening (30-45 min)**
   - Merge approved PRs
   - Run validation tests
   - Update roadmap if needed

**Weekly Tasks:**
1. **Monday:** Sprint planning, prioritize issues
2. **Wednesday:** Mid-sprint check, adjust priorities
3. **Friday:** Sprint review, update roadmap

**Quality Gates:**
- All tests must pass before merge
- Code coverage must not decrease
- Validation metrics must not regress

### For Claude Code

**Task Selection:**
1. Always pick from project board "To Do" column
2. Start with highest priority (P0 > P1 > P2 > P3)
3. Check dependencies before starting

**Implementation Standards:**
```python
# Always include:
# 1. Type hints
def calculate_utility(
    price: float,
    brand_preference: float,
    promotion: bool
) -> float:
    """
    Calculate product utility.
    
    Args:
        price: Current product price
        brand_preference: Customer brand preference (0-1)
        promotion: Whether product is on promotion
    
    Returns:
        Utility score (higher = more preferred)
    
    Example:
        >>> calculate_utility(5.99, 0.8, True)
        12.456
    """
    # 2. Clear documentation
    
    # 3. Defensive programming
    if price <= 0:
        raise ValueError("Price must be positive")
    
    # 4. Implementation
    utility = -2.5 * np.log(price) + 1.2 * brand_preference
    if promotion:
        utility += 0.5
    
    return utility

# 5. Unit test in same PR
def test_calculate_utility():
    utility = calculate_utility(5.99, 0.8, True)
    assert utility > 0
    assert calculate_utility(5.99, 0.8, True) > calculate_utility(5.99, 0.8, False)
```

**Pull Request Template:**
```markdown
## Summary
Brief description of what this PR does

## Issue
Closes #[issue_number]

## Changes
- [ ] Implementation complete
- [ ] Tests added
- [ ] Documentation updated
- [ ] No breaking changes

## Validation
[Paste relevant test output or validation metrics]

## Checklist
- [ ] Code follows style guide
- [ ] All tests pass
- [ ] Coverage maintained/improved
- [ ] Documentation updated
```

### Communication Best Practices

**Issue Creation:**
```markdown
# Title: [Component] Brief description

## Problem
Clear description of the issue or need

## Context
- Why this is important
- Related issues/PRs
- Dependencies

## Proposed Solution
High-level approach

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Additional Notes
Any other relevant information
```

**Code Review Comments:**
- Be specific about what needs to change
- Provide examples when possible
- Link to relevant documentation
- Distinguish between blocking and non-blocking feedback

---

## 11. Release Plan

### v0.1-alpha (Week 2) - First Milestone
**Features:**
- Purchase history system
- Cross-price elasticity
- Basket composition
- Product catalog alignment
- Basic validation (60% pass rate)

**Purpose:** Internal testing, early feedback

---

### v0.5-beta (Week 4) - Second Milestone
**Features:**
- Individual heterogeneity
- Promotional engine
- Non-linear utilities
- Data-driven seasonality
- Improved validation (70% pass rate)

**Purpose:** External testing, feedback from beta users

---

### v1.0 (Week 6-8) - Official Release
**Features:**
- Full calibration
- 80%+ validation pass rate
- Complete documentation
- Example notebooks
- CLI interface

**Purpose:** Public release, academic use

---

### v1.1+ (Ongoing)
**Future Features:**
- Geographic clustering
- Store differentiation
- Advanced visualizations
- Web interface
- Cloud deployment

---

## Appendix A: Quick Reference

### Key Commands

```bash
# Generate synthetic data
python scripts/generate_synthetic.py --config configs/calibrated.yaml

# Run calibration
python scripts/run_calibration.py --data data/raw/dunnhumby --trials 100

# Run validation
python scripts/run_validation.py --synthetic outputs/synthetic_data --real data/processed/dunnhumby

# Run tests
pytest tests/

# Run specific test
pytest tests/unit/test_basket_composer.py -v

# Generate documentation
cd docs && make html

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Configuration Example

```yaml
# configs/calibrated.yaml

generator:
  n_customers: 100000
  n_products: 5000
  n_stores: 20
  simulation_weeks: 52
  random_seed: 42

customer_heterogeneity:
  beta_price:
    distribution: normal
    mean: -2.5
    std: 0.8
  
  beta_quality:
    distribution: lognormal
    mean: 1.2
    std: 0.5

basket_composition:
  trip_purposes:
    - stock_up: 0.25
    - fill_in: 0.35
    - meal_prep: 0.25
    - convenience: 0.15

validation:
  targets_path: data/calibration/targets.npy
  thresholds:
    ks_pvalue: 0.05
    mean_error: 0.10
    std_error: 0.15
```

### Key Metrics Dashboard

Track these metrics weekly:

```python
# Validation Metrics
{
    'overall_pass_rate': 0.75,
    'level1_pass_rate': 0.85,
    'level2_pass_rate': 0.80,
    'level3_pass_rate': 0.70,
    'level4_pass_rate': 0.65,
    
    'ks_tests_passed': 12,
    'ks_tests_total': 15,
    
    'aggregate_errors': {
        'avg_visits': 0.08,
        'avg_basket_size': 0.12,
        'avg_basket_value': 0.09
    }
}

# Performance Metrics
{
    'generation_time_minutes': 18,
    'memory_usage_gb': 16,
    'transactions_per_second': 8500
}

# Code Quality Metrics
{
    'test_coverage': 0.82,
    'tests_passing': 245,
    'tests_total': 250,
    'code_quality_score': 'A'
}
```

---

## Appendix B: Resources

### Documentation
- RetailSynth paper: https://arxiv.org/abs/2312.14095
- Dunnhumby dataset: https://www.dunnhumby.com/source-files/
- Discrete choice models: Train (2009) "Discrete Choice Methods with Simulation"
- Utility theory: McFadden (2001) "Economic Choices"

### Tools
- JAX documentation: https://jax.readthedocs.io/
- NumPyro: https://num.pyro.ai/
- Optuna: https://optuna.org/
- Pytest: https://docs.pytest.org/

### Example Repositories
- RetailSynth GitHub: https://github.com/RetailMarketingAI/retailsynth
- PyMC marketing models: https://github.com/pymc-labs/pymc-marketing
- Discrete choice in Python: https://github.com/timothyb0912/pylogit

---

## Document Version Control

**Version:** 1.0
**Last Updated:** October 31, 2025
**Authors:** Human + Claude (Anthropic)
**Status:** Living Document (update as project progresses)

---




## Next Steps (Updated for v1.1)

**Immediate Actions (Week 0):**
1. [ ] Create GitHub repository
2. [ ] Set up development environment
3. [ ] Initialize project structure (use v1.1 structure above)
4. [ ] Create project board with **updated** issues
5. [ ] Set up CI/CD pipeline
6. [ ] **Download Dunnhumby Complete Journey** (priority!)
7. [ ] Extract product.csv and transaction_data.csv
8. [ ] Schedule Sprint 1.1 kickoff

**First Sprint (Week 1 - Product Catalog):**
1. [ ] Preprocess Dunnhumby product.csv (92K products)
2. [ ] Calculate purchase frequencies from transactions
3. [ ] Implement stratified sampling (20K SKUs)
4. [ ] Ensure major brand coverage
5. [ ] Create product archetypes
6. [ ] Generate category hierarchy JSON
7. [ ] Write validation tests
8. [ ] Initial validation against Dunnhumby distributions

**Second Sprint (Week 2 - Price Elasticity):**
1. [ ] Learn HMM from causal_data.csv
2. [ ] Estimate transition matrices
3. [ ] Compute cross-price elasticity matrixLet's close the following:
4. [ ] Implement arc elasticity model
5. [ ] Integrate all three into utility engine
6. [ ] Validation tests

**Third Sprint (Week 3 - Basket Logic):**
1. [ ] Implement trip purpose framework
2. [ ] Add basket composition rules
3. [ ] Create category constraints
4. [ ] Test: Basket coherence

**Fourth Sprint (Week 4 - Purchase History):**
1. [ ] Implement customer state tracking
2. [ ] Add history-dependent utilities
3. [ ] Create loyalty model
4. [ ] Test: Repeat purchase patterns


---

## Critical Success Factors (Updated)

### Must-Have for 80% Validation:
1. ✅ **20K real product catalog** (not 5K synthetic)
2. ✅ **HMM price dynamics** (not random promotions)
3. ✅ **Cross-price elasticity** (substitution modeling)
4. ✅ **Arc elasticity** (intertemporal behavior)
5. ✅ **Purchase history** (state dependence)
6. ✅ **Basket logic** (trip purpose + constraints)

### Timeline Realism:
- v1.1 estimate: **6-7 weeks** (+1 week buffer for complexity)
- With active collaboration: **Achievable**
- Without Claude Code: Would take 3-4 months

---

**Ready to build the real thing! 🎯**

**Start with Sprint 1.1 on Monday: Extract that 20K product catalog from Dunnhumby!**
