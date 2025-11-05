# Data Calibration & Statistical Validation Guide
## Using Dunnhumby Complete Journey Dataset

**Purpose:** Transform the enhanced_retailsynth_v4_0 code from an unvalidated prototype into a statistically rigorous data generator

**Dataset:** Dunnhumby "The Complete Journey" (~84M transactions, 2,500 households, 2 years)

**Goal:** Match synthetic data distributions to real data within acceptable tolerances

---

## Table of Contents

1. [Overview & Strategy](#overview--strategy)
2. [Phase 1: Data Acquisition & Preparation](#phase-1-data-acquisition--preparation)
3. [Phase 2: Exploratory Data Analysis](#phase-2-exploratory-data-analysis)
4. [Phase 3: Parameter Calibration](#phase-3-parameter-calibration)
5. [Phase 4: Statistical Validation](#phase-4-statistical-validation)
6. [Phase 5: Iterative Refinement](#phase-5-iterative-refinement)
7. [Phase 6: Documentation & Reporting](#phase-6-documentation--reporting)
8. [Implementation Code](#implementation-code)
9. [Success Metrics](#success-metrics)
10. [Troubleshooting](#troubleshooting)

---

## Overview & Strategy

### Calibration Philosophy

**What is Calibration?**
Adjusting synthetic data generator parameters so outputs statistically match real-world distributions.

**Two Approaches:**
1. **Manual Calibration** - Iteratively adjust parameters based on visual/statistical comparisons
2. **Automated Optimization** - Use algorithms (Optuna, genetic algorithms) to find optimal parameters

**We'll use a hybrid approach:**
- Manual for interpretability and domain knowledge
- Automated for fine-tuning difficult-to-fit distributions

### Validation Philosophy

**What is Validation?**
Proving that synthetic data is "close enough" to real data for intended use cases.

**Multi-Level Validation:**
- **Level 1:** Distribution matching (KS tests, chi-square)
- **Level 2:** Aggregate statistics (means, variances, correlations)
- **Level 3:** Behavioral patterns (basket analysis, temporal trends)
- **Level 4:** Predictive accuracy (ML models trained on synthetic perform well on real)

---

## Phase 1: Data Acquisition & Preparation

### Step 1.1: Obtain Dunnhumby Dataset

**Option A: Official Source**
```bash
# Register and download from Dunnhumby
# URL: https://www.dunnhumby.com/source-files/
# Note: May require academic/research credentials
```

**Option B: Kaggle Mirror**
```bash
# Kaggle dataset: dunnhumby - The Complete Journey
kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey
unzip dunnhumby-the-complete-journey.zip -d ./dunnhumby_data/
```

**Dataset Files:**
```
dunnhumby_data/
├── transaction_data.csv       # ~84M rows (main transaction log)
├── hh_demographic.csv          # Household demographics
├── product.csv                 # Product master with categories
├── causal_data.csv             # Promotional information
├── coupon.csv                  # Coupon campaigns
├── coupon_redempt.csv          # Coupon redemptions
└── campaign_desc.csv           # Campaign descriptions
```

### Step 1.2: Data Preprocessing

**Create preprocessing script:**

```python
# File: preprocess_dunnhumby.py

import pandas as pd
import numpy as np
from pathlib import Path

class DunnhumbyPreprocessor:
    """Preprocess Complete Journey data for calibration"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.transactions = None
        self.products = None
        self.households = None
        self.promotions = None
        
    def load_all_data(self):
        """Load all Dunnhumby files"""
        print("Loading Dunnhumby Complete Journey data...")
        
        # Load main files
        self.transactions = pd.read_csv(
            self.data_dir / 'transaction_data.csv',
            dtype={
                'household_key': 'int32',
                'BASKET_ID': 'int32',
                'PRODUCT_ID': 'int32',
                'QUANTITY': 'int16',
                'SALES_VALUE': 'float32',
                'STORE_ID': 'int16',
                'RETAIL_DISC': 'float32',
                'TRANS_TIME': 'int32',
                'WEEK_NO': 'int16',
                'COUPON_DISC': 'float32',
                'COUPON_MATCH_DISC': 'float32'
            }
        )
        
        self.products = pd.read_csv(
            self.data_dir / 'product.csv',
            dtype={
                'PRODUCT_ID': 'int32',
                'MANUFACTURER': 'int32',
                'DEPARTMENT': 'str',
                'BRAND': 'str',
                'COMMODITY_DESC': 'str',
                'SUB_COMMODITY_DESC': 'str',
                'CURR_SIZE_OF_PRODUCT': 'str'
            }
        )
        
        self.households = pd.read_csv(
            self.data_dir / 'hh_demographic.csv',
            dtype={
                'household_key': 'int32',
                'AGE_DESC': 'str',
                'MARITAL_STATUS_CODE': 'str',
                'INCOME_DESC': 'str',
                'HOMEOWNER_DESC': 'str',
                'HH_COMP_DESC': 'str',
                'HOUSEHOLD_SIZE_DESC': 'str',
                'KID_CATEGORY_DESC': 'str'
            }
        )
        
        self.promotions = pd.read_csv(
            self.data_dir / 'causal_data.csv',
            dtype={
                'PRODUCT_ID': 'int32',
                'STORE_ID': 'int16',
                'WEEK_NO': 'int16',
                'DISPLAY': 'str',
                'MAILER': 'str'
            }
        )
        
        print(f"✅ Loaded {len(self.transactions):,} transactions")
        print(f"✅ Loaded {len(self.products):,} products")
        print(f"✅ Loaded {len(self.households):,} households")
        print(f"✅ Loaded {len(self.promotions):,} promotion records")
        
    def clean_and_enrich(self):
        """Clean data and add derived features"""
        print("\n🔧 Cleaning and enriching data...")
        
        # Remove invalid transactions
        self.transactions = self.transactions[
            (self.transactions['SALES_VALUE'] > 0) &
            (self.transactions['QUANTITY'] > 0)
        ]
        
        # Calculate actual price paid
        self.transactions['PRICE_PAID'] = (
            self.transactions['SALES_VALUE'] / 
            self.transactions['QUANTITY']
        )
        
        # Calculate discount percentage
        self.transactions['DISCOUNT_PCT'] = (
            (self.transactions['RETAIL_DISC'] + 
             self.transactions['COUPON_DISC'] + 
             self.transactions['COUPON_MATCH_DISC']) / 
            self.transactions['SALES_VALUE']
        ).clip(0, 1)
        
        # Add date information
        # Dunnhumby uses WEEK_NO (1-104 for 2 years)
        self.transactions['YEAR'] = ((self.transactions['WEEK_NO'] - 1) // 52) + 1
        self.transactions['WEEK_OF_YEAR'] = ((self.transactions['WEEK_NO'] - 1) % 52) + 1
        
        # Merge product information
        self.transactions = self.transactions.merge(
            self.products[['PRODUCT_ID', 'DEPARTMENT', 'COMMODITY_DESC', 
                          'SUB_COMMODITY_DESC', 'MANUFACTURER']],
            on='PRODUCT_ID',
            how='left'
        )
        
        # Merge household demographics
        self.transactions = self.transactions.merge(
            self.households,
            on='household_key',
            how='left'
        )
        
        print(f"✅ Enriched transactions: {len(self.transactions):,} rows")
        
    def create_aggregations(self):
        """Create aggregated views for calibration"""
        print("\n📊 Creating aggregations...")
        
        # Basket-level aggregations
        self.baskets = self.transactions.groupby('BASKET_ID').agg({
            'household_key': 'first',
            'STORE_ID': 'first',
            'WEEK_NO': 'first',
            'PRODUCT_ID': 'nunique',  # Unique items in basket
            'QUANTITY': 'sum',         # Total items
            'SALES_VALUE': 'sum',      # Total spend
            'DISCOUNT_PCT': 'mean'     # Average discount
        }).reset_index()
        
        self.baskets.columns = [
            'basket_id', 'household_key', 'store_id', 'week_no',
            'unique_products', 'total_quantity', 'total_spend', 'avg_discount'
        ]
        
        # Customer-level aggregations
        self.customers = self.transactions.groupby('household_key').agg({
            'BASKET_ID': 'nunique',           # Number of trips
            'PRODUCT_ID': 'nunique',          # Product variety
            'SALES_VALUE': 'sum',             # Total spend
            'WEEK_NO': ['min', 'max'],        # First and last week
            'STORE_ID': lambda x: x.mode()[0] if len(x) > 0 else None  # Primary store
        }).reset_index()
        
        self.customers.columns = [
            'household_key', 'total_trips', 'unique_products',
            'total_spend', 'first_week', 'last_week', 'primary_store'
        ]
        
        # Calculate visit frequency
        self.customers['weeks_active'] = (
            self.customers['last_week'] - self.customers['first_week'] + 1
        )
        self.customers['visits_per_week'] = (
            self.customers['total_trips'] / self.customers['weeks_active']
        )
        
        # Product-level aggregations
        self.product_stats = self.transactions.groupby('PRODUCT_ID').agg({
            'BASKET_ID': 'nunique',      # How many baskets contain this
            'QUANTITY': 'sum',           # Total units sold
            'SALES_VALUE': 'sum',        # Total revenue
            'PRICE_PAID': 'mean',        # Average price
            'DISCOUNT_PCT': 'mean'       # Average discount
        }).reset_index()
        
        self.product_stats.columns = [
            'product_id', 'baskets_purchased', 'total_quantity',
            'total_revenue', 'avg_price', 'avg_discount'
        ]
        
        # Category-level aggregations
        self.category_stats = self.transactions.groupby('DEPARTMENT').agg({
            'BASKET_ID': 'nunique',
            'SALES_VALUE': 'sum',
            'household_key': 'nunique'
        }).reset_index()
        
        self.category_stats.columns = [
            'department', 'baskets_with_category', 'total_revenue', 'unique_customers'
        ]
        
        print(f"✅ Created basket aggregations: {len(self.baskets):,} baskets")
        print(f"✅ Created customer aggregations: {len(self.customers):,} customers")
        print(f"✅ Created product stats: {len(self.product_stats):,} products")
        print(f"✅ Created category stats: {len(self.category_stats):,} categories")
        
    def extract_calibration_targets(self):
        """Extract target distributions for calibration"""
        print("\n🎯 Extracting calibration targets...")
        
        targets = {
            # Customer behavior targets
            'customer': {
                'visits_per_week': {
                    'mean': self.customers['visits_per_week'].mean(),
                    'std': self.customers['visits_per_week'].std(),
                    'median': self.customers['visits_per_week'].median(),
                    'distribution': self.customers['visits_per_week'].values
                },
                'spend_per_trip': {
                    'mean': (self.customers['total_spend'] / self.customers['total_trips']).mean(),
                    'std': (self.customers['total_spend'] / self.customers['total_trips']).std(),
                    'distribution': (self.customers['total_spend'] / self.customers['total_trips']).values
                },
                'product_variety': {
                    'mean': self.customers['unique_products'].mean(),
                    'std': self.customers['unique_products'].std(),
                    'distribution': self.customers['unique_products'].values
                }
            },
            
            # Basket behavior targets
            'basket': {
                'basket_size': {
                    'mean': self.baskets['unique_products'].mean(),
                    'std': self.baskets['unique_products'].std(),
                    'distribution': self.baskets['unique_products'].values
                },
                'basket_quantity': {
                    'mean': self.baskets['total_quantity'].mean(),
                    'std': self.baskets['total_quantity'].std(),
                    'distribution': self.baskets['total_quantity'].values
                },
                'basket_value': {
                    'mean': self.baskets['total_spend'].mean(),
                    'std': self.baskets['total_spend'].std(),
                    'distribution': self.baskets['total_spend'].values
                }
            },
            
            # Product behavior targets
            'product': {
                'purchase_frequency': {
                    'distribution': self.product_stats['baskets_purchased'].values
                },
                'price_distribution': {
                    'mean': self.product_stats['avg_price'].mean(),
                    'std': self.product_stats['avg_price'].std(),
                    'distribution': self.product_stats['avg_price'].values
                },
                'discount_rate': {
                    'mean': self.product_stats['avg_discount'].mean(),
                    'std': self.product_stats['avg_discount'].std(),
                    'distribution': self.product_stats['avg_discount'].values
                }
            },
            
            # Category behavior targets
            'category': {
                'penetration': {
                    'distribution': (
                        self.category_stats['unique_customers'] / 
                        len(self.customers)
                    ).values
                },
                'revenue_share': {
                    'distribution': (
                        self.category_stats['total_revenue'] / 
                        self.category_stats['total_revenue'].sum()
                    ).values
                }
            },
            
            # Temporal patterns
            'temporal': {
                'weekly_transactions': self.transactions.groupby('WEEK_NO').size().values,
                'weekly_revenue': self.transactions.groupby('WEEK_NO')['SALES_VALUE'].sum().values
            }
        }
        
        return targets
    
    def save_calibration_data(self, output_dir: str):
        """Save processed data for calibration"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n💾 Saving calibration data to {output_dir}/")
        
        # Save main datasets
        self.transactions.to_parquet(output_path / 'transactions_processed.parquet')
        self.baskets.to_parquet(output_path / 'baskets.parquet')
        self.customers.to_parquet(output_path / 'customers.parquet')
        self.product_stats.to_parquet(output_path / 'product_stats.parquet')
        self.category_stats.to_parquet(output_path / 'category_stats.parquet')
        
        # Save calibration targets
        targets = self.extract_calibration_targets()
        np.save(output_path / 'calibration_targets.npy', targets, allow_pickle=True)
        
        print("✅ All calibration data saved")
        
        return targets


# Usage
if __name__ == "__main__":
    preprocessor = DunnhumbyPreprocessor('./dunnhumby_data')
    preprocessor.load_all_data()
    preprocessor.clean_and_enrich()
    preprocessor.create_aggregations()
    targets = preprocessor.save_calibration_data('./calibration_data')
    
    print("\n" + "="*70)
    print("CALIBRATION TARGETS EXTRACTED")
    print("="*70)
    print(f"Customer visit frequency: {targets['customer']['visits_per_week']['mean']:.2f} ± {targets['customer']['visits_per_week']['std']:.2f}")
    print(f"Basket size: {targets['basket']['basket_size']['mean']:.2f} ± {targets['basket']['basket_size']['std']:.2f}")
    print(f"Basket value: ${targets['basket']['basket_value']['mean']:.2f} ± ${targets['basket']['basket_value']['std']:.2f}")
```

---

## Phase 2: Exploratory Data Analysis

### Step 2.1: Distribution Analysis

**Create comprehensive EDA notebook:**

```python
# File: eda_dunnhumby.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DunnhumbyEDA:
    """Exploratory Data Analysis for calibration"""
    
    def __init__(self, calibration_data_dir: str):
        self.data_dir = calibration_data_dir
        self.load_processed_data()
        
    def load_processed_data(self):
        """Load preprocessed calibration data"""
        print("Loading processed calibration data...")
        self.transactions = pd.read_parquet(f'{self.data_dir}/transactions_processed.parquet')
        self.baskets = pd.read_parquet(f'{self.data_dir}/baskets.parquet')
        self.customers = pd.read_parquet(f'{self.data_dir}/customers.parquet')
        self.product_stats = pd.read_parquet(f'{self.data_dir}/product_stats.parquet')
        self.targets = np.load(f'{self.data_dir}/calibration_targets.npy', allow_pickle=True).item()
        print("✅ Data loaded")
        
    def analyze_distributions(self):
        """Analyze key distributions"""
        print("\n📊 DISTRIBUTION ANALYSIS")
        print("="*70)
        
        # Customer visit frequency
        visits = self.customers['visits_per_week']
        print(f"\nCustomer Visit Frequency (visits/week):")
        print(f"  Mean: {visits.mean():.3f}")
        print(f"  Median: {visits.median():.3f}")
        print(f"  Std: {visits.std():.3f}")
        print(f"  Min: {visits.min():.3f}, Max: {visits.max():.3f}")
        print(f"  Percentiles: 25th={visits.quantile(0.25):.3f}, 75th={visits.quantile(0.75):.3f}")
        
        # Test for distribution fit
        _, p_value_normal = stats.normaltest(visits)
        _, p_value_lognorm = stats.kstest(visits, 'lognorm', args=stats.lognorm.fit(visits))
        print(f"  Normal test p-value: {p_value_normal:.4f}")
        print(f"  Lognormal test p-value: {p_value_lognorm:.4f}")
        print(f"  → Best fit: {'Lognormal' if p_value_lognorm > p_value_normal else 'Other'}")
        
        # Basket size distribution
        basket_size = self.baskets['unique_products']
        print(f"\nBasket Size (unique products):")
        print(f"  Mean: {basket_size.mean():.3f}")
        print(f"  Median: {basket_size.median():.3f}")
        print(f"  Std: {basket_size.std():.3f}")
        print(f"  Mode: {basket_size.mode()[0]}")
        
        # Basket value distribution
        basket_value = self.baskets['total_spend']
        print(f"\nBasket Value ($):")
        print(f"  Mean: ${basket_value.mean():.2f}")
        print(f"  Median: ${basket_value.median():.2f}")
        print(f"  Std: ${basket_value.std():.2f}")
        print(f"  90th percentile: ${basket_value.quantile(0.9):.2f}")
        
        # Discount patterns
        print(f"\nPromotion/Discount Patterns:")
        promo_rate = (self.transactions['DISCOUNT_PCT'] > 0).mean()
        print(f"  % of items purchased on promotion: {promo_rate*100:.1f}%")
        print(f"  Average discount (when applied): {self.transactions[self.transactions['DISCOUNT_PCT']>0]['DISCOUNT_PCT'].mean()*100:.1f}%")
        
        return {
            'visit_distribution': visits.values,
            'basket_size_distribution': basket_size.values,
            'basket_value_distribution': basket_value.values
        }
    
    def analyze_temporal_patterns(self):
        """Analyze time-based patterns"""
        print("\n📅 TEMPORAL PATTERN ANALYSIS")
        print("="*70)
        
        # Weekly transaction volume
        weekly_txns = self.transactions.groupby('WEEK_NO').size()
        print(f"\nWeekly Transaction Volume:")
        print(f"  Mean: {weekly_txns.mean():.0f} transactions/week")
        print(f"  Std: {weekly_txns.std():.0f}")
        print(f"  CV (coefficient of variation): {(weekly_txns.std()/weekly_txns.mean()):.3f}")
        
        # Detect seasonality
        weekly_revenue = self.transactions.groupby('WEEK_NO')['SALES_VALUE'].sum()
        # Simple seasonal decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(weekly_revenue, model='additive', period=52, extrapolate_trend='freq')
        
        seasonal_strength = decomposition.seasonal.std() / weekly_revenue.std()
        print(f"  Seasonal strength: {seasonal_strength:.3f}")
        print(f"  → {'Strong' if seasonal_strength > 0.15 else 'Moderate' if seasonal_strength > 0.05 else 'Weak'} seasonality detected")
        
        # Day of week patterns (if TRANS_TIME available)
        if 'DAY_OF_WEEK' in self.transactions.columns:
            dow_volume = self.transactions.groupby('DAY_OF_WEEK').size()
            print(f"\nDay of Week Variation:")
            for day, vol in dow_volume.items():
                print(f"  {day}: {vol:,} ({vol/dow_volume.sum()*100:.1f}%)")
        
        return {
            'weekly_transactions': weekly_txns.values,
            'weekly_revenue': weekly_revenue.values,
            'seasonality_component': decomposition.seasonal.values
        }
    
    def analyze_customer_segments(self):
        """Identify customer archetypes"""
        print("\n👥 CUSTOMER SEGMENTATION ANALYSIS")
        print("="*70)
        
        # Calculate customer metrics for segmentation
        customer_features = self.customers.copy()
        customer_features['avg_basket_value'] = customer_features['total_spend'] / customer_features['total_trips']
        customer_features['avg_basket_size'] = (
            self.baskets.groupby('household_key')['unique_products'].mean()
        )
        
        # Simple segmentation based on visit frequency and spend
        # Define quartiles
        freq_q = customer_features['visits_per_week'].quantile([0.33, 0.67])
        spend_q = customer_features['avg_basket_value'].quantile([0.33, 0.67])
        
        def classify_customer(row):
            if row['visits_per_week'] >= freq_q.iloc[1]:
                if row['avg_basket_value'] >= spend_q.iloc[1]:
                    return 'High Frequency - High Value'
                elif row['avg_basket_value'] <= spend_q.iloc[0]:
                    return 'High Frequency - Low Value'
                else:
                    return 'High Frequency - Medium Value'
            elif row['visits_per_week'] <= freq_q.iloc[0]:
                if row['avg_basket_value'] >= spend_q.iloc[1]:
                    return 'Low Frequency - High Value'
                elif row['avg_basket_value'] <= spend_q.iloc[0]:
                    return 'Low Frequency - Low Value'
                else:
                    return 'Low Frequency - Medium Value'
            else:
                return 'Medium Frequency - Mixed Value'
        
        customer_features['segment'] = customer_features.apply(classify_customer, axis=1)
        
        # Segment statistics
        segment_stats = customer_features.groupby('segment').agg({
            'household_key': 'count',
            'visits_per_week': 'mean',
            'avg_basket_value': 'mean',
            'avg_basket_size': 'mean'
        })
        
        segment_stats['percentage'] = segment_stats['household_key'] / len(customer_features) * 100
        
        print("\nCustomer Segments:")
        for segment, stats in segment_stats.iterrows():
            print(f"\n{segment}:")
            print(f"  Customers: {stats['household_key']:.0f} ({stats['percentage']:.1f}%)")
            print(f"  Avg visits/week: {stats['visits_per_week']:.2f}")
            print(f"  Avg basket value: ${stats['avg_basket_value']:.2f}")
            print(f"  Avg basket size: {stats['avg_basket_size']:.1f} items")
        
        return segment_stats
    
    def analyze_price_elasticity(self):
        """Estimate price elasticity from data"""
        print("\n💰 PRICE ELASTICITY ANALYSIS")
        print("="*70)
        
        # For each product, look at price vs quantity relationship
        # Group by product and week
        product_week = self.transactions.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
            'QUANTITY': 'sum',
            'PRICE_PAID': 'mean',
            'household_key': 'nunique'  # Number of unique buyers
        }).reset_index()
        
        # Calculate elasticity for products with sufficient variation
        elasticities = []
        
        for product_id in product_week['PRODUCT_ID'].unique()[:100]:  # Sample first 100 products
            product_data = product_week[product_week['PRODUCT_ID'] == product_id]
            
            if len(product_data) < 10:  # Need minimum observations
                continue
            
            # Log-log regression for elasticity
            log_price = np.log(product_data['PRICE_PAID'].values + 0.01)
            log_quantity = np.log(product_data['QUANTITY'].values + 1)
            
            # Remove infinite values
            valid_mask = np.isfinite(log_price) & np.isfinite(log_quantity)
            log_price = log_price[valid_mask]
            log_quantity = log_quantity[valid_mask]
            
            if len(log_price) >= 10 and log_price.std() > 0.01:  # Need price variation
                # Simple linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_quantity)
                
                if p_value < 0.10:  # Only keep statistically significant
                    elasticities.append({
                        'product_id': product_id,
                        'elasticity': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value
                    })
        
        elasticity_df = pd.DataFrame(elasticities)
        
        if len(elasticity_df) > 0:
            print(f"\nPrice Elasticity Estimates (from {len(elasticity_df)} products):")
            print(f"  Mean elasticity: {elasticity_df['elasticity'].mean():.3f}")
            print(f"  Median elasticity: {elasticity_df['elasticity'].median():.3f}")
            print(f"  Std: {elasticity_df['elasticity'].std():.3f}")
            print(f"  Range: [{elasticity_df['elasticity'].min():.3f}, {elasticity_df['elasticity'].max():.3f}]")
            print(f"  % with negative elasticity (expected): {(elasticity_df['elasticity'] < 0).mean()*100:.1f}%")
            
            return elasticity_df
        else:
            print("  ⚠️ Insufficient price variation to estimate elasticities")
            return None
    
    def create_visualization_report(self, output_dir: str = './eda_visualizations'):
        """Create comprehensive visualization report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Creating visualization report in {output_dir}/")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Key Distribution Analysis', fontsize=16)
        
        # Visit frequency
        axes[0, 0].hist(self.customers['visits_per_week'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Customer Visit Frequency')
        axes[0, 0].set_xlabel('Visits per week')
        axes[0, 0].set_ylabel('Count')
        
        # Basket size
        axes[0, 1].hist(self.baskets['unique_products'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Basket Size Distribution')
        axes[0, 1].set_xlabel('Unique products')
        axes[0, 1].set_ylabel('Count')
        
        # Basket value
        axes[0, 2].hist(self.baskets['total_spend'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Basket Value Distribution')
        axes[0, 2].set_xlabel('Total spend ($)')
        axes[0, 2].set_ylabel('Count')
        
        # Product popularity (log scale)
        axes[1, 0].hist(np.log10(self.product_stats['baskets_purchased'] + 1), 
                       bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Product Popularity (log scale)')
        axes[1, 0].set_xlabel('Log10(baskets purchased)')
        axes[1, 0].set_ylabel('Count')
        
        # Price distribution
        axes[1, 1].hist(self.product_stats['avg_price'], bins=50, 
                       range=(0, self.product_stats['avg_price'].quantile(0.95)),
                       edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Price Distribution (95th percentile)')
        axes[1, 1].set_xlabel('Average price ($)')
        axes[1, 1].set_ylabel('Count')
        
        # Discount rate
        axes[1, 2].hist(self.transactions['DISCOUNT_PCT'], bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Discount Rate Distribution')
        axes[1, 2].set_xlabel('Discount percentage')
        axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_key_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temporal patterns
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('Temporal Patterns', fontsize=16)
        
        # Weekly transaction volume
        weekly_txns = self.transactions.groupby('WEEK_NO').size()
        axes[0].plot(weekly_txns.index, weekly_txns.values, linewidth=2)
        axes[0].set_title('Weekly Transaction Volume')
        axes[0].set_xlabel('Week Number')
        axes[0].set_ylabel('Number of Transactions')
        axes[0].grid(True, alpha=0.3)
        
        # Weekly revenue
        weekly_revenue = self.transactions.groupby('WEEK_NO')['SALES_VALUE'].sum()
        axes[1].plot(weekly_revenue.index, weekly_revenue.values, linewidth=2, color='green')
        axes[1].set_title('Weekly Revenue')
        axes[1].set_xlabel('Week Number')
        axes[1].set_ylabel('Total Revenue ($)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved")


# Usage
if __name__ == "__main__":
    eda = DunnhumbyEDA('./calibration_data')
    
    # Run all analyses
    dist_results = eda.analyze_distributions()
    temporal_results = eda.analyze_temporal_patterns()
    segment_results = eda.analyze_customer_segments()
    elasticity_results = eda.analyze_price_elasticity()
    
    # Create visualization report
    eda.create_visualization_report('./eda_visualizations')
    
    print("\n" + "="*70)
    print("✅ EDA COMPLETE - Review results before calibration")
    print("="*70)
```

---

## Phase 3: Parameter Calibration

### Step 3.1: Manual Calibration Framework

**Create calibration module:**

```python
# File: calibrate_retailsynth.py

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, Tuple
import json

class RetailSynthCalibrator:
    """
    Calibrate enhanced_retailsynth_v4_0 parameters to match Dunnhumby data
    """
    
    def __init__(self, real_data_targets: Dict, synthetic_generator):
        """
        Args:
            real_data_targets: Dictionary of target distributions from real data
            synthetic_generator: Instance of EnhancedRetailSynthV4_0
        """
        self.targets = real_data_targets
        self.generator = synthetic_generator
        self.calibration_history = []
        
    def calculate_ks_distance(self, real_dist: np.ndarray, synthetic_dist: np.ndarray) -> float:
        """
        Calculate Kolmogorov-Smirnov distance between two distributions
        
        Returns KS statistic (0 = perfect match, 1 = complete mismatch)
        """
        ks_stat, _ = stats.ks_2samp(real_dist, synthetic_dist)
        return ks_stat
    
    def calculate_ks_complement(self, real_dist: np.ndarray, synthetic_dist: np.ndarray) -> float:
        """
        Calculate KS-complement metric (higher = better fit)
        Used in RetailSynth paper
        """
        ks_stat = self.calculate_ks_distance(real_dist, synthetic_dist)
        return 1.0 - ks_stat
    
    def calculate_wasserstein_distance(self, real_dist: np.ndarray, synthetic_dist: np.ndarray) -> float:
        """
        Calculate Wasserstein (Earth Mover's) distance
        More sensitive to distribution tails than KS
        """
        return stats.wasserstein_distance(real_dist, synthetic_dist)
    
    def calculate_moment_error(self, real_dist: np.ndarray, synthetic_dist: np.ndarray) -> Dict[str, float]:
        """
        Calculate error in statistical moments
        """
        return {
            'mean_error': abs(np.mean(real_dist) - np.mean(synthetic_dist)) / np.mean(real_dist),
            'std_error': abs(np.std(real_dist) - np.std(synthetic_dist)) / np.std(real_dist),
            'median_error': abs(np.median(real_dist) - np.median(synthetic_dist)) / np.median(real_dist)
        }
    
    def evaluate_fit_quality(self, synthetic_data: Dict) -> Dict:
        """
        Comprehensive evaluation of synthetic data fit to real data
        
        Returns dictionary with fit metrics for all key distributions
        """
        results = {}
        
        # 1. Customer visit frequency
        if 'customer' in self.targets and 'visits_per_week' in self.targets['customer']:
            real_visits = self.targets['customer']['visits_per_week']['distribution']
            synth_visits = synthetic_data['customer']['visits_per_week']
            
            results['visits_per_week'] = {
                'ks_complement': self.calculate_ks_complement(real_visits, synth_visits),
                'wasserstein': self.calculate_wasserstein_distance(real_visits, synth_visits),
                'moments': self.calculate_moment_error(real_visits, synth_visits)
            }
        
        # 2. Basket size
        if 'basket' in self.targets and 'basket_size' in self.targets['basket']:
            real_basket = self.targets['basket']['basket_size']['distribution']
            synth_basket = synthetic_data['basket']['basket_size']
            
            results['basket_size'] = {
                'ks_complement': self.calculate_ks_complement(real_basket, synth_basket),
                'wasserstein': self.calculate_wasserstein_distance(real_basket, synth_basket),
                'moments': self.calculate_moment_error(real_basket, synth_basket)
            }
        
        # 3. Basket value
        if 'basket' in self.targets and 'basket_value' in self.targets['basket']:
            real_value = self.targets['basket']['basket_value']['distribution']
            synth_value = synthetic_data['basket']['basket_value']
            
            results['basket_value'] = {
                'ks_complement': self.calculate_ks_complement(real_value, synth_value),
                'wasserstein': self.calculate_wasserstein_distance(real_value, synth_value),
                'moments': self.calculate_moment_error(real_value, synth_value)
            }
        
        # Calculate aggregate score
        ks_scores = [v['ks_complement'] for v in results.values() if 'ks_complement' in v]
        results['aggregate_ks_score'] = np.mean(ks_scores) if ks_scores else 0.0
        
        return results
    
    def manual_calibration_step(self, 
                                 param_name: str, 
                                 param_value: float,
                                 n_samples: int = 10000) -> Dict:
        """
        Single manual calibration step: set parameter and evaluate fit
        
        Args:
            param_name: Name of parameter to adjust (e.g., 'price_anchor_customers')
            param_value: New value for parameter
            n_samples: Number of synthetic samples to generate for evaluation
        
        Returns:
            Dictionary with fit metrics
        """
        print(f"\n🔧 Testing {param_name} = {param_value}")
        
        # Update generator parameter
        setattr(self.generator.config, param_name, param_value)
        
        # Generate small synthetic sample for evaluation
        # (Implement lightweight sampling method)
        synthetic_sample = self.generator.generate_sample(n_samples=n_samples)
        
        # Evaluate fit
        fit_metrics = self.evaluate_fit_quality(synthetic_sample)
        
        # Log calibration attempt
        self.calibration_history.append({
            'param_name': param_name,
            'param_value': param_value,
            'fit_metrics': fit_metrics,
            'aggregate_score': fit_metrics['aggregate_ks_score']
        })
        
        print(f"  Aggregate KS score: {fit_metrics['aggregate_ks_score']:.4f}")
        
        return fit_metrics
    
    def grid_search_parameter(self,
                             param_name: str,
                             param_range: Tuple[float, float],
                             n_steps: int = 10) -> Dict:
        """
        Grid search over parameter range to find best value
        
        Args:
            param_name: Parameter to optimize
            param_range: (min_value, max_value) tuple
            n_steps: Number of values to try
        
        Returns:
            Dictionary with best parameter value and metrics
        """
        print(f"\n🔍 Grid searching {param_name} in range {param_range}")
        
        param_values = np.linspace(param_range[0], param_range[1], n_steps)
        best_score = -np.inf
        best_value = None
        best_metrics = None
        
        for value in param_values:
            metrics = self.manual_calibration_step(param_name, value, n_samples=5000)
            score = metrics['aggregate_ks_score']
            
            if score > best_score:
                best_score = score
                best_value = value
                best_metrics = metrics
        
        print(f"\n✅ Best {param_name} = {best_value:.4f} (score: {best_score:.4f})")
        
        return {
            'param_name': param_name,
            'best_value': best_value,
            'best_score': best_score,
            'best_metrics': best_metrics
        }
    
    def save_calibration_results(self, filepath: str):
        """Save calibration history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.calibration_history, f, indent=2)
        print(f"✅ Calibration history saved to {filepath}")


# CALIBRATION WORKFLOW
def run_manual_calibration():
    """
    Step-by-step manual calibration workflow
    """
    print("="*70)
    print("MANUAL CALIBRATION WORKFLOW")
    print("="*70)
    
    # 1. Load real data targets
    targets = np.load('./calibration_data/calibration_targets.npy', allow_pickle=True).item()
    
    # 2. Initialize synthetic generator
    from enhanced_retailsynth_v4_0_comprehensive import EnhancedRetailSynthV4_0, EnhancedRetailConfig
    
    config = EnhancedRetailConfig(
        n_customers=10000,  # Start small for calibration
        n_products=1000,
        simulation_weeks=52
    )
    generator = EnhancedRetailSynthV4_0(config)
    
    # 3. Initialize calibrator
    calibrator = RetailSynthCalibrator(targets, generator)
    
    # 4. Calibrate customer mix parameters
    print("\n" + "="*70)
    print("STEP 1: Calibrating Customer Mix")
    print("="*70)
    
    # Based on EDA, we might find real data has:
    # - 30% price-sensitive customers
    # - 20% convenience-focused
    # - 35% planned shoppers
    # - 15% impulse buyers
    
    customer_mix_results = []
    
    # Try different customer mixes
    mixes = [
        {'price_anchor': 0.30, 'convenience': 0.20, 'planned': 0.35, 'impulse': 0.15},
        {'price_anchor': 0.25, 'convenience': 0.25, 'planned': 0.30, 'impulse': 0.20},
        {'price_anchor': 0.35, 'convenience': 0.15, 'planned': 0.40, 'impulse': 0.10},
    ]
    
    for mix in mixes:
        config.price_anchor_customers = mix['price_anchor']
        config.convenience_customers = mix['convenience']
        config.planned_customers = mix['planned']
        config.impulse_customers = mix['impulse']
        
        metrics = calibrator.manual_calibration_step(
            'customer_mix', 
            str(mix), 
            n_samples=5000
        )
        customer_mix_results.append((mix, metrics['aggregate_ks_score']))
    
    best_mix = max(customer_mix_results, key=lambda x: x[1])
    print(f"\n✅ Best customer mix: {best_mix[0]} (score: {best_mix[1]:.4f})")
    
    # 5. Calibrate price sensitivity parameters
    print("\n" + "="*70)
    print("STEP 2: Calibrating Price Sensitivity")
    print("="*70)
    
    # Grid search for optimal price sensitivity
    # (This would require modifying the code to expose price sensitivity as a config parameter)
    
    # 6. Calibrate promotional parameters
    print("\n" + "="*70)
    print("STEP 3: Calibrating Promotional Behavior")
    print("="*70)
    
    # Adjust promotion frequency and depth to match real data
    
    # 7. Save results
    calibrator.save_calibration_results('./calibration_results/manual_calibration.json')
    
    print("\n" + "="*70)
    print("✅ MANUAL CALIBRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review calibration_results/manual_calibration.json")
    print("2. Update config with best parameters")
    print("3. Run full validation (Phase 4)")


if __name__ == "__main__":
    run_manual_calibration()
```

### Step 3.2: Automated Optimization with Optuna

```python
# File: automated_calibration.py

import optuna
from optuna.samplers import TPESampler
import numpy as np
from typing import Dict

class AutomatedCalibration:
    """
    Automated parameter optimization using Optuna
    (Similar to RetailSynth paper approach)
    """
    
    def __init__(self, targets: Dict, generator_class, base_config):
        self.targets = targets
        self.generator_class = generator_class
        self.base_config = base_config
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function: return metric to MAXIMIZE
        
        Higher = better fit to real data
        """
        
        # 1. Suggest parameter values
        config = self.base_config.copy()
        
        # Customer mix parameters (must sum to 1.0)
        price_anchor = trial.suggest_float('price_anchor_pct', 0.15, 0.45)
        convenience = trial.suggest_float('convenience_pct', 0.10, 0.35)
        planned = trial.suggest_float('planned_pct', 0.20, 0.45)
        impulse = 1.0 - (price_anchor + convenience + planned)
        
        if impulse < 0.05 or impulse > 0.30:  # Constraint check
            return -1.0  # Invalid configuration
        
        config.price_anchor_customers = price_anchor
        config.convenience_customers = convenience
        config.planned_customers = planned
        config.impulse_customers = impulse
        
        # Price sensitivity scaling factor
        # (Would need to add this to the config class)
        price_sensitivity_scale = trial.suggest_float('price_sensitivity_scale', 0.5, 2.0)
        
        # Promotional parameters
        promo_frequency = trial.suggest_float('promo_frequency', 0.1, 0.5)
        promo_depth_alpha = trial.suggest_float('promo_depth_alpha', 1.5, 4.0)
        promo_depth_beta = trial.suggest_float('promo_depth_beta', 3.0, 8.0)
        
        # Basket size parameters
        # (Would need to add these to config)
        basket_size_mean = trial.suggest_float('basket_size_mean', 8.0, 15.0)
        basket_size_std = trial.suggest_float('basket_size_std', 3.0, 8.0)
        
        # 2. Generate synthetic sample with these parameters
        generator = self.generator_class(config)
        try:
            synthetic_data = generator.generate_sample(n_samples=5000)
        except Exception as e:
            print(f"  Error generating sample: {e}")
            return -1.0
        
        # 3. Calculate fit metrics
        from calibrate_retailsynth import RetailSynthCalibrator
        calibrator = RetailSynthCalibrator(self.targets, generator)
        fit_metrics = calibrator.evaluate_fit_quality(synthetic_data)
        
        # 4. Return aggregate score (to maximize)
        # Combine multiple metrics with weights
        score = (
            0.4 * fit_metrics.get('visits_per_week', {}).get('ks_complement', 0) +
            0.3 * fit_metrics.get('basket_size', {}).get('ks_complement', 0) +
            0.3 * fit_metrics.get('basket_value', {}).get('ks_complement', 0)
        )
        
        return score
    
    def run_optimization(self, 
                        n_trials: int = 100,
                        study_name: str = 'retailsynth_calibration',
                        storage: str = None) -> optuna.Study:
        """
        Run Optuna optimization
        
        Args:
            n_trials: Number of parameter combinations to try
            study_name: Name for the optimization study
            storage: Optional database URL for persistence
        
        Returns:
            Completed Optuna study object
        """
        print("="*70)
        print(f"AUTOMATED CALIBRATION - {n_trials} TRIALS")
        print("="*70)
        
        # Create study
        sampler = TPESampler(seed=42)  # Tree-structured Parzen Estimator
        
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize fit score
            sampler=sampler,
            storage=storage,
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1  # Parallel trials if desired
        )
        
        # Print results
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\nBest trial:")
        print(f"  Score: {study.best_trial.value:.4f}")
        print(f"\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value:.4f}")
        
        # Save study
        study_df = study.trials_dataframe()
        study_df.to_csv('./calibration_results/optuna_trials.csv', index=False)
        print(f"\n✅ Trial history saved to ./calibration_results/optuna_trials.csv")
        
        # Generate visualization
        try:
            import plotly
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate
            )
            
            fig1 = plot_optimization_history(study)
            fig1.write_html('./calibration_results/optimization_history.html')
            
            fig2 = plot_param_importances(study)
            fig2.write_html('./calibration_results/param_importances.html')
            
            fig3 = plot_parallel_coordinate(study)
            fig3.write_html('./calibration_results/parallel_coordinate.html')
            
            print("✅ Visualization plots saved to ./calibration_results/")
        except ImportError:
            print("⚠️ Install plotly for visualization: pip install plotly")
        
        return study


# Usage
if __name__ == "__main__":
    # Load targets
    targets = np.load('./calibration_data/calibration_targets.npy', allow_pickle=True).item()
    
    # Import generator
    from enhanced_retailsynth_v4_0_comprehensive import EnhancedRetailSynthV4_0, EnhancedRetailConfig
    
    # Base configuration
    base_config = EnhancedRetailConfig(
        n_customers=10000,
        n_products=1000,
        simulation_weeks=52
    )
    
    # Run automated calibration
    auto_cal = AutomatedCalibration(targets, EnhancedRetailSynthV4_0, base_config)
    study = auto_cal.run_optimization(n_trials=100)
    
    print("\n🎉 Automated calibration complete!")
    print("Apply best parameters to your config and run full validation.")
```

---

## Phase 4: Statistical Validation

### Step 4.1: Comprehensive Validation Suite

```python
# File: validate_synthetic_data.py

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class SyntheticDataValidator:
    """
    Comprehensive validation of synthetic data against real data
    """
    
    def __init__(self, real_data_dir: str, synthetic_datasets: Dict):
        """
        Args:
            real_data_dir: Path to preprocessed real data
            synthetic_datasets: Dictionary of generated synthetic datasets
        """
        self.real_data_dir = real_data_dir
        self.synthetic_data = synthetic_datasets
        self.validation_results = {}
        
        # Load real data
        self.load_real_data()
    
    def load_real_data(self):
        """Load preprocessed real data for comparison"""
        print("Loading real data for validation...")
        self.real_transactions = pd.read_parquet(f'{self.real_data_dir}/transactions_processed.parquet')
        self.real_baskets = pd.read_parquet(f'{self.real_data_dir}/baskets.parquet')
        self.real_customers = pd.read_parquet(f'{self.real_data_dir}/customers.parquet')
        print("✅ Real data loaded")
    
    def validate_distributions(self) -> Dict:
        """
        Level 1 Validation: Distribution matching using statistical tests
        """
        print("\n" + "="*70)
        print("LEVEL 1 VALIDATION: DISTRIBUTION MATCHING")
        print("="*70)
        
        results = {}
        
        # 1. Kolmogorov-Smirnov tests
        print("\n1. Kolmogorov-Smirnov Tests (H0: distributions are same)")
        
        # Visit frequency
        real_visits = self.real_customers['visits_per_week'].values
        synth_visits = self.synthetic_data['customers']['visits_per_week'].values
        ks_stat, ks_pval = stats.ks_2samp(real_visits, synth_visits)
        
        results['visits_ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pval,
            'passed': ks_pval > 0.05  # Don't reject H0 at 5% level
        }
        print(f"  Visit frequency: KS={ks_stat:.4f}, p={ks_pval:.4f} {'✅ PASS' if ks_pval > 0.05 else '❌ FAIL'}")
        
        # Basket size
        real_basket_size = self.real_baskets['unique_products'].values
        synth_basket_size = self.synthetic_data['baskets']['unique_products'].values
        ks_stat, ks_pval = stats.ks_2samp(real_basket_size, synth_basket_size)
        
        results['basket_size_ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pval,
            'passed': ks_pval > 0.05
        }
        print(f"  Basket size: KS={ks_stat:.4f}, p={ks_pval:.4f} {'✅ PASS' if ks_pval > 0.05 else '❌ FAIL'}")
        
        # Basket value
        real_basket_value = self.real_baskets['total_spend'].values
        synth_basket_value = self.synthetic_data['baskets']['total_spend'].values
        ks_stat, ks_pval = stats.ks_2samp(real_basket_value, synth_basket_value)
        
        results['basket_value_ks_test'] = {
            'statistic': ks_stat,
            'p_value': ks_pval,
            'passed': ks_pval > 0.05
        }
        print(f"  Basket value: KS={ks_stat:.4f}, p={ks_pval:.4f} {'✅ PASS' if ks_pval > 0.05 else '❌ FAIL'}")
        
        # 2. Chi-square tests for categorical distributions
        print("\n2. Chi-Square Tests for Categorical Distributions")
        
        # Category penetration
        real_cat_penetration = (
            self.real_transactions.groupby('DEPARTMENT')['household_key']
            .nunique()
            .sort_index()
        )
        synth_cat_penetration = (
            self.synthetic_data['transactions'].groupby('category')['customer_id']
            .nunique()
            .sort_index()
        )
        
        # Normalize to same scale
        real_cat_freq = real_cat_penetration / real_cat_penetration.sum()
        synth_cat_freq = synth_cat_penetration / synth_cat_penetration.sum()
        
        # Align categories
        common_cats = set(real_cat_freq.index) & set(synth_cat_freq.index)
        if len(common_cats) > 1:
            real_aligned = real_cat_freq.loc[list(common_cats)].values
            synth_aligned = synth_cat_freq.loc[list(common_cats)].values
            
            chi2_stat, chi2_pval = stats.chisquare(
                f_obs=synth_aligned * 1000,  # Scale up for chi-square
                f_exp=real_aligned * 1000
            )
            
            results['category_penetration_chi2'] = {
                'statistic': chi2_stat,
                'p_value': chi2_pval,
                'passed': chi2_pval > 0.05
            }
            print(f"  Category penetration: χ²={chi2_stat:.4f}, p={chi2_pval:.4f} {'✅ PASS' if chi2_pval > 0.05 else '❌ FAIL'}")
        
        return results
    
    def validate_aggregate_statistics(self) -> Dict:
        """
        Level 2 Validation: Aggregate statistics comparison
        """
        print("\n" + "="*70)
        print("LEVEL 2 VALIDATION: AGGREGATE STATISTICS")
        print("="*70)
        
        results = {}
        
        # Define acceptable error threshold (e.g., ±10%)
        threshold = 0.10
        
        # 1. Customer metrics
        print("\n1. Customer-Level Metrics:")
        
        real_avg_visits = self.real_customers['visits_per_week'].mean()
        synth_avg_visits = self.synthetic_data['customers']['visits_per_week'].mean()
        error = abs(real_avg_visits - synth_avg_visits) / real_avg_visits
        
        results['avg_visits_per_week'] = {
            'real': real_avg_visits,
            'synthetic': synth_avg_visits,
            'error': error,
            'passed': error < threshold
        }
        print(f"  Avg visits/week: Real={real_avg_visits:.3f}, Synth={synth_avg_visits:.3f}, Error={error*100:.1f}% {'✅' if error < threshold else '❌'}")
        
        # Average spend per customer
        real_avg_spend = self.real_customers['total_spend'].mean()
        synth_avg_spend = self.synthetic_data['customers']['total_spend'].mean()
        error = abs(real_avg_spend - synth_avg_spend) / real_avg_spend
        
        results['avg_customer_spend'] = {
            'real': real_avg_spend,
            'synthetic': synth_avg_spend,
            'error': error,
            'passed': error < threshold
        }
        print(f"  Avg customer spend: Real=${real_avg_spend:.2f}, Synth=${synth_avg_spend:.2f}, Error={error*100:.1f}% {'✅' if error < threshold else '❌'}")
        
        # 2. Basket metrics
        print("\n2. Basket-Level Metrics:")
        
        real_avg_basket = self.real_baskets['unique_products'].mean()
        synth_avg_basket = self.synthetic_data['baskets']['unique_products'].mean()
        error = abs(real_avg_basket - synth_avg_basket) / real_avg_basket
        
        results['avg_basket_size'] = {
            'real': real_avg_basket,
            'synthetic': synth_avg_basket,
            'error': error,
            'passed': error < threshold
        }
        print(f"  Avg basket size: Real={real_avg_basket:.2f}, Synth={synth_avg_basket:.2f}, Error={error*100:.1f}% {'✅' if error < threshold else '❌'}")
        
        real_avg_value = self.real_baskets['total_spend'].mean()
        synth_avg_value = self.synthetic_data['baskets']['total_spend'].mean()
        error = abs(real_avg_value - synth_avg_value) / real_avg_value
        
        results['avg_basket_value'] = {
            'real': real_avg_value,
            'synthetic': synth_avg_value,
            'error': error,
            'passed': error < threshold
        }
        print(f"  Avg basket value: Real=${real_avg_value:.2f}, Synth=${synth_avg_value:.2f}, Error={error*100:.1f}% {'✅' if error < threshold else '❌'}")
        
        # 3. Temporal metrics
        print("\n3. Temporal Metrics:")
        
        real_weekly_txns = self.real_transactions.groupby('WEEK_NO').size().mean()
        synth_weekly_txns = self.synthetic_data['transactions'].groupby('week_number').size().mean()
        error = abs(real_weekly_txns - synth_weekly_txns) / real_weekly_txns
        
        results['avg_weekly_transactions'] = {
            'real': real_weekly_txns,
            'synthetic': synth_weekly_txns,
            'error': error,
            'passed': error < threshold
        }
        print(f"  Avg weekly transactions: Real={real_weekly_txns:.0f}, Synth={synth_weekly_txns:.0f}, Error={error*100:.1f}% {'✅' if error < threshold else '❌'}")
        
        return results
    
    def validate_behavioral_patterns(self) -> Dict:
        """
        Level 3 Validation: Complex behavioral patterns
        """
        print("\n" + "="*70)
        print("LEVEL 3 VALIDATION: BEHAVIORAL PATTERNS")
        print("="*70)
        
        results = {}
        
        # 1. Market basket analysis
        print("\n1. Market Basket Association Rules:")
        
        # Calculate top product pairs in real data
        from itertools import combinations
        from collections import Counter
        
        real_baskets_grouped = self.real_transactions.groupby('BASKET_ID')['PRODUCT_ID'].apply(list)
        real_pairs = Counter()
        for basket in real_baskets_grouped:
            if len(basket) >= 2:
                for pair in combinations(sorted(basket), 2):
                    real_pairs[pair] += 1
        
        real_top_pairs = set([pair for pair, count in real_pairs.most_common(50)])
        
        # Calculate in synthetic data
        synth_baskets_grouped = self.synthetic_data['transactions'].groupby('basket_id')['product_id'].apply(list)
        synth_pairs = Counter()
        for basket in synth_baskets_grouped:
            if len(basket) >= 2:
                for pair in combinations(sorted(basket), 2):
                    synth_pairs[pair] += 1
        
        synth_top_pairs = set([pair for pair, count in synth_pairs.most_common(50)])
        
        # Calculate overlap (Jaccard similarity)
        overlap = len(real_top_pairs & synth_top_pairs) / len(real_top_pairs | synth_top_pairs)
        
        results['basket_association_overlap'] = {
            'jaccard_similarity': overlap,
            'passed': overlap > 0.30  # At least 30% overlap in top pairs
        }
        print(f"  Top 50 product pairs overlap: {overlap*100:.1f}% {'✅ PASS' if overlap > 0.30 else '❌ FAIL'}")
        
        # 2. Customer loyalty patterns
        print("\n2. Customer Loyalty Patterns:")
        
        # Store loyalty (% of customers shopping at single store)
        real_single_store_pct = (
            self.real_transactions.groupby('household_key')['STORE_ID']
            .nunique()
            .eq(1)
            .mean()
        )
        
        synth_single_store_pct = (
            self.synthetic_data['transactions'].groupby('customer_id')['store_id']
            .nunique()
            .eq(1)
            .mean()
        )
        
        error = abs(real_single_store_pct - synth_single_store_pct) / real_single_store_pct
        
        results['single_store_loyalty'] = {
            'real': real_single_store_pct,
            'synthetic': synth_single_store_pct,
            'error': error,
            'passed': error < 0.15  # ±15% tolerance
        }
        print(f"  Single-store customers: Real={real_single_store_pct*100:.1f}%, Synth={synth_single_store_pct*100:.1f}%, Error={error*100:.1f}% {'✅' if error < 0.15 else '❌'}")
        
        # 3. Repeat purchase behavior
        print("\n3. Repeat Purchase Behavior:")
        
        # Calculate repeat rate for top products
        real_top_products = self.real_transactions['PRODUCT_ID'].value_counts().head(100).index
        real_repeat_rates = []
        
        for product_id in real_top_products:
            buyers = self.real_transactions[self.real_transactions['PRODUCT_ID'] == product_id]['household_key']
            repeat_rate = (buyers.value_counts() > 1).mean()
            real_repeat_rates.append(repeat_rate)
        
        real_avg_repeat = np.mean(real_repeat_rates)
        
        # Same for synthetic
        synth_top_products = self.synthetic_data['transactions']['product_id'].value_counts().head(100).index
        synth_repeat_rates = []
        
        for product_id in synth_top_products:
            buyers = self.synthetic_data['transactions'][self.synthetic_data['transactions']['product_id'] == product_id]['customer_id']
            repeat_rate = (buyers.value_counts() > 1).mean()
            synth_repeat_rates.append(repeat_rate)
        
        synth_avg_repeat = np.mean(synth_repeat_rates)
        
        error = abs(real_avg_repeat - synth_avg_repeat) / real_avg_repeat
        
        results['product_repeat_rate'] = {
            'real': real_avg_repeat,
            'synthetic': synth_avg_repeat,
            'error': error,
            'passed': error < 0.20
        }
        print(f"  Avg repeat purchase rate: Real={real_avg_repeat*100:.1f}%, Synth={synth_avg_repeat*100:.1f}%, Error={error*100:.1f}% {'✅' if error < 0.20 else '❌'}")
        
        return results
    
    def validate_predictive_accuracy(self) -> Dict:
        """
        Level 4 Validation: ML model performance comparison
        
        Train models on synthetic data, test on real data
        """
        print("\n" + "="*70)
        print("LEVEL 4 VALIDATION: PREDICTIVE ACCURACY")
        print("="*70)
        
        results = {}
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        
        # 1. Churn prediction task
        print("\n1. Churn Prediction Task:")
        
        # Prepare real data
        real_customer_features = self.prepare_churn_features(self.real_customers, self.real_transactions)
        X_real = real_customer_features.drop(['household_key', 'churned'], axis=1)
        y_real = real_customer_features['churned']
        
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.3, random_state=42
        )
        
        # Prepare synthetic data
        synth_customer_features = self.prepare_churn_features(
            self.synthetic_data['customers'], 
            self.synthetic_data['transactions']
        )
        X_synth = synth_customer_features.drop(['customer_id', 'churned'], axis=1)
        y_synth = synth_customer_features['churned']
        
        # Train on synthetic, test on real
        rf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_synth.fit(X_synth, y_synth)
        y_pred_from_synth = rf_synth.predict(X_test_real)
        
        # Train on real, test on real (baseline)
        rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_real.fit(X_train_real, y_train_real)
        y_pred_from_real = rf_real.predict(X_test_real)
        
        # Compare performance
        acc_synth = accuracy_score(y_test_real, y_pred_from_synth)
        acc_real = accuracy_score(y_test_real, y_pred_from_real)
        
        f1_synth = f1_score(y_test_real, y_pred_from_synth, average='weighted')
        f1_real = f1_score(y_test_real, y_pred_from_real, average='weighted')
        
        results['churn_prediction'] = {
            'accuracy_from_synthetic': acc_synth,
            'accuracy_from_real': acc_real,
            'accuracy_degradation': (acc_real - acc_synth) / acc_real,
            'f1_from_synthetic': f1_synth,
            'f1_from_real': f1_real,
            'passed': acc_synth >= 0.80 * acc_real  # Synthetic should achieve ≥80% of real performance
        }
        
        print(f"  Model trained on synthetic data:")
        print(f"    Accuracy: {acc_synth:.3f} (real baseline: {acc_real:.3f})")
        print(f"    F1-score: {f1_synth:.3f} (real baseline: {f1_real:.3f})")
        print(f"    Degradation: {((acc_real - acc_synth) / acc_real)*100:.1f}% {'✅ PASS' if acc_synth >= 0.80 * acc_real else '❌ FAIL'}")
        
        # 2. Basket value prediction
        print("\n2. Basket Value Regression:")
        
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_percentage_error, r2_score
        
        # Prepare features
        real_basket_features = self.prepare_basket_features(self.real_baskets, self.real_transactions)
        X_real_baskets = real_basket_features.drop(['basket_id', 'total_spend'], axis=1)
        y_real_baskets = real_basket_features['total_spend']
        
        X_train_baskets, X_test_baskets, y_train_baskets, y_test_baskets = train_test_split(
            X_real_baskets, y_real_baskets, test_size=0.3, random_state=42
        )
        
        synth_basket_features = self.prepare_basket_features(
            self.synthetic_data['baskets'],
            self.synthetic_data['transactions']
        )
        X_synth_baskets = synth_basket_features.drop(['basket_id', 'total_spend'], axis=1)
        y_synth_baskets = synth_basket_features['total_spend']
        
        # Train models
        gbr_synth = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr_synth.fit(X_synth_baskets, y_synth_baskets)
        y_pred_baskets_synth = gbr_synth.predict(X_test_baskets)
        
        gbr_real = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr_real.fit(X_train_baskets, y_train_baskets)
        y_pred_baskets_real = gbr_real.predict(X_test_baskets)
        
        # Compare
        mape_synth = mean_absolute_percentage_error(y_test_baskets, y_pred_baskets_synth)
        mape_real = mean_absolute_percentage_error(y_test_baskets, y_pred_baskets_real)
        
        r2_synth = r2_score(y_test_baskets, y_pred_baskets_synth)
        r2_real = r2_score(y_test_baskets, y_pred_baskets_real)
        
        results['basket_value_prediction'] = {
            'mape_from_synthetic': mape_synth,
            'mape_from_real': mape_real,
            'r2_from_synthetic': r2_synth,
            'r2_from_real': r2_real,
            'passed': mape_synth <= 1.25 * mape_real  # Within 25% of real model error
        }
        
        print(f"  Model trained on synthetic data:")
        print(f"    MAPE: {mape_synth:.3f} (real baseline: {mape_real:.3f})")
        print(f"    R²: {r2_synth:.3f} (real baseline: {r2_real:.3f})")
        print(f"    {'✅ PASS' if mape_synth <= 1.25 * mape_real else '❌ FAIL'}")
        
        return results
    
    def prepare_churn_features(self, customers_df, transactions_df):
        """Helper: Prepare features for churn prediction"""
        # Define churn as no purchase in last 13 weeks
        max_week = transactions_df['week_number'].max() if 'week_number' in transactions_df.columns else transactions_df['WEEK_NO'].max()
        
        customer_id_col = 'household_key' if 'household_key' in transactions_df.columns else 'customer_id'
        week_col = 'WEEK_NO' if 'WEEK_NO' in transactions_df.columns else 'week_number'
        
        last_purchase = transactions_df.groupby(customer_id_col)[week_col].max()
        
        features = customers_df.copy()
        id_col = 'household_key' if 'household_key' in customers_df.columns else 'customer_id'
        features['last_week'] = features[id_col].map(last_purchase)
        features['churned'] = (max_week - features['last_week']) > 13
        
        # Select numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        return features[[id_col, 'churned'] + list(numeric_cols)]
    
    def prepare_basket_features(self, baskets_df, transactions_df):
        """Helper: Prepare features for basket value prediction"""
        # Use basket-level features
        features = baskets_df.copy()
        
        # Add transaction-level aggregations if helpful
        # For now, just use existing basket features
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        basket_id_col = 'BASKET_ID' if 'BASKET_ID' in features.columns else 'basket_id'
        spend_col = 'total_spend'
        
        return features[[basket_id_col, spend_col] + [c for c in numeric_cols if c not in [basket_id_col, spend_col]]]
    
    def generate_validation_report(self, output_file: str = './validation_report.txt'):
        """Generate comprehensive validation report"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VALIDATION REPORT")
        print("="*70)
        
        # Run all validation levels
        level1 = self.validate_distributions()
        level2 = self.validate_aggregate_statistics()
        level3 = self.validate_behavioral_patterns()
        level4 = self.validate_predictive_accuracy()
        
        # Compile results
        all_results = {
            'level1_distributions': level1,
            'level2_aggregates': level2,
            'level3_behavior': level3,
            'level4_predictive': level4
        }
        
        # Calculate overall pass rate
        all_tests = []
        for level, tests in all_results.items():
            for test_name, result in tests.items():
                if isinstance(result, dict) and 'passed' in result:
                    all_tests.append(result['passed'])
        
        pass_rate = sum(all_tests) / len(all_tests) if all_tests else 0
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SYNTHETIC DATA VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Overall Pass Rate: {pass_rate*100:.1f}% ({sum(all_tests)}/{len(all_tests)} tests passed)\n\n")
            
            f.write("VALIDATION LEVELS:\n")
            f.write("  Level 1: Distribution Matching (statistical tests)\n")
            f.write("  Level 2: Aggregate Statistics (mean, std, etc.)\n")
            f.write("  Level 3: Behavioral Patterns (associations, loyalty)\n")
            f.write("  Level 4: Predictive Accuracy (ML models)\n\n")
            
            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for level, tests in all_results.items():
                f.write(f"\n{level.upper()}\n")
                f.write("-"*70 + "\n")
                for test_name, result in tests.items():
                    f.write(f"\n{test_name}:\n")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {result}\n")
        
        print(f"\n✅ Validation report saved to {output_file}")
        print(f"\nOVERALL RESULT: {pass_rate*100:.1f}% of tests passed")
        
        if pass_rate >= 0.80:
            print("🎉 VALIDATION SUCCESSFUL - Synthetic data is sufficiently accurate")
        elif pass_rate >= 0.60:
            print("⚠️  VALIDATION PARTIAL - Further calibration recommended")
        else:
            print("❌ VALIDATION FAILED - Significant recalibration needed")
        
        return all_results, pass_rate


# Usage
if __name__ == "__main__":
    # Load synthetic datasets (generated with calibrated parameters)
    synthetic_datasets = {
        'transactions': pd.read_parquet('./synthetic_output/transactions.parquet'),
        'baskets': pd.read_parquet('./synthetic_output/baskets.parquet'),
        'customers': pd.read_parquet('./synthetic_output/customers.parquet')
    }
    
    # Run validation
    validator = SyntheticDataValidator(
        real_data_dir='./calibration_data',
        synthetic_datasets=synthetic_datasets
    )
    
    results, pass_rate = validator.generate_validation_report()
    
    print("\n✅ VALIDATION COMPLETE")
    print(f"Pass rate: {pass_rate*100:.1f}%")
```

---

## Phase 5: Iterative Refinement

### Calibration-Validation Loop

```python
def iterative_refinement_workflow(max_iterations: int = 10, target_pass_rate: float = 0.80):
    """
    Iterative loop: Calibrate → Validate → Refine → Repeat
    
    Args:
        max_iterations: Maximum number of refinement iterations
        target_pass_rate: Target validation pass rate (e.g., 80%)
    
    Returns:
        Final calibrated parameters
    """
    print("="*70)
    print("ITERATIVE REFINEMENT WORKFLOW")
    print("="*70)
    
    iteration = 0
    pass_rate = 0.0
    best_params = None
    best_pass_rate = 0.0
    
    while iteration < max_iterations and pass_rate < target_pass_rate:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        # Step 1: Calibration
        print("\n📐 Running calibration...")
        if iteration == 1:
            # First iteration: manual or coarse optimization
            study = run_coarse_optimization(n_trials=50)
        else:
            # Subsequent iterations: fine-tune around best parameters
            study = run_fine_tuning(best_params, n_trials=30)
        
        best_params = study.best_trial.params
        
        # Step 2: Generate full synthetic dataset with best parameters
        print("\n🔄 Generating full synthetic dataset...")
        config = apply_parameters(best_params)
        generator = EnhancedRetailSynthV4_0(config)
        synthetic_datasets = generator.generate_all_datasets()
        
        # Step 3: Validation
        print("\n✅ Running validation...")
        validator = SyntheticDataValidator('./calibration_data', synthetic_datasets)
        results, pass_rate = validator.generate_validation_report(
            f'./validation_iteration_{iteration}.txt'
        )
        
        print(f"\n📊 Iteration {iteration} Results:")
        print(f"   Pass rate: {pass_rate*100:.1f}%")
        
        if pass_rate > best_pass_rate:
            best_pass_rate = pass_rate
            print(f"   🎉 New best pass rate!")
            # Save best parameters
            save_parameters(best_params, f'./best_params_iter_{iteration}.json')
        
        # Step 4: Analyze failures and adjust strategy
        if pass_rate < target_pass_rate:
            print("\n🔍 Analyzing validation failures...")
            failure_analysis = analyze_validation_failures(results)
            print(f"   Key issues: {failure_analysis['top_issues']}")
            
            # Adjust optimization strategy for next iteration
            # (e.g., focus on parameters affecting failed tests)
        
        print(f"\n{'='*70}")
        print(f"Iteration {iteration} complete. Pass rate: {pass_rate*100:.1f}%")
        print(f"{'='*70}")
    
    print("\n" + "="*70)
    print("ITERATIVE REFINEMENT COMPLETE")
    print("="*70)
    print(f"\nFinal pass rate: {pass_rate*100:.1f}%")
    print(f"Best pass rate achieved: {best_pass_rate*100:.1f}%")
    print(f"Iterations run: {iteration}")
    
    if pass_rate >= target_pass_rate:
        print("\n🎉 SUCCESS - Target validation achieved!")
    else:
        print("\n⚠️  Target not reached - consider:")
        print("   1. Reviewing failed tests in validation reports")
        print("   2. Expanding parameter search space")
        print("   3. Adding new behavioral features to generator")
    
    return best_params, best_pass_rate
```

---

## Phase 6: Documentation & Reporting

### Final Calibration Report

```python
def generate_final_calibration_report(best_params, validation_results, output_file='./CALIBRATION_REPORT.md'):
    """
    Generate comprehensive calibration report for publication/documentation
    """
    with open(output_file, 'w') as f:
        f.write("# Enhanced RetailSynth Calibration Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Dataset Used:** Dunnhumby Complete Journey\n")
        f.write(f"- **Calibration Method:** Optuna Bayesian Optimization + Manual Refinement\n")
        f.write(f"- **Validation Pass Rate:** {validation_results['pass_rate']*100:.1f}%\n")
        f.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        f.write("## Calibrated Parameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(best_params, indent=2))
        f.write("\n```\n\n")
        
        f.write("## Validation Results\n\n")
        f.write("### Distribution Matching\n\n")
        f.write("| Metric | KS Statistic | P-Value | Status |\n")
        f.write("|--------|--------------|---------|--------|\n")
        # Add validation details
        
        f.write("\n### Aggregate Statistics\n\n")
        f.write("| Metric | Real | Synthetic | Error | Status |\n")
        f.write("|--------|------|-----------|-------|--------|\n")
        # Add aggregate stats
        
        f.write("\n### Predictive Accuracy\n\n")
        f.write("| Task | Real Model | Synthetic Model | Degradation | Status |\n")
        f.write("|------|------------|-----------------|-------------|--------|\n")
        # Add ML results
        
        f.write("\n## Limitations & Caveats\n\n")
        f.write("- Parameters calibrated to grocery retail (Complete Journey dataset)\n")
        f.write("- May not generalize to other retail categories\n")
        f.write("- Synthetic data should be revalidated for specific use cases\n\n")
        
        f.write("## Usage Recommendations\n\n")
        f.write("### Appropriate Use Cases\n")
        f.write("- Algorithm benchmarking and testing\n")
        f.write("- ML model training (with validation on real data)\n")
        f.write("- Scenario analysis and simulation\n\n")
        
        f.write("### Inappropriate Use Cases\n")
        f.write("- Production forecasting without validation\n")
        f.write("- Financial planning or budgeting\n")
        f.write("- Regulatory compliance reporting\n\n")
        
        f.write("## Citation\n\n")
        f.write("If you use this calibrated synthetic data generator, please cite:\n\n")
        f.write("```\n")
        f.write("Enhanced RetailSynth (2025). Calibrated synthetic retail data generator.\n")
        f.write("Calibration based on Dunnhumby Complete Journey dataset.\n")
        f.write("Original methodology: Xia et al. (2023) RetailSynth.\n")
        f.write("```\n")
    
    print(f"✅ Final calibration report saved to {output_file}")
```

---

## Success Metrics

### Validation Acceptance Criteria

**Minimum Requirements (to claim "validated"):**

| Level | Metric | Target | Priority |
|-------|--------|--------|----------|
| L1 | KS test p-value (visit frequency) | > 0.05 | HIGH |
| L1 | KS test p-value (basket size) | > 0.05 | HIGH |
| L1 | KS test p-value (basket value) | > 0.05 | HIGH |
| L2 | Aggregate metric error | < 10% | HIGH |
| L3 | Basket association overlap | > 30% | MEDIUM |
| L4 | ML model accuracy degradation | < 20% | HIGH |

**Target: ≥ 80% of tests passing**

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: KS Tests Consistently Failing

**Problem:** Distributions don't match statistically

**Diagnosis:**
```python
# Plot Q-Q plots to visualize distribution differences
from scipy.stats import probplot

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
probplot(real_visits, plot=ax[0])
probplot(synth_visits, plot=ax[1])
ax[0].set_title('Real Data Q-Q Plot')
ax[1].set_title('Synthetic Data Q-Q Plot')
```

**Solutions:**
1. Check if wrong distribution family (e.g., normal vs. lognormal)
2. Adjust distribution parameters in code
3. Add mixture models for bimodal/multimodal distributions

#### Issue 2: Aggregate Stats Match but Distributions Don't

**Problem:** Means/std correct but shape is wrong

**Diagnosis:**
```python
# Check higher moments
print("Skewness - Real:", stats.skew(real_visits), "Synth:", stats.skew(synth_visits))
print("Kurtosis - Real:", stats.kurtosis(real_visits), "Synth:", stats.kurtosis(synth_visits))
```

**Solutions:**
1. Use distribution transformations (e.g., Box-Cox)
2. Add skew parameter to generation process
3. Consider mixture of distributions

#### Issue 3: Temporal Patterns Don't Match

**Problem:** Synthetic data lacks seasonality

**Diagnosis:**
```python
# Decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomp_real = seasonal_decompose(real_weekly_revenue, model='additive', period=52)
decomp_synth = seasonal_decompose(synth_weekly_revenue, model='additive', period=52)

# Compare seasonal components
plt.plot(decomp_real.seasonal, label='Real')
plt.plot(decomp_synth.seasonal, label='Synthetic')
plt.legend()
```

**Solutions:**
1. Strengthen seasonality multipliers in code
2. Add explicit holiday boost factors
3. Calibrate seasonal parameters to real data

#### Issue 4: ML Models Perform Poorly on Synthetic Data

**Problem:** Models trained on synthetic fail on real data

**Diagnosis:**
```python
# Check feature importance alignment
feature_importance_real = rf_real.feature_importances_
feature_importance_synth = rf_synth.feature_importances_

# Should be correlated
correlation = np.corrcoef(feature_importance_real, feature_importance_synth)[0, 1]
print(f"Feature importance correlation: {correlation:.3f}")
```

**Solutions:**
1. Missing key behavioral features in synthetic data
2. Feature distributions misaligned
3. Add more complex interaction terms

---

## Summary Checklist

### Pre-Calibration ✅
- [ ] Dunnhumby data downloaded and unzipped
- [ ] Preprocessing script run successfully
- [ ] Calibration targets extracted
- [ ] EDA completed and distributions understood

### Calibration ✅
- [ ] Manual calibration attempted
- [ ] Optuna optimization run (≥100 trials)
- [ ] Best parameters identified
- [ ] Calibration history saved

### Validation ✅
- [ ] Level 1 (distributions) validation run
- [ ] Level 2 (aggregates) validation run
- [ ] Level 3 (behavior) validation run
- [ ] Level 4 (ML models) validation run
- [ ] Validation report generated

### Documentation ✅
- [ ] Calibration parameters documented
- [ ] Validation results documented
- [ ] Limitations documented
- [ ] Usage guidelines created
- [ ] Citation information prepared

### Publication ✅
- [ ] Code updated with calibrated parameters
- [ ] README updated with validation results
- [ ] Limitations clearly stated
- [ ] Academic paper (optional)

---

## Estimated Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Data Prep | 1-2 days | Low |
| EDA | 2-3 days | Medium |
| Manual Calibration | 3-5 days | High |
| Automated Optimization | 1-2 days | Low (compute time) |
| Validation | 2-3 days | Medium |
| Iteration | 5-10 days | High |
| Documentation | 2-3 days | Medium |
| **Total** | **16-28 days** | **~3-4 weeks** |

---

## Final Notes

**This calibration process will transform your synthetic data generator from an unvalidated prototype into a statistically rigorous research tool.**

Key success factors:
1. **Patience**: Calibration is iterative - expect 5-10 refinement cycles
2. **Rigor**: Don't skip validation steps - they're critical for credibility
3. **Documentation**: Record every calibration attempt for reproducibility
4. **Realism**: Accept that 100% match is impossible - aim for "good enough" (80%+ pass rate)
5. **Transparency**: Clearly communicate limitations to users

**Good luck with your calibration! 🎯**
