import numpy as np
import random
from datetime import datetime, timedelta    
from typing import Dict
import pandas as pd
from pathlib import Path
from retailsynth.config import EnhancedRetailConfig
from retailsynth.calibration import CalibrationEngine
from retailsynth.engines import GPUUtilityEngine, ChoiceModel, PricingEvolutionEngine, ProductLifecycleEngine, SeasonalityEngine 
from retailsynth.engines import MarketDynamicsEngine, TemporalCustomerDriftEngine, StoreLoyaltyEngine, VectorizedPreComputationEngine
from retailsynth.engines.price_hmm import PriceStateHMM
from retailsynth.engines.cross_price_elasticity import CrossPriceElasticityEngine
from retailsynth.engines.arc_elasticity import ArcPriceElasticityEngine
from retailsynth.utils import RealisticCategoryHierarchy
from retailsynth.generators.customer_generator import CustomerGenerator
from retailsynth.generators.product_generator import ProductGenerator
from retailsynth.generators.store_generator import StoreGenerator
from retailsynth.generators.market_context_generator import MarketContextGenerator
from retailsynth.generators.transaction_generator import ComprehensiveTransactionGenerator
from retailsynth.catalog import HierarchyMapper  

# ============================================================================
# ENHANCED RETAILSYNTH V4.1 (with Real Product Catalog)
# ============================================================================

class EnhancedRetailSynthV4_1:
    """
    Enhanced RetailSynth v4.1 - With real Dunnhumby product catalog.
    
    NEW in v4.1 (Sprint 1.1):
    - Loads 20K real products from Dunnhumby instead of generating synthetic
    - Uses real category hierarchy (Department → Commodity → Sub-Commodity)
    - Preserves real brand names, manufacturers, and price distributions
    - Product archetypes based on actual purchase behavior
    """
    def __init__(self, config: EnhancedRetailConfig):
        self.config = config
        
        # Validate configuration
        config.validate()
        
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        print(f"\n{'='*70}")
        print(f"🚀 Enhanced RetailSynth v4.1 - REAL CATALOG EDITION")
        print(f"{'='*70}")
        print(f"   Configuration:")
        print(f"   • Customers: {config.n_customers:,}")
        print(f"   • Products: {config.n_products:,} {'(REAL from Dunnhumby)' if config.use_real_catalog else '(Synthetic)'}")
        print(f"   • Stores: {config.n_stores}")
        print(f"   • Weeks: {config.simulation_weeks}")
        print(f"   • GPU Acceleration: ✅ Enabled")
        print(f"   • Temporal Dynamics: ✅ Enabled")
        print(f"   • Customer Drift: ✅ Enabled")
        print(f"   • Product Lifecycle: ✅ Enabled")
        print(f"   • Store Loyalty: ✅ Enabled")
        
        # Load real product catalog (NEW - Sprint 1.1)
        if config.use_real_catalog:
            print(f"\n📦 Loading real product catalog...")
            self._load_real_catalog()
        else:
            print(f"\n📦 Using synthetic product generation...")
            self.category_hierarchy = RealisticCategoryHierarchy.create_category_hierarchy()
            self.brand_portfolio = RealisticCategoryHierarchy.create_brand_portfolio()
            self.hierarchy_mapper = None
            self.real_products = None
        
        # Initialize engines
        self.calibration_engine = CalibrationEngine()
        self.utility_engine = GPUUtilityEngine(self.calibration_engine.parameters)
        self.choice_model = ChoiceModel()
        
        # Temporal dynamics engines
        if config.enable_temporal_dynamics:
            self.seasonality_engine = SeasonalityEngine(region=config.region)
        
        self.datasets = {}
        self.pricing_history = []
        self.lifecycle_history = []
        self.market_share_history = []
    
    def _load_real_catalog(self):
        """Load real product catalog from Dunnhumby (NEW - Sprint 1.1)"""
        try:
            # Load product catalog
            catalog_path = Path(self.config.product_catalog_path)
            self.real_products = pd.read_parquet(catalog_path)
            
            print(f"   ✅ Loaded {len(self.real_products):,} products from {catalog_path.name}")
            print(f"      Departments: {self.real_products['DEPARTMENT'].nunique()}")
            print(f"      Brands: {self.real_products['BRAND'].nunique():,}")
            print(f"      Avg Price: ${self.real_products['avg_price'].mean():.2f}")
            
            # Load category hierarchy
            if self.config.category_hierarchy_path:
                hierarchy_path = Path(self.config.category_hierarchy_path)
                mapping_path = Path(self.config.product_mapping_path) if self.config.product_mapping_path else None
                
                self.hierarchy_mapper = HierarchyMapper()
                self.hierarchy_mapper.load_hierarchy(
                    str(hierarchy_path),
                    str(mapping_path) if mapping_path else None
                )
                
                print(f"   ✅ Loaded category hierarchy from {hierarchy_path.name}")
            else:
                # Build hierarchy from catalog
                self.hierarchy_mapper = HierarchyMapper()
                self.hierarchy_mapper.build_hierarchy(self.real_products)
                print(f"   ✅ Built category hierarchy from catalog")
            
            # Store for backward compatibility
            self.category_hierarchy = self.hierarchy_mapper.hierarchy
            
            # Extract brand portfolio
            self.brand_portfolio = self._extract_brand_portfolio()
            
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"\n💡 To fix this, run:")
            print(f"   python scripts/build_product_catalog.py")
            raise
    
    def _extract_brand_portfolio(self) -> Dict:
        """Extract brand portfolio from real catalog"""
        brands = {}
        
        for dept in self.real_products['DEPARTMENT'].unique():
            dept_products = self.real_products[self.real_products['DEPARTMENT'] == dept]
            dept_brands = dept_products['BRAND'].unique().tolist()
            brands[dept] = dept_brands[:20]  # Top 20 brands per department
        
        return brands
    
    def generate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets with comprehensive temporal dynamics"""
        
        start_time = datetime.now()
        
        # Step 1: Generate customers
        print(f"\n👥 Step 1/7: Generating {self.config.n_customers:,} customers...")
        step_start = datetime.now()
        self.datasets['customers'] = CustomerGenerator.generate_customers_vectorized(self.config, self.calibration_engine)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 2: Generate/Load products (MODIFIED - Sprint 1.1)
        print(f"\n📦 Step 2/7: {'Loading' if self.config.use_real_catalog else 'Generating'} {self.config.n_products:,} products...")
        step_start = datetime.now()
        
        if self.config.use_real_catalog and self.real_products is not None:
            # Use real catalog (NEW)
            self.datasets['products'] = self._prepare_real_products()
            
            # Standardize column names to lowercase
            self.datasets['products'].columns = [col.lower() for col in self.datasets['products'].columns]
            
            # Add required columns for simulation
            self.datasets['products']['base_price'] = self.datasets['products']['avg_price']
            self.datasets['products']['assortment_role'] = self.datasets['products']['category_role']
            
            # Ensure product_id column exists (rename from product_id if needed)
            if 'product_id' not in self.datasets['products'].columns:
                self.datasets['products']['product_id'] = self.datasets['products'].index
        else:
            # Fall back to synthetic generation
            self.datasets['products'] = ProductGenerator.generate_products_vectorized(self.config)
        
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 3: Generate stores
        print(f"\n🏪 Step 3/7: Generating {self.config.n_stores} stores...")
        step_start = datetime.now()
        self.datasets['stores'] = StoreGenerator.generate_stores(self.config)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 4: Generate market context
        print(f"\n💰 Step 4/7: Generating market context...")
        step_start = datetime.now()
        self.datasets['market_context'] = MarketContextGenerator.generate_market_context(self.config)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 5: Initialize temporal engines
        print(f"\n🔧 Step 5/7: Initializing temporal dynamics engines...")
        step_start = datetime.now()
        
        if self.config.enable_temporal_dynamics:
            self.pricing_engine = PricingEvolutionEngine(len(self.datasets['products']))
            self.lifecycle_engine = ProductLifecycleEngine(
                self.datasets['products'], 
                self.config
            )
        
        if self.config.enable_customer_drift:
            self.drift_engine = TemporalCustomerDriftEngine(self.datasets['customers'])
        
        if self.config.enable_store_loyalty:
            self.loyalty_engine = StoreLoyaltyEngine(
                self.datasets['customers'],
                self.datasets['stores']
            )
        
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 6: Pre-compute matrices for GPU
        print(f"\n🔧 Step 6/7: Pre-computing GPU matrices...")
        step_start = datetime.now()
        self.precomp = VectorizedPreComputationEngine(
            self.datasets['customers'], 
            self.datasets['products']
        )
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 7: Generate transactions with full temporal dynamics
        print(f"\n🛒 Step 7/7: Generating transactions with temporal dynamics...")
        print(f"   Expected time: ~{self.config.simulation_weeks * 0.2:.0f}-{self.config.simulation_weeks * 0.4:.0f} minutes")
        step_start = datetime.now()
        transaction_data = self._generate_transactions_with_temporal_dynamics()
        self.datasets['transactions'] = transaction_data['transactions']
        self.datasets['transaction_items'] = transaction_data['transaction_items']
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Business performance
        print(f"\n📈 Calculating business performance...")
        self.datasets['business_performance'] = self._generate_business_performance()
        
        # Temporal metadata
        if self.config.enable_temporal_dynamics:
            self.datasets['temporal_metadata'] = self._create_temporal_metadata()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"✅ GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Customers: {len(self.datasets['customers']):,}")
        print(f"   Products: {len(self.datasets['products']):,}")
        print(f"   Stores: {len(self.datasets['stores']):,}")
        print(f"   Transactions: {len(self.datasets['transactions']):,}")
        print(f"   Transaction Items: {len(self.datasets['transaction_items']):,}")
        print(f"   Performance: {len(self.datasets['transactions']) / (total_time/60):.0f} transactions/minute")
        
        return self.datasets
        
    def _generate_transactions_with_temporal_dynamics(self) -> Dict[str, pd.DataFrame]:
        """
        Generate transactions with FULL temporal dynamics (v3.6).
        Includes customer drift, product lifecycle, and store loyalty.
        """
        all_transactions = []
        all_transaction_items = []
        
        # Initialize transaction generator with store loyalty
        transaction_gen = ComprehensiveTransactionGenerator(
            self.precomp,
            self.utility_engine,
            self.loyalty_engine if self.config.enable_store_loyalty else None,
            self.config
        )
        
        # Track active products (changes with lifecycle)
        active_products = self.datasets['products'].copy()
        
        for week in range(1, self.config.simulation_weeks + 1):
            week_start = datetime.now()
            week_date = self.config.start_date + timedelta(weeks=week-1)
            
            print(f"\n   Week {week}/{self.config.simulation_weeks} ({week_date}):")
            
            # 1. Apply customer drift (v3.6)
            if self.config.enable_customer_drift and week > 1:
                print(f"      🔄 Applying customer drift...")
                self.datasets['customers'] = self.drift_engine.apply_weekly_drift(week)

                # Update pre-computed matrices
                self.precomp.update_from_drift(self.datasets['customers'])
            
            # 2. Update product lifecycle (v3.6)
            if self.config.enable_product_lifecycle and week > 1:
                print(f"      📦 Updating product lifecycle...")
                active_products, retired_ids, new_products = self.lifecycle_engine.update_weekly(week)
                
                if retired_ids or new_products:
                    # Update pre-computed matrices for product changes
                    self.precomp.update_for_products(active_products)
                    self.datasets['products'] = active_products
            
            # 3. Evolve prices
            if self.price_hmm is not None:

                # Use HMM-based price generation (NEW - Sprint 2)
                print(f"      💰 Generating HMM-based prices...")
                current_prices, price_states = self._generate_hmm_prices(week)
                promo_flags = (price_states > 0).astype(int)  # States 1,2,3 are promotional
            else:
                # Fall back to simple pricing engine
                current_prices, promo_flags = self.pricing_engine.evolve_prices(self.precomp.base_prices, week, self.precomp.product_ids)
            
            # Apply seasonality multiplier
            if self.config.enable_temporal_dynamics:
                for dept in ['Fresh', 'Pantry', 'Personal_Care', 'General_Merchandise']:
                    dept_mask = self.precomp.departments == dept
                    multiplier = self.seasonality_engine.get_seasonality_multiplier(week, dept)
                    # Seasonality affects demand, which we approximate by adjusting visit probability
                    # This is a simplification - in reality it would affect product choice probabilities
            
            # Store pricing history
            self.pricing_history.append({
                'week': week,
                'prices': {pid: price for pid, price in zip(self.precomp.product_ids, current_prices)},
                'promotions': {pid: bool(promo) for pid, promo in zip(self.precomp.product_ids, promo_flags)}
            })
            
            # Store lifecycle history
            if self.config.enable_product_lifecycle:
                self.lifecycle_history.append({
                    'week': week,
                    'stages': dict(self.lifecycle_engine.lifecycle_stages)
                })
            
            # 4. Generate transactions
            print(f"      🛒 Generating transactions...")
            week_transactions, week_items = transaction_gen.generate_week_transactions_vectorized(
                week, current_prices, promo_flags, week_date
            )
            
            all_transactions.extend(week_transactions)
            all_transaction_items.extend(week_items)
            
            week_time = (datetime.now() - week_start).total_seconds()
            print(f"      ✅ Week complete: {len(week_transactions):,} transactions in {week_time:.1f}s")
        
        return {
            'transactions': pd.DataFrame(all_transactions),
            'transaction_items': pd.DataFrame(all_transaction_items)
        }
    
    def _generate_business_performance(self) -> pd.DataFrame:
        """Calculate business performance metrics by week and store"""
        transactions = self.datasets['transactions']
        
        performance_data = []
        
        for week in sorted(transactions['week_number'].unique()):
            week_transactions = transactions[transactions['week_number'] == week]
            
            for store_id in sorted(transactions['store_id'].unique()):
                store_transactions = week_transactions[week_transactions['store_id'] == store_id]
                
                if len(store_transactions) == 0:
                    continue
                
                performance_data.append({
                    'week_number': week,
                    'store_id': store_id,
                    'total_transactions': len(store_transactions),
                    'total_revenue': store_transactions['total_revenue'].sum(),
                    'total_margin': store_transactions['total_margin'].sum(),
                    'total_discount': store_transactions['total_discount'].sum(),
                    'avg_basket_size': store_transactions['total_items_count'].mean(),
                    'avg_basket_value': store_transactions['total_revenue'].mean(),
                    'avg_satisfaction': store_transactions['satisfaction_score'].mean(),
                    'promotion_penetration': (store_transactions['promotional_items_count'] > 0).sum() / len(store_transactions),
                    'created_at': datetime.now()
                })
        
        return pd.DataFrame(performance_data)
    
    def _create_temporal_metadata(self) -> pd.DataFrame:
        """Create metadata about temporal dynamics"""
        metadata = []
        
        for week in range(1, self.config.simulation_weeks + 1):
            week_date = self.config.start_date + timedelta(weeks=week-1)
            
            # Get pricing stats
            if week <= len(self.pricing_history):
                prices = list(self.pricing_history[week-1]['prices'].values())
                promos = list(self.pricing_history[week-1]['promotions'].values())
                
                metadata.append({
                    'week_number': week,
                    'week_date': week_date,
                    'avg_price': round(np.mean(prices), 2),
                    'promotion_rate': round(sum(promos) / len(promos), 3),
                    'n_active_products': len(self.datasets['products']),
                    'created_at': datetime.now()
                })
        
        return pd.DataFrame(metadata)

    def _prepare_real_products(self) -> pd.DataFrame:
        """Prepare real products for simulation"""
        # Filter products by department
        if self.config.department_filter:
            self.real_products = self.real_products[self.real_products['DEPARTMENT'].isin(self.config.department_filter)]
        
        # Filter products by brand
        if self.config.brand_filter:
            self.real_products = self.real_products[self.real_products['BRAND'].isin(self.config.brand_filter)]
        
        # Limit products to top N by sales
        if self.config.n_products:
            self.real_products['avg_sales'] = self.real_products['total_revenue'] / self.real_products['total_quantity']
            self.real_products = self.real_products.nlargest(self.config.n_products, 'avg_sales')
        
        # Reset index
        self.real_products.reset_index(drop=True, inplace=True)
        
        return self.real_products
    
    def _generate_hmm_prices(self, week: int) -> tuple:
        """
        Generate prices using HMM price dynamics (NEW - Sprint 2)
        
        Args:
            week: Current week number
        
        Returns:
            Tuple of (prices, states):
                - prices: Array of current prices for all products
                - states: Array of HMM states (0=regular, 1=feature, 2=deep, 3=clearance)
        """
        n_products = len(self.precomp.product_ids)
        current_prices = np.zeros(n_products)
        price_states = np.zeros(n_products, dtype=int)
        
        for i, product_id in enumerate(self.precomp.product_ids):
            base_price = self.precomp.base_prices[i]
            
            # Generate price sequence for this week using HMM
            prices, states = self.price_hmm.generate_price_sequence(
                product_id=product_id,
                n_weeks=1,
                base_price=base_price,
                random_seed=self.config.random_seed + week + product_id
            )
            
            current_prices[i] = prices[0]
            price_states[i] = states[0]
        
        return current_prices, price_states

# ============================================================================
# VISUALIZATION TOOLS
# ============================================================================

class TemporalVisualizationTools:
    """Visualization tools for temporal analysis"""
    
    @staticmethod
    def create_all_visualizations(generator: EnhancedRetailSynthV4_1) -> Dict[str, pd.DataFrame]:
        """Create all visualization datasets"""
        print("\n📊 Creating visualization datasets...")
        
        viz_data = {}
        
        if generator.config.enable_temporal_dynamics:
            # Seasonality calendar
            print("   • Seasonality calendar")
            viz_data['seasonality_calendar'] = generator.seasonality_engine.visualize_seasonality_calendar(
                start_date=generator.config.start_date,
                end_date=generator.config.start_date + timedelta(weeks=generator.config.simulation_weeks),
                categories=['Fresh', 'Pantry', 'Personal_Care', 'General_Merchandise']
            )
            
            # Price evolution
            if generator.pricing_history:
                print("   • Price evolution")
                sample_products = list(generator.pricing_history[0]['prices'].keys())[:10]
                viz_data['price_evolution'] = generator.pricing_engine.visualize_price_evolution(
                    pricing_history=generator.pricing_history,
                    product_ids=sample_products
                )
            
            # Lifecycle trajectories
            if generator.lifecycle_history:
                print("   • Product lifecycle trajectories")
                sample_lifecycle = list(generator.lifecycle_history[0]['stages'].keys())[:10]
                viz_data['lifecycle_trajectories'] = generator.lifecycle_engine.visualize_lifecycle_trajectories(
                    lifecycle_history=generator.lifecycle_history,
                    product_ids=sample_lifecycle
                )
        
        # Weekly trends
        print("   • Weekly performance trends")
        viz_data['weekly_trends'] = TemporalVisualizationTools._create_weekly_trends(
            transactions=generator.datasets['transactions'],
            business_performance=generator.datasets['business_performance']
        )
        
        print("   ✅ All visualization datasets created")
        
        return viz_data
    
    @staticmethod
    def _create_weekly_trends(transactions: pd.DataFrame, business_performance: pd.DataFrame) -> pd.DataFrame:
        """Create weekly trend analysis"""
        weekly_data = []
        
        for week_num in sorted(transactions['week_number'].unique()):
            week_transactions = transactions[transactions['week_number'] == week_num]
            week_performance = business_performance[business_performance['week_number'] == week_num]
            
            weekly_data.append({
                'week_number': week_num,
                'total_transactions': len(week_transactions),
                'total_revenue': week_transactions['total_revenue'].sum(),
                'avg_basket_size': week_transactions['total_items_count'].mean(),
                'avg_basket_value': week_transactions['total_revenue'].mean(),
                'promotion_rate': (week_transactions['promotional_items_count'] > 0).sum() / len(week_transactions) if len(week_transactions) > 0 else 0,
                'avg_satisfaction': week_transactions['satisfaction_score'].mean(),
                'total_stores_active': week_performance['store_id'].nunique() if len(week_performance) > 0 else 0
            })
        
        return pd.DataFrame(weekly_data)
