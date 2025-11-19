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
from retailsynth.engines.customer_state import CustomerStateManager, initialize_customer_states, get_depletion_rates_by_assortment
from retailsynth.engines.purchase_history_engine import PurchaseHistoryEngine
from retailsynth.utils import RealisticCategoryHierarchy
from retailsynth.generators.customer_generator import CustomerGenerator
from retailsynth.generators.product_generator import ProductGenerator
from retailsynth.generators.store_generator import StoreGenerator
from retailsynth.generators.market_context_generator import MarketContextGenerator
from retailsynth.generators.transaction_generator import ComprehensiveTransactionGenerator
from retailsynth.catalog import HierarchyMapper  

# Sprint 1.4: Import basket composition component   
from retailsynth.engines.basket_composer import BasketComposer

# Sprint 2.1: Import promotional engine (Phase 2.1)
from retailsynth.engines.promotional_engine import PromotionalEngine, StorePromoContext
from retailsynth.engines.promo_response import PromoResponseCalculator

# Phase 2.6: Import non-linear utility engine
from retailsynth.engines.nonlinear_utility import NonLinearUtilityEngine, NonLinearUtilityConfig

# Phase 2.7: Import seasonality learning engine
from retailsynth.engines.seasonality_learning import LearnedSeasonalityEngine


# ============================================================================
# ENHANCED RETAILSYNTH V4.1 (with Real Product Catalog + Purchase History)
# ============================================================================

class EnhancedRetailSynthV4_1:
    """
    Enhanced RetailSynth v4.1 - With real Dunnhumby product catalog.
    
    NEW in v4.1 (Sprint 1.1):
    - Loads 20K real products from Dunnhumby instead of generating synthetic
    - Uses real category hierarchy (Department → Commodity → Sub-Commodity)
    - Preserves real brand names, manufacturers, and price distributions
    - Product archetypes based on actual purchase behavior
    
    NEW in Sprint 1.3:
    - Purchase history tracking (brand loyalty, habits, inventory)
    - State-dependent shopping behavior
    - Realistic repeat purchase patterns
    
    NEW in Sprint 2.1:
    - Separate pricing and promotional engines
    - Comprehensive promotional system (mechanics, displays, features)
    - Store-specific promotional contexts
    """
    def __init__(self, config: EnhancedRetailConfig):
        self.config = config
        
        # Validate configuration
        config.validate()
        
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        print(f"\n{'='*70}")
        print(f"🚀 Enhanced RetailSynth v4.1 - REAL CATALOG + PURCHASE HISTORY")
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
        print(f"   • Purchase History: ✅ Enabled (Sprint 1.3)")
        print(f"   • Promotional Engine: ✅ Enabled (Sprint 2.1)")
        
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
        self.utility_engine = GPUUtilityEngine(
            self.calibration_engine.parameters,
            enable_nonlinear=config.enable_nonlinear_utilities,
            use_log_price=config.use_log_price,
            use_quadratic_quality=config.use_quadratic_quality,
            config=config,                      # Phase 2: Enable recursive visit probability
            n_customers=config.n_customers      # Phase 2: For state tracking
        )
        self.choice_model = ChoiceModel()

        # Phase 2.6: Non-linear utility engine (Sprint 2)
        if config.enable_nonlinear_utilities:
            nonlinear_config = NonLinearUtilityConfig(
                use_log_price=config.use_log_price,
                use_reference_prices=config.use_reference_prices,
                use_psychological_thresholds=config.use_psychological_thresholds,
                use_quadratic_quality=config.use_quadratic_quality,
                loss_aversion_lambda=config.loss_aversion_lambda,
                ewma_alpha=config.ewma_alpha
            )
            self.nonlinear_engine = NonLinearUtilityEngine(nonlinear_config)
            print(f"   • Non-Linear Utilities: ✅ Enabled (Phase 2.6)")
        else:
            self.nonlinear_engine = None
            print(f"   • Non-Linear Utilities: ❌ Disabled (using linear)")

        # Phase 2.7: Initialize seasonality engine (learned or hard-coded)
        if config.enable_seasonality_learning:
            self.seasonality_engine = LearnedSeasonalityEngine(
                seasonal_patterns_path=config.seasonal_patterns_path,
                enable_seasonality=config.enable_temporal_dynamics,
                min_confidence=config.seasonality_min_confidence
            )
            # Check if patterns were actually loaded
            if self.seasonality_engine.n_products_with_patterns > 0 or self.seasonality_engine.n_categories_with_patterns > 0:
                print(f"   • Seasonality Learning: ✅ Enabled (Phase 2.7)")
                print(f"      • Product patterns: {self.seasonality_engine.n_products_with_patterns:,}")
                print(f"      • Category patterns: {self.seasonality_engine.n_categories_with_patterns}")
            else:
                print(f"   • Seasonality Learning: ⚠️  Enabled but no patterns loaded")
                print(f"      • Using uniform seasonality (1.0x)")
                print(f"      • Run: python scripts/learn_seasonal_patterns.py")
        elif config.enable_temporal_dynamics:
            self.seasonality_engine = SeasonalityEngine(region=config.region)
            print(f"   • Seasonality: ✅ Enabled (hard-coded patterns)")
        else:
            self.seasonality_engine = None
            print(f"   • Seasonality: ❌ Disabled")
        
        # Sprint 2: Elasticity models (initialized via load_elasticity_models)
        self.price_hmm = None
        self.cross_price_engine = None
        self.arc_elasticity_engine = None

        # Sprint 2.1: Promotional engine (initialized in generate_base_datasets)
        self.promotional_engine = None

        # Sprint 1.3: Purchase history components (initialized later)
        self.state_manager = None
        self.history_engine = None

        # Sprint 1.4: Basket composer (initialized later)
        self.basket_composer = None
        
        self.datasets = {}
        self.pricing_history = []
        self.lifecycle_history = []
        self.market_share_history = []

        self.promo_response_calculator = PromoResponseCalculator()
    
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

    def load_elasticity_models(self, elasticity_dir: str, products: pd.DataFrame):
        """
        Load learned elasticity models from Sprint 2
        
        Args:
            elasticity_dir: Directory containing elasticity model files
                - price_hmm_model.pkl: HMM price dynamics
                - cross_elasticity_matrix.parquet: Cross-price effects
                - arc_elasticity_params.parquet: Arc elasticity parameters
            products: Products DataFrame (needed to initialize engines)
        """
        elasticity_path = Path(elasticity_dir)
        
        print(f"\n🔧 Loading elasticity models from {elasticity_dir}...")
        
        # 1. Load HMM price model
        hmm_path = elasticity_path / 'hmm_parameters.pkl'
        if hmm_path.exists():
            try:
                import pickle
                
                # Load the saved parameters
                with open(hmm_path, 'rb') as f:
                    hmm_params = pickle.load(f)
                
                # Instantiate PriceStateHMM class with products
                self.price_hmm = PriceStateHMM(products_df=products, n_states=4)
                
                # Load the learned parameters into the instance
                self.price_hmm.transition_matrices = hmm_params['transition_matrices']
                self.price_hmm.emission_distributions = hmm_params['emission_distributions']
                self.price_hmm.initial_state_probs = hmm_params['initial_state_probs']
                
                print(f"   ✅ Loaded HMM model: {len(self.price_hmm.transition_matrices):,} products")
            except Exception as e:
                print(f"   ⚠️  Failed to load HMM model: {e}")
                self.price_hmm = None
        else:
            print(f"   ⚠️  HMM model not found: {hmm_path}")
            self.price_hmm = None
        
        # 2. Load cross-price elasticity engine
        cross_matrix_path = elasticity_path / 'cross_elasticity/cross_elasticity_matrix.npz'
        if cross_matrix_path.exists():
            try:
                cross_matrix = np.load(cross_matrix_path)
                
                # Initialize engine with products DataFrame
                self.cross_price_engine = CrossPriceElasticityEngine(products_df=products)
                self.cross_price_engine.cross_elasticity_matrix = cross_matrix
                
                # Load substitute/complement groups if available
                groups_path = elasticity_path / 'cross_elasticity'
                self.cross_price_engine.substitute_groups = pd.read_csv(groups_path / 'substitute_groups.csv', index_col=0)
                self.cross_price_engine.complement_pairs = pd.read_csv(groups_path / 'complement_pairs.csv', index_col=0)
                
                print(f"   ✅ Loaded cross-price elasticity matrix: {len(cross_matrix):,} pairs")
                print(f"   ✅ Loaded substitute groups: {len(self.cross_price_engine.substitute_groups):,}")
                print(f"   ✅ Loaded complement pairs: {len(self.cross_price_engine.complement_pairs):,}")
            except Exception as e:
                print(f"   ⚠️  Failed to load cross-price elasticity: {e}")
                self.cross_price_engine = None
        else:
            print(f"   ⚠️  Cross-price elasticity not found: {cross_matrix_path}")
            self.cross_price_engine = None
        
        # 3. Load arc elasticity parameters
        arc_path = elasticity_path / 'arc_elasticity' / 'arc_elasticity_params.pkl'
        if arc_path.exists() and self.price_hmm is not None:
            try:
                # Load arc elasticity with the HMM model
                self.arc_elasticity_engine = ArcPriceElasticityEngine.load_parameters(
                    arc_path,
                    price_hmm=self.price_hmm
                )
                
                print(f"   ✅ Loaded arc elasticity parameters")
                print(f"      • Inventory decay rate: {self.arc_elasticity_engine.inventory_decay_rate:.1%}/week")
                print(f"      • Future discount factor: {self.arc_elasticity_engine.future_discount_factor:.3f}")
            except Exception as e:
                print(f"   ⚠️  Failed to load arc elasticity: {e}")
                self.arc_elasticity_engine = None
        else:
            if not arc_path.exists():
                print(f"   ⚠️  Arc elasticity not found: {arc_path}")
            elif self.price_hmm is None:
                print(f"   ⚠️  Arc elasticity requires HMM model (load HMM first)")
            self.arc_elasticity_engine = None

    def generate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets with comprehensive temporal dynamics"""
        
        start_time = datetime.now()
        
        # Generate base datasets first (if not already generated)
        if 'products' not in self.datasets or len(self.datasets.get('products', [])) == 0:
            self.generate_base_datasets()
        else:
            print(f"\n✅ Base datasets already generated, skipping to transaction generation...")
        
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
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ All datasets generated in {total_time/60:.1f} minutes")
        
        return self.datasets
    
    def generate_base_datasets(self):
        """
        Generate base datasets (customers, products, stores, market context) WITHOUT transactions.
        This allows elasticity models to be loaded before transaction generation.
        """
        # Step 1: Generate customers
        print(f"\n👥 Step 1/6: Generating {self.config.n_customers:,} customers...")
        step_start = datetime.now()
        self.datasets['customers'] = CustomerGenerator.generate_customers_vectorized(self.config, self.calibration_engine)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 2: Generate/Load products (MODIFIED - Sprint 1.1)
        print(f"\n📦 Step 2/6: {'Loading' if self.config.use_real_catalog else 'Generating'} {self.config.n_products:,} products...")
        step_start = datetime.now()
        
        if self.config.use_real_catalog and self.real_products is not None:
            # Use real catalog (NEW)
            self.datasets['products'] = self._prepare_real_products()
            
            # Standardize column names to lowercase
            self.datasets['products'].columns = [col.lower() for col in self.datasets['products'].columns]
            
            print(self.datasets['products'][self.datasets['products']["product_id"] == 928786])

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
        print(f"\n🏪 Step 3/6: Generating {self.config.n_stores} stores...")
        step_start = datetime.now()
        self.datasets['stores'] = StoreGenerator.generate_stores(self.config)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 4: Generate market context
        print(f"\n💰 Step 4/6: Generating market context...")
        step_start = datetime.now()
        self.datasets['market_context'] = MarketContextGenerator.generate_market_context(self.config)
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 5: Initialize temporal engines
        print(f"\n🔧 Step 5/6: Initializing temporal dynamics engines...")
        step_start = datetime.now()
        
        if self.config.enable_temporal_dynamics:
            self.pricing_engine = PricingEvolutionEngine(len(self.datasets['products']))

        # Sprint 2.1: Initialize promotional engine
        self.promotional_engine = PromotionalEngine(
            hmm_model=self.price_hmm,  # Will be None initially, can be set later
            products_df=self.datasets['products'],
            stores_df=self.datasets['stores'],
            config=None  # Use default configuration
        )
        print(f"   • Promotional Engine: ✅ Initialized")
        
        if self.config.enable_product_lifecycle:
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
        
        # Sprint 1.3: Initialize purchase history components
        self.state_manager = CustomerStateManager(self.datasets['customers']['customer_id'].tolist())
        self.history_engine = PurchaseHistoryEngine(
            products=self.datasets['products'], 
            loyalty_weight=self.config.loyalty_weight, 
            habit_weight=self.config.habit_weight, 
            inventory_weight=self.config.inventory_weight, 
            variety_weight=self.config.variety_weight, 
            price_memory_weight=self.config.price_memory_weight,
            inventory_depletion_rate=self.config.inventory_depletion_rate,
            replenishment_threshold=self.config.replenishment_threshold
        )
        print(f"   • Purchase History: ✅ Enabled")

        # Sprint 1.4: Initialize basket composer
        if self.config.enable_basket_composition and self.config.use_real_catalog:
            self.basket_composer = BasketComposer(products=self.datasets['products'], config=self.config, enable_complements=True, enable_substitutes=True)
            print(f"   • Basket Composition: ✅ Enabled")
        
        # Sprint 2.1: Initialize promotional engine
        self.promotional_engine = PromotionalEngine(
            hmm_model=self.price_hmm,  # Will be None initially, can be set later
            products_df=self.datasets['products'],
            stores_df=self.datasets['stores'],
            config=None  # Use default configuration
        )
        print(f"   • Promotional Engine: ✅ Initialized")
        
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Step 6: Pre-compute matrices for GPU
        print(f"\n🔧 Step 6/6: Pre-computing GPU matrices...")
        step_start = datetime.now()
        self.precomp = VectorizedPreComputationEngine(
            self.datasets['customers'], 
            self.datasets['products']
        )
        print(f"   ✅ Complete in {(datetime.now() - step_start).total_seconds():.1f}s")
        
        # Phase 2.6: Initialize reference prices from base prices
        if self.nonlinear_engine is not None:
            print(f"\n🧮 Phase 2.6: Initializing reference prices...")
            self.nonlinear_engine.initialize_reference_prices(
                self.datasets['products'], 
                price_column='base_price'
            )
            print(f"   ✅ Reference prices initialized for {len(self.datasets['products']):,} products")

    def _generate_transactions_with_temporal_dynamics(self) -> Dict[str, pd.DataFrame]:
        """
        Generate transactions with FULL temporal dynamics (v3.6 + Sprint 1.3).
        Includes customer drift, product lifecycle, store loyalty, and purchase history.
        """
        all_transactions = []
        all_transaction_items = []
        
        # Initialize transaction generator with store loyalty AND purchase history (Sprint 1.3)
        transaction_gen = ComprehensiveTransactionGenerator(
            self.precomp,
            self.utility_engine,
            self.loyalty_engine if self.config.enable_store_loyalty else None,
            self.config,
            state_manager=self.state_manager,  # Sprint 1.3
            history_engine=self.history_engine,  # Sprint 1.3
            basket_composer=self.basket_composer,  # Sprint 1.4
            promo_response_calc=self.promo_response_calculator,  # Sprint 2.5
            nonlinear_engine=self.nonlinear_engine,  # Sprint 2.6
            products=self.datasets['products']  # Phase 2: For SV-based visit probability
        )
        
        # Get depletion rates for inventory tracking (Sprint 1.3)
        depletion_rates = get_depletion_rates_by_assortment(self.config.inventory_depletion_rate)
        
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
            
            # Sprint 1.3: Deplete inventory for all customers
            if week > 1:
                self.state_manager.deplete_all_inventory(depletion_rates)
            
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
                # Sprint 2.1: Use new separated pricing + promotional engines
                print(f"      💰 Generating base prices...")
                current_prices = self.pricing_engine.evolve_prices(
                    self.precomp.base_prices, 
                    week, 
                    self.precomp.product_ids
                )
                
                # Generate promotions per store (Sprint 2.1)
                # For now, use store 1 as default - will iterate over stores in Phase 2.2
                store_id = self.datasets['stores']['store_id'].iloc[0]
                promo_context = self.promotional_engine.generate_store_promotions(
                    store_id=store_id,
                    week_number=week,
                    base_prices=current_prices,
                    product_ids=self.precomp.product_ids
                )
                
                # Apply promotional discounts to get final prices
                final_prices = np.array([
                    self.promotional_engine.get_promotional_price(pid, base_price, promo_context)
                    for pid, base_price in zip(self.precomp.product_ids, current_prices)
                ])
                
                # Create promo flags for backward compatibility
                promo_flags = np.array([
                    1 if pid in promo_context.promoted_products else 0
                    for pid in self.precomp.product_ids
                ])
                
                # Use final prices with promotions applied
                current_prices = final_prices
                
                # Log promotional summary
                summary = self.promotional_engine.get_promo_summary(promo_context)
                print(f"         Promos: {summary['n_promotions']} products, "
                      f"avg discount: {summary['avg_discount']:.1%}, "
                      f"end caps: {summary['n_end_caps']}, "
                      f"in-ad: {summary['n_in_ad']}")
            
            # Phase 2.7: Apply seasonality (learned patterns or hard-coded)
            if self.seasonality_engine is not None:
                if self.config.enable_seasonality_learning:
                    # Use learned product-specific seasonal patterns
                    week_of_year = ((week - 1) % 52) + 1
                    
                    # Get categories for products (if available)
                    category_col = 'commodity_desc' if 'commodity_desc' in self.datasets['products'].columns else 'category'
                    product_categories = None
                    
                    if category_col in self.datasets['products'].columns:
                        prod_df = self.datasets['products'].set_index('product_id')
                        product_categories = np.array([
                            prod_df.loc[pid, category_col] if pid in prod_df.index else 'UNKNOWN'
                            for pid in self.precomp.product_ids
                        ])
                    
                    # Get seasonal multipliers for all products
                    seasonal_multipliers = self.seasonality_engine.get_seasonal_multipliers_vectorized(
                        product_ids=self.precomp.product_ids,
                        week_of_year=week_of_year,
                        categories=product_categories,
                        fallback_value=1.0
                    )
                    
                    # Debug: Check how many products have learned patterns
                    n_with_patterns = np.sum(seasonal_multipliers != 1.0)
                    if week == 1:  # Only print first week to avoid spam
                        print(f"         DEBUG: {n_with_patterns}/{len(seasonal_multipliers)} products have learned seasonal patterns")
                    
                    # Apply seasonality to prices (multiplicative effect on demand → reflected in choice)
                    # Note: We store this for use in transaction generation
                    # The transaction generator will apply these as demand modifiers
                    current_prices = current_prices * (1.0 / seasonal_multipliers)  # Lower price = higher demand
                    
                    # Log seasonality stats
                    avg_seasonal = np.mean(seasonal_multipliers)
                    max_seasonal = np.max(seasonal_multipliers)
                    min_seasonal = np.min(seasonal_multipliers)
                    print(f"         Seasonality: avg={avg_seasonal:.2f}, range=[{min_seasonal:.2f}, {max_seasonal:.2f}]")
                    
                else:
                    # Use hard-coded category-level seasonality (legacy)
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

            # Store Promo Context history For each week, before generating transactions:
            store_promo_contexts = {}
            for store_id in self.datasets['stores']['store_id']:
                store_promo_contexts[store_id] = self.promotional_engine.generate_store_promo_context(
                    store_id=store_id,
                    week_number=week,
                    base_prices=current_prices,
                    product_ids=self.precomp.product_ids
                )
            
            # 4. Generate transactions
            print(f"      🛒 Generating transactions...")
            week_transactions, week_items = transaction_gen.generate_week_transactions_vectorized(
                week, current_prices, promo_flags, week_date, store_promo_contexts
            )
            
            all_transactions.extend(week_transactions)
            all_transaction_items.extend(week_items)
            
            # Phase 2.6: Update reference prices with EWMA after each week
            if self.nonlinear_engine is not None and self.nonlinear_engine.config.use_reference_prices:
                self.nonlinear_engine.update_reference_prices(
                    self.precomp.product_ids,
                    current_prices
                )
            
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
        
        for week_num in sorted(transactions['week_number'].unique()):
            week_transactions = transactions[transactions['week_number'] == week_num]
            
            for store_id in sorted(transactions['store_id'].unique()):
                store_transactions = week_transactions[week_transactions['store_id'] == store_id]
                
                if len(store_transactions) == 0:
                    continue
                
                performance_data.append({
                    'week_number': week_num,
                    'store_id': store_id,
                    'total_transactions': len(store_transactions),
                    'total_revenue': store_transactions['total_revenue'].sum(),
                    'total_margin': store_transactions['total_margin'].sum(),
                    'total_discount': store_transactions['total_discount'].sum(),
                    'avg_basket_size': store_transactions['total_items'].mean(),
                    'avg_basket_value': store_transactions['total_revenue'].mean(),
                    'avg_satisfaction': store_transactions['satisfaction_score'].mean(),
                    'promotion_penetration': (store_transactions['promo_items'] > 0).sum() / len(store_transactions),
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

        # Reset index for sequential access
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

    def _extract_brand_portfolio(self) -> Dict:
        """Extract brand portfolio from real catalog"""
        brands = {}
        
        for dept in self.real_products['DEPARTMENT'].unique():
            dept_products = self.real_products[self.real_products['DEPARTMENT'] == dept]
            dept_brands = dept_products['BRAND'].unique().tolist()
            brands[dept] = dept_brands[:20]  # Top 20 brands per department
        
        return brands
    
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
            
            for store_id in sorted(transactions['store_id'].unique()):
                store_transactions = week_transactions[week_transactions['store_id'] == store_id]
                
                if len(store_transactions) == 0:
                    continue
                
                weekly_data.append({
                    'week_number': week_num,
                    'store_id': store_id,
                    'total_transactions': len(store_transactions),
                    'total_revenue': store_transactions['total_revenue'].sum(),
                    'avg_basket_size': store_transactions['total_items_count'].mean(),
                    'avg_basket_value': store_transactions['total_revenue'].mean(),
                    'avg_satisfaction': store_transactions['satisfaction_score'].mean(),
                    'promotion_rate': (store_transactions['promotional_items_count'] > 0).sum() / len(store_transactions),
                    'total_stores_active': business_performance['store_id'].nunique() if len(business_performance) > 0 else 0
                })
        
        return pd.DataFrame(weekly_data)
