import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
from ..utils.category_hierarchy import RealisticCategoryHierarchy
from ..config import EnhancedRetailConfig

# ============================================================================
# PRODUCT LIFECYCLE ENGINE (v3.2 + v3.6 ENHANCEMENTS)
# ============================================================================

class ProductLifecycleEngine:
    """
    Models complete product lifecycle with retirement and launches (v3.6).
    Stages: Launch → Growth → Maturity → Decline → Retirement
    """
    
    def __init__(self, products_df: pd.DataFrame, config: EnhancedRetailConfig):
        self.products = products_df.copy()
        self.config = config
        self.active_products = products_df.copy()
        self.retired_products = pd.DataFrame()
        self.next_product_id = products_df['product_id'].max() + 1
        
        # Lifecycle tracking
        self.lifecycle_stages = self._assign_initial_lifecycle_stages()
        self.weeks_in_stage = {pid: 0 for pid in products_df['product_id']}
        self.launch_queue = []
    
    def _assign_initial_lifecycle_stages(self) -> Dict:
        """Assign realistic initial lifecycle stages"""
        stages = {}
        for _, product in self.products.iterrows():
            # Most products start in maturity, some in growth, few in launch/decline
            stage = np.random.choice(
                ['launch', 'growth', 'maturity', 'decline'],
                p=[0.05, 0.20, 0.60, 0.15]
            )
            stages[product['product_id']] = {
                'stage': stage,
                'weeks_in_stage': np.random.randint(0, 20)
            }
        return stages
    
    def update_lifecycle_stage(self, product_id: int, week_number: int) -> str:
        """
        Update product lifecycle stage based on elapsed time.
        Returns current stage.
        """
        if product_id not in self.lifecycle_stages:
            return 'maturity'
        
        current_stage = self.lifecycle_stages[product_id]['stage']
        weeks_in_stage = self.lifecycle_stages[product_id]['weeks_in_stage']
        weeks_in_stage += 1
        
        # Stage transitions based on typical product lifecycle
        if current_stage == 'launch' and weeks_in_stage > 8:
            if np.random.random() < 0.3:  # 30% chance to move to growth
                current_stage = 'growth'
                weeks_in_stage = 0
        
        elif current_stage == 'growth' and weeks_in_stage > 12:
            if np.random.random() < 0.4:  # 40% chance to mature
                current_stage = 'maturity'
                weeks_in_stage = 0
        
        elif current_stage == 'maturity' and weeks_in_stage > 26:
            if np.random.random() < 0.15:  # 15% chance to decline
                current_stage = 'decline'
                weeks_in_stage = 0
        
        # Update tracking
        self.lifecycle_stages[product_id] = {
            'stage': current_stage,
            'weeks_in_stage': weeks_in_stage
        }
        
        return current_stage
    
    def get_lifecycle_multiplier(self, product_id: int) -> float:
        """
        Get demand multiplier based on lifecycle stage.
        Launch: 0.6-0.8, Growth: 1.1-1.3, Maturity: 0.95-1.05, Decline: 0.5-0.7
        """
        if product_id not in self.lifecycle_stages:
            return 1.0
        
        stage = self.lifecycle_stages[product_id]['stage']
        
        multipliers = {
            'launch': np.random.uniform(0.6, 0.8),
            'growth': np.random.uniform(1.1, 1.3),
            'maturity': np.random.uniform(0.95, 1.05),
            'decline': np.random.uniform(0.5, 0.7)
        }
        
        return multipliers.get(stage, 1.0)
    
    def update_weekly(self, week_number: int) -> Tuple[pd.DataFrame, List[int], List[Dict]]:
        """
        Update lifecycle stages, retire products, launch new ones (v3.6).
        Returns: (active_products_df, retired_product_ids, new_products_list)
        """
        retired_ids = []
        new_products = []
        
        # Update all product stages
        for product_id in self.active_products['product_id'].values:
            stage = self.update_lifecycle_stage(product_id, week_number)
            
            # Check for retirement (products in decline for configured weeks)
            if stage == 'decline':
                weeks_in_decline = self.lifecycle_stages[product_id]['weeks_in_stage']
                if weeks_in_decline >= self.config.product_retirement_weeks:
                    retired_ids.append(product_id)
        
        # Retire products
        if retired_ids:
            retiring_products = self.active_products[
                self.active_products['product_id'].isin(retired_ids)
            ]
            self.retired_products = pd.concat([self.retired_products, retiring_products], 
                                             ignore_index=True)
            self.active_products = self.active_products[
                ~self.active_products['product_id'].isin(retired_ids)
            ]
            
            print(f"   Week {week_number}: Retired {len(retired_ids)} products")
        
        # Launch new products to replace retired ones and grow catalog
        n_new_products = len(retired_ids) + int(
            len(self.active_products) * self.config.new_product_launch_rate * np.random.random()
        )
        
        if n_new_products > 0:
            new_products = self._generate_new_products(n_new_products, week_number)
            new_products_df = pd.DataFrame(new_products)
            self.active_products = pd.concat([self.active_products, new_products_df], 
                                            ignore_index=True)
            
            print(f"   Week {week_number}: Launched {len(new_products)} new products")
        
        return self.active_products, retired_ids, new_products
    
    def _generate_new_products(self, n_products: int, week_number: int) -> List[Dict]:
        """Generate new product launches"""
        category_hierarchy = RealisticCategoryHierarchy.create_category_hierarchy()
        all_brands = RealisticCategoryHierarchy.get_all_brands()
        
        new_products = []
        for i in range(n_products):
            product_id = self.next_product_id
            self.next_product_id += 1
            
            # Select random category path
            dept = np.random.choice(list(category_hierarchy.keys()))
            cat = np.random.choice(list(category_hierarchy[dept].keys()))
            subcat = np.random.choice(list(category_hierarchy[dept][cat].keys()))
            items = category_hierarchy[dept][cat][subcat]
            item = np.random.choice(items)
            
            # Product attributes
            product = {
                'product_id': product_id,
                'department': dept,
                'category': cat,
                'subcategory': subcat,
                'product_name': f"{item} New",
                'brand': np.random.choice(all_brands),
                'base_price': round(np.random.uniform(2.0, 25.0), 2),
                'assortment_role': np.random.choice(
                    ['lpg_line', 'front_basket', 'mid_basket', 'back_basket'],
                    p=[0.15, 0.25, 0.40, 0.20]
                ),
                'launch_week': week_number,
                'created_at': datetime.now()
            }
            
            new_products.append(product)
            
            # Initialize lifecycle stage
            self.lifecycle_stages[product_id] = {
                'stage': 'launch',
                'weeks_in_stage': 0
            }
        
        return new_products
    
    def visualize_lifecycle_trajectories(self, lifecycle_history: List[Dict],
                                       product_ids: List[int]) -> pd.DataFrame:
        """Create visualization dataset for lifecycle trajectories"""
        trajectory_data = []
        
        for week_data in lifecycle_history:
            week_number = week_data['week']
            stages = week_data['stages']
            
            for product_id in product_ids:
                if product_id in stages:
                    trajectory_data.append({
                        'week_number': week_number,
                        'product_id': product_id,
                        'lifecycle_stage': stages[product_id]['stage'],
                        'weeks_in_stage': stages[product_id]['weeks_in_stage']
                    })
        
        return pd.DataFrame(trajectory_data)
