import numpy as np
import pandas as pd
from retailsynth.utils import RealisticCategoryHierarchy
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig

# ============================================================================
# PRODUCT GENERATOR (v3.2)
# ============================================================================

class ProductGenerator:
    """
    Generates product profiles with realistic attributes and pricing.
    """
    
    @staticmethod
    def generate_products_vectorized(config: EnhancedRetailConfig) -> pd.DataFrame:
        """Generate products with vectorized operations"""
        print(f"      Generating {config.n_products:,} products (vectorized)...")
        start_time = datetime.now()
        
        category_hierarchy = RealisticCategoryHierarchy.create_category_hierarchy()
        all_brands = RealisticCategoryHierarchy.get_all_brands()
        
        products = []
        product_id = 1
        
        # Flatten category hierarchy for sampling
        category_paths = []
        for dept, categories in category_hierarchy.items():
            for cat, subcats in categories.items():
                for subcat, items in subcats.items():
                    for item in items:
                        category_paths.append((dept, cat, subcat, item))
        
        # Sample products from category paths
        n_products = config.n_products
        selected_paths = [category_paths[i % len(category_paths)] for i in range(n_products)]
        
        # Vectorized price generation by department
        dept_list = [path[0] for path in selected_paths]
        prices = np.zeros(n_products)
        
        # Use Log-Normal distribution (better for retail prices)
        # Real prices are right-skewed: many cheap items, few expensive
        for dept in ['Fresh', 'Pantry', 'Personal_Care', 'General_Merchandise']:
            dept_mask = np.array([d == dept for d in dept_list])
            n_dept = dept_mask.sum()
            
            if dept == 'Fresh':
                # Mean ~$5, range $1-15
                mu, sigma = 1.5, 0.6
                prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
                prices[dept_mask] = np.clip(prices[dept_mask], 1.0, 15.0)
            elif dept == 'Pantry':
                # Mean ~$6, range $1.5-20
                mu, sigma = 1.7, 0.6
                prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
                prices[dept_mask] = np.clip(prices[dept_mask], 1.5, 20.0)
            elif dept == 'Personal_Care':
                # Mean ~$10, range $3-30
                mu, sigma = 2.2, 0.6
                prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
                prices[dept_mask] = np.clip(prices[dept_mask], 3.0, 30.0)
            else:  # General_Merchandise
                # Mean ~$15, range $5-50
                mu, sigma = 2.6, 0.6
                prices[dept_mask] = np.random.lognormal(mu, sigma, n_dept)
                prices[dept_mask] = np.clip(prices[dept_mask], 5.0, 50.0)

        # Vectorized role assignment
        roles = np.random.choice(
            ['lpg_line', 'front_basket', 'mid_basket', 'back_basket'],
            size=n_products,
            p=[0.15, 0.25, 0.40, 0.20]
        )
        
        # Vectorized brand assignment
        brands = np.random.choice(all_brands, size=n_products)
        
        for i in range(n_products):
            dept, cat, subcat, item = selected_paths[i]
            
            products.append({
                'product_id': product_id,
                'department': dept,
                'category': cat,
                'subcategory': subcat,
                'product_name': item,
                'brand': brands[i],
                'base_price': round(float(prices[i]), 2),
                'assortment_role': roles[i],
                'created_at': datetime.now()
            })
            
            product_id += 1
        
        print(f"         ✅ Generated in {(datetime.now() - start_time).total_seconds():.1f}s")
        return pd.DataFrame(products)
