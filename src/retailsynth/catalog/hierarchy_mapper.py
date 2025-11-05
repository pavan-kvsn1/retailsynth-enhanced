"""
Hierarchy Mapper for RetailSynth Enhanced.

Builds and manages the product category hierarchy from Dunnhumby data.
Creates a 3-level hierarchy: Department → Commodity → Sub-Commodity
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

class HierarchyMapper:
    """
    Build and manage product category hierarchy.
    
    Hierarchy Structure:
    - Level 1: Department (e.g., GROCERY, DRUG GM, PRODUCE)
    - Level 2: Commodity (e.g., SOFT DRINKS, PAIN REMEDIES)
    - Level 3: Sub-Commodity (e.g., CARBONATED SOFT DRINKS)
    """
    
    def __init__(self):
        """Initialize HierarchyMapper."""
        self.hierarchy = {}
        self.product_to_category = {}
        self.category_stats = {}
    
    def build_hierarchy(self, catalog_df: pd.DataFrame) -> Dict:
        """
        Build hierarchical category structure from product catalog.
        
        Args:
            catalog_df: Product catalog DataFrame with hierarchy columns
            
        Returns:
            Nested dictionary representing hierarchy
        """
        print("Building category hierarchy...")
        
        hierarchy = {}
        product_mapping = {}
        
        for _, row in catalog_df.iterrows():
            dept = str(row['DEPARTMENT'])
            commodity = str(row['COMMODITY_DESC'])
            sub_commodity = str(row['SUB_COMMODITY_DESC'])
            product_id = int(row['PRODUCT_ID'])
            
            # Build nested structure
            if dept not in hierarchy:
                hierarchy[dept] = {}
            
            if commodity not in hierarchy[dept]:
                hierarchy[dept][commodity] = {}
            
            if sub_commodity not in hierarchy[dept][commodity]:
                hierarchy[dept][commodity][sub_commodity] = []
            
            # Add product details
            product_info = {
                'product_id': product_id,
                'brand': str(row['BRAND']),
                'manufacturer': str(row.get('MANUFACTURER', 'Unknown')),
                'size': str(row.get('CURR_SIZE_OF_PRODUCT', '')),
                'avg_price': float(row.get('avg_price', 0.0)),
            }
            
            hierarchy[dept][commodity][sub_commodity].append(product_info)
            
            # Store reverse mapping
            product_mapping[product_id] = {
                'department': dept,
                'commodity': commodity,
                'sub_commodity': sub_commodity
            }
        
        self.hierarchy = hierarchy
        self.product_to_category = product_mapping
        
        # Calculate statistics
        self._calculate_hierarchy_stats(catalog_df)
        
        print(f"✅ Built hierarchy:")
        print(f"   Departments: {len(hierarchy)}")
        print(f"   Commodities: {sum(len(commodities) for commodities in hierarchy.values())}")
        print(f"   Sub-Commodities: {sum(len(subs) for dept in hierarchy.values() for subs in dept.values())}")
        
        return hierarchy
    
    def _calculate_hierarchy_stats(self, catalog_df: pd.DataFrame):
        """Calculate statistics for each category level."""
        stats = {}
        
        # Department stats
        dept_stats = catalog_df.groupby('DEPARTMENT').agg({
            'PRODUCT_ID': 'count',
            'total_revenue': 'sum',
            'avg_price': 'mean',
            'purchase_frequency': 'sum'
        }).to_dict('index')
        
        stats['departments'] = dept_stats
        
        # Commodity stats
        commodity_stats = catalog_df.groupby(['DEPARTMENT', 'COMMODITY_DESC']).agg({
            'PRODUCT_ID': 'count',
            'total_revenue': 'sum',
            'avg_price': 'mean'
        }).to_dict('index')
        
        stats['commodities'] = commodity_stats
        
        self.category_stats = stats
    
    def get_products_in_category(self, 
                                 department: str = None,
                                 commodity: str = None,
                                 sub_commodity: str = None) -> List[int]:
        """
        Get all product IDs in a category.
        
        Args:
            department: Department name (optional)
            commodity: Commodity name (optional, requires department)
            sub_commodity: Sub-commodity name (optional, requires commodity)
            
        Returns:
            List of product IDs
        """
        products = []
        
        if department is None:
            # Return all products
            for dept in self.hierarchy.values():
                for comm in dept.values():
                    for sub in comm.values():
                        products.extend([p['product_id'] for p in sub])
        
        elif commodity is None:
            # Return all products in department
            if department in self.hierarchy:
                for comm in self.hierarchy[department].values():
                    for sub in comm.values():
                        products.extend([p['product_id'] for p in sub])
        
        elif sub_commodity is None:
            # Return all products in commodity
            if department in self.hierarchy and commodity in self.hierarchy[department]:
                for sub in self.hierarchy[department][commodity].values():
                    products.extend([p['product_id'] for p in sub])
        
        else:
            # Return products in specific sub-commodity
            if (department in self.hierarchy and 
                commodity in self.hierarchy[department] and
                sub_commodity in self.hierarchy[department][commodity]):
                products = [p['product_id'] for p in 
                           self.hierarchy[department][commodity][sub_commodity]]
        
        return products
    
    def get_category_for_product(self, product_id: int) -> Dict[str, str]:
        """
        Get category information for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dictionary with department, commodity, sub_commodity
        """
        return self.product_to_category.get(product_id, {})
    
    def get_sibling_products(self, product_id: int, level: str = 'sub_commodity') -> List[int]:
        """
        Get products in the same category as the given product.
        
        Args:
            product_id: Product ID
            level: Category level ('department', 'commodity', 'sub_commodity')
            
        Returns:
            List of sibling product IDs
        """
        category = self.get_category_for_product(product_id)
        
        if not category:
            return []
        
        if level == 'department':
            return self.get_products_in_category(department=category['department'])
        elif level == 'commodity':
            return self.get_products_in_category(
                department=category['department'],
                commodity=category['commodity']
            )
        else:  # sub_commodity
            return self.get_products_in_category(
                department=category['department'],
                commodity=category['commodity'],
                sub_commodity=category['sub_commodity']
            )
    
    def save_hierarchy(self, output_path: str):
        """
        Save hierarchy to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.hierarchy, f, indent=2)
        
        print(f"✅ Saved hierarchy to {output_path}")
    
    def save_product_mapping(self, output_path: str):
        """
        Save product-to-category mapping.
        
        Args:
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Convert int keys to strings for JSON
        mapping_str_keys = {str(k): v for k, v in self.product_to_category.items()}
        
        with open(output_file, 'w') as f:
            json.dump(mapping_str_keys, f, indent=2)
        
        print(f"✅ Saved product mapping to {output_path}")
    
    def load_hierarchy(self, hierarchy_path: str, mapping_path: str = None):
        """
        Load hierarchy from JSON files.
        
        Args:
            hierarchy_path: Path to hierarchy JSON
            mapping_path: Path to product mapping JSON (optional)
        """
        with open(hierarchy_path, 'r') as f:
            self.hierarchy = json.load(f)
        
        if mapping_path:
            with open(mapping_path, 'r') as f:
                mapping_str_keys = json.load(f)
                # Convert string keys back to int
                self.product_to_category = {
                    int(k): v for k, v in mapping_str_keys.items()
                }
        
        print(f"✅ Loaded hierarchy from {hierarchy_path}")
    
    def print_hierarchy_summary(self):
        """Print a summary of the hierarchy structure."""
        print("\n" + "="*70)
        print("CATEGORY HIERARCHY SUMMARY")
        print("="*70)
        
        for dept, commodities in sorted(self.hierarchy.items()):
            n_commodities = len(commodities)
            n_products = sum(
                len(products) 
                for comm in commodities.values() 
                for products in comm.values()
            )
            
            print(f"\n{dept}")
            print(f"  Commodities: {n_commodities}")
            print(f"  Products: {n_products}")
            
            # Show top 3 commodities by product count
            commodity_counts = {
                comm: sum(len(products) for products in subs.values())
                for comm, subs in commodities.items()
            }
            top_commodities = sorted(
                commodity_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for comm, count in top_commodities:
                print(f"    - {comm}: {count} products")
        
        print("="*70 + "\n")