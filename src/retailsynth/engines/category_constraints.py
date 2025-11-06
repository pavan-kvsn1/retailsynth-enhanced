"""
Category Constraints (Sprint 1.4)

Defines rules for realistic basket composition:
- Category co-occurrence patterns
- Complement relationships (peanut butter → jelly)
- Substitute relationships (Coke ↔ Pepsi)
- Quantity constraints per category
- Category diversity requirements

Based on Dunnhumby shopping patterns and retail research.

Author: RetailSynth Team
Date: November 2024
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CategoryConstraint:
    """
    Constraints for a product category
    
    Attributes:
        max_products: Maximum products from this category in one basket
        min_products: Minimum products if category is selected
        max_quantity_per_product: Max quantity of any single product
        typical_quantity: Typical quantity purchased
    """
    max_products: int
    min_products: int
    max_quantity_per_product: int
    typical_quantity: int


# Category-specific constraints
CATEGORY_CONSTRAINTS = {
    'DAIRY': CategoryConstraint(
        max_products=6,
        min_products=1,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'PRODUCE': CategoryConstraint(
        max_products=8,
        min_products=2,
        max_quantity_per_product=3,
        typical_quantity=2
    ),
    'MEAT': CategoryConstraint(
        max_products=4,
        min_products=1,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'GROCERY': CategoryConstraint(
        max_products=10,
        min_products=2,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'FROZEN': CategoryConstraint(
        max_products=5,
        min_products=1,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'BAKERY': CategoryConstraint(
        max_products=3,
        min_products=1,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'BEVERAGE': CategoryConstraint(
        max_products=4,
        min_products=1,
        max_quantity_per_product=3,
        typical_quantity=2
    ),
    'SNACKS': CategoryConstraint(
        max_products=5,
        min_products=1,
        max_quantity_per_product=2,
        typical_quantity=1
    ),
    'DELI': CategoryConstraint(
        max_products=3,
        min_products=1,
        max_quantity_per_product=1,
        typical_quantity=1
    )
}


# Complement relationships (buying A increases probability of buying B)
COMPLEMENT_PAIRS = [
    # Breakfast combos
    ('CEREAL', 'MILK'),
    ('BREAD', 'BUTTER'),
    ('BREAD', 'JAM'),
    ('EGGS', 'BACON'),
    ('PANCAKE MIX', 'SYRUP'),
    
    # Meal combos
    ('PASTA', 'PASTA SAUCE'),
    ('RICE', 'BEANS'),
    ('CHIPS', 'DIP'),
    ('HOT DOGS', 'HOT DOG BUNS'),
    ('HAMBURGER', 'HAMBURGER BUNS'),
    
    # Beverage combos
    ('COFFEE', 'CREAMER'),
    ('TEA', 'HONEY'),
    ('SODA', 'ICE CREAM'),
    
    # Cooking combos
    ('CHICKEN', 'VEGETABLES'),
    ('BEEF', 'POTATOES'),
    ('FISH', 'LEMON'),
    ('SALAD', 'SALAD DRESSING'),
    
    # Snack combos
    ('PEANUT BUTTER', 'JELLY'),
    ('CRACKERS', 'CHEESE'),
    ('COOKIES', 'MILK'),
    ('POPCORN', 'BUTTER')
]


# Category-level complements
CATEGORY_COMPLEMENTS = {
    'MEAT': ['PRODUCE', 'GROCERY'],
    'PASTA': ['GROCERY'],  # Pasta sauce
    'BAKERY': ['DAIRY'],   # Bread + butter
    'CEREAL': ['DAIRY'],   # Cereal + milk
    'SNACKS': ['BEVERAGE']
}


# Substitute relationships (buying A reduces probability of buying B)
SUBSTITUTE_GROUPS = {
    'milk_alternatives': ['WHOLE MILK', 'SKIM MILK', '2% MILK', 'ALMOND MILK', 'SOY MILK'],
    'cola': ['COCA COLA', 'PEPSI', 'RC COLA'],
    'bread': ['WHITE BREAD', 'WHEAT BREAD', 'SOURDOUGH'],
    'juice': ['ORANGE JUICE', 'APPLE JUICE', 'GRAPE JUICE'],
    'chips': ['LAYS', 'DORITOS', 'PRINGLES'],
    'cereal': ['CHEERIOS', 'FROSTED FLAKES', 'LUCKY CHARMS']
}


class CategoryConstraintEngine:
    """
    Enforces category constraints and relationships in basket composition
    """
    
    def __init__(self):
        self.category_constraints = CATEGORY_CONSTRAINTS
        self.complement_pairs = COMPLEMENT_PAIRS
        self.category_complements = CATEGORY_COMPLEMENTS
        self.substitute_groups = SUBSTITUTE_GROUPS
        
        # Build reverse lookup for complements
        self._build_complement_lookup()
    
    def _build_complement_lookup(self):
        """Build efficient lookup for complement relationships"""
        self.product_complements = {}
        
        for product_a, product_b in self.complement_pairs:
            if product_a not in self.product_complements:
                self.product_complements[product_a] = []
            if product_b not in self.product_complements:
                self.product_complements[product_b] = []
            
            self.product_complements[product_a].append(product_b)
            self.product_complements[product_b].append(product_a)
    
    def get_category_constraint(self, category: str) -> CategoryConstraint:
        """
        Get constraints for a category
        
        Args:
            category: Category name
        
        Returns:
            CategoryConstraint object
        """
        return self.category_constraints.get(
            category,
            CategoryConstraint(
                max_products=5,
                min_products=1,
                max_quantity_per_product=2,
                typical_quantity=1
            )
        )
    
    def get_complements(self, product_name: str) -> List[str]:
        """
        Get complement products for a given product
        
        Args:
            product_name: Product name or commodity
        
        Returns:
            List of complement product names
        """
        complements = []
        
        # Check product-level complements
        for product_a, product_b in self.complement_pairs:
            if product_a.upper() in product_name.upper():
                complements.append(product_b)
            elif product_b.upper() in product_name.upper():
                complements.append(product_a)
        
        return complements
    
    def get_category_complements(self, category: str) -> List[str]:
        """
        Get complementary categories
        
        Args:
            category: Category name
        
        Returns:
            List of complementary category names
        """
        return self.category_complements.get(category, [])
    
    def is_substitute(self, product_a: str, product_b: str) -> bool:
        """
        Check if two products are substitutes
        
        Args:
            product_a: First product name
            product_b: Second product name
        
        Returns:
            True if products are substitutes
        """
        for group_name, products in self.substitute_groups.items():
            products_upper = [p.upper() for p in products]
            
            if (any(p in product_a.upper() for p in products_upper) and
                any(p in product_b.upper() for p in products_upper)):
                return True
        
        return False
    
    def get_substitute_group(self, product_name: str) -> Optional[List[str]]:
        """
        Get substitute group for a product
        
        Args:
            product_name: Product name
        
        Returns:
            List of substitute products, or None if no group found
        """
        for group_name, products in self.substitute_groups.items():
            if any(p.upper() in product_name.upper() for p in products):
                return products
        
        return None
    
    def validate_basket(
        self,
        basket: Dict[str, List[Tuple[str, int]]],
        raise_error: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate basket against constraints
        
        Args:
            basket: Dict mapping category -> [(product_name, quantity)]
            raise_error: Whether to raise exception on validation failure
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Check category constraints
        for category, products in basket.items():
            constraint = self.get_category_constraint(category)
            
            # Check max products
            if len(products) > constraint.max_products:
                violations.append(
                    f"Category {category} has {len(products)} products "
                    f"(max {constraint.max_products})"
                )
            
            # Check min products
            if len(products) < constraint.min_products:
                violations.append(
                    f"Category {category} has {len(products)} products "
                    f"(min {constraint.min_products})"
                )
            
            # Check quantity constraints
            for product_name, quantity in products:
                if quantity > constraint.max_quantity_per_product:
                    violations.append(
                        f"Product {product_name} has quantity {quantity} "
                        f"(max {constraint.max_quantity_per_product})"
                    )
        
        # Check for multiple substitutes
        all_products = [p for products in basket.values() for p, _ in products]
        for i, product_a in enumerate(all_products):
            for product_b in all_products[i+1:]:
                if self.is_substitute(product_a, product_b):
                    violations.append(
                        f"Basket contains substitutes: {product_a} and {product_b}"
                    )
        
        is_valid = len(violations) == 0
        
        if not is_valid and raise_error:
            raise ValueError(f"Basket validation failed:\n" + "\n".join(violations))
        
        return is_valid, violations
    
    def suggest_complements(
        self,
        current_basket: Dict[str, List[str]],
        available_products: Dict[str, List[str]],
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Suggest complementary products based on current basket
        
        Args:
            current_basket: Dict mapping category -> [product_names]
            available_products: Dict mapping category -> [available_product_names]
            max_suggestions: Maximum number of suggestions
        
        Returns:
            List of suggested product names
        """
        suggestions = []
        
        # Get all products in basket
        basket_products = [p for products in current_basket.values() for p in products]
        
        # Find complements
        for product in basket_products:
            complements = self.get_complements(product)
            
            for complement in complements:
                # Check if complement is available and not already in basket
                for category, products in available_products.items():
                    for available_product in products:
                        if (complement.upper() in available_product.upper() and
                            available_product not in basket_products):
                            suggestions.append(available_product)
                            
                            if len(suggestions) >= max_suggestions:
                                return suggestions
        
        return suggestions[:max_suggestions]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_typical_basket_categories(trip_type: str) -> List[str]:
    """
    Get typical categories for a trip type
    
    Args:
        trip_type: Trip purpose type
    
    Returns:
        List of category names
    """
    typical_categories = {
        'stock_up': ['DAIRY', 'PRODUCE', 'MEAT', 'GROCERY', 'FROZEN', 'BAKERY'],
        'fill_in': ['DAIRY', 'PRODUCE', 'BAKERY'],
        'meal_prep': ['MEAT', 'PRODUCE', 'GROCERY'],
        'convenience': ['BEVERAGE', 'SNACKS', 'DAIRY'],
        'special_occasion': ['MEAT', 'PRODUCE', 'BEVERAGE', 'FROZEN', 'BAKERY']
    }
    
    return typical_categories.get(trip_type, ['DAIRY', 'PRODUCE', 'GROCERY'])
