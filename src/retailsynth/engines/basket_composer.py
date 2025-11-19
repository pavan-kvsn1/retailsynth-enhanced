"""
Basket Composer (Sprint 1.4)

Generates realistic, coherent shopping baskets based on:
- Trip purpose (stock-up, fill-in, meal-prep, etc.)
- Category constraints and requirements
- Complement/substitute relationships
- Customer utilities and preferences
- Purchase history and habits

Replaces independent product sampling with structured basket composition.

Author: RetailSynth Team
Date: November 2024
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from collections import defaultdict

from .trip_purpose import TripPurpose, TripPurposeSelector, TripCharacteristics
from .category_constraints import CategoryConstraintEngine, CategoryConstraint
from .customer_state import CustomerState
from retailsynth.config import EnhancedRetailConfig


class BasketComposer:
    """
    Composes realistic shopping baskets based on trip purpose and constraints
    
    Process:
    1. Determine trip purpose → basket size
    2. Select required categories
    3. Sample products within each category
    4. Add complementary items
    5. Enforce quantity constraints
    6. Validate basket coherence
    """
    
    def __init__(
        self,
        products: pd.DataFrame,
        config: Optional[EnhancedRetailConfig] = None,
        enable_complements: bool = True,
        enable_substitutes: bool = True
    ):
        """
        Initialize basket composer
        
        Args:
            products: Product catalog with columns:
                - product_id
                - commodity_desc (category)
                - sub_commodity_desc
                - brand
                - curr_size_of_product
            config: Configuration object with tunable parameters
            enable_complements: Whether to add complementary items
            enable_substitutes: Whether to enforce substitute constraints
        """
        self.products = products
        self.config = config
        self.enable_complements = enable_complements
        self.enable_substitutes = enable_substitutes
        
        # Use config parameters if available
        if self.config:
            self.complement_probability = self.config.complement_probability
            self.substitute_avoidance = self.config.substitute_avoidance
            self.category_diversity_weight = self.config.category_diversity_weight
        else:
            # Fallback defaults
            self.complement_probability = 0.4
            self.substitute_avoidance = 0.8
            self.category_diversity_weight = 0.3
        
        # Initialize engines
        self.trip_selector = TripPurposeSelector(config=self.config)  # Pass config for trip characteristics
        self.constraint_engine = CategoryConstraintEngine()
        
        # Build product lookups
        self._build_product_mappings()
    
    def _build_product_mappings(self):
        """Build efficient lookup structures"""
        # Map category -> list of product IDs
        self.category_to_products = defaultdict(list)
        
        # Category mapping: Real Dunnhumby categories → Generic categories
        self.category_mapping = {
            # Beverages
            'SOFT DRINKS': 'BEVERAGES',
            'JUICE-FRUIT/VEGETABLE': 'BEVERAGES',
            'WATER': 'BEVERAGES',
            'SPORTS DRINKS': 'BEVERAGES',
            'TEA': 'BEVERAGES',
            'COFFEE': 'BEVERAGES',
            
            # Dairy
            'CHEESE': 'DAIRY',
            'YOGURT': 'DAIRY',
            'MILK': 'DAIRY',
            'BUTTER AND MARGARINE': 'DAIRY',
            'SOUR CREAM/DIPS': 'DAIRY',
            'EGGS': 'DAIRY',
            
            # Snacks
            'BAG SNACKS': 'SNACKS',
            'CRACKERS/MISC BKD FD': 'SNACKS',
            'CANDY - PACKAGED': 'SNACKS',
            'COOKIES': 'SNACKS',
            'NUTS': 'SNACKS',
            
            # Bakery
            'BAKED BREAD/BUNS/ROLLS': 'BAKERY',
            'FRESH BAKERY': 'BAKERY',
            
            # Meat
            'FRZN MEAT/MEAT DINNERS': 'MEAT',
            'PROCESSED MEATS': 'MEAT',
            'BEEF': 'MEAT',
            'PORK': 'MEAT',
            'POULTRY': 'MEAT',
            
            # Produce
            'VEGETABLES - SHELF STABLE': 'PRODUCE',
            'FRUIT - SHELF STABLE': 'PRODUCE',
            
            # Pantry
            'SOUP': 'PANTRY',
            'PASTA': 'PANTRY',
            'RICE': 'PANTRY',
            'CANNED VEGETABLES': 'PANTRY',
            'CANNED FRUIT': 'PANTRY',
            'CONDIMENTS/GRAVIES/SAUCES': 'PANTRY',
            
            # Frozen
            'FROZEN PIZZA': 'FROZEN',
            'ICE CREAM/MILK/SHERBETS': 'FROZEN',
            'FROZEN VEGETABLES': 'FROZEN',
            
            # Breakfast
            'COLD CEREAL': 'BREAKFAST',
            'HOT CEREAL': 'BREAKFAST',
            'BREAKFAST FOOD': 'BREAKFAST',
            
            # Personal Care
            'HAIR CARE PRODUCTS': 'PERSONAL_CARE',
            'ORAL HYGIENE PRODUCTS': 'PERSONAL_CARE',
            'MAKEUP AND TREATMENT': 'PERSONAL_CARE',
            'SKIN CARE PRODUCTS': 'PERSONAL_CARE',
            
            # Household
            'LAUNDRY SUPPLIES': 'HOUSEHOLD',
            'PAPER PRODUCTS': 'HOUSEHOLD',
            'CLEANING PRODUCTS': 'HOUSEHOLD',
            
            # Baby
            'BABY FOODS': 'BABY',
            'DIAPERS': 'BABY',
        }
        
        for _, row in self.products.iterrows():
            category = row.get('commodity_desc', 'GROCERY')
            if pd.notna(category):
                category = category.upper()
                
                # Map to generic category if mapping exists
                generic_category = self.category_mapping.get(category, 'GROCERY')
                
                # Store under both real and generic categories
                self.category_to_products[category].append(row['product_id'])
                if generic_category != category:
                    self.category_to_products[generic_category].append(row['product_id'])
        
        # Debug: Print category distribution
        print(f"\n🔍 Basket Composer: Found {len(self.category_to_products)} categories:")
        for cat, products in sorted(self.category_to_products.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"   • {cat}: {len(products)} products")
        
        # Map product_id -> category
        self.product_to_category = dict(zip(
            self.products['product_id'],
            self.products['commodity_desc'].str.upper()
        ))
        
        # Map product_id -> sub_category
        self.product_to_subcategory = dict(zip(
            self.products['product_id'],
            self.products.get('sub_commodity_desc', 'Unknown')
        ))
        
        # Map product_id -> product name
        self.product_to_name = dict(zip(
            self.products['product_id'],
            self.products.get('commodity_desc', 'Unknown')
        ))
    
    def generate_basket(
        self,
        customer_id: int,
        shopping_personality: str,
        utilities: np.ndarray,
        product_ids: np.ndarray,
        customer_state: Optional[CustomerState] = None,
        week_number: int = 1,
        day_of_week: Optional[int] = None,
        promo_flags: Optional[np.ndarray] = None
    ) -> List[Tuple[int, int]]:
        """
        Generate a coherent shopping basket
        
        Args:
            customer_id: Customer ID
            shopping_personality: Customer type (price_anchor, convenience, planned, impulse)
            utilities: Utility values for all products
            product_ids: Array of product IDs corresponding to utilities
            customer_state: Optional customer state for history-aware composition
            week_number: Current week (for seasonality)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            promo_flags: Optional promotional flags array (1.0 = on promo, 0.0 = not)
        
        Returns:
            List of (product_id, quantity) tuples
        """
        # Store promo flags and product IDs for quantity boost logic
        self._current_promo_flags = promo_flags
        self._current_product_ids = product_ids
        
        # Step 1: Determine trip purpose
        weeks_since_last = self._get_weeks_since_last_trip(customer_state)
        trip_purpose = self.trip_selector.select_trip_purpose(
            shopping_personality=shopping_personality,
            weeks_since_last_trip=weeks_since_last,
            day_of_week=day_of_week,
            week_of_year=week_number
        )
        
        # Step 2: Determine basket size
        target_basket_size = self.trip_selector.get_basket_size(trip_purpose)
        
        # Step 3: Get required and optional categories
        required_categories = self.trip_selector.get_required_categories(trip_purpose)
        optional_categories = self.trip_selector.get_optional_categories(trip_purpose)
        min_categories = self.trip_selector.get_min_categories(trip_purpose)
        
        # Step 4: Build basket by category
        basket = self._compose_basket_by_category(
            trip_purpose=trip_purpose,
            required_categories=required_categories,
            optional_categories=optional_categories,
            min_categories=min_categories,
            target_size=target_basket_size,
            utilities=utilities,
            product_ids=product_ids,
            customer_state=customer_state
        )
        
        # Step 5: Add complementary items if enabled
        if self.enable_complements and len(basket) < target_basket_size:
            basket = self._add_complements(
                basket=basket,
                utilities=utilities,
                product_ids=product_ids,
                max_additions=target_basket_size - len(basket)
            )
        
        # Step 6: Enforce quantity constraints
        basket = self._enforce_quantity_constraints(basket, trip_purpose)
        
        return basket
    
    def _get_weeks_since_last_trip(self, customer_state: Optional[CustomerState]) -> int:
        """Calculate weeks since customer's last shopping trip"""
        if customer_state is None:
            return 1
        
        # Get most recent purchase across all products
        if not customer_state.last_purchase_week:
            return 2  # New customer, assume needs stock-up
        
        last_week = max(customer_state.last_purchase_week.values())
        weeks_since = customer_state.current_week - last_week
        
        return max(0, weeks_since)
    
    def _compose_basket_by_category(
        self,
        trip_purpose: TripPurpose,
        required_categories: List[str],
        optional_categories: List[str],
        min_categories: int,
        target_size: int,
        utilities: np.ndarray,
        product_ids: np.ndarray,
        customer_state: Optional[CustomerState]
    ) -> List[Tuple[int, int]]:
        """
        Compose basket by selecting products from categories
        
        Strategy:
        1. Fill required categories first
        2. Add optional categories until target size reached
        3. Use utilities to select products within categories
        4. Respect category constraints
        """
        basket = []
        selected_categories = set()
        
        # Create utility lookup
        utility_map = dict(zip(product_ids, utilities))
        
        # Step 1: Fill required categories
        for category in required_categories:
            if category not in self.category_to_products:
                continue
            
            products = self._select_products_from_category(
                category=category,
                utility_map=utility_map,
                trip_purpose=trip_purpose,
                customer_state=customer_state,
                already_selected=set(p for p, _ in basket)
            )
            
            basket.extend(products)
            selected_categories.add(category)
        
        # Step 2: Add optional categories until we reach target size or min categories
        remaining_size = target_size - len(basket)
        available_optional = [c for c in optional_categories if c in self.category_to_products]
        
        # Sort optional categories by average utility
        category_utilities = []
        for category in available_optional:
            cat_products = self.category_to_products[category]
            cat_utilities = [utility_map.get(p, 0) for p in cat_products]
            avg_utility = np.mean(cat_utilities) if cat_utilities else 0
            category_utilities.append((category, avg_utility))
        
        category_utilities.sort(key=lambda x: x[1], reverse=True)
        
        for category, _ in category_utilities:
            if len(basket) >= target_size:
                break
            
            if len(selected_categories) >= min_categories and len(basket) >= target_size * 0.8:
                break
            
            products = self._select_products_from_category(
                category=category,
                utility_map=utility_map,
                trip_purpose=trip_purpose,
                customer_state=customer_state,
                already_selected=set(p for p, _ in basket),
                max_products=min(3, remaining_size)
            )
            
            basket.extend(products)
            selected_categories.add(category)
            remaining_size = target_size - len(basket)
        
        return basket
    
    def _select_products_from_category(
        self,
        category: str,
        utility_map: Dict[int, float],
        trip_purpose: TripPurpose,
        customer_state: Optional[CustomerState],
        already_selected: Set[int],
        max_products: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Select products from a specific category
        
        Args:
            category: Category name
            utility_map: Dict mapping product_id -> utility
            trip_purpose: Trip purpose
            customer_state: Customer state for history-aware selection
            already_selected: Set of already selected product IDs
            max_products: Maximum products to select (None = use constraint)
        
        Returns:
            List of (product_id, quantity) tuples
        """
        # Get category constraint
        constraint = self.constraint_engine.get_category_constraint(category)
        
        if max_products is None:
            # Sample number of products from category
            n_products = np.random.randint(
                constraint.min_products,
                min(constraint.max_products, len(self.category_to_products[category])) + 1
            )
        else:
            n_products = min(max_products, constraint.max_products)
        
        # Get available products in category
        available_products = [
            p for p in self.category_to_products[category]
            if p not in already_selected and p in utility_map
        ]
        
        if not available_products:
            return []
        
        # Get utilities for available products
        product_utilities = np.array([utility_map[p] for p in available_products])
        
        # Apply purchase history boost if available
        if customer_state is not None:
            for i, product_id in enumerate(available_products):
                # Boost utility for previously purchased products
                if product_id in customer_state.purchase_count:
                    habit_strength = customer_state.habit_strength.get(product_id, 0)
                    product_utilities[i] += habit_strength * 2.0  # Habit boost
        
        # Convert utilities to probabilities (softmax)
        exp_utilities = np.exp(product_utilities - np.max(product_utilities))
        probabilities = exp_utilities / exp_utilities.sum()
        
        # Check for valid probabilities
        valid_probs = probabilities > 1e-10  # Avoid numerical zeros
        n_valid = np.sum(valid_probs)
        
        if n_valid == 0:
            # Fallback: uniform probabilities if all utilities are too low
            probabilities = np.ones(len(available_products)) / len(available_products)
            n_valid = len(available_products)
        
        # Sample products (can't sample more than available valid products)
        n_to_select = min(n_products, len(available_products), n_valid)
        
        if n_to_select == 0:
            return []
        
        selected_indices = np.random.choice(
            len(available_products),
            size=n_to_select,
            replace=False,
            p=probabilities
        )
        
        # Assign quantities
        selected_products = []
        for idx in selected_indices:
            product_id = available_products[idx]
            
            # FIX Priority 2A: Use config-based quantity distribution
            if self.config:
                # Sample from config distribution (tunable!)
                base_quantity = max(1, int(np.random.normal(
                    self.config.quantity_mean,
                    self.config.quantity_std
                )))
                
                # FIX Priority 2B: Apply promotional quantity boost
                if self._current_promo_flags is not None and self._current_product_ids is not None:
                    # Find if this product is on promotion
                    product_idx = np.where(self._current_product_ids == product_id)[0]
                    if len(product_idx) > 0 and self._current_promo_flags[product_idx[0]] > 0:
                        # Product is on promotion - apply boost
                        # Not all customers stockpile - 60% probability
                        if np.random.random() < 0.6:
                            base_quantity = int(base_quantity * self.config.promotion_quantity_boost)
                
                # Apply category constraint as upper bound
                quantity = min(base_quantity, constraint.max_quantity_per_product, self.config.quantity_max)
                
                # Stock-up trips: increase quantity probability (tunable via trip characteristics)
                if trip_purpose == TripPurpose.STOCK_UP and np.random.random() < 0.3:
                    quantity = min(quantity + 1, constraint.max_quantity_per_product, self.config.quantity_max)
            else:
                # Fallback: Original hardcoded logic
                if customer_state and product_id in customer_state.purchase_count:
                    quantity = constraint.typical_quantity
                else:
                    quantity = 1
                
                if trip_purpose == TripPurpose.STOCK_UP and np.random.random() < 0.3:
                    quantity = min(quantity + 1, constraint.max_quantity_per_product)
            
            selected_products.append((product_id, quantity))
        
        return selected_products
    
    def _add_complements(
        self,
        basket: List[Tuple[int, int]],
        utilities: np.ndarray,
        product_ids: np.ndarray,
        max_additions: int
    ) -> List[Tuple[int, int]]:
        """
        Add complementary products to basket
        
        Args:
            basket: Current basket
            utilities: Product utilities
            product_ids: Product IDs
            max_additions: Maximum products to add
        
        Returns:
            Updated basket with complements
        """
        if max_additions <= 0:
            return basket
        
        utility_map = dict(zip(product_ids, utilities))
        basket_product_ids = set(p for p, _ in basket)
        additions = []
        
        # For each product in basket, find complements
        for product_id, _ in basket:
            if len(additions) >= max_additions:
                break
            
            # Get product name/category
            product_name = self.product_to_name.get(product_id, '')
            
            # Find complements
            complements = self.constraint_engine.get_complements(product_name)
            
            for complement_name in complements:
                if len(additions) >= max_additions:
                    break
                
                # Find products matching complement
                for candidate_id in product_ids:
                    if candidate_id in basket_product_ids or candidate_id in [p for p, _ in additions]:
                        continue
                    
                    candidate_name = self.product_to_name.get(candidate_id, '')
                    
                    if complement_name.upper() in candidate_name.upper():
                        # Add complement with probability based on utility
                        complement_utility = utility_map.get(candidate_id, 0)
                        add_prob = 1.0 / (1.0 + np.exp(-complement_utility))  # Sigmoid
                        
                        if np.random.random() < add_prob * self.complement_probability:  # 70% of sigmoid prob
                            additions.append((candidate_id, 1))
                            basket_product_ids.add(candidate_id)
                            break
        
        return basket + additions
    
    def _enforce_quantity_constraints(
        self,
        basket: List[Tuple[int, int]],
        trip_purpose: TripPurpose
    ) -> List[Tuple[int, int]]:
        """
        Enforce quantity constraints on basket
        
        Rules:
        - No more than max_same_product of any single product
        - Respect category quantity constraints
        """
        max_same_product = self.trip_selector.get_max_same_product(trip_purpose)
        
        # Group by product and sum quantities
        product_quantities = defaultdict(int)
        for product_id, quantity in basket:
            product_quantities[product_id] += quantity
        
        # Enforce max quantity per product
        constrained_basket = []
        for product_id, total_quantity in product_quantities.items():
            # Get category constraint
            category = self.product_to_category.get(product_id, 'GROCERY')
            constraint = self.constraint_engine.get_category_constraint(category)
            
            # Apply constraints
            final_quantity = min(
                total_quantity,
                max_same_product,
                constraint.max_quantity_per_product
            )
            
            constrained_basket.append((product_id, final_quantity))
        
        return constrained_basket
    
    def validate_basket(
        self,
        basket: List[Tuple[int, int]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate basket coherence
        
        Args:
            basket: List of (product_id, quantity) tuples
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        # Group by category
        category_products = defaultdict(list)
        for product_id, quantity in basket:
            category = self.product_to_category.get(product_id, 'GROCERY')
            product_name = self.product_to_name.get(product_id, f'Product_{product_id}')
            category_products[category].append((product_name, quantity))
        
        # Validate using constraint engine
        is_valid, constraint_violations = self.constraint_engine.validate_basket(
            category_products,
            raise_error=False
        )
        
        violations.extend(constraint_violations)
        
        # Additional checks
        if len(basket) == 0:
            violations.append("Empty basket")
        
        if len(basket) > 100:
            violations.append(f"Basket too large: {len(basket)} items")
        
        return len(violations) == 0, violations
    
    def get_basket_statistics(
        self,
        basket: List[Tuple[int, int]]
    ) -> Dict:
        """
        Calculate basket statistics
        
        Args:
            basket: List of (product_id, quantity) tuples
        
        Returns:
            Dictionary with basket metrics
        """
        if not basket:
            return {
                'total_items': 0,
                'unique_products': 0,
                'num_categories': 0,
                'avg_quantity': 0,
                'max_quantity': 0
            }
        
        # Count categories
        categories = set()
        for product_id, _ in basket:
            category = self.product_to_category.get(product_id, 'GROCERY')
            categories.add(category)
        
        # Calculate quantities
        quantities = [q for _, q in basket]
        
        return {
            'total_items': sum(quantities),
            'unique_products': len(basket),
            'num_categories': len(categories),
            'avg_quantity': np.mean(quantities),
            'max_quantity': max(quantities),
            'categories': list(categories)
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_basket_coherence_score(
    basket: List[Tuple[int, int]],
    products: pd.DataFrame,
    constraint_engine: CategoryConstraintEngine
) -> float:
    """
    Calculate coherence score for a basket (0-1)
    
    Higher score = more realistic basket
    
    Factors:
    - Category diversity (not all from one category)
    - Complement presence (pasta + sauce)
    - No excessive quantities
    - Reasonable basket size
    
    Args:
        basket: List of (product_id, quantity) tuples
        products: Product catalog
        constraint_engine: Constraint engine
    
    Returns:
        Coherence score (0-1)
    """
    if not basket:
        return 0.0
    
    score = 1.0
    
    # Factor 1: Category diversity (0.3 weight)
    product_to_category = dict(zip(products['product_id'], products['commodity_desc']))
    categories = set(product_to_category.get(p, 'GROCERY') for p, _ in basket)
    
    diversity_score = min(len(categories) / 5.0, 1.0)  # Ideal: 5+ categories
    score *= (0.7 + 0.3 * diversity_score)
    
    # Factor 2: Quantity reasonableness (0.2 weight)
    max_quantity = max(q for _, q in basket)
    if max_quantity > 5:
        score *= 0.8  # Penalize excessive quantities
    
    # Factor 3: Basket size reasonableness (0.2 weight)
    total_items = sum(q for _, q in basket)
    if total_items < 2 or total_items > 60:
        score *= 0.8
    
    # Factor 4: No duplicate substitutes (0.3 weight)
    product_names = [product_to_category.get(p, '') for p, _ in basket]
    for i, name_a in enumerate(product_names):
        for name_b in product_names[i+1:]:
            if constraint_engine.is_substitute(name_a, name_b):
                score *= 0.7  # Penalize substitutes
    
    return np.clip(score, 0.0, 1.0)
