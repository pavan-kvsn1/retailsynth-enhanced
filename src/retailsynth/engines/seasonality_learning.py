"""
Seasonality Learning Engine (Phase 2.7)

This module learns and applies product-specific seasonal patterns from historical data.

Key Features:
1. Product-specific seasonal indices (52 weeks)
2. Category-level fallback patterns
3. Multiplicative seasonality (demand multipliers)
4. Holiday and event detection

Replaces hard-coded seasonality with data-driven patterns learned from Dunnhumby.

Author: RetailSynth Team
Sprint: 2, Phase: 2.7
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import pickle
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SeasonalPattern:
    """Seasonal pattern for a product or category"""
    entity_id: int  # Product ID or category ID
    entity_type: str  # 'product' or 'category'
    weekly_indices: np.ndarray  # 52 values, one per week
    baseline: float  # Average demand level
    n_observations: int  # Number of data points used
    confidence: float  # Confidence score [0, 1]


class LearnedSeasonalityEngine:
    """
    Applies learned seasonal patterns to transaction generation
    
    Uses product-specific patterns when available, falls back to
    category patterns for products with insufficient data.
    """
    
    def __init__(self, 
                 seasonal_patterns_path: Optional[str] = None,
                 enable_seasonality: bool = True,
                 min_confidence: float = 0.3):
        """
        Initialize seasonality engine
        
        Args:
            seasonal_patterns_path: Path to learned seasonal patterns (.pkl)
            enable_seasonality: Whether to apply seasonal effects
            min_confidence: Minimum confidence to use a pattern
        """
        self.enable_seasonality = enable_seasonality
        self.min_confidence = min_confidence
        
        # Storage for patterns
        self.product_patterns: Dict[int, SeasonalPattern] = {}
        self.category_patterns: Dict[str, SeasonalPattern] = {}
        
        # Statistics
        self.n_products_with_patterns = 0
        self.n_categories_with_patterns = 0
        
        # Load patterns if path provided
        if seasonal_patterns_path:
            self.load_patterns(seasonal_patterns_path)
        
        logger.info(f"LearnedSeasonalityEngine initialized")
        logger.info(f"  • Seasonality: {'✅ Enabled' if enable_seasonality else '❌ Disabled'}")
        logger.info(f"  • Product patterns: {self.n_products_with_patterns:,}")
        logger.info(f"  • Category patterns: {self.n_categories_with_patterns}")
    
    def load_patterns(self, patterns_path: str):
        """
        Load learned seasonal patterns from file
        
        Args:
            patterns_path: Path to pickle file with patterns
        """
        path = Path(patterns_path)
        
        if not path.exists():
            logger.warning(f"Seasonal patterns not found: {patterns_path}")
            logger.warning("  → Using uniform seasonality (no learned patterns)")
            return
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Load product patterns
            if 'product_patterns' in data:
                sample_keys = []
                for prod_id, pattern_dict in data['product_patterns'].items():
                    # Handle both pickle (int keys) and JSON (string keys)
                    if isinstance(prod_id, str):
                        prod_id = int(prod_id)
                    
                    self.product_patterns[prod_id] = SeasonalPattern(**pattern_dict)
                    
                    if len(sample_keys) < 3:
                        sample_keys.append((prod_id, type(prod_id).__name__))
                
                logger.info(f"   • Products: {len(self.product_patterns):,}")
                if sample_keys:
                    logger.info(f"   • Sample IDs: {sample_keys}")
            
            # Load category patterns
            if 'category_patterns' in data:
                for cat_id, pattern_dict in data['category_patterns'].items():
                    # Category keys stay as strings
                    self.category_patterns[cat_id] = SeasonalPattern(**pattern_dict)
            
            self.n_products_with_patterns = len(self.product_patterns)
            self.n_categories_with_patterns = len(self.category_patterns)
            
            logger.info(f"✅ Loaded seasonal patterns from {path.name}")
            logger.info(f"   • Products: {self.n_products_with_patterns:,}")
            logger.info(f"   • Categories: {self.n_categories_with_patterns}")
            
        except Exception as e:
            logger.error(f"Failed to load seasonal patterns: {e}")
            logger.warning("  → Using uniform seasonality")
    
    def get_seasonal_multiplier(self,
                                product_id: int,
                                week_of_year: int,
                                category: Optional[str] = None,
                                fallback_value: float = 1.0) -> float:
        """
        Get seasonal demand multiplier for a product in a specific week
        
        Args:
            product_id: Product ID
            week_of_year: Week number (1-52)
            category: Product category (for fallback)
            fallback_value: Default multiplier if no pattern found
        
        Returns:
            Seasonal multiplier (1.0 = baseline, >1.0 = high season, <1.0 = low season)
        """
        if not self.enable_seasonality:
            return fallback_value
        
        # Ensure week is in valid range
        week_idx = ((week_of_year - 1) % 52)
        
        # Try product-specific pattern first
        if product_id in self.product_patterns:
            pattern = self.product_patterns[product_id]
            
            # Only use if confidence is high enough
            if pattern.confidence >= self.min_confidence:
                return float(pattern.weekly_indices[week_idx])
        
        # Fallback to category pattern
        if category and category in self.category_patterns:
            pattern = self.category_patterns[category]
            
            if pattern.confidence >= self.min_confidence:
                return float(pattern.weekly_indices[week_idx])
        
        # No pattern found - return baseline
        return fallback_value
    
    def get_seasonal_multipliers_vectorized(self,
                                           product_ids: np.ndarray,
                                           week_of_year: int,
                                           categories: Optional[np.ndarray] = None,
                                           fallback_value: float = 1.0) -> np.ndarray:
        """
        Get seasonal multipliers for multiple products at once (vectorized)
        
        Args:
            product_ids: Array of product IDs
            week_of_year: Week number (1-52)
            categories: Array of categories (same length as product_ids)
            fallback_value: Default multiplier
        
        Returns:
            Array of seasonal multipliers
        """
        # Debug: Check if function is called
        if not hasattr(self, '_function_called'):
            self._function_called = True
            print(f"         SEASONALITY: get_seasonal_multipliers_vectorized() CALLED")
            print(f"            • enable_seasonality = {self.enable_seasonality}")
            print(f"            • n_products = {len(product_ids)}")
            print(f"            • week_of_year = {week_of_year}")
        
        if not self.enable_seasonality:
            return np.full(len(product_ids), fallback_value)
        
        multipliers = np.full(len(product_ids), fallback_value)
        week_idx = ((week_of_year - 1) % 52)

        # Debug first call
        if not hasattr(self, '_debug_logged'):
            self._debug_logged = True
            if len(product_ids) > 0:
                sample_id = product_ids[0]
                sample_id_int = int(sample_id)  # Convert numpy type to Python int
                print(f"         SEASONALITY DEBUG:")
                print(f"            • Lookup prod_id={sample_id} (numpy type={type(sample_id).__name__})")
                print(f"            • Converted to: {sample_id_int} (type={type(sample_id_int).__name__})")
                print(f"            • In patterns dict: {sample_id_int in self.product_patterns}")
                print(f"            • Pattern dict has {len(self.product_patterns)} products")
                if len(self.product_patterns) > 0:
                    first_key = next(iter(self.product_patterns.keys()))
                    print(f"            • First pattern key: {first_key} (type={type(first_key).__name__})")
                    # Check if any of the first 10 product_ids match
                    matches = sum(1 for pid in product_ids[:10] if int(pid) in self.product_patterns)
                    print(f"            • Matches in first 10 products: {matches}/10")
        
        
        for i, prod_id in enumerate(product_ids):
            # Convert numpy int to Python int for dictionary lookup
            prod_id = int(prod_id)
            
            # Try product pattern
            if prod_id in self.product_patterns:
                pattern = self.product_patterns[prod_id]
                if pattern.confidence >= self.min_confidence:
                    multipliers[i] = pattern.weekly_indices[week_idx]
                    continue
            
            # Fallback to category
            if categories is not None:
                cat = categories[i]
                if cat in self.category_patterns:
                    pattern = self.category_patterns[cat]
                    if pattern.confidence >= self.min_confidence:
                        multipliers[i] = pattern.weekly_indices[week_idx]
        
        return multipliers
    
    def get_pattern_info(self, product_id: int, category: Optional[str] = None) -> Dict:
        """
        Get information about seasonal pattern for a product
        
        Args:
            product_id: Product ID
            category: Product category (for fallback)
        
        Returns:
            Dict with pattern information
        """
        info = {
            'has_product_pattern': False,
            'has_category_pattern': False,
            'pattern_type': 'none',
            'confidence': 0.0,
            'peak_week': None,
            'trough_week': None,
            'seasonality_strength': 0.0
        }
        
        pattern = None
        
        # Check product pattern
        if product_id in self.product_patterns:
            pattern = self.product_patterns[product_id]
            info['has_product_pattern'] = True
            info['pattern_type'] = 'product'
        
        # Check category pattern
        elif category and category in self.category_patterns:
            pattern = self.category_patterns[category]
            info['has_category_pattern'] = True
            info['pattern_type'] = 'category'
        
        # Extract pattern info
        if pattern:
            info['confidence'] = pattern.confidence
            info['peak_week'] = int(np.argmax(pattern.weekly_indices)) + 1
            info['trough_week'] = int(np.argmin(pattern.weekly_indices)) + 1
            info['seasonality_strength'] = float(np.std(pattern.weekly_indices))
        
        return info
    
    def get_coverage_stats(self, product_ids: List[int], 
                          categories: Optional[List[str]] = None) -> Dict:
        """
        Get statistics about pattern coverage
        
        Args:
            product_ids: List of product IDs in dataset
            categories: List of categories (optional)
        
        Returns:
            Dict with coverage statistics
        """
        n_products = len(product_ids)
        n_with_product_pattern = sum(1 for pid in product_ids if pid in self.product_patterns)
        
        n_with_category_pattern = 0
        if categories:
            n_with_category_pattern = sum(
                1 for i, pid in enumerate(product_ids)
                if pid not in self.product_patterns and categories[i] in self.category_patterns
            )
        
        n_with_any_pattern = n_with_product_pattern + n_with_category_pattern
        
        return {
            'n_products': n_products,
            'n_with_product_pattern': n_with_product_pattern,
            'n_with_category_pattern': n_with_category_pattern,
            'n_with_any_pattern': n_with_any_pattern,
            'product_coverage': n_with_product_pattern / n_products if n_products > 0 else 0.0,
            'total_coverage': n_with_any_pattern / n_products if n_products > 0 else 0.0
        }
    
    def save_patterns(self, output_path: str):
        """
        Save patterns to file
        
        Args:
            output_path: Path to save patterns (.pkl)
        """
        data = {
            'product_patterns': {
                pid: {
                    'entity_id': p.entity_id,
                    'entity_type': p.entity_type,
                    'weekly_indices': p.weekly_indices,
                    'baseline': p.baseline,
                    'n_observations': p.n_observations,
                    'confidence': p.confidence
                }
                for pid, p in self.product_patterns.items()
            },
            'category_patterns': {
                cat: {
                    'entity_id': p.entity_id,
                    'entity_type': p.entity_type,
                    'weekly_indices': p.weekly_indices,
                    'baseline': p.baseline,
                    'n_observations': p.n_observations,
                    'confidence': p.confidence
                }
                for cat, p in self.category_patterns.items()
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✅ Saved seasonal patterns to {output_path}")


# Convenience function for demonstrating seasonality effects
def demonstrate_seasonality():
    """Demonstrate seasonal patterns with example"""
    print("="*70)
    print("SEASONALITY LEARNING DEMONSTRATION")
    print("="*70)
    
    # Create mock seasonal patterns
    engine = LearnedSeasonalityEngine(enable_seasonality=True)
    
    # Add example product pattern (holiday spikes)
    holiday_pattern = np.ones(52)
    holiday_pattern[46:52] = [1.2, 1.3, 1.5, 2.0, 1.8, 1.4]  # Thanksgiving + Christmas
    holiday_pattern[0:2] = [1.3, 1.1]  # New Year
    
    engine.product_patterns[12345] = SeasonalPattern(
        entity_id=12345,
        entity_type='product',
        weekly_indices=holiday_pattern,
        baseline=100.0,
        n_observations=1000,
        confidence=0.9
    )
    
    print("\n📊 Example: Holiday Season Pattern (Product 12345)")
    print(f"{'Week':<8} {'Multiplier':<12} {'Bar':<30}")
    print("-" * 50)
    
    for week in [1, 12, 24, 36, 47, 48, 49, 50, 51, 52]:
        mult = engine.get_seasonal_multiplier(12345, week)
        bar = "█" * int(mult * 20)
        print(f"{week:<8} {mult:<12.2f} {bar}")
    
    print("\n💡 Interpretation:")
    print("  • Weeks 47-52: Holiday season (1.5x - 2.0x demand)")
    print("  • Week 1-2: New Year (1.1x - 1.3x demand)")
    print("  • Other weeks: Baseline demand (1.0x)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    # Run demonstration
    demonstrate_seasonality()
