"""
RetailSynth Main Module
======================
Main module for RetailSynth.
"""

# Use lazy imports to avoid requiring JAX for catalog-only usage
def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless needed."""
    if name == 'EnhancedRetailSynthV4_1':
        from .generators import EnhancedRetailSynthV4_1
        return EnhancedRetailSynthV4_1
    elif name == 'EnhancedRetailConfig':
        from .config import EnhancedRetailConfig
        return EnhancedRetailConfig
    elif name == 'CalibrationEngine':
        from .calibration import CalibrationEngine
        return CalibrationEngine
    elif name == 'GPUUtilityEngine':
        from .engines import GPUUtilityEngine
        return GPUUtilityEngine
    elif name == 'ChoiceModel':
        from .engines import ChoiceModel
        return ChoiceModel
    elif name == 'PricingEvolutionEngine':
        from .engines import PricingEvolutionEngine
        return PricingEvolutionEngine
    elif name == 'ProductLifecycleEngine':
        from .engines import ProductLifecycleEngine
        return ProductLifecycleEngine
    elif name == 'SeasonalityEngine':
        from .engines import SeasonalityEngine
        return SeasonalityEngine
    elif name == 'MarketDynamicsEngine':
        from .engines import MarketDynamicsEngine
        return MarketDynamicsEngine
    elif name == 'TemporalCustomerDriftEngine':
        from .engines import TemporalCustomerDriftEngine
        return TemporalCustomerDriftEngine
    elif name == 'StoreLoyaltyEngine':
        from .engines import StoreLoyaltyEngine
        return StoreLoyaltyEngine
    elif name == 'RealisticCategoryHierarchy':
        from .utils import RealisticCategoryHierarchy
        return RealisticCategoryHierarchy
    elif name == 'CustomerGenerator':
        from .generators import CustomerGenerator
        return CustomerGenerator
    elif name == 'ProductGenerator':
        from .generators import ProductGenerator
        return ProductGenerator
    elif name == 'StoreGenerator':
        from .generators import StoreGenerator
        return StoreGenerator
    elif name == 'MarketContextGenerator':
        from .generators import MarketContextGenerator
        return MarketContextGenerator
    elif name == 'ComprehensiveTransactionGenerator':
        from .generators import ComprehensiveTransactionGenerator
        return ComprehensiveTransactionGenerator
    elif name == 'ProductCatalogBuilder':
        from .catalog import ProductCatalogBuilder
        return ProductCatalogBuilder
    elif name == 'HierarchyMapper':
        from .catalog import HierarchyMapper
        return HierarchyMapper
    elif name == 'ArchetypeClassifier':
        from .catalog import ArchetypeClassifier
        return ArchetypeClassifier
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    'EnhancedRetailSynthV4_1',
    'EnhancedRetailConfig',
    'CalibrationEngine',
    'GPUUtilityEngine',
    'ChoiceModel',
    'PricingEvolutionEngine',
    'ProductLifecycleEngine',
    'SeasonalityEngine',
    'MarketDynamicsEngine',
    'TemporalCustomerDriftEngine',
    'StoreLoyaltyEngine',
    'RealisticCategoryHierarchy',
    'CustomerGenerator',
    'ProductGenerator',
    'StoreGenerator',
    'MarketContextGenerator',
    'ComprehensiveTransactionGenerator',
    'ProductCatalogBuilder',
    'HierarchyMapper',
    'ArchetypeClassifier',
]

__version__ = '4.1'
