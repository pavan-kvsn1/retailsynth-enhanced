"""
RetailSynth Engines Module
==========================
Core behavioral and simulation engines for retail transaction generation.
"""

from .choice_engine import ChoiceModel
from .drift_engine import TemporalCustomerDriftEngine
from .lifecycle_engine import ProductLifecycleEngine
from .loyalty_engine import StoreLoyaltyEngine
from .market_engine import MarketDynamicsEngine
from .precomputation_engine import VectorizedPreComputationEngine
from .pricing_engine import PricingEvolutionEngine
from .seasonality_engine import SeasonalityEngine
from .utility_engine import GPUUtilityEngine
from .base_price_hmm import BasePriceHMM

__all__ = [
    # Choice modeling
    'ChoiceModel',
    
    # Customer behavior engines
    'TemporalCustomerDriftEngine',
    'StoreLoyaltyEngine',
    
    # Product engines
    'ProductLifecycleEngine',
    'PricingEvolutionEngine',
    
    # Market dynamics
    'MarketDynamicsEngine',
    'SeasonalityEngine',
    
    # Computation engines
    'VectorizedPreComputationEngine',
    'GPUUtilityEngine',
    
    # Pricing & Promotions (Sprint 2)
    'BasePriceHMM',
]

__version__ = '3.7'