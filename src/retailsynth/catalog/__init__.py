"""
Product catalog module for RetailSynth Enhanced.

This module handles extraction and processing of real product catalogs
from the Dunnhumby Complete Journey dataset.
"""

from .product_catalog_builder import ProductCatalogBuilder
from .hierarchy_mapper import HierarchyMapper
from .archetype_classifier import ArchetypeClassifier

__all__ = [
    'ProductCatalogBuilder',
    'HierarchyMapper',
    'ArchetypeClassifier',
]