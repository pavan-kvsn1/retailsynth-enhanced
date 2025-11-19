"""
Quick test to verify modular structure works
"""
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth import EnhancedRetailConfig, EnhancedRetailSynthV4_1

from retailsynth.config import EnhancedRetailConfig
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1

config = EnhancedRetailConfig(
    n_customers=1000,
    n_products=1000,
    n_stores=10,
    simulation_weeks=2,
    use_real_catalog=True,
    product_catalog_path='data/processed/product_catalog/product_catalog_20k.parquet'  
)

generator = EnhancedRetailSynthV4_1(config)
datasets = generator.generate_all_datasets() 

# Load the trained HMM model
generator.load_elasticity_models(
    elasticity_dir='data/processed/elasticity_models',
    products=generator.datasets['products']
)