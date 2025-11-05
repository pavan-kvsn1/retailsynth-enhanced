"""
Quick test to verify modular structure works
"""
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth import EnhancedRetailConfig, EnhancedRetailSynthV4_1

# Small test configuration
config = EnhancedRetailConfig(
    n_customers=1000,
    n_products=100,
    n_stores=5,
    simulation_weeks=4,
    random_seed=42
)

# Generate data
print("Testing modular generation...")
generator = EnhancedRetailSynthV4_1(config)
datasets = generator.generate_all_datasets()

# Verify outputs
print("\n✅ Test Results:")
print(f"   Customers: {len(datasets['customers']):,}")
print(f"   Products: {len(datasets['products']):,}")
print(f"   Stores: {len(datasets['stores'])}")
print(f"   Transactions: {len(datasets['transactions']):,}")
print(f"   Transaction Items: {len(datasets['transaction_items']):,}")

print("\n✅ Modular structure is working!")