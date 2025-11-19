#!/usr/bin/env python3
"""
Build Product Catalog Script

Extracts 20K representative SKUs from Dunnhumby's 92K products.

Usage:
    python scripts/build_product_catalog.py
    python scripts/build_product_catalog.py --n-skus 15000 --output-dir data/processed/custom_catalog
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.catalog import ProductCatalogBuilder, HierarchyMapper, ArchetypeClassifier


def main():
    parser = argparse.ArgumentParser(
        description='Build representative product catalog from Dunnhumby data'
    )
    parser.add_argument(
        '--products-path',
        type=str,
        default='data/raw/dunnhumby/product.csv',
        help='Path to Dunnhumby product.csv'
    )
    parser.add_argument(
        '--transactions-path',
        type=str,
        default='data/raw/dunnhumby/transaction_data.csv',
        help='Path to Dunnhumby transaction_data.csv'
    )
    parser.add_argument(
        '--n-skus',
        type=int,
        default=25000,
        help='Target number of SKUs (default: 25000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/product_catalog',
        help='Output directory for catalog files'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("RETAILSYNTH ENHANCED - PRODUCT CATALOG BUILDER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Products path: {args.products_path}")
    print(f"  Transactions path: {args.transactions_path}")
    print(f"  Target SKUs: {args.n_skus:,}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.random_seed}")
    print()
    
    # Step 1: Build catalog
    print("\n" + "="*70)
    print("STEP 1: LOAD AND SAMPLE PRODUCTS")
    print("="*70)
    
    builder = ProductCatalogBuilder(
        n_target_skus=args.n_skus,
        random_seed=args.random_seed
    )
    
    # Load Dunnhumby data
    builder.load_dunnhumby_data(
        products_path=args.products_path,
        transactions_path=args.transactions_path
    )
    
    # Create representative sample
    catalog = builder.create_representative_sample()
    
    # Validate sample
    metrics = builder.validate_sample()
    
    # Step 2: Build hierarchy
    print("\n" + "="*70)
    print("STEP 2: BUILD CATEGORY HIERARCHY")
    print("="*70)
    
    hierarchy_mapper = HierarchyMapper()
    hierarchy = hierarchy_mapper.build_hierarchy(catalog)
    hierarchy_mapper.print_hierarchy_summary()
    
    # Step 3: Classify archetypes
    print("\n" + "="*70)
    print("STEP 3: CLASSIFY PRODUCT ARCHETYPES")
    print("="*70)
    
    archetype_classifier = ArchetypeClassifier()
    catalog_with_archetypes = archetype_classifier.classify_products(catalog)
    archetype_classifier.print_archetype_summary()
    
    # Step 4: Save everything
    print("\n" + "="*70)
    print("STEP 4: SAVE CATALOG FILES")
    print("="*70)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Remove any existing empty files
    catalog_file = output_path / 'product_catalog_20k.parquet'
    if catalog_file.exists() and catalog_file.stat().st_size < 100:
        catalog_file.unlink()
        print(f"  Removed corrupted file: {catalog_file}")
    
    # Save catalog
    catalog_with_archetypes.to_parquet(
        catalog_file,
        index=False
    )
    print(f"✅ Saved catalog: {catalog_file}")
    
    # Save hierarchy
    hierarchy_mapper.save_hierarchy(
        output_path / 'category_hierarchy.json'
    )
    hierarchy_mapper.save_product_mapping(
        output_path / 'product_to_category.json'
    )
    
    # Save archetypes
    archetype_classifier.save_archetypes(
        output_path / 'product_archetypes.csv'
    )
    
    # Save catalog with all metadata
    builder.representative_catalog = catalog_with_archetypes
    builder.save_catalog(args.output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("CATALOG BUILD COMPLETE!")
    print("="*70)
    print(f"\nOutput files in {args.output_dir}:")
    print(f"  ✅ product_catalog_20k.parquet - Main catalog")
    print(f"  ✅ category_hierarchy.json - Category structure")
    print(f"  ✅ product_to_category.json - Product mappings")
    print(f"  ✅ product_archetypes.csv - Archetype definitions")
    print(f"  ✅ catalog_summary.json - Summary statistics")
    
    print(f"\n📊 Validation Score: {metrics['validation_score']*100:.0f}%")
    
    if metrics['validation_score'] >= 0.75:
        print("✅ VALIDATION PASSED - Catalog is ready for use!")
    else:
        print("⚠️  VALIDATION WARNING - Some metrics below threshold")
    
    print("\n🚀 Next steps:")
    print("   1. Run: python scripts/learn_price_elasticity.py")
    print("   2. Start Sprint 1.2: Price Elasticity with HMM")
    print()


if __name__ == '__main__':
    main()