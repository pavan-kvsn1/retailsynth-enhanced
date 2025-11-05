
"""
Learn Price Elasticity Parameters from Dunnhumby Data (Sprint 2)

This script learns all three elasticity models from Dunnhumby Complete Journey:
1. HMM Price States (transition matrices, emission distributions)
2. Cross-Price Elasticity (substitution/complementarity matrix)
3. Arc Elasticity (intertemporal effects via HMM)

Usage:
    python scripts/learn_price_elasticity.py \
        --products data/processed/product_catalog/representative_catalog.csv \
        --transactions data/raw/dunnhumby/transaction_data.csv \
        --causal data/raw/dunnhumby/causal_data.csv \
        --output outputs/models/elasticity/

Output:
    - hmm_parameters.pkl: HMM transition matrices and emission distributions
    - cross_elasticity/: Cross-price elasticity matrix and relationships
    - Summary statistics and validation metrics
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.engines.price_hmm import PriceStateHMM
from retailsynth.engines.cross_price_elasticity import CrossPriceElasticityEngine
from retailsynth.engines.arc_elasticity import ArcPriceElasticityEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(products_path: str, 
              transactions_path: str, 
              causal_path: str = None) -> tuple:
    """
    Load Dunnhumby data files
    
    Args:
        products_path: Path to product catalog CSV
        transactions_path: Path to transaction data CSV
        causal_path: Optional path to causal/promotional data CSV
    
    Returns:
        Tuple of (products_df, transactions_df, causal_df)
    """
    logger.info("Loading data files...")
    
    # Load products
    products_df = pd.read_csv(products_path)
    logger.info(f"Loaded {len(products_df):,} products")
    
    # Load transactions (with dtype optimization)
    logger.info("Loading transaction data (this may take a while)...")
    transactions_df = pd.read_csv(
        transactions_path,
        dtype={
            'household_key': 'int32',
            'BASKET_ID': 'int32',
            'PRODUCT_ID': 'int32',
            'QUANTITY': 'int16',
            'SALES_VALUE': 'float32',
            'STORE_ID': 'int16',
            'RETAIL_DISC': 'float32',
            'TRANS_TIME': 'int32',
            'WEEK_NO': 'int16',
            'COUPON_DISC': 'float32',
            'COUPON_MATCH_DISC': 'float32'
        }
    )
    logger.info(f"Loaded {len(transactions_df):,} transactions")
    
    # Load causal data if provided
    causal_df = None
    if causal_path and Path(causal_path).exists():
        logger.info("Loading promotional data...")
        causal_df = pd.read_csv(
            causal_path,
            dtype={
                'PRODUCT_ID': 'int32',
                'STORE_ID': 'int16',
                'WEEK_NO': 'int16'
            }
        )
        logger.info(f"Loaded {len(causal_df):,} promotional records")
    else:
        logger.warning("No causal data provided - HMM will use discounts only")
    
    return products_df, transactions_df, causal_df

def learn_hmm_parameters(products_df: pd.DataFrame,
                        transactions_df: pd.DataFrame,
                        causal_df: pd.DataFrame = None,
                        output_dir: Path = None) -> PriceStateHMM:
    """
    Learn HMM price state parameters
    
    Args:
        products_df: Product catalog
        transactions_df: Transaction data
        causal_df: Optional promotional data
        output_dir: Directory to save parameters
    
    Returns:
        Trained PriceStateHMM instance
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Learning HMM Price State Parameters")
    logger.info("="*70)
    
    # Initialize HMM
    hmm = PriceStateHMM(products_df, n_states=4)
    
    # Learn from data
    hmm.learn_from_data(transactions_df, causal_df)
    
    # Save parameters
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        hmm.save_parameters(output_dir / 'hmm_parameters.pkl')
    
    return hmm

def learn_cross_price_elasticity(products_df: pd.DataFrame,
                                 transactions_df: pd.DataFrame,
                                 output_dir: Path = None) -> CrossPriceElasticityEngine:
    """
    Learn cross-price elasticity matrix
    
    Args:
        products_df: Product catalog
        transactions_df: Transaction data
        output_dir: Directory to save parameters
    
    Returns:
        Trained CrossPriceElasticityEngine instance
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Learning Cross-Price Elasticity Matrix")
    logger.info("="*70)
    
    # Initialize engine
    cross_price = CrossPriceElasticityEngine(products_df)
    
    # Estimate from data
    cross_price.estimate_from_data(
        transactions_df,
        min_observations=10,
        top_competitors=5,
        elasticity_threshold=0.1
    )
    
    # Save parameters
    if output_dir:
        cross_dir = output_dir / 'cross_elasticity'
        cross_price.save_parameters(cross_dir)
    
    return cross_price

def initialize_arc_elasticity(hmm: PriceStateHMM) -> ArcPriceElasticityEngine:
    """
    Initialize arc elasticity engine with learned HMM
    
    Args:
        hmm: Trained PriceStateHMM instance
    
    Returns:
        ArcPriceElasticityEngine instance
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Initializing Arc Elasticity Engine")
    logger.info("="*70)
    
    # Initialize with HMM
    arc_elasticity = ArcPriceElasticityEngine(
        price_hmm=hmm,
        inventory_decay_rate=0.25,  # 25% per week
        future_discount_factor=0.95  # Weekly discount
    )
    
    logger.info("✅ Arc elasticity engine initialized with HMM parameters")
    
    return arc_elasticity

def generate_validation_report(hmm: PriceStateHMM,
                               cross_price: CrossPriceElasticityEngine,
                               products_df: pd.DataFrame,
                               output_dir: Path):
    """
    Generate validation report with summary statistics
    
    Args:
        hmm: Trained HMM
        cross_price: Trained cross-price engine
        products_df: Product catalog
        output_dir: Directory to save report
    """
    logger.info("\n" + "="*70)
    logger.info("GENERATING VALIDATION REPORT")
    logger.info("="*70)
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("PRICE ELASTICITY LEARNING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*70)
    
    # HMM Statistics
    report_lines.append("\n1. HMM PRICE STATE MODEL")
    report_lines.append("-" * 70)
    report_lines.append(f"Products with learned parameters: {len(hmm.transition_matrices):,}")
    report_lines.append(f"Coverage: {len(hmm.transition_matrices)/len(products_df):.1%}")
    
    if len(hmm.transition_matrices) > 0:
        # Average transition matrix
        avg_transition = np.mean(
            [tm for tm in hmm.transition_matrices.values()],
            axis=0
        )
        
        report_lines.append("\nAverage Transition Matrix:")
        report_lines.append(f"  Regular -> Regular:       {avg_transition[0,0]:.3f}")
        report_lines.append(f"  Regular -> Feature:       {avg_transition[0,1]:.3f}")
        report_lines.append(f"  Regular -> Deep:          {avg_transition[0,2]:.3f}")
        report_lines.append(f"  Regular -> Clearance:     {avg_transition[0,3]:.3f}")
        report_lines.append(f"  Feature -> Regular:       {avg_transition[1,0]:.3f}")
        report_lines.append(f"  Feature -> Feature:       {avg_transition[1,1]:.3f}")
        report_lines.append(f"  Feature -> Deep:          {avg_transition[1,2]:.3f}")
        report_lines.append(f"  Feature -> Clearance:     {avg_transition[1,3]:.3f}")
        report_lines.append(f"  Deep -> Regular:          {avg_transition[2,0]:.3f}")
        report_lines.append(f"  Deep -> Feature:          {avg_transition[2,1]:.3f}")
        report_lines.append(f"  Deep -> Deep:             {avg_transition[2,2]:.3f}")
        report_lines.append(f"  Deep -> Clearance:        {avg_transition[2,3]:.3f}")
        report_lines.append(f"  Clearance -> Regular:     {avg_transition[3,0]:.3f}")
        report_lines.append(f"  Clearance -> Feature:     {avg_transition[3,1]:.3f}")
        report_lines.append(f"  Clearance -> Deep:        {avg_transition[3,2]:.3f}")
        report_lines.append(f"  Clearance -> Clearance:   {avg_transition[3,3]:.3f}")
        
        # State prevalence
        state_prevalence = {0: 0, 1: 0, 2: 0, 3: 0}
        for probs in hmm.initial_state_probs.values():
            for state, prob in probs.items():
                state_prevalence[state] += prob
        
        total = sum(state_prevalence.values())
        report_lines.append("\nAverage State Prevalence:")
        report_lines.append(f"  Regular (State 0):        {state_prevalence[0]/total:.1%}")
        report_lines.append(f"  Feature (State 1):        {state_prevalence[1]/total:.1%}")
        report_lines.append(f"  Deep Discount (State 2):  {state_prevalence[2]/total:.1%}")
        report_lines.append(f"  Clearance (State 3):      {state_prevalence[3]/total:.1%}")
    
    # Cross-Price Elasticity Statistics
    report_lines.append("\n2. CROSS-PRICE ELASTICITY MATRIX")
    report_lines.append("-" * 70)
    
    if cross_price.cross_elasticity_matrix is not None:
        n_products = len(products_df)
        nnz = cross_price.cross_elasticity_matrix.nnz
        
        report_lines.append(f"Matrix size: {n_products:,} x {n_products:,}")
        report_lines.append(f"Non-zero elasticities: {nnz:,}")
        report_lines.append(f"Sparsity: {1 - nnz / (n_products**2):.4f}")
        report_lines.append(f"Density: {nnz / (n_products**2):.4f}")
        
        report_lines.append("\nProduct Relationships:")
        report_lines.append(f"  Substitute pairs: {len(cross_price.substitute_groups):,}")
        report_lines.append(f"  Complement pairs: {len(cross_price.complement_pairs):,}")
        
        # Example substitutes
        if len(cross_price.substitute_groups) > 0:
            report_lines.append("\nTop 5 Substitute Pairs (by elasticity):")
            top_subs = cross_price.substitute_groups.nlargest(5, 'elasticity')
            for _, row in top_subs.iterrows():
                report_lines.append(
                    f"  Product {row['product_i']} ↔ {row['product_j']}: "
                    f"ε = {row['elasticity']:.3f}"
                )
        
        # Example complements
        if len(cross_price.complement_pairs) > 0:
            report_lines.append("\nTop 5 Complement Pairs (by |elasticity|):")
            top_comps = cross_price.complement_pairs.nsmallest(5, 'elasticity')
            for _, row in top_comps.iterrows():
                report_lines.append(
                    f"  Product {row['product_i']} ↔ {row['product_j']}: "
                    f"ε = {row['elasticity']:.3f}"
                )
    
    # Arc Elasticity
    report_lines.append("\n3. ARC ELASTICITY (INTERTEMPORAL)")
    report_lines.append("-" * 70)
    report_lines.append("✅ Arc elasticity engine initialized with HMM parameters")
    report_lines.append("   - Inventory decay rate: 25%/week")
    report_lines.append("   - Future discount factor: 0.95")
    report_lines.append("   - Uses HMM for price expectations")
    
    # Validation Metrics
    report_lines.append("\n4. VALIDATION METRICS")
    report_lines.append("-" * 70)
    
    hmm_coverage = len(hmm.transition_matrices) / len(products_df)
    cross_coverage = cross_price.cross_elasticity_matrix.nnz / (len(products_df) ** 2)
    
    report_lines.append(f"HMM Parameter Coverage: {hmm_coverage:.1%}")
    report_lines.append(f"Cross-Price Density: {cross_coverage:.4f}")
    
    if hmm_coverage > 0.80:
        report_lines.append("✅ HMM coverage exceeds 80% threshold")
    else:
        report_lines.append("⚠️  HMM coverage below 80% - may need more transaction data")
    
    if len(cross_price.substitute_groups) > 100:
        report_lines.append("✅ Sufficient substitute relationships identified")
    else:
        report_lines.append("⚠️  Few substitute relationships - check category diversity")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("LEARNING COMPLETE")
    report_lines.append("="*70)
    
    # Print report
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # Save report
    report_path = output_dir / 'learning_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"\n✅ Report saved to {report_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Learn price elasticity parameters from Dunnhumby data")
    parser.add_argument('--products', type=str, default='data/raw/dunnhumby/product.csv', help='Path to product catalog CSV')
    parser.add_argument('--transactions', type=str, default='data/raw/dunnhumby/transaction_data.csv', help='Path to transaction data CSV')
    parser.add_argument('--causal', type=str, default='data/raw/dunnhumby/causal.csv', help='Path to causal/promotional data CSV (optional)')
    parser.add_argument('--output', type=str, default='data/processed/elasticity', help='Output directory for learned parameters')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("PRICE ELASTICITY LEARNING PIPELINE (SPRINT 2)")
    logger.info("="*70)
    
    # Load data
    products_df, transactions_df, causal_df = load_data(args.products, args.transactions, args.causal)
    
    # Step 1: Learn HMM parameters
    hmm = learn_hmm_parameters(products_df, transactions_df, causal_df, output_dir)
    
    # Step 2: Learn cross-price elasticity
    cross_price = learn_cross_price_elasticity(products_df, transactions_df, output_dir)
    
    # Step 3: Initialize arc elasticity
    arc_elasticity = initialize_arc_elasticity(hmm)
    
    # Generate validation report
    generate_validation_report(hmm, cross_price, products_df, output_dir)
    
    logger.info("\n" + "="*70)
    logger.info("✅ ALL ELASTICITY MODELS LEARNED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nFiles created:")
    logger.info(f"  - hmm_parameters.pkl")
    logger.info(f"  - cross_elasticity/cross_elasticity_matrix.npz")
    logger.info(f"  - cross_elasticity/substitute_groups.csv")
    logger.info(f"  - cross_elasticity/complement_pairs.csv")
    logger.info(f"  - learning_report.txt")
    logger.info("\nNext steps:")
    logger.info("  1. Review learning_report.txt for validation metrics")
    logger.info("  2. Integrate elasticity engines into transaction generator")
    logger.info("  3. Run validation tests (Sprint 2 validation)")


if __name__ == '__main__':
    main()