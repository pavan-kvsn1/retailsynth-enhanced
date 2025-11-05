"""
Cross-Price Elasticity Engine (Sprint 2.2)

This module estimates and applies cross-price elasticity effects between products.
Cross-price elasticity measures substitution/complementarity:

    ε_ij = (∂Q_i / ∂P_j) * (P_j / Q_i)

Where:
- ε_ij > 0: Substitutes (Coke ↔ Pepsi)
- ε_ij < 0: Complements (Chips ↔ Dip)
- ε_ij ≈ 0: Independent

Uses log-log regression on Dunnhumby data:
    log(Q_i,t) = α_i + β_i * log(P_i,t) + Σ_j γ_ij * log(P_j,t) + ε_t

Where γ_ij is the cross-price elasticity between products i and j.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from sklearn.linear_model import Ridge
import pickle
import logging

logger = logging.getLogger(__name__)


class CrossPriceElasticityEngine:
    """
    Estimate and apply cross-price elasticity effects
    
    Learns sparse elasticity matrix from transaction data, identifying
    substitute and complement relationships between products.
    
    Attributes:
        products (pd.DataFrame): Product catalog
        cross_elasticity_matrix (csr_matrix): Sparse elasticity matrix
        substitute_groups (pd.DataFrame): Identified substitute pairs
        complement_pairs (pd.DataFrame): Identified complement pairs
    """
    
    def __init__(self, products_df: pd.DataFrame):
        """
        Initialize cross-price elasticity engine
        
        Args:
            products_df: Product catalog with PRODUCT_ID, COMMODITY_DESC
        """
        self.products = products_df.reset_index(drop=True)
        self.cross_elasticity_matrix = None
        self.substitute_groups = None
        self.complement_pairs = None
        self.product_id_to_idx = {
            pid: idx for idx, pid in enumerate(self.products['PRODUCT_ID'])
        }
        
        logger.info(f"Initialized CrossPriceElasticityEngine with {len(products_df)} products")
    
    def estimate_from_data(self, 
                          transactions_df: pd.DataFrame,
                          min_observations: int = 10,
                          top_competitors: int = 5,
                          elasticity_threshold: float = 0.1):
        """
        Estimate cross-price elasticity from Dunnhumby transactions
        
        Args:
            transactions_df: Transaction data with PRODUCT_ID, WEEK_NO, QUANTITY, SALES_VALUE
            min_observations: Minimum observations required for estimation
            top_competitors: Number of top competitors to consider per product
            elasticity_threshold: Minimum absolute elasticity to store
        
        Process:
            1. Aggregate to product-week level
            2. For each product, identify potential substitutes/complements
            3. Run log-log regression to estimate elasticities
            4. Build sparse elasticity matrix
        """
        logger.info("Estimating cross-price elasticity matrix...")
        
        # Step 1: Product-week aggregation
        logger.info("Aggregating to product-week level...")
        product_week = transactions_df.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({'QUANTITY': 'sum',
                                                                               'SALES_VALUE': 'sum',
                                                                               'household_key': 'nunique'}).reset_index()
        
        product_week['price'] = product_week['SALES_VALUE'] / product_week['QUANTITY']
        product_week['quantity_per_customer'] = product_week['QUANTITY'] / product_week['household_key']
        
        # Step 2: Initialize sparse matrix
        n_products = len(self.products)
        elasticity_matrix = lil_matrix((n_products, n_products))
        
        # Step 3: Estimate elasticities by commodity category
        logger.info("Estimating elasticities by category...")
        commodities = self.products['COMMODITY_DESC'].unique()
        
        # Iterate over each commodity
        for commodity in tqdm(commodities, desc="Processing categories"):
            # Get products in this commodity
            commodity_products = self.products[self.products['COMMODITY_DESC'] == commodity]['PRODUCT_ID'].values

            # Skip if there are less than 2 products in the commodity
            if len(commodity_products) < 2:
                continue

            # Get data for these products
            commodity_data = product_week[product_week['PRODUCT_ID'].isin(commodity_products)]

            # Skip if there is no data for this commodity
            if len(commodity_data) == 0:
                continue

            # Pivot to wide format (products as columns)
            try:
                quantity_wide = commodity_data.pivot(index='WEEK_NO', columns='PRODUCT_ID', values='quantity_per_customer').fillna(0)
                price_wide = commodity_data.pivot(index='WEEK_NO', columns='PRODUCT_ID', values='price').fillna(method='ffill').fillna(method='bfill')
            except Exception as e:
                logger.warning(f"Skipping commodity {commodity}: {e}")
                continue
            
            # Log transformation
            log_quantity = np.log(quantity_wide + 1)
            log_price = np.log(price_wide + 0.01)
            
            # Estimate elasticities for each product in category
            for focal_product in commodity_products:
                if focal_product not in log_quantity.columns:
                    continue

                # Estimate elasticities for this product
                self._estimate_product_elasticities(
                    focal_product=focal_product,                #<-- Focal product
                    log_quantity=log_quantity,                    #<-- Log-transformed quantity
                    log_price=log_price,                          #<-- Log-transformed price
                    commodity_products=commodity_products,        #<-- Products in this commodity
                    elasticity_matrix=elasticity_matrix,          #<-- Elasticity matrix
                    min_observations=min_observations,            #<-- Minimum observations
                    top_competitors=top_competitors,              #<-- Number of top competitors
                    elasticity_threshold=elasticity_threshold     #<-- Elasticity threshold
                )
        
        # Convert to CSR for efficient computation
        self.cross_elasticity_matrix = elasticity_matrix.tocsr()
        
        logger.info(f"✅ Estimated cross-price elasticity matrix")
        logger.info(f"   Products: {n_products:,}")
        logger.info(f"   Non-zero elasticities: {self.cross_elasticity_matrix.nnz:,}")
        logger.info(f"   Sparsity: {1 - self.cross_elasticity_matrix.nnz / (n_products**2):.4f}")
        
        # Identify substitute/complement groups
        self._identify_product_relationships()
    
    def _estimate_product_elasticities(self,
                                      focal_product: int,
                                      log_quantity: pd.DataFrame,
                                      log_price: pd.DataFrame,
                                      commodity_products: np.ndarray,
                                      elasticity_matrix: lil_matrix,
                                      min_observations: int,
                                      top_competitors: int,
                                      elasticity_threshold: float):
        """
        Estimate cross-price elasticities for a single focal product
        
        Args:
            focal_product: Product ID to estimate elasticities for
            log_quantity: Log-transformed quantity data (wide format)
            log_price: Log-transformed price data (wide format)
            commodity_products: Array of product IDs in same commodity
            elasticity_matrix: Sparse matrix to populate
            min_observations: Minimum observations required
            top_competitors: Number of competitors to include
            elasticity_threshold: Minimum elasticity to store
        """
        y = log_quantity[focal_product].values
        
        # Build feature matrix (own price + competitor prices)
        X_cols = []
        feature_products = []
        
        # Own price
        if focal_product in log_price.columns:
            X_cols.append(log_price[focal_product].values)
            feature_products.append(focal_product)
        
        # Competitor prices (top N by volume)
        competitors = [p for p in commodity_products if p != focal_product]
        
        if len(competitors) > 0:
            # Get competitor volumes
            competitor_volumes = log_quantity[[c for c in competitors if c in log_quantity.columns]].sum().sort_values(ascending=False)
            #Get the Top N competitors by volume
            top_comp = competitor_volumes.head(top_competitors).index
            
            #Add the top competitor products price to the feature matrix
            for comp_product in top_comp:
                if comp_product in log_price.columns:
                    X_cols.append(log_price[comp_product].values)
                    feature_products.append(comp_product)
        
        if len(X_cols) < 2:  # Need at least own price + 1 competitor
            return
        
        #Stack the feature matrix
        X = np.column_stack(X_cols)
        
        # Handle missing values
        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid_mask.sum() < min_observations:
            return
        
        # Regression with regularization
        try:
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(X[valid_mask], y[valid_mask])
            
            # Extract elasticities
            focal_idx = self.product_id_to_idx[focal_product]
            
            for i, feature_product in enumerate(feature_products):
                elasticity = model.coef_[i]
                feature_idx = self.product_id_to_idx[feature_product]
                
                # Only store if significant
                if abs(elasticity) > elasticity_threshold:
                    elasticity_matrix[focal_idx, feature_idx] = elasticity
        
        except Exception as e:
            logger.debug(f"Regression failed for product {focal_product}: {e}")
    
    def _identify_product_relationships(self,
                                       substitute_threshold: float = 0.2,
                                       complement_threshold: float = -0.2):
        """
        Classify product pairs as substitutes or complements
        
        Args:
            substitute_threshold: Minimum elasticity for substitutes
            complement_threshold: Maximum elasticity for complements
        """
        from scipy.sparse import find
        
        # Extract non-zero elasticities
        rows, cols, elasticities = find(self.cross_elasticity_matrix)
        
        # Classify
        substitute_pairs = []
        complement_pairs = []
        
        for i, j, elasticity in zip(rows, cols, elasticities):
            if i == j:  # Skip own-price
                continue
            
            if elasticity > substitute_threshold:  # Substitutes
                substitute_pairs.append((
                    self.products.iloc[i]['PRODUCT_ID'],
                    self.products.iloc[j]['PRODUCT_ID'],
                    elasticity
                ))
            elif elasticity < complement_threshold:  # Complements
                complement_pairs.append((
                    self.products.iloc[i]['PRODUCT_ID'],
                    self.products.iloc[j]['PRODUCT_ID'],
                    elasticity
                ))
        
        self.substitute_groups = pd.DataFrame(
            substitute_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        ) if substitute_pairs else pd.DataFrame(columns=['product_i', 'product_j', 'elasticity'])
        
        self.complement_pairs = pd.DataFrame(
            complement_pairs,
            columns=['product_i', 'product_j', 'elasticity']
        ) if complement_pairs else pd.DataFrame(columns=['product_i', 'product_j', 'elasticity'])
        
        logger.info(f"\nProduct relationships identified:")
        logger.info(f"  Substitute pairs: {len(self.substitute_groups):,}")
        logger.info(f"  Complement pairs: {len(self.complement_pairs):,}")
    
    def apply_cross_price_effects(self,
                                  focal_product_id: int,
                                  base_utility: float,
                                  current_prices: Dict[int, float],
                                  reference_prices: Dict[int, float]) -> float:
        """
        Adjust utility based on cross-price effects
        
        Args:
            focal_product_id: Product being evaluated
            base_utility: Base utility before cross-price adjustment
            current_prices: Current prices for all products
            reference_prices: Reference prices (e.g., average price)
        
        Returns:
            Adjusted utility incorporating cross-price effects
        """
        if self.cross_elasticity_matrix is None:
            logger.warning("Cross-price elasticity matrix not estimated yet")
            return base_utility
        
        if focal_product_id not in self.product_id_to_idx:
            return base_utility
        
        focal_idx = self.product_id_to_idx[focal_product_id]
        
        # Get cross-price elasticities for this product
        elasticities = self.cross_elasticity_matrix[focal_idx].toarray().flatten()
        
        # Calculate price change effects
        cross_price_adjustment = 0.0
        
        for j, elasticity in enumerate(elasticities):
            if abs(elasticity) < 0.01:  # Skip near-zero
                continue
            
            other_product_id = self.products.iloc[j]['PRODUCT_ID']
            
            if other_product_id in current_prices and other_product_id in reference_prices:
                # Price change percentage
                price_change_pct = (
                    current_prices[other_product_id] - reference_prices[other_product_id]
                ) / (reference_prices[other_product_id] + 0.01)  # Avoid division by zero
                
                # Utility adjustment = elasticity * price_change
                # Positive elasticity (substitute): competitor price ↑ → focal utility ↑
                # Negative elasticity (complement): complement price ↑ → focal utility ↓
                cross_price_adjustment += elasticity * price_change_pct
        
        # Apply adjustment
        adjusted_utility = base_utility + cross_price_adjustment
        
        return adjusted_utility
    
    def get_substitutes(self, product_id: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Get top substitute products for a given product
        
        Args:
            product_id: Product ID
            top_n: Number of top substitutes to return
        
        Returns:
            List of (substitute_product_id, elasticity) tuples
        """
        if product_id not in self.product_id_to_idx:
            return []
        
        focal_idx = self.product_id_to_idx[product_id]
        elasticities = self.cross_elasticity_matrix[focal_idx].toarray().flatten()
        
        # Get positive elasticities (substitutes)
        substitutes = []
        for j, elasticity in enumerate(elasticities):
            if j != focal_idx and elasticity > 0.2:
                other_product_id = self.products.iloc[j]['PRODUCT_ID']
                substitutes.append((other_product_id, elasticity))
        
        # Sort by elasticity (descending)
        substitutes.sort(key=lambda x: x[1], reverse=True)
        
        return substitutes[:top_n]
    
    def get_complements(self, product_id: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Get top complement products for a given product
        
        Args:
            product_id: Product ID
            top_n: Number of top complements to return
        
        Returns:
            List of (complement_product_id, elasticity) tuples
        """
        if product_id not in self.product_id_to_idx:
            return []
        
        focal_idx = self.product_id_to_idx[product_id]
        elasticities = self.cross_elasticity_matrix[focal_idx].toarray().flatten()
        
        # Get negative elasticities (complements)
        complements = []
        for j, elasticity in enumerate(elasticities):
            if j != focal_idx and elasticity < -0.2:
                other_product_id = self.products.iloc[j]['PRODUCT_ID']
                complements.append((other_product_id, abs(elasticity)))
        
        # Sort by absolute elasticity (descending)
        complements.sort(key=lambda x: x[1], reverse=True)
        
        return complements[:top_n]
    
    def save_parameters(self, output_dir: str):
        """
        Save learned elasticity parameters
        
        Args:
            output_dir: Directory to save parameters
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sparse matrix
        if self.cross_elasticity_matrix is not None:
            save_npz(
                output_dir / 'cross_elasticity_matrix.npz',
                self.cross_elasticity_matrix
            )
        
        # Save product relationships
        if self.substitute_groups is not None:
            self.substitute_groups.to_csv(
                output_dir / 'substitute_groups.csv',
                index=False
            )
        
        if self.complement_pairs is not None:
            self.complement_pairs.to_csv(
                output_dir / 'complement_pairs.csv',
                index=False
            )
        
        # Save metadata
        metadata = {
            'product_id_to_idx': self.product_id_to_idx,
            'n_products': len(self.products)
        }
        
        with open(output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ Saved cross-price elasticity parameters to {output_dir}")
    
    def load_parameters(self, input_dir: str):
        """
        Load learned elasticity parameters
        
        Args:
            input_dir: Directory containing saved parameters
        """
        input_dir = Path(input_dir)
        
        # Load sparse matrix
        matrix_path = input_dir / 'cross_elasticity_matrix.npz'
        if matrix_path.exists():
            self.cross_elasticity_matrix = load_npz(matrix_path)
        
        # Load product relationships
        subs_path = input_dir / 'substitute_groups.csv'
        if subs_path.exists():
            self.substitute_groups = pd.read_csv(subs_path)
        
        comps_path = input_dir / 'complement_pairs.csv'
        if comps_path.exists():
            self.complement_pairs = pd.read_csv(comps_path)
        
        # Load metadata
        with open(input_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.product_id_to_idx = metadata['product_id_to_idx']
        
        logger.info(f"✅ Loaded cross-price elasticity parameters from {input_dir}")
        if self.cross_elasticity_matrix is not None:
            logger.info(f"   Non-zero elasticities: {self.cross_elasticity_matrix.nnz:,}")