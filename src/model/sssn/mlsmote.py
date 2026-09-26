import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def get_minority_instances(X: npt.NDArray[Any], y: npt.NDArray[Any], minor_class_indices: list[int]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Filter X and y for instances belonging to any of the specified minority classes."""
    mask = np.any(y[:, minor_class_indices] == 1, axis=1)
    return X[mask], y[mask]

def apply_mlsmote(
    X: npt.NDArray[Any], 
    y: npt.NDArray[Any], 
    target_count: int = 1000, 
    k_neighbors: int = 5, 
    random_state: int = 42
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Apply Multi-Label Synthetic Minority Over-sampling Technique (MLSMOTE).
    
    Identifies minority instances based on class frequencies, finds K-nearest neighbors
    in the feature space, and generates synthetic samples using linear interpolation for
    features and logical OR (union) for the multi-label targets.
    
    Args:
        X: Feature matrix (e.g., 512D spatial embeddings).
        y: Multi-label binary targets.
        target_count: Number of synthetic instances to generate.
        k_neighbors: Number of nearest neighbors to use.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple of (X_augmented, y_augmented) containing both original and synthetic data.
    """
    rng = np.random.RandomState(random_state)
    
    # Identify class frequencies
    class_counts = np.sum(y, axis=0)
    median_count = np.median(class_counts)
    
    # Minority classes: those with less than median frequency (or highly skewed)
    # We define minority classes as those with less than median_count
    minor_class_indices = np.where(class_counts < median_count)[0].tolist()
    
    if not minor_class_indices:
        logger.info("No minority classes found for MLSMOTE. Returning original data.")
        return X, y
        
    logger.info("Minority classes identified: %s", minor_class_indices)
    
    X_min, y_min = get_minority_instances(X, y, minor_class_indices)
    
    if len(X_min) < k_neighbors + 1:
        logger.warning("Not enough minority instances for %d neighbors. Using %d.", k_neighbors, max(1, len(X_min) - 1))
        k_neighbors = max(1, len(X_min) - 1)
        
    if k_neighbors < 1:
        logger.warning("Too few instances to perform MLSMOTE. Returning original data.")
        return X, y
        
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(X_min)
    
    synthetic_X = []
    synthetic_y = []
    
    # Generate synthetic samples
    for _ in range(target_count):
        # Pick a random minority instance
        idx = rng.randint(0, len(X_min))
        instance_X = X_min[idx]
        instance_y = y_min[idx]
        
        # Find neighbors (first one is the instance itself)
        neighbors = nn.kneighbors([instance_X], return_distance=False)[0][1:]
        
        # Pick a random neighbor
        neighbor_idx = rng.choice(neighbors)
        neighbor_X = X_min[neighbor_idx]
        neighbor_y = y_min[neighbor_idx]
        
        # Interpolate features
        ratio = rng.random()
        new_X = instance_X + ratio * (neighbor_X - instance_X)
        
        # Logical OR for labels
        new_y = np.logical_or(instance_y, neighbor_y).astype(int)
        
        synthetic_X.append(new_X)
        synthetic_y.append(new_y)
        
    logger.info("Generated %d synthetic instances via MLSMOTE.", target_count)
    
    X_augmented = np.vstack([X, np.array(synthetic_X)])
    y_augmented = np.vstack([y, np.array(synthetic_y)])
    
    return X_augmented, y_augmented
