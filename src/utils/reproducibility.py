"""Ensure reproducibility across runs"""

import random
import numpy as np
import tensorflow as tf
import os


def set_random_seeds(seed=42):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Configure TensorFlow for deterministic operations
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    print(f"✓ Random seeds set to {seed}")
    print(f"✓ Deterministic operations enabled")
