"""
Base feature loader abstract class.

Defines the interface for feature extractors that process recording IDs and return features
while maintaining consistent dimensionality.
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFeatureLoader(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for feature extractors.

    Defines the interface for extractors that process recording IDs and return features
    while maintaining consistent dimensionality.
    """

    @abstractmethod
    def fit(self, X, y=None) -> "BaseFeatureLoader":
        """Load and prepare feature data."""
        pass

    @abstractmethod
    def transform(self, recording_ids_list: list[list[int]]) -> np.ndarray:
        """
        Extract features maintaining input dimensionality.

        Args:
            recording_ids_list: Shape (n_samples, n_recordings_per_sample)

        Returns:
            np.ndarray: Shape (n_samples, n_recordings_per_sample, n_features_per_recording)
        """
        pass

    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Return the number of features per recording this extractor produces."""
        pass
