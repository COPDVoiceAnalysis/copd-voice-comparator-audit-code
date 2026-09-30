"""
PCA preprocessing component for parselmouth and wav2vec2 features.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class W2VParselPCA(BaseEstimator, TransformerMixin):
    """
    PCA transformer for parsel and wav2vec2 features.
    """

    def __init__(
        self,
        wav2vec_n_components: int | None = None,
        parsel_n_components: int | None = None,
        random_state: int | None = None,
        parsel_feature_count: int | None = None,
        demographic_feature_count: int | None = None,
        regularization: float = 1e-8,
    ):
        """
        Initialize PCA transformer with pydantic configuration or individual parameters.

        Args:
            wav2vec_n_components: Number of PCA components for Wav2Vec features
            parsel_n_components: Number of PCA components for Parselmouth features
            random_state: Random state for reproducibility
            parsel_feature_count: Number of parselmouth features (for splitting concatenated features)
            demographic_feature_count: Number of demographic features (for splitting concatenated features)
            regularization: Regularization parameter for numerical stability
        """
        self.wav2vec_n_components = wav2vec_n_components
        self.parsel_n_components = parsel_n_components
        self.random_state = random_state or 42
        self.parsel_feature_count = parsel_feature_count
        self.demographic_feature_count = demographic_feature_count or 0
        self.regularization = regularization

        # PCA components - initialized in fit()
        self.wav2vec_pca_ = None
        self.parsel_pca_ = None

        logger.debug(f"W2VParselPCA initialized with config: {self}")

    def _compute_split_boundaries(self, n_total_features: int):
        """Compute feature split boundaries dynamically.

        When upstream steps (e.g. VarianceThreshold) remove columns, the total
        feature count shrinks but parsel_feature_count / demographic_feature_count
        stay fixed, which would mis-align the split.  We therefore cap
        parsel_feature_count to what is actually available and derive the
        wav2vec segment as the remainder.
        """
        parsel_end = min(
            self.parsel_feature_count if self.parsel_feature_count else 0,
            n_total_features - self.demographic_feature_count,
        )
        # Ensure parsel_end is non-negative
        parsel_end = max(parsel_end, 0)
        wav2vec_end = n_total_features - self.demographic_feature_count
        # Ensure wav2vec_end >= parsel_end
        wav2vec_end = max(wav2vec_end, parsel_end)
        return parsel_end, wav2vec_end

    def fit(self, X, y=None):
        """
        Fit PCA transformers on the feature data.

        Args:
            X: Concatenated features array from FeaturePreloader [parselmouth | wav2vec | demographic]
            y: Target data (not used)

        Returns:
            self
        """
        # Split concatenated features into parsel, wav2vec, and demographic
        parsel_end, wav2vec_end = self._compute_split_boundaries(X.shape[1])
        # Store actual boundaries after potential upstream column removal
        self.actual_parsel_end_ = parsel_end
        self.actual_wav2vec_end_ = wav2vec_end

        if parsel_end > 0:
            X_parsel = X[:, :parsel_end]
        else:
            X_parsel = np.empty((X.shape[0], 0))

        if wav2vec_end > parsel_end:
            X_wav2vec = X[:, parsel_end:wav2vec_end]
        else:
            X_wav2vec = np.empty((X.shape[0], 0))

        if self.demographic_feature_count > 0:
            X_demographic = X[:, wav2vec_end:]
        else:
            X_demographic = np.empty((X.shape[0], 0))

        logger.debug(f"W2VParselPCA fit() - X_parsel shape: {X_parsel.shape}")
        logger.debug(f"W2VParselPCA fit() - X_wav2vec shape: {X_wav2vec.shape}")
        logger.debug(f"W2VParselPCA fit() - X_demographic shape: {X_demographic.shape}")
        logger.debug(
            f"W2VParselPCA fit() - wav2vec_n_components: {self.wav2vec_n_components}"
        )
        logger.debug(
            f"W2VParselPCA fit() - parsel_n_components: {self.parsel_n_components}"
        )

        # Fit Wav2Vec2 PCA if data is available and components are specified
        if (
            X_wav2vec.size > 0
            and X_wav2vec.shape[1] > 0
            and self.wav2vec_n_components is not None
            and self.wav2vec_n_components > 0
        ):
            # Check if PCA is feasible
            max_possible_components = min(X_wav2vec.shape[0], X_wav2vec.shape[1]) - 1

            if (
                self.wav2vec_n_components >= max_possible_components
                or max_possible_components <= 0
            ):
                logger.debug(
                    f"Wav2Vec2 PCA skipped: requested {self.wav2vec_n_components} components, "
                    f"but only {X_wav2vec.shape[1]} features and {X_wav2vec.shape[0]} samples available. "
                    f"Using raw features instead."
                )
                self.wav2vec_pca_ = None
            else:
                logger.debug("Fitting Wav2Vec2 PCA...")

                # Preprocess for numerical stability
                # X_wav2vec_stable = self._preprocess_for_numerical_stability(X_wav2vec)

                # Update n_components if features were removed
                actual_n_components = min(self.wav2vec_n_components, X_wav2vec.shape[1])

                self.wav2vec_pca_ = PCA(
                    n_components=actual_n_components,
                    random_state=self.random_state,
                    svd_solver="full",
                    whiten=False,  # TODO what does this do?
                )

                # Compute and store noise scale from training data
                self.wav2vec_noise_scale_ = self._compute_noise_scale(X_wav2vec)
                X_wav2vec_reg = self._add_regularization_noise(X_wav2vec, self.wav2vec_noise_scale_)

                # Suppress numerical warnings during PCA fitting
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*encountered in matmul.*",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*divide by zero.*", category=RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*overflow.*", category=RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*invalid value.*", category=RuntimeWarning
                    )
                    self.wav2vec_pca_.fit(X_wav2vec_reg)
        else:
            logger.debug("Skipping Wav2Vec2 PCA (empty data or disabled)")
            self.wav2vec_pca_ = None

        # Fit Parselmouth PCA if data is available and components are specified
        if (
            X_parsel.size > 0
            and X_parsel.shape[1] > 0
            and self.parsel_n_components is not None
            and self.parsel_n_components > 0
        ):
            # Check if PCA is feasible
            max_possible_components = min(X_parsel.shape[0], X_parsel.shape[1]) - 1

            if (
                self.parsel_n_components >= max_possible_components
                or max_possible_components <= 0
            ):
                logger.debug(
                    f"Parselmouth PCA skipped: requested {self.parsel_n_components} components, "
                    f"but only {X_parsel.shape[1]} features and {X_parsel.shape[0]} samples available. "
                    f"Using raw features instead."
                )
                self.parsel_pca_ = None
            else:
                logger.debug("Fitting Parselmouth PCA...")

                # Preprocess for numerical stability
                # X_parsel_stable = self._preprocess_for_numerical_stability(X_parsel)

                # Update n_components if features were removed
                actual_n_components = min(self.parsel_n_components, X_parsel.shape[1])

                self.parsel_pca_ = PCA(
                    n_components=actual_n_components,
                    random_state=self.random_state,
                    svd_solver="full",
                )

                # Compute and store noise scale from training data
                self.parsel_noise_scale_ = self._compute_noise_scale(X_parsel)
                X_parsel_reg = self._add_regularization_noise(X_parsel, self.parsel_noise_scale_)

                # Suppress numerical warnings during PCA fitting
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*encountered in matmul.*",
                        category=RuntimeWarning,
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*divide by zero.*", category=RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*overflow.*", category=RuntimeWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message=".*invalid value.*", category=RuntimeWarning
                    )
                    self.parsel_pca_.fit(X_parsel_reg)
        else:
            logger.debug("Skipping Parselmouth PCA (empty data or disabled)")
            self.parsel_pca_ = None

        # Baseline-only: both voice blocks empty is allowed; transform() will pass X through
        if X_wav2vec.shape[1] == 0 and X_parsel.shape[1] == 0:
            logger.debug(
                "Both voice feature sets are empty (baseline-only); PCA step will pass through."
            )

        return self

    def transform(self, X):
        """
        Apply PCA transformation to the features.

        Args:
            X: Concatenated features array from FeaturePreloader [parselmouth | wav2vec | demographic]

        Returns:
            Combined feature array after PCA transformation [parselmouth_pca | wav2vec_pca | demographic]
        """
        # Use fitted boundaries (set during fit) to handle upstream column removal
        parsel_end = getattr(self, "actual_parsel_end_", None)
        wav2vec_end = getattr(self, "actual_wav2vec_end_", None)
        if parsel_end is None or wav2vec_end is None:
            # Fallback for unfitted transformer or backward compatibility
            parsel_end, wav2vec_end = self._compute_split_boundaries(X.shape[1])

        if parsel_end > 0:
            X_parsel = X[:, :parsel_end]
        else:
            X_parsel = np.empty((X.shape[0], 0))

        if wav2vec_end > parsel_end:
            X_wav2vec = X[:, parsel_end:wav2vec_end]
        else:
            X_wav2vec = np.empty((X.shape[0], 0))

        if self.demographic_feature_count > 0:
            X_demographic = X[:, wav2vec_end:]
        else:
            X_demographic = np.empty((X.shape[0], 0))

        # Transform Parselmouth features
        if self.parsel_pca_ is not None:
            # Reuse noise scale computed during fit for consistency
            parsel_ns = getattr(self, "parsel_noise_scale_", None)
            X_parsel_reg = self._add_regularization_noise(X_parsel, parsel_ns)

            # Suppress numerical warnings during PCA transform
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*encountered in matmul.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore", message=".*divide by zero.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=".*overflow.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=".*invalid value.*", category=RuntimeWarning
                )
                X_parsel_output = self.parsel_pca_.transform(X_parsel_reg)
        else:
            X_parsel_output = X_parsel

        # Transform Wav2Vec2 features
        if self.wav2vec_pca_ is not None:
            # Reuse noise scale computed during fit for consistency
            wav2vec_ns = getattr(self, "wav2vec_noise_scale_", None)
            X_wav2vec_reg = self._add_regularization_noise(X_wav2vec, wav2vec_ns)

            # Suppress numerical warnings during PCA transform
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*encountered in matmul.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore", message=".*divide by zero.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=".*overflow.*", category=RuntimeWarning
                )
                warnings.filterwarnings(
                    "ignore", message=".*invalid value.*", category=RuntimeWarning
                )
                X_wav2vec_output = self.wav2vec_pca_.transform(X_wav2vec_reg)
        else:
            X_wav2vec_output = X_wav2vec

        # Demographic features pass through without transformation
        X_demographic_output = X_demographic

        # Combine features: [parselmouth_pca | wav2vec_pca | demographic]
        return np.hstack([X_parsel_output, X_wav2vec_output, X_demographic_output])

    def _compute_noise_scale(self, X):
        """Compute noise scale from training data (called during fit only)."""
        noise_scale = self.regularization * np.std(X, axis=0, keepdims=True)
        min_noise = self.regularization * 1e-6
        return np.maximum(noise_scale, min_noise)

    def _add_regularization_noise(self, X, noise_scale=None):
        """
        Add small regularization noise to prevent numerical instability in PCA.

        Args:
            X: Input feature matrix
            noise_scale: Pre-computed noise scale (from fit). If None, computed from X.

        Returns:
            Feature matrix with added regularization noise
        """
        if self.regularization <= 0:
            return X

        # Use the same random state for reproducibility
        rng = np.random.RandomState(self.random_state)

        if noise_scale is None:
            noise_scale = self._compute_noise_scale(X)

        noise = rng.normal(0, noise_scale, X.shape)

        return X + noise

    def _preprocess_for_numerical_stability(self, X):
        """
        Preprocess features to improve numerical stability for PCA.

        Args:
            X: Input feature matrix

        Returns:
            Preprocessed feature matrix
        """
        # Remove features with zero variance
        feature_vars = np.var(X, axis=0)
        zero_var_mask = feature_vars > 1e-12  # More conservative threshold

        if not np.all(zero_var_mask):
            removed_count = np.sum(~zero_var_mask)
            logger.debug(f"Removing {removed_count} near-zero variance features")
            X = X[:, zero_var_mask]

        # Remove perfectly correlated features
        if X.shape[1] > 1:
            try:
                corr_matrix = np.corrcoef(X.T)
                # Find upper triangle indices (excluding diagonal)
                upper_tri = np.triu_indices_from(corr_matrix, k=1)
                high_corr_pairs = np.where(np.abs(corr_matrix[upper_tri]) > 0.99)[0]

                if len(high_corr_pairs) > 0:
                    # Remove the second feature in each highly correlated pair
                    features_to_remove = set()
                    for idx in high_corr_pairs:
                        row_idx, col_idx = upper_tri[0][idx], upper_tri[1][idx]
                        features_to_remove.add(col_idx)  # Remove the later feature

                    if features_to_remove:
                        keep_mask = np.ones(X.shape[1], dtype=bool)
                        keep_mask[list(features_to_remove)] = False
                        logger.debug(
                            f"Removing {len(features_to_remove)} highly correlated features"
                        )
                        X = X[:, keep_mask]
            except Exception as e:
                logger.debug(f"Correlation analysis failed, skipping: {e}")

        return X
