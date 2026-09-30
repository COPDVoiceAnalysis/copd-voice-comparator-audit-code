"""
Person-aware cross-validation strategies to prevent data leakage.

This module provides cross-validation strategies that ensure all chunks/samples
from the same person (audio_id) stay in the same fold, preventing data leakage
in machine learning experiments with chunked audio data.
"""

import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.utils.validation import check_array
import logging

logger = logging.getLogger(__name__)


class PersonAwareStratifiedKFold(BaseCrossValidator):
    """
    Person-aware stratified K-fold cross-validation.

    This cross-validation strategy ensures that all samples from the same person
    (identified by audio_id) are placed in the same fold, preventing data leakage
    while maintaining class balance as much as possible.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits

    def split(self, X, y=None, groups=None, audio_ids=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), default=None
            Target variable for supervised learning
        groups : array-like of shape (n_samples,), default=None
            Group labels (audio IDs) for sklearn compatibility. Preferred over audio_ids.
        audio_ids : array-like of shape (n_samples,), default=None
            Audio IDs identifying which person each sample belongs to (legacy, use groups instead)

        Yields
        ------
        train : ndarray
            Training set indices for that split
        test : ndarray
            Testing set indices for that split
        """
        # Prefer groups (sklearn convention) over audio_ids (legacy)
        if groups is not None and audio_ids is None:
            audio_ids = groups
        if audio_ids is None:
            logger.warning(
                "No audio_ids provided to PersonAwareStratifiedKFold. "
                "Falling back to standard StratifiedKFold."
            )
            # Fallback to standard stratified k-fold
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            yield from skf.split(X, y)
            return

        X = check_array(X, accept_sparse=["csc", "csr"], ensure_2d=True, force_all_finite=False)
        audio_ids = np.array(audio_ids)

        if y is not None:
            y = np.array(y)
            if len(y) != len(X):
                raise ValueError("X and y must have the same number of samples")

        if len(audio_ids) != len(X):
            raise ValueError("X and audio_ids must have the same number of samples")

        # Group samples by audio_id and get their labels
        person_groups = defaultdict(list)
        person_labels = {}

        for idx, audio_id in enumerate(audio_ids):
            person_groups[audio_id].append(idx)
            if y is not None:
                person_labels[audio_id] = y[
                    idx
                ]  # All samples from same person have same label

        unique_audio_ids = list(person_groups.keys())
        n_persons = len(unique_audio_ids)

        logger.debug(
            f"PersonAwareStratifiedKFold: {n_persons} unique persons, {len(X)} total samples"
        )

        if n_persons < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than the number "
                f"of unique persons={n_persons}"
            )

        # Create person-level labels for stratification
        if y is not None:
            person_y = np.array(
                [person_labels[audio_id] for audio_id in unique_audio_ids]
            )

            # Check if we can do stratified splitting
            unique_labels, label_counts = np.unique(person_y, return_counts=True)
            min_class_count = min(label_counts)

            if min_class_count < self.n_splits:
                logger.warning(
                    f"Minimum class count ({min_class_count}) is less than n_splits ({self.n_splits}). "
                    f"Using regular KFold instead of StratifiedKFold at person level."
                )
                # Use regular k-fold at person level
                from sklearn.model_selection import KFold

                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
                person_splits = list(kf.split(unique_audio_ids))
            else:
                # Use stratified k-fold at person level
                skf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
                person_splits = list(skf.split(unique_audio_ids, person_y))
        else:
            # No labels provided, use regular k-fold at person level
            from sklearn.model_selection import KFold

            kf = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            person_splits = list(kf.split(unique_audio_ids))

        # Convert person-level splits to sample-level splits
        for train_person_idx, test_person_idx in person_splits:
            train_audio_ids = [unique_audio_ids[i] for i in train_person_idx]
            test_audio_ids = [unique_audio_ids[i] for i in test_person_idx]

            # Get all sample indices for training persons
            train_indices = []
            for audio_id in train_audio_ids:
                train_indices.extend(person_groups[audio_id])

            # Get all sample indices for test persons
            test_indices = []
            for audio_id in test_audio_ids:
                test_indices.extend(person_groups[audio_id])

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            # Validate no overlap
            if len(set(train_indices) & set(test_indices)) > 0:
                raise ValueError(
                    "Train and test indices overlap - this should not happen"
                )

            # Log fold statistics
            if y is not None:
                train_labels = y[train_indices]
                test_labels = y[test_indices]
                train_dist = np.bincount(train_labels) / len(train_labels)
                test_dist = np.bincount(test_labels) / len(test_labels)
                logger.debug(
                    f"Fold: {len(train_audio_ids)} train persons ({len(train_indices)} samples), "
                    f"{len(test_audio_ids)} test persons ({len(test_indices)} samples). "
                    f"Train dist: {train_dist}, Test dist: {test_dist}"
                )

            yield train_indices, test_indices


class PersonAwareRepeatedStratifiedKFold(BaseCrossValidator):
    """
    Person-aware repeated stratified K-fold cross-validation.

    Repeats PersonAwareStratifiedKFold n times with different randomization.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds per repetition
    n_repeats : int, default=10
        Number of repetitions
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None, audio_ids=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), default=None
            Target variable for supervised learning
        groups : array-like of shape (n_samples,), default=None
            Group labels (audio IDs) for sklearn compatibility. Preferred over audio_ids.
        audio_ids : array-like of shape (n_samples,), default=None
            Audio IDs identifying which person each sample belongs to (legacy, use groups instead)

        Yields
        ------
        train : ndarray
            Training set indices for that split
        test : ndarray
            Testing set indices for that split
        """
        # Prefer groups (sklearn convention) over audio_ids (legacy)
        if groups is not None and audio_ids is None:
            audio_ids = groups

        rng = np.random.RandomState(self.random_state)

        for repeat in range(self.n_repeats):
            # Use different random state for each repetition
            repeat_random_state = (
                None if self.random_state is None else rng.randint(0, 2**31)
            )

            cv = PersonAwareStratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=repeat_random_state
            )

            yield from cv.split(X, y, audio_ids=audio_ids)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits * self.n_repeats


def detect_chunked_data(metadata_df) -> bool:
    """
    Detect if the dataset contains chunked data from split_poem.py.

    Chunked data is identified by the presence of non-null chunk_number values,
    which are added by the split_poem.py script when audio files are split
    into smaller segments.

    Parameters
    ----------
    metadata_df : pandas.DataFrame or Metadata object
        Metadata containing chunk_number field

    Returns
    -------
    bool
        True if chunked data is detected (chunk_number field exists and has non-null values)
    """
    # Handle both DataFrame and Metadata object
    if hasattr(metadata_df, "to_dataframe"):
        df = metadata_df.to_dataframe()
    else:
        df = metadata_df

    # Check if chunk_number column exists and has non-null values
    has_chunk_column = "chunk_number" in df.columns
    has_chunks = has_chunk_column and df["chunk_number"].notna().any()

    if has_chunks:
        # Get statistics about chunked data
        chunked_rows = df[df["chunk_number"].notna()]
        unique_persons = len(chunked_rows["audio_id"].unique())
        total_chunks = len(chunked_rows)
        max_chunks_per_person = (
            chunked_rows.groupby("audio_id")["chunk_number"].count().max()
        )

        logger.info(
            f"Chunked data detected: {unique_persons} persons with chunks, {total_chunks} total chunks"
        )
        logger.info(f"Max chunks per person: {max_chunks_per_person}")
    else:
        total_samples = len(df)
        unique_persons = len(df["audio_id"].unique())
        logger.info(
            f"No chunked data detected: {unique_persons} unique persons, {total_samples} total samples"
        )

    logger.info(f"Chunked data detected: {has_chunks}")
    return has_chunks


def validate_no_person_leakage(
    train_audio_ids: np.ndarray, test_audio_ids: np.ndarray
) -> bool:
    """
    Validate that no person appears in both training and test sets.

    Parameters
    ----------
    train_audio_ids : array-like
        Audio IDs in training set
    test_audio_ids : array-like
        Audio IDs in test set

    Returns
    -------
    bool
        True if no leakage detected, False otherwise
    """
    train_persons = set(train_audio_ids)
    test_persons = set(test_audio_ids)
    overlap = train_persons & test_persons

    if overlap:
        logger.error(
            f"Person leakage detected! {len(overlap)} persons appear in both train and test:"
        )
        logger.error(
            f"Overlapping audio_ids: {sorted(list(overlap))[:10]}..."
        )  # Show first 10
        return False

    logger.info(
        f"No person leakage detected. {len(train_persons)} train persons, {len(test_persons)} test persons"
    )
    return True
