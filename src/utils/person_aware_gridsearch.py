"""
Person-aware GridSearchCV wrapper for hyperparameter optimization.

This module provides a GridSearchCV wrapper that can pass audio_ids to
person-aware cross-validation strategies, preventing data leakage during
hyperparameter optimization when working with chunked audio data.
"""

from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


class PersonAwareGridSearchCV(GridSearchCV):
    """
    GridSearchCV wrapper that supports person-aware cross-validation.

    This class extends sklearn's GridSearchCV to pass audio_ids to cross-validation
    strategies that support person-aware splitting, such as PersonAwareStratifiedKFold
    and PersonAwareRepeatedStratifiedKFold.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data
    param_grid : dict or list of dictionaries
        Dictionary with parameters names as keys and lists of parameter settings
    audio_ids : array-like, optional
        Audio IDs identifying which person each sample belongs to.
        Only used if the CV strategy supports audio_ids parameter.
    **kwargs : dict
        Additional parameters passed to GridSearchCV

    Examples
    --------
    >>> from src.utils.person_aware_cv import PersonAwareRepeatedStratifiedKFold
    >>> cv = PersonAwareRepeatedStratifiedKFold(n_splits=3, n_repeats=2)
    >>> grid_search = PersonAwareGridSearchCV(
    ...     estimator=pipeline,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ...     audio_ids=audio_ids
    ... )
    >>> grid_search.fit(X_train, y_train)
    """

    def __init__(self, estimator, param_grid, *, audio_ids=None, **kwargs):
        super().__init__(estimator, param_grid, **kwargs)
        self.audio_ids = audio_ids

    def fit(self, X, y=None, **fit_params):
        """
        Fit the GridSearchCV with person-aware cross-validation if supported.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), default=None
            Target variable
        **fit_params : dict
            Additional parameters passed to the fit method

        Returns
        -------
        self : object
            Returns self for method chaining
        """
        # Get the CV strategy (GridSearchCV stores it as cv attribute)
        cv_strategy = getattr(self, "cv", None)

        # Check if CV strategy supports audio_ids parameter
        cv_supports_audio_ids = (
            cv_strategy is not None
            and hasattr(cv_strategy, "split")
            and "audio_ids" in cv_strategy.split.__code__.co_varnames
        )

        if (
            cv_supports_audio_ids
            and self.audio_ids is not None
            and cv_strategy is not None
        ):
            logger.debug(
                "Using person-aware cross-validation for hyperparameter optimization"
            )

            # Store original split method
            original_split = cv_strategy.split

            # Create wrapper that passes audio_ids
            def patched_split(X, y=None, groups=None):
                return original_split(X, y, groups, audio_ids=self.audio_ids)

            # Temporarily replace split method
            cv_strategy.split = patched_split

            try:
                result = super().fit(X, y, **fit_params)
            finally:
                # Restore original split method to avoid side effects
                cv_strategy.split = original_split

            return result
        else:
            # Fallback to standard GridSearchCV behavior
            if self.audio_ids is not None:
                logger.debug(
                    "CV strategy does not support audio_ids - using standard GridSearchCV"
                )
            return super().fit(X, y, **fit_params)
