"""
Streamlined evaluation utilities with direct MLFlow integration.

This module provides a minimal EvaluationManager that focuses on evaluation logic
and logs all results directly to MLFlow, eliminating redundant file-based storage.
"""

import numpy as np
import pandas as pd
import tempfile
import os
from numpy.typing import NDArray
from typing import Any, cast
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from src.models.metadata import Metadata
import logging
from src.analysis.language_bias_analysis import LanguageBiasAnalyzer
from src.utils.mlflow_tracker import MLFlowTracker
from src.utils.plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
from src.utils.custom_metrics import (
    ppv_deploy,
    npv_deploy,
    expected_fp_per_1000,
)
from src.utils.sample_weight_strategies import compute_evaluation_strata_weights
from src.utils.stratum_weights import (
    assign_age_band,
    COARSE_AGE_EDGES,
    COARSE_AGE_LABELS,
)

logger = logging.getLogger(__name__)


def _mlflow_safe_stratum_key(key: str) -> str:
    """Sanitize stratum key for MLflow metric names (alphanumerics, _, -, ., :, / only)."""
    return key.replace("<=", "le").replace(">", "gt")


def _aggregate_chunks_to_persons(
    y_pred_proba: np.ndarray,
    y_true: np.ndarray,
    audio_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate chunk-level predictions to person level via mean probability pooling.

    For unchunked data (unique audio_ids) this is a no-op.

    Returns:
        person_proba: (n_persons,) mean predicted probability per person
        person_y: (n_persons,) true label per person (all chunks share the same label)
        person_ids: (n_persons,) unique audio_ids in stable order
    """
    df = pd.DataFrame({
        "audio_id": audio_ids,
        "proba": y_pred_proba,
        "y": y_true,
    })
    grouped = df.groupby("audio_id", sort=False).agg(
        proba=("proba", "mean"),
        y=("y", "first"),
    )
    return (
        grouped["proba"].to_numpy(),
        grouped["y"].to_numpy(),
        grouped.index.to_numpy(),
    )


class MLFlowEvaluationManager:
    """
    Streamlined evaluation manager with direct MLFlow integration.

    This class handles evaluation logic and logs all results directly to MLFlow,
    eliminating the need for file-based storage and manual result aggregation.
    """

    def __init__(
        self,
        mlflow_tracker: MLFlowTracker,
        metadata: Metadata | None = None,
        feature_config=None,
        preloader=None,
        demographic_feature_columns: list[str] | None = None,
        include_voice_features: bool = True,
    ):
        """
        Initialize MLFlow-integrated evaluation manager.

        Args:
            mlflow_tracker: MLFlowTracker instance for logging
            metadata: Metadata object for language bias analysis (optional)
            feature_config: FeatureConfig for lazy loading original features
            preloader: FeaturePreloader instance for lazy loading
            demographic_feature_columns: Same as training (e.g. ["age", "sex", "bmi"] or with comorbidities)
            include_voice_features: Same as training (False for baseline-only)
        """
        self.mlflow_tracker = mlflow_tracker
        self.metadata = metadata
        self.feature_config = feature_config
        self.preloader = preloader
        self.demographic_feature_columns = (
            demographic_feature_columns
            if demographic_feature_columns is not None
            else ["age", "sex", "bmi"]
        )
        self.include_voice_features = include_voice_features

        # Track summary statistics for final logging
        self.fold_count = 0
        self.summary_stats = {
            # Unweighted (study population)
            "aurocs": [],
            "mccs": [],
            "balanced_accuracies": [],
            "average_precs": [],
            # Strata-weighted (target population)
            "weighted_aurocs": [],
            "weighted_mccs": [],
            "weighted_balanced_accuracies": [],
            "weighted_average_precs": [],
            # Deployment metrics
            "ppv_deploy": [],
            "npv_deploy": [],
            "expected_fp_per_1000": [],
            # Fairness metrics
            "score_age_correlation_controls": [],
            "auroc_stratum": [],  # list of dicts stratum_key -> auroc
        }
        self.pi_deploy = 0.2  # for PPV/NPV/Expected FP reporting

        # Store all predictions for final language bias analysis
        self.all_predictions = []

    def evaluate_fold(
        self,
        model,
        chunked_test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        fold: int,
        classifier_name: str,
        label_encoder,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate a single fold.

        For chunked data, predictions are made on chunk-level features (matching
        the training distribution) and aggregated to person level via mean
        probability pooling before computing metrics.  For unchunked data the
        aggregation is a no-op.

        Strata weights for target-population-weighted metrics are computed internally
        from metadata (age/sex), independent of the training weight strategy.

        Args:
            model: Trained model to evaluate
            chunked_test_data: (X_test, y_test, audio_ids) for test data (chunk-level if chunked)
            fold: Fold number
            classifier_name: Name of the classifier
            label_encoder: Label encoder for converting predictions
            threshold: Decision threshold for binary predictions

        Returns:
            Dictionary with evaluation results
        """
        self.fold_count = fold

        X_test, y_test, audio_ids = chunked_test_data

        # Predict at chunk level (matches training feature distribution)
        y_pred_proba_chunks = model.predict_proba(X_test)[:, 1]

        # Aggregate to person level via mean probability pooling
        person_proba, person_y, person_ids = _aggregate_chunks_to_persons(
            y_pred_proba_chunks, y_test, audio_ids
        )

        logger.info(
            "Fold %d: predicted %d chunks → aggregated to %d persons",
            fold, len(audio_ids), len(person_ids),
        )

        # Compute strata weights at person level
        person_strata_weight = (
            compute_evaluation_strata_weights(self.metadata, person_ids, person_y)
            if self.metadata is not None else None
        )

        # Evaluate on person-level aggregated data
        person_test_data = (person_proba, person_y, person_ids)
        metrics = self._evaluate_person_level(
            person_test_data,
            fold,
            classifier_name,
            label_encoder,
            threshold=threshold,
            sample_weight=person_strata_weight,
        )

        # Log to MLFlow with step indexing
        mlflow_metrics = {
            "auroc": metrics["auroc_score"],
            "mcc": metrics["mcc_score"],
            "balanced_accuracy": metrics["balanced_accuracy_score"],
            "average_prec": metrics["average_prec_score"],
            "n_samples": len(person_y),
            "threshold": threshold,
            # Strata-weighted metrics (target population)
            "weighted_auroc": metrics["weighted_auroc"],
            "weighted_mcc": metrics["weighted_mcc"],
            "weighted_balanced_accuracy": metrics["weighted_balanced_accuracy"],
        }
        self.mlflow_tracker.log_fold_metrics(fold, mlflow_metrics)

        # Store for final summary
        self.summary_stats["aurocs"].append(metrics["auroc_score"])
        self.summary_stats["mccs"].append(metrics["mcc_score"])
        self.summary_stats["balanced_accuracies"].append(metrics["balanced_accuracy_score"])
        self.summary_stats["average_precs"].append(metrics["average_prec_score"])
        self.summary_stats["weighted_aurocs"].append(metrics["weighted_auroc"])
        self.summary_stats["weighted_mccs"].append(metrics["weighted_mcc"])
        self.summary_stats["weighted_balanced_accuracies"].append(metrics["weighted_balanced_accuracy"])
        self.summary_stats["weighted_average_precs"].append(metrics["weighted_average_prec"])
        self.summary_stats["ppv_deploy"].append(metrics["ppv_deploy"])
        self.summary_stats["npv_deploy"].append(metrics["npv_deploy"])
        self.summary_stats["expected_fp_per_1000"].append(metrics["expected_fp_per_1000"])
        self.summary_stats["score_age_correlation_controls"].append(
            metrics["score_age_correlation_controls"]
        )
        self.summary_stats["auroc_stratum"].append(metrics["auroc_stratum"])

        # Store predictions for final language bias analysis
        self.all_predictions.append(metrics["predictions_df"])

        return metrics

    def _load_original_features_for_patients(
        self, audio_ids: NDArray[np.int_]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load original features for specific patients on-demand.

        Args:
            audio_ids: Array of patient identifiers (audio_ids) from chunked test set

        Returns:
            Tuple of (original_features, y_original, original_audio_ids)
        """
        if (
            self.metadata is None
            or self.feature_config is None
            or self.preloader is None
        ):
            raise ValueError(
                "Metadata, feature_config, and preloader required for lazy loading"
            )

        logger.info(
            f"Lazy loading original features for {len(np.unique(audio_ids))} patients"
        )

        # Generate original recording identifier sets for these specific patients
        unique_audio_ids: list = cast(list, np.unique(audio_ids).tolist())

        original_recording_sets = (
            self.metadata.get_original_recording_identifier_sets_subset(
                unique_audio_ids, self.feature_config
            )
        )

        # Load features for original recordings (same as training: columns + include_voice)
        original_demographic_features = self.metadata.get_demographic_features(
            original_recording_sets, self.demographic_feature_columns
        )
        original_features, _, _ = self.preloader.load_features(
            original_recording_sets,
            original_demographic_features,
            include_voice_features=self.include_voice_features,
        )

        # Get labels and audio_ids for original data (same order as recording_sets)
        y_original = np.array(
            [
                self.metadata.df[self.metadata.df["audio_id"] == rs.audio_id].iloc[0][
                    "class_idx"
                ]
                for rs in original_recording_sets
            ]
        )
        original_audio_ids = np.array([rs.audio_id for rs in original_recording_sets])

        logger.info(
            f"Loaded original features: {original_features.shape}, {len(original_audio_ids)} patients"
        )
        return original_features, y_original, original_audio_ids

    def finalize_experiment(self, label_encoder):
        """
        Log final summary metrics and generate comprehensive outer fold results as MLFlow artifacts.

        Args:
            label_encoder: Label encoder for class names
        """
        # Log evaluation summary
        aurocs = self.summary_stats["aurocs"]
        mccs = self.summary_stats["mccs"]
        balanced_accuracies = self.summary_stats["balanced_accuracies"]
        average_precs = self.summary_stats["average_precs"]
        ppv_vals = self.summary_stats["ppv_deploy"]
        npv_vals = self.summary_stats["npv_deploy"]
        expected_fp_vals = self.summary_stats["expected_fp_per_1000"]
        score_age_vals = self.summary_stats["score_age_correlation_controls"]
        auroc_stratum_vals = self.summary_stats["auroc_stratum"]

        if aurocs:
            final_metrics = {
                "auroc_mean": np.mean(aurocs),
                "auroc_std": np.std(aurocs),
                "mcc_mean": np.mean(mccs),
                "mcc_std": np.std(mccs),
                "average_prec_mean": np.mean(average_precs),
                "average_prec_std": np.std(average_precs),
                "balanced_accuracy_mean": np.mean(balanced_accuracies),
                "balanced_accuracy_std": np.std(balanced_accuracies),
            }

            # Strata-weighted metrics (target population estimates)
            for metric_key, metric_name in [
                ("weighted_aurocs", "weighted_auroc"),
                ("weighted_mccs", "weighted_mcc"),
                ("weighted_balanced_accuracies", "weighted_balanced_accuracy"),
                ("weighted_average_precs", "weighted_average_prec"),
            ]:
                valid = [x for x in self.summary_stats[metric_key] if np.isfinite(x)]
                if valid:
                    final_metrics[f"{metric_name}_mean"] = float(np.mean(valid))
                    final_metrics[f"{metric_name}_std"] = float(np.std(valid))
            if ppv_vals:
                final_metrics["ppv_deploy_mean"] = float(np.nanmean(ppv_vals))
                final_metrics["ppv_deploy_std"] = float(np.nanstd(ppv_vals))
            if npv_vals:
                final_metrics["npv_deploy_mean"] = float(np.nanmean(npv_vals))
                final_metrics["npv_deploy_std"] = float(np.nanstd(npv_vals))
            if expected_fp_vals:
                final_metrics["expected_fp_per_1000_mean"] = float(np.nanmean(expected_fp_vals))
                final_metrics["expected_fp_per_1000_std"] = float(np.nanstd(expected_fp_vals))
            if score_age_vals:
                valid = [x for x in score_age_vals if np.isfinite(x)]
                if valid:
                    final_metrics["score_age_correlation_controls_mean"] = float(np.mean(valid))
                    final_metrics["score_age_correlation_controls_std"] = float(np.std(valid))
            if auroc_stratum_vals:
                all_keys = set()
                for d in auroc_stratum_vals:
                    all_keys.update(d.keys())
                for k in sorted(all_keys):
                    vals = [d[k] for d in auroc_stratum_vals if k in d]
                    if vals:
                        safe_k = _mlflow_safe_stratum_key(k)
                        final_metrics[f"auroc_stratum_{safe_k}_mean"] = float(np.mean(vals))
                        final_metrics[f"auroc_stratum_{safe_k}_std"] = float(np.std(vals))
            self.mlflow_tracker.log_final_metrics(final_metrics)

        # Save fold-level predictions as artifact (for ROC curves, calibration analysis, etc.)
        self._save_fold_predictions_artifact()

        # Generate final confusion matrices
        self._generate_final_confusion_matrices(label_encoder)

        # Perform final comprehensive language bias analysis
        self._perform_final_language_bias_analysis(label_encoder)

        logger.info(
            "Experiment finalized - all results and outer fold summaries logged to MLFlow"
        )

    def _evaluate_person_level(
        self,
        person_test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        fold: int,
        classifier_name: str,
        label_encoder,
        threshold: float,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate person-level (aggregated) predictions.

        Args:
            person_test_data: (y_pred_proba, y_true, audio_ids) at person level
        """
        y_pred_proba, y_test, audio_ids = person_test_data
        y_pred = (y_pred_proba >= threshold).astype(int)
        return self._compute_metrics(
            y_pred_proba, y_pred, y_test, audio_ids,
            fold, classifier_name, label_encoder, sample_weight,
        )

    def _evaluate_single_dataset(
        self,
        model,
        test_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        fold: int,
        classifier_name: str,
        label_encoder,
        threshold: float,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on a single dataset. Optionally use sample_weight for weighted metrics."""
        X_test, y_test, audio_ids = test_data

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        return self._compute_metrics(
            y_pred_proba, y_pred, y_test, audio_ids,
            fold, classifier_name, label_encoder, sample_weight,
        )

    def _compute_metrics(
        self,
        y_pred_proba: np.ndarray,
        y_pred: np.ndarray,
        y_test: np.ndarray,
        audio_ids: np.ndarray,
        fold: int,
        classifier_name: str,
        label_encoder,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute all evaluation metrics from predictions."""

        # Calculate unweighted metrics (study population)
        mcc_score = matthews_corrcoef(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        average_prec = average_precision_score(y_test, y_pred_proba, average="weighted")
        auroc_score = roc_auc_score(y_test, y_pred_proba)

        # Calculate strata-weighted metrics (target population estimates)
        if sample_weight is not None and len(sample_weight) == len(y_test):
            weighted_auroc = roc_auc_score(y_test, y_pred_proba, sample_weight=sample_weight)
            weighted_average_prec = average_precision_score(
                y_test, y_pred_proba, sample_weight=sample_weight
            )
        else:
            weighted_auroc = float("nan")
            weighted_average_prec = float("nan")

        # Create predictions dataframe for language bias analysis
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_true_labels = label_encoder.inverse_transform(y_test)

        predictions_df = pd.DataFrame(
            {
                "audio_id": audio_ids,
                "true": y_test,
                "pred": y_pred,
                "true_label": y_true_labels,
                "pred_label": y_pred_labels,
                "fold": fold,
                "classifier": classifier_name,
            }
        )

        # Add probabilities (binary: class 0 = 1 - p, class 1 = p)
        predictions_df[f"proba_{label_encoder.classes_[0]}"] = 1.0 - y_pred_proba
        predictions_df[f"proba_{label_encoder.classes_[1]}"] = y_pred_proba

        # Add strata weight for downstream strata-weighted ROC curves
        if sample_weight is not None and len(sample_weight) == len(y_test):
            predictions_df["strata_weight"] = sample_weight

        # Deployment metrics at pi_deploy (default 0.2): PPV, NPV, Expected FP per 1000
        # Use strata weights for TPR/FPR so they reflect the target population
        cases = y_test == 1
        controls = y_test == 0
        if sample_weight is not None and len(sample_weight) == len(y_test):
            w = sample_weight
            tp_w = (w[cases] * (y_pred[cases] == 1)).sum()
            fn_w = (w[cases] * (y_pred[cases] == 0)).sum()
            fp_w = (w[controls] * (y_pred[controls] == 1)).sum()
            tn_w = (w[controls] * (y_pred[controls] == 0)).sum()
        else:
            tp_w = float((y_pred[cases] == 1).sum()) if cases.any() else 0.0
            fn_w = float((y_pred[cases] == 0).sum()) if cases.any() else 0.0
            fp_w = float((y_pred[controls] == 1).sum()) if controls.any() else 0.0
            tn_w = float((y_pred[controls] == 0).sum()) if controls.any() else 0.0
        tpr = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
        fpr = fp_w / (fp_w + tn_w) if (fp_w + tn_w) > 0 else 0.0
        tnr = 1.0 - fpr

        # Weighted balanced accuracy: (TPR_w + TNR_w) / 2
        weighted_balanced_accuracy = (tpr + tnr) / 2.0

        # Weighted MCC from weighted confusion matrix entries
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        num = tp_w * tn_w - fp_w * fn_w
        den = np.sqrt(
            (tp_w + fp_w) * (tp_w + fn_w) * (tn_w + fp_w) * (tn_w + fn_w)
        )
        weighted_mcc = float(num / den) if den > 0 else float("nan")

        pi = self.pi_deploy
        ppv = ppv_deploy(pi, tpr, fpr)
        npv = npv_deploy(pi, tpr, fpr)
        exp_fp_1k = expected_fp_per_1000(pi, fpr)

        # Stratified AUC and score-age correlation (controls) when metadata available
        auroc_stratum: dict[str, float] = {}
        score_age_corr = float("nan")
        if self.metadata is not None and len(audio_ids) > 0:
            try:
                meta = self.metadata.get_metadata_for_audio_id_sequence(
                    audio_ids, columns=["age", "sex"]
                )
                age = pd.to_numeric(meta["age"], errors="coerce").to_numpy()
                sex = meta["sex"].to_numpy().astype(int)
                age_bands = assign_age_band(
                    age, edges=COARSE_AGE_EDGES, labels=COARSE_AGE_LABELS
                )
                for band in COARSE_AGE_LABELS:
                    for s in (0, 1):
                        mask = (age_bands == band) & (sex == s)
                        if mask.sum() >= 2 and len(np.unique(y_test[mask])) >= 2:
                            try:
                                auroc_stratum[f"{band}_sex{s}"] = roc_auc_score(
                                    y_test[mask], y_pred_proba[mask]
                                )
                            except Exception:
                                pass
                # Score-age correlation (controls only)
                ctrl = y_test == 0
                if ctrl.sum() >= 2 and np.isfinite(age[ctrl]).sum() >= 2:
                    score_age_corr = float(
                        np.corrcoef(y_pred_proba[ctrl], age[ctrl])[0, 1]
                    )
                    if not np.isfinite(score_age_corr):
                        score_age_corr = float("nan")
            except Exception as e:
                logger.debug("Stratified AUC / score-age skipped: %s", e)

        return {
            "predictions_df": predictions_df,
            # Unweighted metrics (study population)
            "auroc_score": auroc_score,
            "average_prec_score": average_prec,
            "mcc_score": mcc_score,
            "balanced_accuracy_score": balanced_accuracy,
            # Strata-weighted metrics (target population)
            "weighted_auroc": weighted_auroc,
            "weighted_mcc": weighted_mcc,
            "weighted_balanced_accuracy": weighted_balanced_accuracy,
            "weighted_average_prec": weighted_average_prec,
            # Deployment metrics
            "ppv_deploy": ppv,
            "npv_deploy": npv,
            "expected_fp_per_1000": exp_fp_1k,
            # Fairness metrics
            "auroc_stratum": auroc_stratum,
            "score_age_correlation_controls": score_age_corr,
        }

    def _save_fold_predictions_artifact(self):
        """
        Save all fold-level predictions as a parquet artifact for downstream analysis
        (ROC curves, calibration plots, threshold sensitivity analysis, etc.).
        """
        if not self.all_predictions:
            logger.warning("No predictions collected - skipping predictions artifact")
            return

        try:
            combined = pd.concat(self.all_predictions, ignore_index=True)
            with tempfile.NamedTemporaryFile(
                suffix=".parquet", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            try:
                combined.to_parquet(temp_path, index=False)
                self.mlflow_tracker.log_file_artifact(temp_path, "fold_predictions")
                logger.info(
                    "Saved fold-level predictions artifact: %d rows across %d folds",
                    len(combined),
                    combined["fold"].nunique(),
                )
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.warning("Failed to save fold predictions artifact: %s", e)

    def _generate_final_confusion_matrices(self, label_encoder):
        """
        Generate final confusion matrix from aggregated predictions across all folds.

        Args:
            label_encoder: Label encoder for class names
        """
        if not self.all_predictions:
            logger.warning(
                "No predictions collected - skipping confusion matrix generation"
            )
            return

        try:
            logger.info("Generating final confusion matrix")

            # Combine all fold predictions
            combined_predictions = pd.concat(
                self.all_predictions, ignore_index=True
            )

            # Extract true and predicted labels
            y_true_numeric = combined_predictions["true"].values
            y_pred_numeric = combined_predictions["pred"].values

            # Convert back to string labels for confusion matrix
            y_true = label_encoder.inverse_transform(y_true_numeric)
            y_pred = label_encoder.inverse_transform(y_pred_numeric)
            labels = list(label_encoder.classes_)

            # Create temporary file for confusion matrix
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            try:
                plt.clf()
                plot_confusion_matrix(y_true, y_pred, labels, temp_path)
                self.mlflow_tracker.log_file_artifact(temp_path, "confusion_matrices")
                logger.info("Confusion matrix generated")

            except Exception as e:
                logger.warning(f"Failed to generate confusion matrix: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                plt.clf()

        except Exception as e:
            logger.warning(f"Final confusion matrix generation failed: {e}")

    def _perform_final_language_bias_analysis(self, label_encoder):
        """
        Perform comprehensive language bias analysis on all predictions from all folds.
        Only runs if there are multiple languages in the dataset.

        Args:
            label_encoder: Label encoder for class names
        """
        if self.metadata is None:
            logger.debug("No metadata provided - skipping language bias analysis")
            return

        if not self.all_predictions:
            logger.warning("No predictions collected - skipping language bias analysis")
            return

        try:
            # Check if there are multiple languages in the dataset
            metadata_df = self.metadata.to_dataframe()
            if "language" not in metadata_df.columns:
                logger.info(
                    "No language column found in metadata - skipping language bias analysis"
                )
                return

            unique_languages = metadata_df["language"].nunique()
            if unique_languages <= 1:
                logger.info(
                    f"Only {unique_languages} language detected in dataset - skipping language bias analysis"
                )
                return

            logger.info(
                f"Performing comprehensive language bias analysis across all folds ({unique_languages} languages detected)"
            )

            # Combine all fold predictions
            combined_predictions = pd.concat(
                self.all_predictions, ignore_index=True
            )

            logger.info(
                f"Analyzing {len(combined_predictions)} predictions"
            )

            # Create temporary directory for bias analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                bias_analyzer = LanguageBiasAnalyzer(output_dir=temp_dir)

                # Join predictions with metadata on audio_id
                predictions_with_metadata = pd.merge(
                    combined_predictions, metadata_df, on="audio_id", how="inner"
                )

                if len(predictions_with_metadata) == 0:
                    logger.warning(
                        "No predictions could be mapped to metadata - skipping bias analysis"
                    )
                    return

                # Perform dataset analysis
                dataset_analysis = (
                    bias_analyzer.analyze_dataset_language_distribution(metadata_df)
                )

                # Perform prediction analysis
                class_names = list(label_encoder.classes_)
                prediction_analysis = (
                    bias_analyzer.analyze_prediction_language_bias(
                        predictions_with_metadata,
                        predictions_with_metadata["pred"].values,
                        predictions_with_metadata["true"].values,
                        class_names,
                    )
                )

                # Generate bias report
                bias_analyzer.generate_bias_report(
                    dataset_analysis, prediction_analysis
                )

                # Log bias analysis files as MLFlow artifacts
                for filename in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, filename)
                    if os.path.isfile(file_path):
                        self.mlflow_tracker.log_file_artifact(
                            file_path, "language_bias_analysis"
                        )

                logger.info("Language bias analysis completed")

        except Exception as e:
            logger.warning(f"Final language bias analysis failed: {e}")
