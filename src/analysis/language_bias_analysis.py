"""
Language bias analysis module for voice biomarker classification.

This module provides comprehensive analysis of language distributions across
disease categories and model predictions to detect potential language bias
in the voice biomarker classification model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Any
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageBiasAnalyzer:
    """
    Comprehensive language bias analysis for voice biomarker classification.

    This class provides methods to:
    1. Analyze language distributions in the dataset
    2. Detect potential language bias in model predictions
    3. Generate statistical tests and visualizations
    4. Provide recommendations for bias mitigation
    """

    def __init__(self, output_dir: str = "outputs/language_analysis"):
        """
        Initialize the language bias analyzer.

        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style with custom color palette
        plt.style.use("default")

        # Create custom color palette with distinct colors for en/de
        self.custom_colors = {
            "de": "#1f77b4",
            "en": "#ff7f0e",
            "unknown": "#d62728",
            "it": "#2ca02c",
            "es": "#9467bd",
            "pt": "#8c564b",
            "fr": "#e377c2",
            "el": "#7f7f7f",
            "ru": "#bcbd22",
        }

    def analyze_dataset_language_distribution(
        self, metadata_df: pd.DataFrame, save_plots: bool = True
    ) -> dict[str, Any]:
        """
        Analyze language distribution across disease categories in the dataset.

        Args:
            metadata_df: DataFrame with metadata including language and lung_disease_main
            save_plots: Whether to save visualization plots

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing dataset language distribution...")

        # Filter out longitudinal data for cleaner analysis
        analysis_df = metadata_df[metadata_df.longitudinal == 0].copy()

        # Fix Charite data language assignment if missing
        charite_mask = analysis_df["data_source"] == "charite"
        missing_lang_mask = analysis_df["language"].isna() | (
            analysis_df["language"] == ""
        )
        charite_missing_lang = charite_mask & missing_lang_mask

        if charite_missing_lang.any():
            logger.info(
                f"Fixing {charite_missing_lang.sum()} Charite records with missing language - setting to 'de'"
            )
            analysis_df.loc[charite_missing_lang, "language"] = "de"

        # Handle missing language values
        analysis_df["language_clean"] = analysis_df["language"].fillna("unknown")

        # PATIENT-LEVEL ANALYSIS: Aggregate by audio_id to avoid chunk bias
        logger.info("Performing patient-level language analysis to avoid chunk bias...")

        # Get one record per patient (audio_id) with their language and disease info
        patient_level_df = (
            analysis_df.groupby("audio_id")
            .agg(
                {
                    "language_clean": "first",  # Language should be same for all chunks of a patient
                    "lung_disease_main": "first",  # Disease should be same for all chunks of a patient
                    "data_source": "first",  # Data source should be same for all chunks of a patient
                    "longitudinal": "first",  # Should be same for all chunks of a patient
                }
            )
            .reset_index()
        )

        logger.info(f"Chunk-level samples: {len(analysis_df)}")
        logger.info(f"Patient-level samples: {len(patient_level_df)}")

        # Overall language distribution (patient-level)
        lang_dist = patient_level_df["language_clean"].value_counts()
        logger.info(f"Patient-level language distribution:\n{lang_dist}")

        # Also calculate chunk-level for comparison
        chunk_lang_dist = analysis_df["language_clean"].value_counts()
        logger.info(f"Chunk-level language distribution:\n{chunk_lang_dist}")

        # Language distribution by disease (use patient-level data)
        disease_lang_crosstab = pd.crosstab(
            patient_level_df["lung_disease_main"],
            patient_level_df["language_clean"],
            margins=True,
        )

        # Calculate percentages
        disease_lang_pct = (
            pd.crosstab(
                patient_level_df["lung_disease_main"],
                patient_level_df["language_clean"],
                normalize="index",
            )
            * 100
        )

        # Statistical test for independence
        chi2_result = chi2_contingency(
            disease_lang_crosstab.iloc[:-1, :-1]  # Exclude margins
        )
        chi2_stat = chi2_result[0]
        chi2_p = chi2_result[1]
        chi2_dof = chi2_result[2]

        # Create visualizations
        if save_plots:
            self._plot_language_distribution(
                patient_level_df, disease_lang_crosstab, disease_lang_pct
            )

        # Analyze data sources (use patient-level data)
        source_lang_dist = self._analyze_source_language_distribution(patient_level_df)

        results = {
            "overall_language_distribution": lang_dist.to_dict(),
            "chunk_level_language_distribution": chunk_lang_dist.to_dict(),
            "disease_language_crosstab": disease_lang_crosstab,
            "disease_language_percentages": disease_lang_pct,
            "chi2_test": {
                "statistic": chi2_stat,
                "p_value": chi2_p,
                "degrees_of_freedom": chi2_dof,
                "significant": chi2_p < 0.05,
            },
            "source_language_distribution": source_lang_dist,
            "n_individuals": len(patient_level_df),
            "total_chunks": len(analysis_df),
            "unique_languages": len(lang_dist),
        }

        # Log key findings
        self._log_dataset_findings(results)

        return results

    def analyze_prediction_language_bias(
        self,
        metadata_df: pd.DataFrame,
        predictions: Any,
        true_labels: Any,
        label_names: list[str],
        save_plots: bool = True,
    ) -> dict[str, Any]:
        """
        Analyze language bias in model predictions.

        Args:
            metadata_df: DataFrame with metadata including language
            predictions: Model predictions (same order as metadata_df)
            true_labels: True labels (same order as metadata_df)
            label_names: Names of the classes
            save_plots: Whether to save visualization plots

        Returns:
            Dictionary containing bias analysis results
        """
        logger.info("Analyzing prediction language bias...")

        # Create analysis dataframe
        analysis_df = metadata_df.copy()
        analysis_df["prediction"] = predictions
        analysis_df["true_label"] = true_labels
        analysis_df["language_clean"] = analysis_df["language"].fillna("unknown")

        # Filter out longitudinal data
        analysis_df = analysis_df[analysis_df.longitudinal == 0]

        # Overall prediction accuracy by language
        lang_accuracy = self._calculate_language_accuracy(analysis_df)

        # Confusion matrices by language
        lang_confusion_matrices = self._calculate_language_confusion_matrices(
            analysis_df, label_names
        )

        # Statistical tests for bias
        bias_tests = self._perform_bias_statistical_tests(analysis_df)

        # Prediction distribution analysis
        pred_dist_analysis = self._analyze_prediction_distributions(analysis_df)

        # Create visualizations
        if save_plots:
            self._plot_prediction_bias_analysis(
                analysis_df, lang_accuracy, lang_confusion_matrices, label_names
            )

        results = {
            "language_accuracy": lang_accuracy,
            "language_confusion_matrices": lang_confusion_matrices,
            "bias_statistical_tests": bias_tests,
            "prediction_distributions": pred_dist_analysis,
            "overall_accuracy": (
                analysis_df["prediction"] == analysis_df["true_label"]
            ).mean(),
        }

        # Log key findings
        self._log_prediction_findings(results)

        return results

    def generate_bias_report(
        self,
        dataset_analysis: dict[str, Any],
        prediction_analysis: dict[str, Any] | None = None,
        save_report: bool = True,
    ) -> str:
        """
        Generate a comprehensive bias analysis report.

        Args:
            dataset_analysis: Results from analyze_dataset_language_distribution
            prediction_analysis: Results from analyze_prediction_language_bias
            save_report: Whether to save the report to file

        Returns:
            Report text
        """
        logger.info("Generating comprehensive bias report...")

        report_lines = [
            "# Language Bias Analysis Report",
            "=" * 50,
            "",
            "## Dataset Language Distribution Analysis",
            "-" * 40,
            "",
        ]

        # Dataset analysis section
        report_lines.extend(self._format_dataset_analysis(dataset_analysis))

        # Prediction analysis section (if available)
        if prediction_analysis:
            report_lines.extend(["", "## Model Prediction Bias Analysis", "-" * 40, ""])
            report_lines.extend(self._format_prediction_analysis(prediction_analysis))

        # Conclusions and recommendations
        report_lines.extend(["", "## Conclusions and Recommendations", "-" * 40, ""])
        report_lines.extend(
            self._generate_conclusions_and_recommendations(
                dataset_analysis, prediction_analysis
            )
        )

        report_text = "\n".join(report_lines)

        if save_report:
            report_path = self.output_dir / "language_bias_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Bias report saved to {report_path}")

        return report_text

    def _analyze_source_language_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze language distribution by data source."""
        source_lang_crosstab = pd.crosstab(
            df["data_source"], df["language_clean"], margins=True
        )

        source_lang_pct = (
            pd.crosstab(df["data_source"], df["language_clean"], normalize="index")
            * 100
        )

        return {"crosstab": source_lang_crosstab, "percentages": source_lang_pct}

    def _calculate_language_accuracy(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate prediction accuracy by language."""
        lang_accuracy = {}

        for lang in df["language_clean"].unique():
            lang_data = df[df["language_clean"] == lang]
            if len(lang_data) > 0:
                accuracy = (lang_data["prediction"] == lang_data["true_label"]).mean()
                lang_accuracy[lang] = accuracy

        return lang_accuracy

    def _calculate_language_confusion_matrices(
        self, df: pd.DataFrame, label_names: list[str]
    ) -> dict[str, np.ndarray]:
        """Calculate confusion matrices by language."""
        lang_confusion_matrices = {}

        for lang in df["language_clean"].unique():
            lang_data = df[df["language_clean"] == lang]
            if len(lang_data) > 0:
                cm = confusion_matrix(
                    lang_data["true_label"],
                    lang_data["prediction"],
                    labels=range(len(label_names)),
                )
                lang_confusion_matrices[lang] = cm

        return lang_confusion_matrices

    def _perform_bias_statistical_tests(self, df: pd.DataFrame) -> dict[str, Any]:
        """Perform statistical tests for language bias."""
        tests = {}

        # Test if prediction accuracy differs significantly by language
        languages = df["language_clean"].unique()
        if len(languages) > 1:
            # Chi-square test for independence between language and correct prediction
            correct_pred = (df["prediction"] == df["true_label"]).astype(int)
            lang_correct_crosstab = pd.crosstab(df["language_clean"], correct_pred)

            if lang_correct_crosstab.shape == (len(languages), 2):
                res = chi2_contingency(lang_correct_crosstab)
                chi2_stat = res.statistic # type: ignore
                chi2_p = res.pvalue # type: ignore
                tests["accuracy_independence"] = {
                    "test": "chi2",
                    "statistic": chi2_stat,
                    "p_value": chi2_p,
                    "significant": chi2_p < 0.05,
                    "interpretation": "Prediction accuracy differs significantly by language"
                    if chi2_p < 0.05
                    else "No significant difference in accuracy by language",
                }

        # Test for specific language pairs if we have German and English
        if "de" in languages and "en" in languages:
            de_data = df[df["language_clean"] == "de"]
            en_data = df[df["language_clean"] == "en"]

            de_correct = (de_data["prediction"] == de_data["true_label"]).sum()
            de_total = len(de_data)
            en_correct = (en_data["prediction"] == en_data["true_label"]).sum()
            en_total = len(en_data)

            # Fisher's exact test for German vs English accuracy
            contingency_table = [
                [de_correct, de_total - de_correct],
                [en_correct, en_total - en_correct],
            ]

            try:
                fisher_result = fisher_exact(contingency_table)
                odds_ratio = fisher_result.statistic # type: ignore
                fisher_p = fisher_result.pvalue # type: ignore
                tests["german_vs_english"] = {
                    "test": "fisher_exact",
                    "odds_ratio": odds_ratio,
                    "p_value": fisher_p,
                    "significant": fisher_p < 0.05,
                    "interpretation": f"German accuracy significantly {'higher' if odds_ratio > 1 else 'lower'} than English"
                    if fisher_p < 0.05
                    else "No significant difference between German and English accuracy",
                }
            except Exception as e:
                logger.warning(f"Could not perform Fisher's exact test: {e}")

        return tests

    def _analyze_prediction_distributions(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze how predictions are distributed across languages."""
        # Prediction distribution by language
        pred_lang_crosstab = pd.crosstab(
            df["language_clean"], df["prediction"], margins=True
        )

        pred_lang_pct = (
            pd.crosstab(df["language_clean"], df["prediction"], normalize="index") * 100
        )

        return {
            "prediction_by_language_crosstab": pred_lang_crosstab,
            "prediction_by_language_percentages": pred_lang_pct,
        }

    def _plot_language_distribution(
        self, df: pd.DataFrame, crosstab: pd.DataFrame, percentages: pd.DataFrame
    ):
        """Create visualizations for language distribution analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Dataset Language Distribution Analysis", fontsize=16)

        # Get colors for languages present in data
        lang_counts = df["language_clean"].value_counts()
        colors = [self.custom_colors.get(lang, "#808080") for lang in lang_counts.index]

        # Overall language distribution
        axes[0, 0].pie(
            lang_counts.values,
            labels=lang_counts.index,
            autopct="%1.1f%%",
            colors=colors,
        )
        axes[0, 0].set_title("Overall Language Distribution")

        # Language by disease (counts)
        crosstab_plot = crosstab.iloc[:-1, :-1]  # Exclude margins
        plot_colors = [
            self.custom_colors.get(col, "#808080") for col in crosstab_plot.columns
        ]
        crosstab_plot.plot(kind="bar", ax=axes[0, 1], color=plot_colors)
        axes[0, 1].set_title("Language Distribution by Disease (Counts)")
        axes[0, 1].set_xlabel("Disease")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend(title="Language")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Language by disease (percentages)
        percentages.plot(kind="bar", stacked=True, ax=axes[1, 0], color=plot_colors)
        axes[1, 0].set_title("Language Distribution by Disease (Percentages)")
        axes[1, 0].set_xlabel("Disease")
        axes[1, 0].set_ylabel("Percentage")
        axes[1, 0].legend(title="Language")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Data source by language
        source_lang = pd.crosstab(df["data_source"], df["language_clean"])
        source_colors = [
            self.custom_colors.get(col, "#808080") for col in source_lang.columns
        ]
        source_lang.plot(kind="bar", ax=axes[1, 1], color=source_colors)
        axes[1, 1].set_title("Language Distribution by Data Source")
        axes[1, 1].set_xlabel("Data Source")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].legend(title="Language")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "dataset_language_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_prediction_bias_analysis(
        self,
        df: pd.DataFrame,
        lang_accuracy: dict[str, float],
        lang_confusion_matrices: dict[str, np.ndarray],
        label_names: list[str],
    ):
        """Create visualizations for prediction bias analysis."""
        # Accuracy by language
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Prediction Language Bias Analysis", fontsize=16)

        # Accuracy by language bar plot
        languages = list(lang_accuracy.keys())
        accuracies = list(lang_accuracy.values())

        bars = axes[0, 0].bar(languages, accuracies)
        axes[0, 0].set_title("Prediction Accuracy by Language")
        axes[0, 0].set_xlabel("Language")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        # Prediction distribution by language
        pred_dist = pd.crosstab(
            df["language_clean"], df["prediction"], normalize="index"
        )
        pred_dist.plot(kind="bar", stacked=True, ax=axes[0, 1])
        axes[0, 1].set_title("Prediction Distribution by Language")
        axes[0, 1].set_xlabel("Language")
        axes[0, 1].set_ylabel("Proportion")
        axes[0, 1].legend(title="Prediction", labels=label_names)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Sample sizes by language
        sample_sizes = df["language_clean"].value_counts()
        axes[1, 0].bar(sample_sizes.index, sample_sizes.values)
        axes[1, 0].set_title("Sample Sizes by Language")
        axes[1, 0].set_xlabel("Language")
        axes[1, 0].set_ylabel("Number of Samples")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Language vs True Label distribution
        true_dist = pd.crosstab(
            df["language_clean"], df["true_label"], normalize="index"
        )
        true_dist.plot(kind="bar", stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title("True Label Distribution by Language")
        axes[1, 1].set_xlabel("Language")
        axes[1, 1].set_ylabel("Proportion")
        axes[1, 1].legend(title="True Label", labels=label_names)
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "prediction_language_bias.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create confusion matrix plots for each language (if we have multiple languages)
        if len(lang_confusion_matrices) > 1:
            n_langs = len(lang_confusion_matrices)
            n_cols = min(3, n_langs)
            n_rows = (n_langs + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()

            for i, (lang, cm) in enumerate(lang_confusion_matrices.items()):
                if i < len(axes):
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=label_names,
                        yticklabels=label_names,
                        ax=axes[i],
                    )
                    axes[i].set_title(f"Confusion Matrix - {lang}")
                    axes[i].set_xlabel("Predicted")
                    axes[i].set_ylabel("True")

            # Hide unused subplots
            for i in range(len(lang_confusion_matrices), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "confusion_matrices_by_language.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def _format_dataset_analysis(self, analysis: dict[str, Any]) -> list[str]:
        """Format dataset analysis results for the report."""
        lines = []

        lines.append(f"Total samples analyzed: {analysis['n_individuals']}")
        lines.append(f"Unique languages found: {analysis['unique_languages']}")
        lines.append("")

        lines.append("Language distribution:")
        for lang, count in analysis["overall_language_distribution"].items():
            pct = (count / analysis["n_individuals"]) * 100
            lines.append(f"  {lang}: {count} ({pct:.1f}%)")
        lines.append("")

        lines.append("Statistical test for language-disease independence:")
        chi2_test = analysis["chi2_test"]
        lines.append(f"  Chi-square statistic: {chi2_test['statistic']:.4f}")
        lines.append(f"  p-value: {chi2_test['p_value']:.4f}")
        lines.append(f"  Significant: {'Yes' if chi2_test['significant'] else 'No'}")

        if chi2_test["significant"]:
            lines.append("  → Language and disease categories are NOT independent")
            lines.append(
                "  → This suggests potential confounding between language and disease"
            )
        else:
            lines.append("  → Language and disease categories appear independent")

        return lines

    def _format_prediction_analysis(self, analysis: dict[str, Any]) -> list[str]:
        """Format prediction analysis results for the report."""
        lines = []

        lines.append(f"Overall model accuracy: {analysis['overall_accuracy']:.3f}")
        lines.append("")

        lines.append("Accuracy by language:")
        for lang, acc in analysis["language_accuracy"].items():
            lines.append(f"  {lang}: {acc:.3f}")
        lines.append("")

        # Statistical tests
        if "bias_statistical_tests" in analysis:
            lines.append("Statistical tests for bias:")
            for test_name, test_result in analysis["bias_statistical_tests"].items():
                lines.append(f"  {test_name}:")
                lines.append(f"    Test: {test_result['test']}")
                lines.append(f"    p-value: {test_result['p_value']:.4f}")
                lines.append(
                    f"    Significant: {'Yes' if test_result['significant'] else 'No'}"
                )
                lines.append(f"    Interpretation: {test_result['interpretation']}")
                lines.append("")

        return lines

    def _generate_conclusions_and_recommendations(
        self,
        dataset_analysis: dict[str, Any],
        prediction_analysis: dict[str, Any] | None,
    ) -> list[str]:
        """Generate conclusions and recommendations based on the analysis."""
        lines = []

        # Dataset-level conclusions
        chi2_significant = dataset_analysis["chi2_test"]["significant"]

        if chi2_significant:
            lines.append("⚠️  POTENTIAL LANGUAGE BIAS DETECTED:")
            lines.append(
                "   - Language and disease categories are not independent in the dataset"
            )
            lines.append(
                "   - This creates risk of the model learning language patterns instead of disease patterns"
            )
            lines.append("")
        else:
            lines.append("✅ Dataset language distribution appears balanced:")
            lines.append(
                "   - No significant association between language and disease categories"
            )
            lines.append("")

        # Prediction-level conclusions
        acc_diff = 0.0  # Initialize default value
        if prediction_analysis and prediction_analysis["language_accuracy"]:
            accuracies = list(prediction_analysis["language_accuracy"].values())
            if accuracies:  # Check if we have any accuracies
                max_acc = max(accuracies)
                min_acc = min(accuracies)
                acc_diff = max_acc - min_acc

                if acc_diff > 0.1:  # 10% difference threshold
                    lines.append("⚠️  SIGNIFICANT ACCURACY DIFFERENCES BY LANGUAGE:")
                    lines.append(
                        f"   - Accuracy range: {min_acc:.3f} to {max_acc:.3f} (difference: {acc_diff:.3f})"
                    )
                    lines.append(
                        "   - This suggests the model may be biased toward certain languages"
                    )
                    lines.append("")
                else:
                    lines.append("✅ Model accuracy is consistent across languages:")
                    lines.append(
                        f"   - Accuracy range: {min_acc:.3f} to {max_acc:.3f} (difference: {acc_diff:.3f})"
                    )
                    lines.append("")

            # Check statistical tests
            if "bias_statistical_tests" in prediction_analysis:
                significant_tests = [
                    test
                    for test, result in prediction_analysis[
                        "bias_statistical_tests"
                    ].items()
                    if result["significant"]
                ]

                if significant_tests:
                    lines.append("⚠️  STATISTICAL EVIDENCE OF LANGUAGE BIAS:")
                    for test in significant_tests:
                        result = prediction_analysis["bias_statistical_tests"][test]
                        lines.append(f"   - {result['interpretation']}")
                    lines.append("")

        return lines

    def _log_dataset_findings(self, results: dict[str, Any]):
        """Log key findings from dataset analysis."""
        logger.info("=== DATASET LANGUAGE ANALYSIS FINDINGS ===")
        logger.info(f"Total samples: {results['n_individuals']}")
        logger.info(f"Unique languages: {results['unique_languages']}")

        chi2_test = results["chi2_test"]
        logger.info(f"Language-disease independence test: p={chi2_test['p_value']:.4f}")

        if chi2_test["significant"]:
            logger.warning(
                "⚠️  Language and disease are NOT independent - potential bias risk!"
            )
        else:
            logger.info("✅ Language and disease appear independent")

    def _log_prediction_findings(self, results: dict[str, Any]):
        """Log key findings from prediction analysis."""
        logger.info("=== PREDICTION LANGUAGE BIAS FINDINGS ===")
        logger.info(f"Overall accuracy: {results['overall_accuracy']:.3f}")

        accuracies = list(results["language_accuracy"].values())
        languages = list(results["language_accuracy"].keys())

        for lang, acc in zip(languages, accuracies):
            logger.info(f"Accuracy for {lang}: {acc:.3f}")

        acc_diff = max(accuracies) - min(accuracies)
        if acc_diff > 0.1:
            logger.warning(
                f"⚠️  Large accuracy difference across languages: {acc_diff:.3f}"
            )
        else:
            logger.info(
                f"✅ Consistent accuracy across languages (diff: {acc_diff:.3f})"
            )


def analyze_language_bias_from_files(
    metadata_path: str,
    predictions_path: str | None = None,
    output_dir: str = "outputs/language_analysis",
) -> dict[str, Any]:
    """
    Convenience function to run language bias analysis from file paths.

    Args:
        metadata_path: Path to metadata CSV file
        predictions_path: Path to predictions CSV file (optional)
        output_dir: Output directory for analysis results

    Returns:
        Dictionary containing all analysis results
    """
    analyzer = LanguageBiasAnalyzer(output_dir)

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata with {len(metadata_df)} records")

    # Analyze dataset
    dataset_results = analyzer.analyze_dataset_language_distribution(metadata_df)

    prediction_results = None
    if predictions_path and Path(predictions_path).exists():
        # Load predictions (assuming CSV with columns: prediction, true_label)
        pred_df = pd.read_csv(predictions_path)

        # Ensure same length
        if len(pred_df) == len(metadata_df):
            prediction_results = analyzer.analyze_prediction_language_bias(
                metadata_df,
                pred_df["prediction"].values,
                pred_df["true_label"].values,
                label_names=["control", "copd"],  # Adjust as needed
            )
        else:
            logger.warning(
                f"Prediction file length ({len(pred_df)}) doesn't match metadata ({len(metadata_df)})"
            )

    # Generate comprehensive report
    report = analyzer.generate_bias_report(dataset_results, prediction_results)

    return {
        "dataset_analysis": dataset_results,
        "prediction_analysis": prediction_results,
        "report": report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze language bias in voice biomarker data"
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV file")
    parser.add_argument("--predictions", help="Path to predictions CSV file (optional)")
    parser.add_argument(
        "--output_dir", default="outputs/language_analysis", help="Output directory"
    )

    args = parser.parse_args()

    results = analyze_language_bias_from_files(
        args.metadata, args.predictions, args.output_dir
    )

    print("Language bias analysis completed!")
    print(f"Results saved to: {args.output_dir}")
