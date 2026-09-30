"""
Configuration constants and mappings for metadata extraction.
Centralizes all hardcoded values, column mappings, and patterns.
"""

import re
from typing import Any

# Excel sheet names
EXCEL_SHEETS = {
    "main": "Bereinigt",
    "single_recordings": "Details einmalige Aufnahmen",
    "longitudinal_recordings": "Details long. Aufnahmen",
}

# Column name mappings for main data sheet
MAIN_COLUMN_MAPPING = {
    "studien-id": "study_id",
    "audio-id": "audio_id",
    "geburtsjahr": "birth_year",
    "sex": "sex",
    "körpergröße(cm)": "height_cm",
    "gewicht(kg)": "weight_kg",
    "bmi": "bmi",
    "Ready_to_Use": "ready_to_use",
    "lungenerkrankung-haupt": "lung_disease_main",
    "lungenerkrankung2": "lung_disease_2",
    "lungenerkrankung3": "lung_disease_3",
    "lungenerkrankung4": "lung_disease_4",
    "controll": "control",
    "longitudinal": "longitudinal",
    "copd:exazerbiert?": "copd_exacerbated",
    "copdexazerbationschwer": "copd_exacerbation_severe",
    "copdexazerbationmittel": "copd_exacerbation_moderate",
    "copdexazerbationleicht": "copd_exacerbation_mild",
    "copd:goldi-iv?": "copd_gold_stage",
    "copd:gruppea/b/e?": "copd_group_a_b_e",
    "packyears": "pack_years",
}

# Column mappings for single recordings sheet
SINGLE_RECORDINGS_COLUMN_MAPPING = {
    "studien-id": "study_id",
    "datum": "date",
}

# Column mappings for longitudinal recordings sheet
LONGITUDINAL_COLUMN_MAPPING = {
    "studien-id": "study_id",
    "audio-id": "audio_id",
}

# Disease name standardization mapping
DISEASE_MAPPING = {
    "asthma": "asthma",
    "keine bekannt": "control",
    "restriktion": "restriction",
    "copd": "copd",
    "lungenfibrose": "fibrosis",
    "ohs": "ohs",
    "ph": "ph",
    "cteph": "cteph",
    "sarkoidose": "sarcoidosis",
    "exogen-allergische alveolitis": "hypersensitivity_pneumonitis",
    "granulomatose mit polyangiitis": "granulomatosis_w_polyangiitis",
    "pneumonie influenza a": "influenza_a_pneumonia",
    "bronchiektasen": "bronchiectasis",
    "lungenemphysem": "emphysema",
    "z.n. mykoplasmeninfektion": "post_mycoplasma_infection",
    "lungen-ca": "lung_cancer",
}

# Columns that contain disease information
DISEASE_COLUMNS = [
    "lung_disease_main",
    "lung_disease_2",
    "lung_disease_3",
    "lung_disease_4",
]

# Columns to drop from main data
COLUMNS_TO_DROP = [
    "copd_exacerbation_severe",
    "copd_exacerbation_moderate",
    "copd_exacerbation_mild",
]

# Missing value representations to standardize
MISSING_VALUE_REPRESENTATIONS = [
    "nicht vorhanden",
    "unbekannt",
    "nan",
    "/",
    "",
    " ",
    "null",
    "NULL",
    "NaN",
    "NAN",
]

# Columns that should be converted to numeric
NUMERIC_COLUMNS = ["birth_year", "height_cm", "weight_kg", "longitudinal"]

# Audio file patterns
AUDIO_FILE_PATTERNS = {
    "single_recording": re.compile(r"\d{1,4}_[A-Za-z]+\.wav$"),
    "longitudinal_recording": re.compile(r"\d{1,4}_[A-Za-z]+_\d{8}\.wav$"),
}

# Date format configurations
DATE_FORMATS = {
    "input_format": "%d/%m/%Y",
    "output_format": "%Y-%m-%d",
    "filename_format": "%Y%m%d",
}

# Longitudinal recording date column pattern
LONGITUDINAL_DATE_COLUMNS = [f"datumstimmaufnahme{i}" for i in range(1, 10)]
LONGITUDINAL_TIME_COLUMNS = [f"uhrzeitstimmaufnahme{i}" for i in range(1, 10)]

# Data type configurations for validation
DATA_TYPE_CONFIG = {
    "audio_id": "int",
    "study_id": "str",
    "birth_year": "float",
    "height_cm": "float",
    "weight_kg": "float",
    "longitudinal": "int",
    "control": "int",
    "ready_to_use": "int",
}

# Validation rules
VALIDATION_RULES = {
    "audio_id_positive": True,
    "no_question_marks": True,
    "control_group_consistency": True,
    "unique_recording_identifiers": True,
}
