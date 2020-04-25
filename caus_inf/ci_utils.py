"""
Utiltiies for the causal inference/eicu project
"""

# See dpson/eicu_preproc/save_model_inputs.py
g_pt_endpoints = ["full_score_1", "full_score_6",
        "full_score_12", "full_score_24",
        "hospital_discharge_expired_1", "hospital_discharge_expired_6",
        "hospital_discharge_expired_12", "hospital_discharge_expired_24",
        "unit_discharge_expired_1", "unit_discharge_expired_6",
        "unit_discharge_expired_12", "unit_discharge_expired_24"]

# Prefixes used in the batch files
g_prefixes = { "vitalPeriodic" : "vs",
        "vitalAperiodic" : "avs",
        "lab" : "lab" }

# Included variable file names
g_includes = { "vitalPeriodic" : "included_per_variables.txt",
        "vitalAperiodic" : "included_aper_variables.txt",
        "lab" : "included_lab_variables.txt" }

