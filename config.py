"""
==========================================================
config.py

Global Configuration File

Project:
AI-based Explainable Multimodal Cattle Health Monitoring

==========================================================
"""

# -----------------------------------------
# Random Seed
# -----------------------------------------

RANDOM_SEED = 42

# -----------------------------------------
# Number of Cattle
# -----------------------------------------

NUM_CATTLE = 100

# -----------------------------------------
# Number of Simulation Days
# -----------------------------------------

SIMULATION_DAYS = 60

# -----------------------------------------
# Sampling Interval
# -----------------------------------------

SAMPLING_INTERVAL_MIN = 10

# 10 minutes according to literature

SAMPLES_PER_DAY = int(24 * 60 / SAMPLING_INTERVAL_MIN)

TOTAL_SAMPLES = SIMULATION_DAYS * SAMPLES_PER_DAY

# -----------------------------------------
# Healthy Environmental Range
# -----------------------------------------

AMBIENT_TEMP_RANGE = (18, 28)

HUMIDITY_RANGE = (45, 75)

# -----------------------------------------
# THI Threshold
# -----------------------------------------

THI_SAFE = 72
THI_MODERATE = 79
THI_SEVERE = 89

# -----------------------------------------
# Output Folder
# -----------------------------------------

OUTPUT_FOLDER = "generated"

# -----------------------------------------
# File Names
# -----------------------------------------

PROFILE_FILE = "generated/cow_profiles.csv"

BASELINE_FILE = "generated/healthy_baseline.csv"