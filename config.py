# config.py

import psutil
from matplotlib import rcParams

# --- Global Configuration Constants ---
FILE_PATH = 'gearset20_0.csv'
STEP_LENGTH = 1024
KNN_K = 5
EPOCHS = 200
LEARNING_RATE = 0.01

# --- Utility Tools ---
def print_memory_usage(stage):
    """Prints the current memory usage of the process."""
    process = psutil.Process()
    memory_info = process.memory_info()
    # Prints the Resident Set Size (RSS) memory usage in MB
    print(f'{stage} - Memory Usage: {memory_info.rss / 1024 ** 2:.2f} MB')

# Set matplotlib font to support Chinese characters (SimHei)
rcParams['font.sans-serif'] = ['SimHei']
# Resolve the issue where the negative sign is displayed incorrectly
rcParams['axes.unicode_minus'] = False