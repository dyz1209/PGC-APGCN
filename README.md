# PGC-APGCN
## Overview
The project code for fault diagnosis of wind turbine gearbox using perception graph construction and adaptive pruning graph convolutional network is designed with a modular structure, which facilitates maintenance and expansion. In this section, we demonstrate the performance of this method using the publicly available SEU 20-0 dataset. Due to privacy and security concerns, the data from the Tianqiaoshan wind farm cannot be disclosed here.

## Prerequisites
This project requires the following Python libraries. Please ensure they are installed in your virtual environment:

| Library           | Version (Verified)   | Purpose                      | Installation Command              |
|-------------------|----------------------|------------------------------|-----------------------------------|
| `torch`           | 2.8.0+cpu            | Core deep learning framework  | `pip install torch==2.8.0+cpu`    |
| `torch-geometric` | 2.6.1                | GCN model implementation      | Please refer to PyG official site |
| `pandas`          | 2.3.1                | Data processing               | `pip install pandas==2.3.1`       |
| `numpy`           | 2.1.3                | Scientific computing          | `pip install numpy==2.1.3`        |
| `scikit-learn`    | 1.7.0                | PCA, normalization, K-NN (BallTree) | `pip install scikit-learn==1.7.0` |
| `matplotlib`      | 3.10.6               | Plotting (PCA, confusion matrix) | `pip install matplotlib==3.10.6`  |
| `seaborn`         | 0.13.2               | Confusion matrix visualization | `pip install seaborn==0.13.2`     |
| `psutil`          | 7.1.0                | Memory monitoring             | `pip install psutil==7.1.0`       |


## Project Structure
The project is modular and consists of the following four main files:

- **`config.py`**: Stores global configuration constants (such as `FILE_PATH`, `EPOCHS`, `LEARNING_RATE`, etc.) and memory monitoring tools.
- **`features.py`**: Feature engineering module that handles data loading, time-series segmentation, normalization, and PCA dimensionality reduction based on variance contribution.
- **`graph_model.py`**: Core model that includes the construction of K-NN graphs by class, feature pre-enhancement of neighbors, and the definition of a two-layer GCN model.
- **`main.py`**: Execution and training module that coordinates all other modules, defines the training loop, early stopping mechanism, and final evaluation.
- **`gearset20_0.csv`**: Dataset file (required).

## Usage

1. **Data Preparation**: Name the time-series dataset file `gearset20_0.csv` and place it in the root directory of the project.
2. **Configuration Check**: Check and modify parameters in `config.py` as needed, especially `EPOCHS` (set to 200 to match early stopping) and `STEP_LENGTH` (set to 1024).
3. **Run the Project**: After activating the virtual environment in the terminal, run the main execution file:
   ```bash
   python main.py
