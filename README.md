# GNN-SP (Graph Neural Network for Survival Path)

This repository contains the code for the GNN-SP pipeline, a Heterogeneous Graph Neural Network approach for survival analysis and prediction, which includes a feature update module via graph-based message passing and a prediction task module.

## 1. System requirements

### Operating Systems
The software has been tested on and is compatible with:
* macOS (tested on macOS 13+)
* Linux (Ubuntu 20.04/22.04)
* Windows 10/11

### Software Dependencies
The pipeline requires **Python 3.8+**. The core dependencies and their tested versions include:
* `torch` (tested on 2.0.0+)
* `torch_geometric` (tested on 2.3.0+)
* `pandas` (tested on 1.5.0+)
* `numpy` (tested on 1.23.0+)
* `scikit-learn` (tested on 1.2.0+)
* `lifelines` (tested on 0.27.0+)

### Non-standard Hardware
* No non-standard hardware is strictly required. 
* **Optional but recommended:** A CUDA-enabled GPU (NVIDIA) for faster graph neural network training. The code will automatically detect and utilize CUDA if available (`torch.device('cuda')`); otherwise, it will default to CPU.

## 2. Installation guide

### Instructions
1. Clone or download the repository to your local machine.
2. It is highly recommended to create a virtual environment (e.g., using `conda` or `venv`):
   ```bash
   conda create -n gnn_sp_env python=3.9
   conda activate gnn_sp_env
   ```
3. Install the required packages using `pip`:
   ```bash
   pip install pandas numpy scikit-learn lifelines
   ```
4. Install PyTorch and PyTorch Geometric according to your specific OS and CUDA version. For example (CPU version):
   ```bash
   pip install torch torchvision torchaudio
   pip install torch_geometric
   ```

### Typical Install Time
On a "normal" desktop computer with a standard internet connection, the installation process typically takes **5 to 10 minutes**.

## 3. Demo

### Instructions to run on data
Before running the demo, ensure you update the file paths in the script to match your local environment:
* Line 16: Update the working directory path `os.chdir(...)` to where your code and data are located.
* Line 35: Update the `result_root` path to where you want the output saved.

To run the demo, execute the Python script:
```bash
python "GNN-SP code.py"
```

### Expected Output
The expected output files will be automatically generated and saved in your designated **`Results`** folder (e.g., `/Users/.../Desktop/Results/`). The outputs include:
* `results_summary.xlsx`: A comprehensive summary of the model's C-index performance across datasets.
* `[task_name]_best_cindex_final_best.xlsx`: The best C-index metrics with 95% Confidence Intervals for train, validation, and test sets.
* `[task_name]_cindex_epoch_history_final_best.xlsx`: Epoch-by-epoch training history.
* `[task_name]_test_survival_risk_final_best.csv`: Patient-level predicted survival risk scores.

### Expected Run Time
For the provided demo dataset (ts9 data: `f1t1f2t2f3456789.xlsx`), the expected run time on a "normal" desktop computer is approximately **2 minutes**. 
*(Note: Running other time-step data (ts1-ts8) may take longer, as earlier time-steps typically contain a larger volume of patient data).*

## 4. Instructions for use

### How to run the software on your data
1. **Data Format:** Prepare your clinical data in an Excel (`.xlsx`) format. Ensure your columns match the expected clinical features defined in the code (e.g., `ID`, `BCLC`, `duration`, `is_death`, `5year-S`, and respective time-step features like `1st_ViableLesion`, `1st_resection`).
2. **Running specific Time-Steps (ts1 - ts9):** The code is designed to seamlessly run data from ts1 through ts9, provided the data files are placed in the correct working directory. 
   * **Running Demo Data Only:** Currently, around **Line 431**, the script is set up to only run the demo data (ts9):
       ```python
       filenames = ["f1t1f2t2f3456789.xlsx"]
       ```
   * **Running All Data:** If you have prepared the complete datasets for all time-steps, you can run them sequentially by **deleting the `#` symbols** to uncomment the code block around **Lines 426-430**:
       ```python
       filenames = [
           "f1t1.xlsx", "f1t1f2t2.xlsx", "f1t1f2t2f3.xlsx",
           "f1t1f2t2f34.xlsx", "f1t1f2t2f345.xlsx", "f1t1f2t2f3456.xlsx",
           "f1t1f2t2f34567.xlsx", "f1t1f2t2f345678.xlsx", "f1t1f2t2f3456789.xlsx"
       ]
       ```

### Reproduction instructions (Optional)
To reproduce the exact quantitative results presented in the manuscript, do not alter the random seed initialization block (Lines 42-47). The seed is fixed at `20250715`, which ensures that the stratified data splits, model parameter initialization, and bootstrap C-index calculations remain deterministic across runs.
