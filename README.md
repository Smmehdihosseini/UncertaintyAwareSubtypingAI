# Uncertainty-Aware AI for Tumor Subtyping with Histology and Immunohistochemistry: A Multi-Center Study in Renal Cell Carcinoma

![Python](https://img.shields.io/badge/Python-3.9-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.2-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Research--Code-orange)

This is the official repository for the paper:

*"Uncertainty-Aware AI for Tumor Subtyping with Histology and Immunohistochemistry: A Multi-Center Study in Renal Cell Carcinoma"*.

## Table of Contents

- [Installation Guide](#installation-guide)
  - [1. Clone the Repository](#clone-the-repository)
  - [2. Install CUDA](#install-cuda)
  - [3. Create a Virtual Environment](#create-a-virtual-environment)
    - [Option 1: Using venv](#option-1-using-venv)
    - [Option 2: Using conda](#option-2-using-conda)
    - [Option 3: Using pyenv + pyenv-virtualenv](#option-3-using-pyenv--pyenv-virtualenv)
  - [4. Install Python Packages](#install-python-packages)
  - [5. Set Environment Variables](#set-environment-variables)
    - [CUDA Device Selection](#cuda-device-selection)
    - [CPU Thread Control](#cpu-thread-control)
    - [OpenSlide (Windows Only)](#openslide-windows-only)
  - [6. File Requirements](#file-requirements)
    - [Patients Metadata File](#patients-metadata-file)
    - [Tree Pair Classes](#tree-pair-classes)
    - [Configuration File](#configuration-file)

- [Usage](#usage)
  - [A. Prepare Data](#prepare-data)
  - [B. Training the Model](#training-the-model)
  - [C. Set Hyperparameters](#set-hyperparameters)
  - [D. WSIs Evaluation](#wsis-evaluation)
  - [E. IHC Module](#ihc-module)
  - [F. Framework Evaluation: Pre vs Post IHC Analysis](#framework-evaluation)

- [License](#license)
- [Citation](#citation)
- [Contacts](#contacts)

## Installation Guide <a id="installation-guide"></a>

This guide explains how to set up the environment and install the required dependencies for the project.

### 1. Clone the Repository <a id="clone-the-repository"></a>

First, clone the repository from GitHub and move into the project directory:

```bash
git clone https://github.com/ُSmmehdihosseini/UncertaintyAwareSubtypingAI.git
cd UncertaintyAwareSubtypingAI
```

### 2. Install CUDA <a id="install-cuda"></a>

This project requires **CUDA 11.2** for GPU acceleration. Make sure CUDA is installed and accessible before installing packages and running any training or evaluation scripts. You can verify your CUDA installation with:

You can verify your CUDA installation with:

```bash
nvcc --version
```

or

```bash
nvidia-smi
```

If CUDA is not installed, download and install CUDA Toolkit 11.2 from the official NVIDIA website:

https://developer.nvidia.com/cuda-11.2.0-download-archive

After installation, verify that CUDA is available in your system `PATH`.

> **Note:**  
> The Python dependencies listed in requirements.txt (e.g., TensorFlow) are compiled to work with **CUDA 11.2**. Installing a different CUDA version may lead to compatibility issues. 

### 3. Create a Virtual Environment <a id="create-a-virtual-environment"></a>

It is recommended to use a Python virtual environment to avoid dependency conflicts:

#### Option 1: Using `venv` <a id="option-1-using-venv"></a>

Create the environment:

```bash
python3 -m venv uat
```

Activate the environment:

**Linux / macOS**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

#### Option 2: Using `conda` <a id="option-2-using-conda"></a>

Create a new environment with a specific Python version:

```bash
conda create -n uat python=3.9.18
```
Activate the environment:

```bash
conda activate uat
```

#### Option 3: Using `pyenv + pyenv-virtualenv` <a id="option-3-using-pyenv--pyenv-virtualenv"></a>

Install the Python version (if not already installed):

```bash
pyenv install 3.9.18
```
Create a virtual environment:

```bash
pyenv virtualenv 3.9.18 uat
```
Activate the environment:

```bash
pyenv activate uat
```

### 4. Install Python packages <a id="install-python-packages"></a>

After activating the virtual environment, install the required Python packages using the `requirements.txt` file.

Install dependencies:

```bash
pip install -r requirements.txt
```
Verify the installation (optional):

```bash
pip list
```

> **Note about OpenSlide (Windows Users):**  
> If you are using Windows, installing the Python `openslide` package with `pip` is not sufficient by itself. You must also download the **OpenSlide binary libraries** (`openslide-win64`) and add the `bin` directory to your system `PATH`.  
>
> 1. Download the Windows binaries from: https://openslide.org/download/  
> 2. Extract the archive (e.g., `openslide-win64-xxxx`).  
> 3. Add the `bin` folder (e.g., `openslide-win64/bin`) to your system `PATH`, or load it in Python before importing `openslide` (See See [OpenSlide (Windows Only)](#openslide-windows-only)).

### 5. Set Environment Variables <a id="set-environment-variables"></a>

Some scripts rely on environment variables to control GPU visibility and runtime configuration. These variables should be set before running any training or evaluation scripts.

#### Cuda Device Selection <a id="cuda-device-selection"></a>
The `CUDA_VISIBLE_DEVICES` variable specifies which GPUs are accessible to the framework:

```bash
export CUDA_VISIBLE_DEVICES=0
```

To enable multiple GPUs:
```bash
export CUDA_VISIBLE_DEVICES=0,1
```

> **Note:**  
> The number of GPUs specified here should match the number of values defined in the `gpu_memory` field inside `_info/config.json`.
>
> For example:
> ```bash
> export CUDA_VISIBLE_DEVICES=0,1
> ```
> with `"gpu_memory": "4,8"`
>
> means that GPU 0 will allocate 4 GB and GPU 1 will allocate 8 GB.

#### CPU Thread Control <a id="cpu-thread-control"></a>


To avoid excessive CPU usage from numerical libraries (NumPy, MKL, OpenBLAS, etc.), the following environment variables can be set to limit the number of threads according to your system specifications:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

These variables help ensure stable CPU utilization and prevent oversubscription during data loading and preprocessing.

#### OpenSlide (Windows Only) <a id="openslide-windows-only"></a>

If using Windows, the OpenSlide binary path may need to be specified so that Python can locate the required dynamic libraries:

```bash
set OPENSLIDE_PATH=path\to\openslide-win64-xxxx\bin
```

> **Note:**  
> Make sure that the directory contains the OpenSlide bin files distributed with the official Windows release.

### 6. File Requirements <a id="file-requirements"></a>

#### 1. Patients Metadata File <a id="patients-metadata-file"></a>

This file includes the list of all patient IDs to be considered in training, validation, and testing phases (See [Table. 1](#ids_table_example)) his file should be located in the `_info/ids.csv` directory and should include the following information:

- **subtype**: Specifies the identified patient subtype. The valid values are:
    - **ccRCC**
    - **pRCC**
    - **CHROMO**
    - **ONCOCYTOMA**
    
- **roi_exist**: Indicating whether annotations exist for the specific patient. If no annotations exist, the corresponding patient will automatically be assigned to the test set. The valid values are:
    - **TRUE**
    - **FALSE**

- **center**: The institution or medical center from which the patient originates.

- **batch**: Indicates the cohort to which a patient belongs. This field allows patients to be grouped and makes it easy to include or exclude entire cohorts during data preparation.

  > **Note:**  
  > Each slide must include the ID of the corresponding patient into its name, e.g. `HPA1.28.A1.svs`.

**Table 1. `_info/ids.csv` Example:** <a id="ids_table_example"></a>

| id | subtype | roi_exist | center | batch |
| --- | --- | --- | --- | --- |
| HPA1 | ccRCC | True | CenterA | 1 |
| HPA2 | pRCC | True | CenterA | 1 |
| HPA3 | CHROMO | False | CenterA | 1 |
| HPA4 | ONCOCYTOMA | False | CenterA | 1 |
| ... | ... | ... | ... | ...

#### 2. Tree Pair Classes:  <a id="tree-pair-classes"></a>

This file lists all possible classes that may appear at different stages of MCExpertDT, such as having `ccRCC` and `pRCC` in the first class of the Node stage while `CHROMO` and `ONCOCYTOMA` are in the second class. Non-tumor classes should also be included. The file should be located in the `_info/tree_pair_dict.json` directory by default. Here is an example:

```json
{
    "Root": {
        "class_0": [
            "normal"
        ],
        "class_1": [
            "ccRCC",
            "pRCC",
            "CHROMO",
            "ONCOCYTOMA"
        ]
    },
    "Node": {
        "class_0": [
            "ccRCC",
            "pRCC"
        ],
        "class_1": [
            "CHROMO",
            "ONCOCYTOMA"
        ]
    },
    "Leaf1": {
        "class_0": [
            "CHROMO"
        ],
        "class_1": [
            "ONCOCYTOMA"
        ]
    },
    "Leaf2": {
        "class_0": [
            "ccRCC"
        ],
        "class_1": [
            "pRCC"
        ]
    }
}
```

#### 3. Configuration File:  <a id="configuration-file"></a>

This file contains all directories used occasionally in the scripts. It has be located in the `_info/config.json` directory of the repository. The required arguments are as follows:

- **gpu_memory**: GPU memory allocation for each GPU device used by the framework and previously set in ` CUDA_VISIBLE_DEVICES`.   The value must be provided as a **string**, where each GPU memory allocation is separated by commas. Each number corresponds to the memory (in GB) reserved for a specific GPU.  
  For example, `"4"` means 4 GB will be allocated for a single GPU, while `"4,8"` means 4 GB for GPU 0 and 8 GB for GPU 1.

- **wsis_dir**: Main directory containing all Whole Slide Images (WSIs). The dataset should be organized in this directory according to the expected folder structure used by the framework. Inside this directory, there should be separate folders for each subtype (e.g., ccRCC). Within each subtype folder, there must be a corresponding ccRCC_xml folder that contains the XML annotation files for the WSIs located in that subtype’s parent directory.

- **info_dir**: Directory containing metadata and configuration files related to the dataset, cases ids, models parameters, etc.

- **dfs_dir**: Directory used to store DataFrame files generated during data preparation. These files typically contain information about image patches, patch locations, and dataset splits used during training and validation.

- **weights_dir**: Directory where MCExpertDT model weights and checkpoints are saved during training and later loaded during inference or evaluation.

- **cache_dir**: Directory used to cache intermediate data (such as extracted crops or processed patches) to significantly improve data loading speed during training and inference.

- **results_dir**: Directory where experiment outputs and model evaluation results are stored. This may include WSI-level predictions, crop-level predictions, and experiment summaries.

- **figures_dir**: Directory where generated figures and visualizations (such as prediction maps or intermediate model outputs) are saved.

- **preds_dir**: Directory where the prediction outputs of the model are saved as NumPy arrays. These files can later be used for post-processing, analysis, or generating annotation files.

- **log_dir**: Directory where training and inference logs are stored. These logs typically include experiment configurations, runtime messages, and debugging information.

```json
{
        "gpu_memory": "4,8",
        "wsis_dir": "/path/to/dataset/wsis",
        "info_dir": "/path/to/folder/_info",
        "dfs_dir": "/path/to/folder/_dataframes",
        "weights_dir": "/path/to/folder/_weights",
        "cache_dir": "/path/to/folder/_cache",
        "results_dir": "/path/to/folder/_results",
        "figures_dir": "/path/to/folder/_figures",
        "preds_dir": "/path/to/folder/_predictions",
        "log_dir": "/path/to/folder/_logs"
}
```

## Usage <a id="sec.usage"></a>

### A. Prepare Data <a id="prepare-data"></a>

1. First, please refer to the [Installation Guide](#installation-guide) section and make sure to you have already done all the steps to get started.
2. Open `prepare_data.py` and review all arguments, modifying them according to your requirements and system limitations or directly address them during runtime. Then run:
    ```bash
    python prepare_data.py
    ```

    Upon execution, `dfs_dir` directory will include folders for each fold (e.g., `Fold1`, `Fold2`, ...). Each fold contains the training and validation splits as well as stage-specific crop files generated for that fold.

    Inside each fold directory, the following files are created:

    - **FoldX_train_df.csv:** Contains detailed information for all crops designated for **training** in fold `X`.
    - **FoldX_val_df.csv:** Contains detailed information for all crops designated for **validation** in fold `X`.

    Additionally, stage-specific training datasets are generated for each fold:

    - **FoldX_Root.csv:** Contains all **training** crops used for the **Root** stage in fold `X`.
    - **FoldX_Node.csv:** Contains all **training** crops used for the **Node** stage in fold `X`.
    - **FoldX_Leaf1.csv:** Contains all **training** crops used for the **Leaf 1** stage in fold `X`.
    - **FoldX_Leaf2.csv:** Contains all **training** crops used for the **Leaf 2** stage in fold `X`.

    Finally, the following file remains at the main directory level (outside the fold folders):

    - **crops_df.csv:** Contains all available crops information that can be used for training and validation.

### B. Train the Model <a id="train-the-model"></a>

3. You can now train each stage of MCExpertDT. This requires running train.py separately for each stage and each fold. You can set `--cross_val` field to the fold id (recommended to set also the `--weights_id` argument to the fold id) and later set the `--stage` argumen to **Root**, **Node**, **Leaf1** and **Leaf2**. Review all script arguments before execution and modify them if necessary:

    ```bash
    python train.py --stage <stage_value> --cross_val FoldX --weights_id FoldX --model_type mc
    ```

    - After training all the folds and stages, you can find the model checkpoints under `Weights_dir` directory.

### C. Set Hyperparameters <a id="set-hyperparameters"></a>

4. First, you should evaluate the stage-level models on the training crops to get the uncertainty threshold. Set a proper name `--runtime_id` for evaluation results. The MC settings will be read by this id from `model_params.json` file. Run:

    ```bash
    python eval_crops.py --runtime_id <runtime_id>
    ```
    > **Note:**  
    A good naming convention for `--runtime_id` could be like e.g. `Fold1_mc10`.

5. Open `eval_crops.ipynb` and find the '**Crop Analysis**' section. Replace the name of the `eval_id` with the one you specified in previous step, also choose the stage as `stage_eval` that you want to get the best threshold and metric. Execute all the cells and get the results. Head to last section '**Write Parameters Based on Above Evaluations**. You can now fill properly these parameters to include in your `model_params.json` file to be used for further experiments.

### D. WSIs Evaluation <a id="wsis-evaluation"></a>

6. Open `eval_slide.py` file review the arguments. You will find them almost similar to the previous steps. Run the script for `Train` slides. Also set the `--runtime_id` to the one you specified in previous step (Try to do this so you have consistent naming among different steps and avoid confusion). :

    ```bash
    python eval_slide.py --eval_cases Train --runtime_id <runtime_id>
    ```

    > **Note:**  
    As you can see here there is a `--stain_transfer` argument in the eval_slide script. This repository will be updated soon regarding the cyclegan training, data preparation, etc.

### E. IHC Module <a id="ihc-module"></a>

7. Find the `./ihc` folder and `ihc_model.ipynb` notebook to do train and validate and the IHC module. The optimal model resulted in the grid search have to be used in the `framework.ipynb` notebook for a full framework analysis.

### F. Framework Evaluation: Pre vs Post IHC Analysis <a id="framework-evaluation"></a>

8. Open `framework.ipynb` and follow the steps to load the results obtained previously in each module of framework and set up the confidence score. Later on, you will have the specified results of pre vs post IHC module.

##  License  <a id="license"></a>

This project is released under the MIT License. You are free to use, modify, and distribute this code for research and commercial purposes, provided that the original copyright notice and license are included. See the LICENSE
file for the full license text.

> **Note:**  
> If you use any part of this codebase in your research, publications, or derivative works, please cite our paper listed in the [Citation](#citation)
section. 

## Citation (⚠️ To be Updated) <a id="citation"></a>

If you use this code or find this repository useful in your research, please cite our paper:

```bibtex
@article{hosseini2026,
    title={Uncertainty-Aware AI for Tumor Subtyping with Histology and      Immunohistochemistry: A Multi-Center Study in Renal Cell Carcinoma},
    author={Hosseini, Seyed Mohammad Mehdi and Hannelet, Paul and Di Cataldo, Santa and Descombes, Xavier and Sibony, Mathilde and Dacausin, Myriam and Ponzio, Francesco and Ambrosetti, Damien},
    journal={To be updated},
    year={2026},
    volume = {To be updated},
    number = {To be updated},
    pages = {To be updated},
    doi = {To be updated},
    publisher = {To be updated},
    url = {To be updated},
    note = {To be updated}
}
```


## Contacts <a id="contacts"></a>

For questions, issues, or collaboration inquiries regarding this repository, please contact:

**Seyed Mohammad Mehdi Hosseini**  
Research Fellow  
Politecnico di Torino, DIST | DAUIN  
Email: mehdi.hosseini@polito.it  

and/or

**Francesco Ponzio**   
Assistant Professor (RTDa)   
Politecnico di Torino, DAUIN  
Email: francesco.ponzio@polito.it