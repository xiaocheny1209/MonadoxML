# MonadoxML
This is a machine learning training and inference pipeline for mental state classification using EEG signal data. This project processes raw EEG data, extracts relevant features, and applies classification models to analyze cognitive and emotional states.  

The pipeline is modular and can be extended with additional signal processing techniques, EEG feature extraction methods and ML/DL models.


## Installation
### Prerequisites
- Python 3.10 or newer
- Conda

### Code installation
Clone the repository, create a conda environment and install the required dependencies:
```bash
git clone https://github.com/xiaocheny1209/MonadoxML.git
cd MonadoxML
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

## Dataset Instructions
Before running the code, you need to download the dataset manually and place it in the appropriate directory. Below are the steps to follow:

### 1. Download the Dataset

- Download **Data.zip** from FACED Dataset: [Download link for FACED dataset](<https://www.synapse.org/Synapse:syn50614194/files/>)
or 
- Download STEW Dataset from IEEEDataPort: [Download link for STEW dataset](<https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset?check_logged_in=1>)


### 2. Folder Structure

Once you have downloaded the dataset, unzip it, rename it as "FACED" and put it under the **data/** folder.
or rename STEW dataset.zip to "STEW" and put it under the **data/** folder.


## Usage
Run the pipeline
```bash
python src/main.py
```

Run the pipeline for STEW dataset
```bash
python src/main_stew.py
```

## Model Performance

### SVM with Difference Entropy (Cross-Validation over FACED Dataset)

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 0.8810              |
| 2    | 0.9286              |
| 3    | 0.9048              |
| 4    | 0.8929              |
| 5    | 0.9405              |
| 6    | 0.8929              |
| 7    | 0.8810              |
| 8    | 0.8929              |
| 9    | 0.8571              |
| 10   | 0.8333              |

**Best fold**: 5 with Validation Accuracy: **0.9405**

---

### SVM with Power Spectral Density (Cross-Validation over FACED Dataset)

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 0.7024              |
| 2    | 0.6667              |
| 3    | 0.6548              |
| 4    | 0.6905              |
| 5    | 0.6905              |
| 6    | 0.6548              |
| 7    | 0.6905              |
| 8    | 0.7024              |
| 9    | 0.6429              |
| 10   | 0.7024              |

**Best fold**: 1 with Validation Accuracy: **0.7024**

### SVM with Power Spectral Density (Cross-Validation over STEW Dataset)

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 0.9244              |
| 2    | 0.9244              |
| 3    | 0.9437              |
| 4    | 0.9363              |
| 5    | 0.9393              |
| 6    | 0.9378              |
| 7    | 0.9304              |
| 8    | 0.9230              |
| 9    | 0.9333              |
| 10   | 0.9422              |

**Best fold**: 3 with Validation Accuracy: **0.9437**

### Random Forest Classifier (Cross-Validation over STEW Dataset)

| Fold | Validation Accuracy |
|------|---------------------|
| 1    | 0.9304              |
| 2    | 0.9274              |
| 3    | 0.9511              |
| 4    | 0.9422              |
| 5    | 0.9481              |
| 6    | 0.9437              |
| 7    | 0.9363              |
| 8    | 0.9333              |
| 9    | 0.9393              |
| 10   | 0.9481              |

**Best fold**: 3 with Validation Accuracy: **0.9511**
Please follow the commented instructions next to the corresponding lines in main_stew.py and feature_extraction_stew.py to swtich between SVM and Random Forest Tree. (To change folds in line 43 of main_stew.py change "random_state = 42" to "random_state=None")

## License
This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.


## Contributing
We welcome contributions! Feel free to submit a pull request or open an issue.

1. Clone the repo
2. Create a new branch: `git checkout -b branch-name`
3. Commit changes: `git commit -m "commit message"`
4. Push to branch: `git push origin branch-name`
5. Open a pull request


## Acknowledgments

The **FACED Dataset** and the code used in this repository are based on the following paper:

Chen, J., Wang, X., Huang, C. *et al.* A Large Finer-grained Affective Computing EEG Dataset. *Sci Data* **10**, 740 (2023). https://doi.org/10.1038/s41597-023-02650-w  
Please refer to the official paper for more details on the dataset.

The **STEW Dataset** and the code used in this repository are based on the following paper: 
W. L. Lim, O. Sourina and L. P. Wang, "STEW: Simultaneous Task EEG Workload Data Set," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 11, pp. 2106-2114, Nov. 2018, doi: 10.1109/TNSRE.2018.2872924.
keywords: {Electroencephalography;Task analysis;Multitasking;Databases;Open Access;Electronic mail;Frequency measurement;Electroencephalography (EEG);mental workload;open access dataset},
Please refer to the official paper for more details on the dataset.

We thank the authors and contributors of these resources for their valuable work.

