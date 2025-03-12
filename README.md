# MonadoxML
This is a machine learning training and inference pipeline for mental state classification using EEG signal data. This project processes raw EEG data, extracts relevant features, and applies classification models to analyze cognitive and emotional states.  

The pipeline is **modular** and can be extended with additional signal processing techniques (e.g., **wavelet transform, FFT**) for EEG feature extraction.


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

## Usage
Run the pipeline
```bash
python src/main.py
```

## Model Performance
| Model | Accuracy | Inference time (ms) |
|--------|---------|-----------|
| Random Forest | 87% | 1000 |
| SVM | 91% | 1000 |


## License
This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](LICENSE) file for details.


## Contributing
We welcome contributions! Feel free to submit a pull request or open an issue.

1. Clone the repo
2. Create a new branch: `git checkout -b branch-name`
3. Commit changes: `git commit -m "commit message"`
4. Push to branch: `git push origin branch-name`
5. Open a pull request
