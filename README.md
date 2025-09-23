# Polymers_FTW

Fine-tuning of TransPolymer for prediction of polymer properties from SMILES data. 

This repository contains files that allow for LoRA fine tuning of TransPolymer for the prediction of five polymer properties: 

1. Glass Transition Temperature (Tg)
2. Crystallization Temperature (Tc)
3. Radius of Gyration (Rg)
4. Density 
5. Free Fractional Volume (FFV)

The LoRA fine-tuning framework presented here was created for the NeurIPS - Open Polymer Prediction 2025 competition hosted by Notre Dame University on Kaggle (https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/overview), where it won a silver medal. 
---

## Features

- Fine-tunes the [TransPolymer](https://github.com/ChangwenXu98/TransPolymer.git) model for property prediction using SMILES data.
- Jupyter Notebook and Python-based workflow.
- Includes basic visualization tools to assess Pearson corrlation and loss for adjustment of LoRA parameters for training. 

---

## Installation

### Prerequisites

- See the requirements.txt file

### Setup

Create a python environment
```bash
python -m venv your_environment_name
```

Clone the repository:
```bash
git clone https://github.com/fvillafu125/polymers_ftw.git
cd polymers_ftw
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Or install packages manually:
```bash
pip install torch rdkit pandas ...
```

---

## Usage

1. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open and run the provided notebooks in sequence.
3. To fine-tune on your own data, update the appropriate data paths and parameters in the notebook or scripts.

**Example notebook:**  
`notebooks/finetune_transpolymer.ipynb`

---

## Project Structure

```plaintext
polymers_ftw/
├── data/                 # Datasets and data preprocessing scripts
├── notebooks/            # Jupyter Notebooks for experiments and analysis
├── src/                  # Core Python modules
├── results/              # Results, metrics, and figures
├── requirements.txt      # List of required Python packages
└── README.md             # Project documentation
```

---

## Data

- Input data: SMILES strings and associated target properties.
- Place your data in the `data/` directory.
- Data format example (CSV):
    | SMILES        | property1 | property2 |
    |---------------|-----------|-----------|
    | CC(=O)OC1=CC=... | 0.123     | 5.67      |

- (Optional) Preprocessing scripts and instructions.

---

## Training and Evaluation

- Instructions for running training and evaluation scripts or notebooks.
- Configurable hyperparameters: learning rate, batch size, epochs, etc.
- Example command or notebook cell to start training.

---

## Results

- Summarize key results (tables, metrics, example plots).
- (Optional) Add images or figures to illustrate performance.
- (Optional) Link to pre-trained models or supplementary materials.

---

## Contributing

Contributions are welcome!  
Please open an issue or submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) (if available) for more details.

---

## License

This project is licensed under the [MIT License](LICENSE).  
(Or update with your preferred license.)

---

## Acknowledgements

- Based on [TransPolymer](https://github.com/fvillafu125/TransPolymer)
- List any collaborators, datasets, or libraries used.
- Funding sources (if any).

---

## Contact

For questions, suggestions, or collaborations, please contact:  
[Your Name] ([your.email@domain.com](mailto:your.email@domain.com))
