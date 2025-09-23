# Polymers_FTW

Fine-tuning of TransPolymer for prediction of polymer properties from SMILES data. 

This repository contains files that allow for LoRA fine-tuning of TransPolymer for the prediction of five polymer properties: 

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
- Includes basic visualization tools to assess Pearson correlation and loss for adjustment of LoRA training parameters. 

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

## Dataset

The primary and supplementary training data was provided by the Kaggle competition organizers (https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data), and is provided here in the data/neurips-open-polymer-prediction-2025 folder. It was augmented with rdkit using the data_prep.py script provided in the repository and broken into to 5 sets of train and test data for each target property (Tg, Tc, Rg, density, FFV) with a 0.9/0.1 training/test split. These datasets for downstream LoRA fine-tuning are provided in the data/ folder. 

For Tg, optimal results were obtained by normalizing both the train and test sets by the largest value across both sets (see normalizer.py). There are also other train and test sets for Tg that were adjusted with z-normalizaiton (see znormalizer.py). Both normalized and z-normalized Tg data are available in the data/ folder. 

The general format of the training and test data sets used for downstream LoRA fine-tuning is as follows:

    |  SMILES       | property |
    |---------------|----------|
    | CC(=O)OC1=CC=... | 0.123 |

---

## Training and Evaluation

### Fine-tuning

1. Edit the training parameters in the config_finetune.yaml file
2. Run the following command:
```bash
python DownStream_LoRA.py
```
Note that the default loss function for downstream training is MSE.

### Inference

Inference can be performed with either the included notebook, inference.ipynb.

### Model Checkpoints

The pre-trained TransPolymer model checkpoint is included in ckpt/pretrain.pt. The LoRA fine-tuned model checkpoints are included in ckpt/neurips.pt

### Data Visualization

The provided correlation.ipynb notebook allows for visualization of Pearson correlation between actual test and predicted properties. The loss_plot.ipynb notebook allows for plotting training and test loss. 

---

## Results

- Summarize key results (tables, metrics, example plots).
- (Optional) Add images or figures to illustrate performance.
- (Optional) Link to pre-trained models or supplementary materials.

---

## Contributing

Contributions are welcome!  
Please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).  

---

## Acknowledgements

- Based on [TransPolymer](https://github.com/ChangwenXu98/TransPolymer.git)
```
@article{xu2023transpolymer,
  title={TransPolymer: a Transformer-based language model for polymer property predictions},
  author={Xu, Changwen and Wang, Yuyang and Barati Farimani, Amir},
  journal={npj Computational Materials},
  volume={9},
  number={1},
  pages={64},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
- Fine-tuning performed using Low Rank Adaptation (LoRA)
```
@misc{hu2021loralowrankadaptationlarge,
      title={LoRA: Low-Rank Adaptation of Large Language Models}, 
      author={Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
      year={2021},
      eprint={2106.09685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2106.09685}, 
}
```

---

## Contact

For questions, suggestions, or collaborations, please contact:  
[Fernando Villafuerte] ([fjoaquin.villafuerte@gmail.com](mailto:fjoaquin.villafuerte@gmail.com))
