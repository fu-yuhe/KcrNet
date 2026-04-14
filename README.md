# KcrNet: An Interpretable Lysine Crotonylation Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation of **KcrNet**, a lightweight deep learning framework for accurate prediction of lysine crotonylation (Kcr) sites. KcrNet integrates ESM2 protein language model embeddings with handcrafted physicochemical and CTD features, and employs MLM domain-adaptive fine-tuning.

## Paper

If you use this code or dataset in your research, please cite our paper:

> Yuhe Fu, Jia Zheng. *KcrNet: An interpretable Kcr predictor integrating multi-feature fusion and MLM domain adaptation.* (Preprint/Submitted)

## Quick Start

### 1. Clone the repository
git clone https://github.com/fu-yuhe/KcrNet.git
cd KcrNet

### 2. Install dependencies
pip install -r requirements.txt

### 3. Data preparation
The training data is already included in the `cleaned_data/` folder:
- train_pos.fasta : positive samples (Kcr sites)
- train_neg.fasta : negative samples (non-Kcr sites)

### 4. Run training
python kcrmodel.py

The script will automatically perform MLM domain-adaptive pretraining and 5-fold cross-validation.

## Repository Structure
KcrNet/
├── kcrmodel.py                # Main training script
├── cleaned_data/              # Processed dataset
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── .gitignore                 # Ignored files
└── README.md                  # This file

## Results
Our model achieves an average 5-fold cross-validation accuracy of ~84.96% and AUC of ~91.54% on the human training set.

## Contact
For questions or collaborations, please contact: 1396554623@qq.com