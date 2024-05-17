# Paraphrase Generation with Distillation and VAE

This project aims to build a paraphrase generation model using a combination of model distillation and a Conditional Variational Autoencoder (CVAE). The teacher model is a trimmed version of Pegasus-large, and the student model is a trimmed version of Pegasus-xsum.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Project Overview

The project involves:
1. Data loading and preprocessing.
2. Custom dataset handling.
3. Model trimming for distillation.
4. Training with mixed precision and gradient accumulation.
5. Evaluation of model performance using BERTScore and BLEU metrics.

## Setup

### Prerequisites

Make sure you have the following installed:
- Python 3.7 or higher
- PyTorch
- Transformers
- scikit-learn
- pandas
- tqdm
- TensorBoard
- NLTK
- BERTScore

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ZhengQin2001/NLP_Assignment_SN20010701.git
   cd paraphrase-generation
   ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

### Training
To train the model, run the following command:
  ```bash
  python main.py train
  ```
This will start the training process, save model checkpoints, and log training metrics to TensorBoard.

### Evaluation
To evaluate the trained model, run the following command:
  ```bash
  python main.py evaluate
  ```
This will load the trained model and evaluate its performance on the validation and test datasets.

### Directory Structure

```
  .
  ├── config.py
      Contains configurations such as file paths, column mappings, and model parameters.
  ├── data_processing.py
      Functions for data loading and preprocessing.
  ├── dataset.py
      Custom dataset class for handling data.
  ├── evaluate.py
      Evaluation logic for the model.
  ├── loss.py
      Custom loss functions and evaluation metrics.
  ├── main.py
      Entry point to run training or evaluation.
  ├── models.py
      Model definitions.
  ├── model_utils.py
      Functions for trimming model architectures.
  ├── README.md
      Project overview and instructions.
  ├── requirements.txt
      Required packages and dependencies.
  ├── train.py
      Training loop.
```
  
## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers): For providing the pre-trained transformer models and tokenizers used in this project.
- [BERTScore](https://github.com/Tiiiger/bert_score): For the evaluation metric that helps in assessing the quality of generated paraphrases.
- [NLTK](https://www.nltk.org/): For the natural language processing tools and libraries used in the project.


This `README.md` file provides a comprehensive overview of the paraphrase generation project, setup instructions, and usage guidelines. Make sure to replace the placeholder repository URL with the actual URL of your repository. Additionally, create a `requirements.txt` file with the necessary package dependencies. Here’s an example `requirements.txt` file:

```plaintext
torch
transformers
scikit-learn
pandas
tqdm
tensorboard
nltk
bert-score
```
