# 🤗 BERT Model for Sentiment Analysis

This repository contains a Jupyter notebook implementing a BERT model for sentiment analysis on movie reviews.

## 📖 Overview

- Using the BERT Transformer model for binary sentiment classification.
- Loading and preprocessing IMDB movie review dataset.
- Tokenization using Hugging Face's BERT tokenizer.
- Dataset class to prepare inputs for the model.
- Fine-tuning BERT model with PyTorch.
- Training loop with accuracy and loss tracking.
- Saving and loading the trained model.
- Inference and prediction on sample text inputs.
- Plotting training loss and accuracy curves.
- Optionally comparing DistilBERT and RoBERTa models.

## 🛠 Requirements

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- matplotlib
- tqdm

Install packages using:

!pip install torch transformers scikit-learn pandas matplotlib tqdm


## 🚀 Usage

1. Download the [IMDB dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. Open `BERT-Model.ipynb` in Jupyter Notebook or JupyterLab.
3. Follow the cells step-by-step to preprocess data, train, and evaluate the model.
4. Use the prediction function to classify new text inputs.

## ⚠️ Note

Some model weights are newly initialized and fine-tuning is required for better prediction performance.

## 🔗 References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [IMDB dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## 👨‍💻 Author

Developed by Sayed Hamidullah Fazlly

M. Sc. Web & Data Science (On-going)

University of Koblenz, Germany

Email: 11hamidullah@gmail.com / sayedhamidullah@uni-konlenz.de

## 📄 License

MIT License
