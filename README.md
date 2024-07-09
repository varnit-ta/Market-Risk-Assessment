# Market Risk Assessment Using Deep Learning
This repository contains the implementation of a deep learning-based approach to assess market risk, inspired by the research paper titled ["Market Risk Assessment Using Deep Learning Model and Fog Computing Infrastructure"](Market_Risk_LSTM.pdf) published in the International Journal of Advances in Engineering Architecture Science and Technology. The paper emphasizes the application of machine learning techniques, particularly Long Short-Term Memory (LSTM) networks, to enhance traditional risk management practices in the financial sector.

## Overview

The project focuses on developing a market risk assessment model utilizing LSTM networks, which are a type of recurrent neural network (RNN) well-suited for analyzing sequential data. The aim is to predict future market movements by analyzing historical stock prices and other relevant financial indicators. This README provides an overview of the implementation, data preprocessing, model training, and evaluation processes.



## Table of Contents

1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Definition](#model-definition)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Predicting Risk](#predicting-risk)
7. [Requirements](#requirements)
8. [Usage](#usage)
9. [Results](#results)
10. [References](#references)

## Overview

This project aims to develop a market risk assessment tool using deep learning techniques. The model takes various financial features as input and predicts the risk percentage. The features include:

- Volatility
- Beta
- Market Cap
- P/E Ratio
- Dividend Yield
- Current Ratio
- 52 Week High
- 52 Week Low

## Data Preprocessing

The first step involves loading the dataset, handling missing values, normalizing the data, and splitting it into training and testing sets.

1. **Loading the Data**:
   The dataset is loaded using pandas. The features and target variable (risk percentage) are extracted.

2. **Handling Missing Values**:
   Missing values in the dataset are filled using the mean of the respective columns.

3. **Normalizing the Data**:
   The features are normalized to a range between 0 and 1 using MinMaxScaler from scikit-learn.

4. **Splitting the Data**:
   The dataset is split into training and testing sets using an 80-20 split ratio.

5. **Converting to Tensors**:
   The data is converted to PyTorch tensors for compatibility with the PyTorch model.

## Model Definition

An LSTM-based neural network model is defined with several hyperparameters:

- **Input Size**: Number of input features.
- **Hidden Size**: Number of neurons in the hidden layers.
- **Number of Layers**: Number of LSTM layers.
- **Output Size**: Number of outputs (1 for risk percentage).
- **Dropout Rate**: Dropout rate for regularization.
- **Learning Rate**: Learning rate for the optimizer.
- **Number of Epochs**: Number of epochs for training.

The LSTM model is defined using PyTorch's `nn.Module`. It consists of LSTM layers followed by a fully connected layer. Dropout is applied to prevent overfitting.


```python
class MarketRiskLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(MarketRiskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.relu(out[:, -1, :])
        out = self.dropout(out)
        out = self.batch_norm(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out
```

## Training the Model

The model is trained using the following steps:

1. **Loss Function and Optimizer**:
   Mean Squared Error (MSE) is used as the loss function. The Adam optimizer is used to update the model weights.

2. **Training Loop**:
   The model is trained for a specified number of epochs. In each epoch, the model's predictions are compared with the actual target values to calculate the loss. The gradients are computed, and the model weights are updated accordingly.

3. **Printing Training Progress**:
   The training progress is monitored by printing the loss value at regular intervals (every 100 epochs).

## Evaluating the Model

After training, the model's performance is evaluated on the test set. The test loss is calculated to measure how well the model generalizes to unseen data.

## Predicting Risk

To predict the risk percentage for new data, the following steps are followed:

1. **User Input**:
   The user is prompted to enter values for each feature.

2. **Preprocessing the Input**:
   The user input is processed and normalized in the same way as the training data.

3. **Making Predictions**:
   The processed input is fed into the trained model to get the predicted risk percentage.

## Requirements

To run the code, you need the following libraries:

- pandas
- torch
- matplotlib
- scikit-learn
- yfinance
- tqdm
- numpy

## Usage

1. **Install Dependencies**:
   Install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**:
   Run the main script to train the model and make predictions:
   ```bash
   python main.py
   ```

3. **Follow the Prompts**: 
   Enter the required financial feature values to get the risk percentage prediction.

## Results

The model's performance can be monitored through the printed loss values during training. After training, the test loss provides an indication of how well the model generalizes to unseen data.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

---
