# Animal Sleep Prediction with Random Forest Regressor

This project aims to predict the total sleep time of various animals using a machine learning model. The dataset used contains information such as body weight, brain weight, and sleep patterns of different animals. A **Random Forest Regressor** is used to model the relationship between these features and the total sleep time.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Evaluation Metrics](#evaluation-metrics)
- [Outlier Analysis](#outlier-analysis)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to use machine learning to predict the **total sleep time** (`sleep_total`) of various animal species based on their characteristics. It uses features like body weight, brain weight, sleep cycle, and dietary categories (carnivores, herbivores, etc.).

## Dataset
The dataset used is `msleep.csv`, which contains:
- `bodywt`: Body weight of the animal.
- `brainwt`: Brain weight of the animal.
- `sleep_rem`: REM sleep time.
- `sleep_cycle`: Sleep cycle time.
- `conservation`: Conservation status of the animal.
- `vore`: Dietary category (carnivore, herbivore, etc.).

Missing values in the dataset are handled by filling them with mean values.

## Data Preprocessing
- Log transformation is applied to `bodywt` and `brainwt` to reduce skewness.
- One-hot encoding is used to convert categorical variables like `vore` and `conservation` into numerical form.
- Features are scaled using `StandardScaler` for better model performance.

## Model
A **Random Forest Regressor** is used for this project with the following parameters:
- `n_estimators`: 200
- `max_depth`: 10
- `random_state`: 42

The model is trained on 80% of the data, with the remaining 20% used for testing.

## Evaluation Metrics
The model's performance is evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-Squared (R²) Score**

## Outlier Analysis
An outlier analysis is performed to identify the data point with the largest prediction error, providing insight into which animals the model struggles to predict accurately.

## Results
The predictions for each animal are compared against the actual sleep totals. The results show how closely the model can predict the sleep behavior of various species based on the provided features.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/animal-sleep-prediction.git
   cd animal-sleep-prediction
