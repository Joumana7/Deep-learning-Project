Parkinson's Disease Detection and Severity Estimation using Deep Learning

This repository contains the implementation of a deep learning-based system for Parkinson's Disease (PD) detection and severity estimation using speech signal data. The project addresses two main tasks:

    Binary classification of individuals as either Healthy or PD.

    Regression-based estimation of disease severity using the Unified Parkinson’s Disease Rating Scale (UPDRS).

Dataset

The dataset used is the Parkinson's Speech Dataset with Multiple Types of Audio Recordings from the UCI Machine Learning Repository. It includes:

    Training Set:

        Voice recordings from 20 PD patients and 20 healthy individuals.

        Each subject contributed 26 speech samples (vowels, digits, words, sentences).

        26 extracted features per sample.

        Includes UPDRS score and class label for each sample.

    Test Set:

        Recordings from 28 PD patients (sustained vowels 'a' and 'o').

        26 features extracted per sample.

        No class labels provided.

 Project Structure

The project includes the following steps:
1. Data Preprocessing

    Loaded using pandas.

    Renamed and type-cast columns.

    Trimmed to ensure samples per subject are consistent (26).

    Standardized using StandardScaler.

    Reshaped to 3D format: (subjects, samples, features) → (40, 26, 26) for training, (6, 26, 26) for testing.

    Prepared target variables for:

        Classification (y_train_class)

        Regression (y_train_updrs, y_test_updrs)

2. Model Architectures

We implemented five deep learning architectures using TensorFlow/Keras:

    LSTM: Bidirectional LSTM with LayerNormalization, L2 regularization, Dropout.

    GRU: Bidirectional GRU with BatchNormalization, Dropout.

    BiLSTM: Similar to LSTM but tuned separately.

    SimpleRNN: Stacked SimpleRNNs with BatchNormalization and Dropout.

    DenseNet1D: Conv1D-based architecture inspired by DenseNet for 1D data.

Each architecture is modularized into separate build_<model>() functions for reusability.

3. Model Training

    Used Adam optimizer with task-specific losses:

        Classification: sparse_categorical_crossentropy

        Regression: mean_squared_error (MSE)

    Metrics:

        Classification: accuracy

        Regression: mean_absolute_error (MAE)

    Callbacks: EarlyStopping, ReduceLROnPlateau

    Validation split from training data using train_test_split.

4. Evaluation

    Results stored and compared for both tasks.

    Metrics:

        Classification: Accuracy and Loss

        Regression: MAE and MSE

 Results Summary
Model	Classification Accuracy	Regression MAE
LSTM	~85%	~9.10
GRU	~85%	8.81 
BiLSTM	~82.5%	~9.30
SimpleRNN	87.5% ~9.20
DenseNet1D	~77.5%	~10.5

    SimpleRNN performed best for classification.

    GRU achieved the best performance in UPDRS regression.

Insights & Discussion

    Recurrent architectures (LSTM, GRU, etc.) performed significantly better than DenseNet1D, affirming the importance of temporal modeling in speech-based medical diagnosis.

    The results suggest that even simple recurrent models can be highly effective when properly trained and regularized.

    Demonstrates feasibility of using deep learning for non-invasive PD detection from voice data.

Getting Started
Prerequisites

Install required Python packages:

pip install numpy pandas scikit-learn tensorflow matplotlib



 File Structure

parkinsons-detection-dl/
├── data/
│   ├── train_data.txt
│   └── test_data.txt
├── models/
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── bilstm_model.py
│   ├── simpler_nn_model.py
│   └── densenet1d_model.py
├── utils/
│   ├── preprocessing.py
│   └── training.py
├── train_models.py
├── README.md
└── requirements.txt

 References

    Dataset: UCI Parkinson's Speech Dataset

    TensorFlow

    Keras Documentation

    Relevant literature on PD and speech-based diagnosis.
