# Customer Churn Prediction Model

A deep learning model built with TensorFlow to predict customer churn in banking. The project implements a neural network that processes customer data to identify clients at risk of leaving the bank. Features include data preprocessing, TFRecord pipeline optimization, model training with validation, performance evaluation using metrics like AUC-ROC, and a production-ready inference pipeline.

## Features

- Data preprocessing pipeline with label encoding and feature scaling
- Optimized data input using TFRecords format
- Three-layer neural network implementation with ReLU activation
- Model training with mini-batch gradient descent
- Real-time performance monitoring (accuracy, loss)
- Model evaluation with confusion matrix, precision, recall, F1-score, and AUC
- Production-ready inference pipeline with TensorFlow Serving support

## Project Structure

```
.
├── data/
│   ├── raw/                  # Raw CSV data
│   └── processed/            # TFRecord files
├── models/
│   ├── base_model.py        # Base neural network architecture
│   ├── train_model.py       # Training specific operations
│   └── inference_model.py   # Inference specific operations
├── checkpoints/             # Saved model weights
├── model-export/            # SavedModel format for deployment
└── src/
    ├── training.py          # Main training script
    ├── inference.py         # Inference pipeline
    ├── dataset.py           # Data input pipeline
    ├── preprocess.py        # Data preprocessing
    ├── performance.py       # Evaluation metrics
    └── tf_records_writer.py # TFRecord conversion utility
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

2. Install dependencies:
```bash
pip install tensorflow sklearn pandas numpy matplotlib
```

## Usage

1. Prepare the dataset:
```bash
python src/preprocess.py
python src/tf_records_writer.py
```

2. Train the model:
```bash
python src/training.py --n_epoch 10 --learning_rate 0.005 --batch_size 64
```

3. Export model for inference:
```bash
python src/inference.py --model_version 1
```

## Model Architecture

The neural network consists of:
- Input layer: 18 features
- Hidden layer 1: 50 units with ReLU activation
- Hidden layer 2: 50 units with ReLU activation
- Output layer: 2 units with softmax activation

## Performance Metrics

The model evaluation includes:
- Confusion Matrix
- Precision and Recall
- F1 Score
- AUC-ROC Curve

## Configuration

Key hyperparameters can be adjusted through flags:
- `--n_epoch`: Number of training epochs
- `--learning_rate`: Learning rate for Adam optimizer
- `--batch_size`: Mini-batch size
- `--l2_reg`: Enable/disable L2 regularization
- `--alpha`: Regularization strength

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
