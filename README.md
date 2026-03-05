# CancerVision AI - Breast Cancer Prediction

A machine learning-powered Streamlit web application that predicts breast cancer risk based on mammography features.

## Features

- **Machine Learning Model**: Trained on the breast cancer dataset with a scikit-learn pipeline
- **Beautiful UI**: Animated gradient background with modern design
- **Three Risk Levels**: Low Risk, Moderate Risk, and High Risk predictions
- **Interactive Inputs**: Adjust mammography features or generate random samples
- **Real-time Predictions**: Instant cancer probability predictions with visual feedback

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/santhoshchintapenta/CancerVision-AI-breast-cancer-prediction-.git
cd breast_cancer_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### How to Use:
1. Enter mammography feature values or use the random sample generator
2. Click "Predict Risk" to get a prediction
3. View the risk level and cancer probability
4. Follow the recommendations based on the result

## Features Used

- **Radius**: Mean size of the tumor
- **Texture**: Standard deviation of grayscale values
- **Perimeter**: Outer edge length of the tumor
- **Area**: Size of the tumor region
- **Smoothness**: Local variation in radius lengths
- **Compactness**: Perimeter² / Area - 1.0

## Risk Categories

- **Low Risk** (< 40%): Regular check-ups recommended
- **Moderate Risk** (40-70%): Schedule follow-up appointments
- **High Risk** (> 70%): Immediate medical consultation advised

## Model Details

- **Algorithm**: Logistic Regression with StandardScaler preprocessing
- **Training Data**: Breast Cancer Wisconsin Dataset
- **Accuracy**: Validated on test set

## Disclaimer

This application is for educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for medical advice.

## License

MIT License

## Author

Santhosh Chintapenta

## Contact

For questions or issues, please open an issue on GitHub.
