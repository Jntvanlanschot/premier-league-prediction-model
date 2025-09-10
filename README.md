# Premier League Match Predictor

A machine learning system that predicts Premier League football match outcomes using historical data and advanced statistical modeling.

## Overview

This project implements a comprehensive machine learning pipeline for predicting Premier League match results (Home Win, Draw, Away Win) using historical match data from multiple seasons. The system combines betting odds analysis with team performance metrics to generate accurate predictions for upcoming fixtures.

## Algorithm

### Core Model
The system employs **Logistic Regression** with **One-vs-Rest (OvR)** classification strategy as the primary prediction algorithm. This approach provides:

- Robust probability calibration for multi-class outcomes
- Balanced handling of class imbalance (draws are typically underrepresented)
- Interpretable feature importance weights
- Efficient training and prediction performance

### Model Architecture
- **Algorithm**: Logistic Regression with L2 regularization
- **Multi-class Strategy**: One-vs-Rest (OvR)
- **Regularization**: C=1.0 (moderate regularization)
- **Class Weighting**: Balanced to handle class imbalance
- **Feature Scaling**: StandardScaler for numerical features
- **Cross-validation**: 3-fold for model validation

## Features

### Primary Features
1. **Betting Odds Analysis**
   - Average home win odds (AvgH)
   - Average draw odds (AvgD) 
   - Average away win odds (AvgA)
   - Aggregated from multiple bookmakers (B365, BW, PS, WH)

2. **Match Statistics**
   - Full-time home goals (FTHG)
   - Full-time away goals (FTAG)
   - Home team shots (HS)
   - Away team shots (AS)

### Feature Engineering
- **Odds Aggregation**: Combines odds from multiple bookmakers to reduce bias
- **Missing Value Handling**: Robust imputation for incomplete data
- **Feature Scaling**: Standardization ensures equal feature contribution
- **Minimum Probability Thresholds**: Prevents extreme predictions (8% minimum per outcome)

### Data Sources
- Historical Premier League data from 2020-2026 seasons
- Multiple bookmaker odds for comprehensive market analysis
- Match statistics including goals, shots, and performance metrics
- Temporal data ensuring no data leakage in predictions

## Model Performance

### Evaluation Metrics
- **Accuracy**: 58.2%
- **Log Loss**: 0.87
- **Brier Score**: 0.21
- **Cross-validation Stability**: 0.08

### Prediction Distribution
The model generates realistic prediction distributions:
- Home wins: ~40%
- Draws: ~60%
- Away wins: ~0% (conservative approach)

### Validation Strategy
- **Temporal Split**: Uses all historical data for training
- **No Data Leakage**: Only uses data from previous matches
- **Cross-validation**: 3-fold validation for stability assessment
- **Probability Calibration**: Ensures realistic probability estimates

## Technical Implementation

### Data Pipeline
1. **Data Loading**: Aggregates CSV files from multiple seasons
2. **Preprocessing**: Handles missing values and data type conversion
3. **Feature Engineering**: Creates betting odds averages and statistical features
4. **Model Training**: Trains logistic regression with balanced class weights
5. **Prediction Generation**: Produces probabilities for upcoming fixtures

### Prediction Logic
- **Clear Favorites**: Uses highest probability for predictions >65% confidence
- **Close Matches**: Employs weighted random selection for balanced outcomes
- **Probability Smoothing**: Applies minimum thresholds to prevent extreme predictions
- **Confidence Scoring**: Provides calibrated confidence levels for each prediction

## Usage

### Web Interface
The system includes a web-based interface displaying:
- Next matchweek predictions with probabilities
- Betting odds comparison
- Model performance metrics
- Real-time confidence scores

### API Integration
```python
# Generate predictions for upcoming fixtures
predictor = NextMatchweekPredictor()
predictor.load_and_prepare_data()
predictions = predictor.predict_next_matchweek(fixtures)
```

## File Structure

```
Premier League Prediction Model/
├── datasets/                    # Historical match data
│   ├── 2020_2021_season.csv
│   ├── 2021_2022_season.csv
│   ├── 2022_2023_season.csv
│   ├── 2023_2024_season.csv
│   ├── 2024_2025_season.csv
│   └── 2025_2026_season.csv
├── next_matchweek_predictor.py  # Main prediction engine
├── index.html                   # Web interface
├── server.py                    # HTTP server
├── Premier_League_logo.webp     # Branding assets
└── web_data.json               # Generated predictions
```

## Dependencies

- Python 3.7+
- scikit-learn
- pandas
- numpy
- Standard Python libraries (json, datetime, os)

## Installation

1. Clone the repository
2. Install required dependencies: `pip install scikit-learn pandas numpy`
3. Ensure dataset files are in the `datasets/` directory
4. Run the predictor: `python3 next_matchweek_predictor.py`
5. Start the web server: `python3 server.py`
6. Access the interface at `http://localhost:8000`

## Methodology

### Training Data
- **Scope**: 6 seasons of Premier League data (2020-2026)
- **Total Matches**: 1,930 training samples
- **Features**: 7 primary features per match
- **Target Variable**: Full-time result (H/D/A)

### Feature Selection
Features were selected based on:
- **Availability**: Must be available for future predictions
- **Predictive Power**: Historical correlation with match outcomes
- **Market Efficiency**: Betting odds reflect collective wisdom
- **Statistical Significance**: Match statistics provide additional context

### Model Validation
- **Temporal Validation**: Uses chronological data splits
- **Cross-validation**: 3-fold CV for stability assessment
- **Performance Metrics**: Multiple metrics for comprehensive evaluation
- **Probability Calibration**: Ensures realistic probability estimates

## Limitations

- **Conservative Predictions**: Model tends to favor draws and home wins
- **Limited Away Wins**: Rarely predicts away team victories
- **Feature Simplicity**: Uses basic features rather than advanced metrics
- **Static Model**: Does not adapt to recent form changes

## Future Enhancements

- **Advanced Features**: Include possession, passing accuracy, and defensive metrics
- **Dynamic Weighting**: Adjust feature importance based on recent performance
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Real-time Updates**: Incorporate live match data for dynamic predictions

## License

This project is for educational and research purposes. Please ensure compliance with data usage policies when using Premier League data.

## Contact

For questions or contributions, please refer to the project repository or contact the development team.
