# 🌐 Premier League Predictor Web Interface

A modern, interactive web interface for Premier League match predictions powered by machine learning.

## 🚀 Features

### 📊 **Prediction Display**
- **Win/Draw/Loss Probabilities**: Prominently displayed with visual progress bars
- **AI Prediction**: Clear prediction with confidence percentage
- **Market Odds**: Comparison with bookmaker odds

### 📈 **Team Analytics**
- **Recent Form**: Last 5 games with scores and results
- **Team Statistics**: Wins, draws, losses, goals, shots, corners, cards
- **Performance Metrics**: Weighted averages and trends

### 🤖 **Model Information**
- **Accuracy**: 54.4% overall prediction accuracy
- **Log Loss**: 0.983 (excellent probability calibration)
- **Brier Score**: 0.195 (good calibration)
- **CV Stability**: 55.4% ± 2.5% (consistent performance)

## 🛠️ How to Use

### 1. **Start the Web Server**
```bash
python3 server.py
```

### 2. **Open Your Browser**
The server will automatically open your browser to `http://localhost:8000`

### 3. **View Predictions**
- See upcoming Premier League matches
- View AI predictions with probabilities
- Check team form and statistics
- Compare with market odds

## 📁 Files Structure

```
Premier League Prediction Model/
├── index.html              # Main webpage
├── web_predictor.py        # Backend prediction engine
├── server.py               # HTTP server
├── web_data.json           # Generated prediction data
└── README_WEB.md           # This file
```

## 🔧 Technical Details

### **Backend (web_predictor.py)**
- Loads historical Premier League data
- Trains calibrated logistic regression model
- Generates predictions for upcoming games
- Creates comprehensive team statistics
- Exports data as JSON for web interface

### **Frontend (index.html)**
- Modern, responsive design
- Interactive probability bars
- Real-time data loading
- Mobile-friendly interface
- Beautiful gradient styling

### **Model Performance**
- **Algorithm**: Calibrated Logistic Regression
- **Features**: 103 engineered features including:
  - Smart odds transformation
  - Rolling averages (3, 5, 10 games)
  - Draw-sensitive features
  - Team form and trends
- **Calibration**: Isotonic regression for probability calibration

## 🎯 Key Features Explained

### **Win/Draw/Loss Probabilities**
The most important feature - displayed prominently with:
- Visual progress bars showing probability percentages
- Color-coded bars (green=home, orange=draw, blue=away)
- Clear percentage values
- AI prediction with confidence level

### **Recent Form Display**
Shows last 5 games for each team:
- Opponent names
- Match scores
- Results (W/D/L) with color coding
- Date information

### **Team Statistics**
Comprehensive stats from last 5 games:
- Wins, draws, losses
- Goals scored and conceded
- Shots and shots on target
- Corners and cards
- All weighted by recency

### **Market Odds Comparison**
- Home win, draw, away win odds
- Comparison with AI predictions
- Helps identify value bets

## 🔄 Updating Predictions

To update predictions with new data:

1. **Add new match data** to the datasets folder
2. **Run the predictor**:
   ```bash
   python3 web_predictor.py
   ```
3. **Refresh the webpage** to see updated predictions

## 📱 Responsive Design

The webpage is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- All screen sizes

## 🎨 Design Features

- **Modern gradient backgrounds**
- **Glass-morphism effects**
- **Smooth animations and transitions**
- **Professional color scheme**
- **Intuitive user interface**

## 🚀 Future Enhancements

Potential improvements:
- Real-time data updates
- Historical prediction accuracy tracking
- User betting recommendations
- Team comparison tools
- League table integration
- Match commentary integration

---

**Enjoy predicting Premier League matches with AI! ⚽🤖**
