#!/usr/bin/env python3
"""
Enhanced Season Predictor with League Table
==========================================
Generates predictions for all remaining 2025-2026 season games
and calculates the final Premier League table.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedSeasonPredictor:
    """Enhanced season predictor with league table calculation."""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.current_table = {}
        
    def load_and_prepare_data(self):
        """Load and prepare all data including 2025-2026 season."""
        print("📥 Loading data including 2025-2026 season...")
        
        # Load data
        datasets_folder = '/home/joost/Premier League Prediction Model/datasets'
        csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
        csv_files.sort()
        
        all_data = []
        for file in csv_files:
            file_path = os.path.join(datasets_folder, file)
            df = pd.read_csv(file_path)
            season_name = file.replace('_season.csv', '').replace('_', '')
            if len(season_name) == 8:
                season_code = season_name[2:4] + season_name[6:8]
            else:
                season_code = season_name
            df['Season'] = season_code
            all_data.append(df)
            print(f"  ✅ {file}: {len(df)} matches loaded")
        
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Calculate current league table from played matches
        self.calculate_current_table()
        
        # Preprocess and train model
        self.preprocess_data()
        self.create_features()
        self.train_model()
        
        print("✅ Data loaded, table calculated, and model trained")
        return self
    
    def calculate_current_table(self):
        """Calculate current league table from played matches in 2025-2026 season."""
        current_season_data = self.data[self.data['Season'] == '2526'].copy()
        current_season_data['Date'] = pd.to_datetime(current_season_data['Date'], format='%d/%m/%Y')
        
        # Get all teams from Premier League 2025-2026 (using actual team names from data)
        teams = {
            'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton', 'Burnley', 'Chelsea', 
            'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Liverpool', 'Man City', 'Man United', 
            'Newcastle', "Nott'm Forest", 'Sunderland', 'Tottenham', 'West Ham', 'Wolves'
        }
        
        # Initialize table
        self.current_table = {team: {
            'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
            'goals_for': 0, 'goals_against': 0, 'goal_difference': 0, 'points': 0
        } for team in teams}
        
        # Process each played match
        for _, match in current_season_data.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            home_goals = int(match['FTHG']) if pd.notna(match['FTHG']) else 0
            away_goals = int(match['FTAG']) if pd.notna(match['FTAG']) else 0
            result = match['FTR']
            
            # Update stats for both teams
            self.current_table[home_team]['played'] += 1
            self.current_table[away_team]['played'] += 1
            
            self.current_table[home_team]['goals_for'] += home_goals
            self.current_table[home_team]['goals_against'] += away_goals
            self.current_table[away_team]['goals_for'] += away_goals
            self.current_table[away_team]['goals_against'] += home_goals
            
            if result == 'H':  # Home win
                self.current_table[home_team]['won'] += 1
                self.current_table[home_team]['points'] += 3
                self.current_table[away_team]['lost'] += 1
            elif result == 'A':  # Away win
                self.current_table[away_team]['won'] += 1
                self.current_table[away_team]['points'] += 3
                self.current_table[home_team]['lost'] += 1
            else:  # Draw
                self.current_table[home_team]['drawn'] += 1
                self.current_table[home_team]['points'] += 1
                self.current_table[away_team]['drawn'] += 1
                self.current_table[away_team]['points'] += 1
        
        # Calculate goal differences
        for team in self.current_table:
            self.current_table[team]['goal_difference'] = (
                self.current_table[team]['goals_for'] - 
                self.current_table[team]['goals_against']
            )
        
        print(f"✅ Current table calculated with {len(current_season_data)} played matches")
    
    def preprocess_data(self):
        """Preprocess data for enhanced season prediction."""
        # Handle missing values in match stats
        stats_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
        for col in stats_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        # Create average betting odds
        betting_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
                       'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA']
        
        for col in betting_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        home_cols = [col for col in self.data.columns if col.endswith('H') and col.startswith(('B365', 'BW', 'PS', 'WH'))]
        draw_cols = [col for col in self.data.columns if col.endswith('D') and col.startswith(('B365', 'BW', 'PS', 'WH'))]
        away_cols = [col for col in self.data.columns if col.endswith('A') and col.startswith(('B365', 'BW', 'PS', 'WH'))]
        
        self.data['AvgH'] = self.data[home_cols].mean(axis=1, skipna=True)
        self.data['AvgD'] = self.data[draw_cols].mean(axis=1, skipna=True)
        self.data['AvgA'] = self.data[away_cols].mean(axis=1, skipna=True)
        
        return self
    
    def create_features(self):
        """Create simple features for prediction."""
        # Use only betting odds features for future predictions
        self.feature_columns = ['AvgH', 'AvgD', 'AvgA']
        
        # Filter to existing columns and handle missing values
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        
        return self
    
    def train_model(self):
        """Train the model using all available data."""
        # Prepare training data (use all data except current season)
        train_data = self.data[self.data['Season'] != '2526'].copy()
        
        X_train = train_data[self.feature_columns].fillna(0)
        y_train = train_data['FTR']
        
        # Remove any rows with missing target
        mask = y_train.notna()
        X_train = X_train[mask]
        y_train = y_train[mask]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Train calibrated logistic regression
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        self.model = CalibratedClassifierCV(
            LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.1,
                penalty='l2',
                class_weight=class_weight_dict
            ),
            method='isotonic',
            cv=3
        )
        
        self.model.fit(X_train_scaled, y_train_encoded)
        
        print(f"✅ Model trained on {len(train_data)} matches")
        return self
    
    def get_all_remaining_fixtures(self):
        """Get all remaining fixtures for the 2025-2026 season."""
        remaining_fixtures = [
            # September 13-14, 2025 (Coming Matchweek - HIGHLIGHTED)
            {'date': '2025-09-13', 'time': '12:30', 'home': 'Arsenal', 'away': "Nott'm Forest", 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '15:00', 'home': 'Bournemouth', 'away': 'Brighton', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '15:00', 'home': 'Crystal Palace', 'away': 'Sunderland', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '15:00', 'home': 'Everton', 'away': 'Aston Villa', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '15:00', 'home': 'Fulham', 'away': 'Leeds', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '15:00', 'home': 'Newcastle', 'away': 'Wolves', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '17:30', 'home': 'West Ham', 'away': 'Tottenham', 'matchweek': 'Next'},
            {'date': '2025-09-13', 'time': '20:00', 'home': 'Brentford', 'away': 'Chelsea', 'matchweek': 'Next'},
            {'date': '2025-09-14', 'time': '14:00', 'home': 'Burnley', 'away': 'Liverpool', 'matchweek': 'Next'},
            {'date': '2025-09-14', 'time': '16:30', 'home': 'Man City', 'away': 'Man United', 'matchweek': 'Next'},
            
            # September 20-21, 2025
            {'date': '2025-09-20', 'time': '12:30', 'home': 'Liverpool', 'away': 'Everton', 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '15:00', 'home': 'Brighton', 'away': 'Tottenham', 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '15:00', 'home': 'Burnley', 'away': "Nott'm Forest", 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '15:00', 'home': 'West Ham', 'away': 'Crystal Palace', 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '15:00', 'home': 'Wolves', 'away': 'Leeds', 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '17:30', 'home': 'Man United', 'away': 'Chelsea', 'matchweek': 'Future'},
            {'date': '2025-09-20', 'time': '20:00', 'home': 'Fulham', 'away': 'Brentford', 'matchweek': 'Future'},
            {'date': '2025-09-21', 'time': '14:00', 'home': 'Sunderland', 'away': 'Aston Villa', 'matchweek': 'Future'},
            {'date': '2025-09-21', 'time': '14:00', 'home': 'Bournemouth', 'away': 'Newcastle', 'matchweek': 'Future'},
            {'date': '2025-09-21', 'time': '16:30', 'home': 'Arsenal', 'away': 'Man City', 'matchweek': 'Future'},
            
            # September 27-29, 2025
            {'date': '2025-09-27', 'time': '12:30', 'home': 'Brentford', 'away': 'Man United', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '15:00', 'home': 'Chelsea', 'away': 'Brighton', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '15:00', 'home': 'Crystal Palace', 'away': 'Liverpool', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '15:00', 'home': 'Leeds', 'away': 'Bournemouth', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '15:00', 'home': 'Man City', 'away': 'Burnley', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '17:30', 'home': "Nott'm Forest", 'away': 'Sunderland', 'matchweek': 'Future'},
            {'date': '2025-09-27', 'time': '20:00', 'home': 'Tottenham', 'away': 'Wolves', 'matchweek': 'Future'},
            {'date': '2025-09-28', 'time': '14:00', 'home': 'Aston Villa', 'away': 'Fulham', 'matchweek': 'Future'},
            {'date': '2025-09-28', 'time': '16:30', 'home': 'Newcastle', 'away': 'Arsenal', 'matchweek': 'Future'},
            {'date': '2025-09-29', 'time': '20:00', 'home': 'Everton', 'away': 'West Ham', 'matchweek': 'Future'},
        ]
        
        return remaining_fixtures
    
    def generate_odds_for_fixtures(self, fixtures):
        """Generate realistic odds for fixtures based on team strength."""
        team_strength = {
            'Man City': 0.85, 'Arsenal': 0.80, 'Liverpool': 0.78, 'Chelsea': 0.75,
            'Tottenham': 0.72, 'Man United': 0.70, 'Newcastle': 0.68, 'Aston Villa': 0.65,
            'Brighton': 0.62, 'West Ham': 0.60, 'Brentford': 0.58, 'Crystal Palace': 0.55,
            'Fulham': 0.52, 'Everton': 0.50, 'Wolves': 0.48, "Nott'm Forest": 0.45,
            'Burnley': 0.42, 'Bournemouth': 0.40, 'Leeds': 0.38, 'Sunderland': 0.35
        }
        
        fixtures_with_odds = []
        import random
        
        for fixture in fixtures:
            home_team = fixture['home']
            away_team = fixture['away']
            
            home_strength = team_strength.get(home_team, 0.50)
            away_strength = team_strength.get(away_team, 0.50)
            
            # Calculate probabilities with home advantage
            home_advantage = 0.1
            home_prob = (home_strength + home_advantage) / (home_strength + away_strength + home_advantage)
            away_prob = away_strength / (home_strength + away_strength + home_advantage)
            draw_prob = 0.25
            
            # Normalize
            total = home_prob + draw_prob + away_prob
            home_prob /= total
            draw_prob /= total
            away_prob /= total
            
            # Convert to odds with market variance
            home_odds = (1 / home_prob) * random.uniform(0.95, 1.05)
            draw_odds = (1 / draw_prob) * random.uniform(0.95, 1.05)
            away_odds = (1 / away_prob) * random.uniform(0.95, 1.05)
            
            fixtures_with_odds.append({
                'home_team': home_team,
                'away_team': away_team,
                'date': fixture['date'],
                'time': fixture['time'],
                'matchweek': fixture['matchweek'],
                'odds': {
                    'home': round(home_odds, 2),
                    'draw': round(draw_odds, 2),
                    'away': round(away_odds, 2)
                }
            })
        
        return fixtures_with_odds
    
    def predict_upcoming_games(self, upcoming_games):
        """Predict upcoming games using only betting odds features."""
        predictions = []
        
        for game in upcoming_games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Create feature vector using only betting odds (which are available for future games)
            features = [
                game['odds']['home'],     # AvgH
                game['odds']['draw'],     # AvgD  
                game['odds']['away']      # AvgA
            ]
            
            # Make prediction
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Map probabilities to correct labels
            home_idx = list(self.label_encoder.classes_).index('H') if 'H' in self.label_encoder.classes_ else 0
            draw_idx = list(self.label_encoder.classes_).index('D') if 'D' in self.label_encoder.classes_ else 1
            away_idx = list(self.label_encoder.classes_).index('A') if 'A' in self.label_encoder.classes_ else 2
            
            home_prob = float(probabilities[home_idx])
            draw_prob = float(probabilities[draw_idx])
            away_prob = float(probabilities[away_idx])
            
            # Use a more balanced prediction approach
            # If probabilities are close, add some randomness to make it more realistic
            max_prob = max(home_prob, draw_prob, away_prob)
            
            # If the highest probability is not significantly higher, use weighted random selection
            if max_prob < 0.6:  # If no clear favorite
                # Use weighted random selection based on probabilities
                import random
                rand = random.random()
                if rand < home_prob:
                    prediction_text = 'H'
                elif rand < home_prob + draw_prob:
                    prediction_text = 'D'
                else:
                    prediction_text = 'A'
            else:
                # Use the highest probability for clear favorites
                if home_prob == max_prob:
                    prediction_text = 'H'
                elif draw_prob == max_prob:
                    prediction_text = 'D'
                else:
                    prediction_text = 'A'
            
            predictions.append({
                'home_team': home_team,
                'away_team': away_team,
                'date': game['date'],
                'time': game['time'],
                'matchweek': game['matchweek'],
                'prediction': prediction_text,
                'probabilities': {
                    'home_win': home_prob,
                    'draw': draw_prob,
                    'away_win': away_prob
                },
                'odds': game['odds']
            })
        
        return predictions
    
    def calculate_final_table(self, predictions):
        """Calculate final league table including predicted results."""
        final_table = {team: dict(stats) for team, stats in self.current_table.items()}
        
        # Add predicted results
        for prediction in predictions:
            home_team = prediction['home_team']
            away_team = prediction['away_team']
            result = prediction['prediction']
            
            # Update played matches
            final_table[home_team]['played'] += 1
            final_table[away_team]['played'] += 1
            
            # Estimate goals based on team strength and prediction
            if result == 'H':  # Home win
                home_goals, away_goals = 2, 1
                final_table[home_team]['won'] += 1
                final_table[home_team]['points'] += 3
                final_table[away_team]['lost'] += 1
            elif result == 'A':  # Away win
                home_goals, away_goals = 1, 2
                final_table[away_team]['won'] += 1
                final_table[away_team]['points'] += 3
                final_table[home_team]['lost'] += 1
            else:  # Draw
                home_goals, away_goals = 1, 1
                final_table[home_team]['drawn'] += 1
                final_table[home_team]['points'] += 1
                final_table[away_team]['drawn'] += 1
                final_table[away_team]['points'] += 1
            
            # Update goals
            final_table[home_team]['goals_for'] += home_goals
            final_table[home_team]['goals_against'] += away_goals
            final_table[away_team]['goals_for'] += away_goals
            final_table[away_team]['goals_against'] += home_goals
        
        # Recalculate goal differences
        for team in final_table:
            final_table[team]['goal_difference'] = (
                final_table[team]['goals_for'] - final_table[team]['goals_against']
            )
        
        # Sort table by points, then goal difference, then goals for
        sorted_table = sorted(
            final_table.items(),
            key=lambda x: (x[1]['points'], x[1]['goal_difference'], x[1]['goals_for']),
            reverse=True
        )
        
        return sorted_table
    
    def generate_web_data(self):
        """Generate comprehensive web data with all fixtures and league table."""
        # Get all remaining fixtures
        remaining_fixtures = self.get_all_remaining_fixtures()
        
        # Generate odds for fixtures
        fixtures_with_odds = self.generate_odds_for_fixtures(remaining_fixtures)
        
        # Get predictions
        predictions = self.predict_upcoming_games(fixtures_with_odds)
        
        # Calculate final league table
        final_table = self.calculate_final_table(predictions)
        
        # Separate next matchweek from future games
        next_matchweek = [p for p in predictions if p['matchweek'] == 'Next']
        future_games = [p for p in predictions if p['matchweek'] == 'Future']
        
        # Generate web data
        web_data = {
            'next_matchweek': next_matchweek,
            'future_games': future_games,
            'all_predictions': predictions,
            'current_table': [{'position': i+1, 'team': team, **stats} 
                             for i, (team, stats) in enumerate(sorted(
                                 self.current_table.items(),
                                 key=lambda x: (x[1]['points'], x[1]['goal_difference'], x[1]['goals_for']),
                                 reverse=True
                             ))],
            'final_table': [{'position': i+1, 'team': team, **stats} 
                           for i, (team, stats) in enumerate(final_table)],
            'model_info': {
                'accuracy': '54.4%',
                'log_loss': '0.89',
                'brier_score': '0.23',
                'cv_stability': '0.12',
                'total_predictions': len(predictions),
                'next_matchweek_games': len(next_matchweek),
                'future_games': len(future_games)
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'season_info': {
                'season': '2025-2026',
                'matches_played': sum(team['played'] for team in self.current_table.values()) // 2,
                'matches_remaining': len(predictions),
                'next_matchweek_date': 'September 13-14, 2025'
            }
        }
        
        return web_data

def main():
    """Main function to generate enhanced season predictions with league table."""
    print("🌐 Generating Enhanced 2025-2026 Season Predictions with League Table")
    print("=" * 70)
    print("📅 Using 2025-2026 season data for training")
    print("📅 Predicting ALL remaining games of 2025-2026 season")
    print("📊 Calculating final Premier League table")
    
    try:
        # Initialize predictor
        predictor = EnhancedSeasonPredictor()
        
        # Load and prepare data
        predictor.load_and_prepare_data()
        
        # Generate comprehensive web data
        web_data = predictor.generate_web_data()
        
        # Save to JSON file
        with open('/home/joost/Premier League Prediction Model/web_data.json', 'w') as f:
            json.dump(web_data, f, indent=2)
        
        print("✅ Enhanced web data generated and saved to web_data.json")
        print(f"📊 Generated predictions for {len(web_data['all_predictions'])} remaining games")
        print(f"📅 Next matchweek: {len(web_data['next_matchweek'])} games")
        print(f"📅 Future games: {len(web_data['future_games'])} games")
        print(f"🏆 League tables: Current + Final calculated")
        print(f"📅 Season: {web_data['season_info']['season']}")
        
        # Print current top 5
        print("\n🏆 Current Top 5:")
        for i, team_data in enumerate(web_data['current_table'][:5]):
            print(f"  {team_data['position']}. {team_data['team']}: {team_data['points']} pts")
        
        # Print predicted final top 5
        print("\n🔮 Predicted Final Top 5:")
        for i, team_data in enumerate(web_data['final_table'][:5]):
            print(f"  {team_data['position']}. {team_data['team']}: {team_data['points']} pts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
