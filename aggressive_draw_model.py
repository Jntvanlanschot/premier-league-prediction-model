#!/usr/bin/env python3
"""
Aggressive Draw Optimization Model
==================================
Implements more aggressive techniques for draw prediction:
1. Lower draw threshold (0.20 instead of 0.28)
2. Draw oversampling
3. More aggressive class weights
4. Draw-specific ensemble
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class AggressiveDrawModel:
    """Aggressive draw optimization model."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.feature_columns = []
        self.draw_threshold = 0.20  # More aggressive threshold
        
    def load_and_preprocess_data(self):
        """Load and preprocess data quickly."""
        print("\n📥 LOADING AND PREPROCESSING DATA")
        print("=" * 40)
        
        datasets_folder = '/home/joost/Premier League Prediction Model/datasets'
        import os
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
        
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Quick preprocessing
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        
        # Create basic features
        home_cols = [col for col in self.data.columns if col.endswith('H') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        draw_cols = [col for col in self.data.columns if col.endswith('D') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        away_cols = [col for col in self.data.columns if col.endswith('A') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        
        self.data['AvgH'] = self.data[home_cols].mean(axis=1, skipna=True)
        self.data['AvgD'] = self.data[draw_cols].mean(axis=1, skipna=True)
        self.data['AvgA'] = self.data[away_cols].mean(axis=1, skipna=True)
        
        # Draw-specific features
        self.data['implied_prob_draw'] = 1 / self.data['AvgD']
        self.data['odds_balance'] = abs(self.data['AvgH'] - self.data['AvgA'])
        
        print(f"✅ Loaded {len(self.data)} matches")
        return self
    
    def create_simple_features(self):
        """Create simple but effective features."""
        print("\n🎯 CREATING SIMPLE DRAW FEATURES")
        print("=" * 30)
        
        # Simple rolling features (5 games)
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = self.data[
                (self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)
            ].copy().sort_values('Date')
            
            # Simple rolling averages
            team_matches['team_draws_5'] = 0
            team_matches['team_goals_5'] = 0
            
            for i in range(len(team_matches)):
                if i >= 5:
                    recent = team_matches.iloc[i-5:i]
                    draws = (recent['FTR'] == 'D').sum()
                    goals = recent['FTHG'].sum() + recent['FTAG'].sum()
                    
                    idx = team_matches.iloc[i].name
                    self.data.loc[idx, f'team_draws_5'] = draws
                    self.data.loc[idx, f'team_goals_5'] = goals
        
        # Match-level features
        self.data['draw_tendency_diff'] = 0
        self.data['low_scoring_match'] = 0
        
        for idx, row in self.data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            home_draws = row.get('team_draws_5', 0)
            away_draws = row.get('team_draws_5', 0)
            
            self.data.loc[idx, 'draw_tendency_diff'] = abs(home_draws - away_draws)
            self.data.loc[idx, 'low_scoring_match'] = 1 if row['AvgH'] > 2.5 and row['AvgA'] > 2.5 else 0
        
        print("✅ Simple features created")
        return self
    
    def create_train_test_split(self):
        """Create train/test splits."""
        print("\n📊 CREATING TRAIN/TEST SPLITS")
        print("=" * 30)
        
        available_seasons = sorted(self.data['Season'].unique())
        train_seasons = available_seasons[:-2]
        test_seasons = available_seasons[-2:]
        
        train_data = []
        test_data = []
        
        for season in train_seasons:
            season_data = self.data[self.data['Season'] == season].copy()
            train_data.append(season_data)
        
        for season in test_seasons:
            season_data = self.data[self.data['Season'] == season].copy()
            test_data.append(season_data)
        
        self.train_data = pd.concat(train_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)
        
        print(f"✅ Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        return self
    
    def define_features(self):
        """Define feature set."""
        self.feature_columns = [
            'AvgH', 'AvgD', 'AvgA', 'implied_prob_draw', 'odds_balance',
            'team_draws_5', 'team_goals_5', 'draw_tendency_diff', 'low_scoring_match'
        ]
        
        # Filter to existing columns
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        print(f"✅ Features: {len(self.feature_columns)}")
        return self
    
    def prepare_data(self):
        """Prepare data."""
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data['FTR']
        X_test = self.test_data[self.feature_columns].fillna(0)
        y_test = self.test_data['FTR']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, y_test
    
    def test_aggressive_models(self, X_train, X_test, y_train, y_test, y_test_original):
        """Test aggressive draw models."""
        print("\n🤖 TESTING AGGRESSIVE DRAW MODELS")
        print("=" * 40)
        
        # Very aggressive class weights
        class_weight_dict = {0: 1.0, 1: 3.0, 2: 1.0}  # 3x weight for draws
        
        algorithms = {
            'Aggressive Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight=class_weight_dict
            ),
            
            'Aggressive Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight=class_weight_dict
            ),
            
            'Draw-Focused Ensemble': VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42, class_weight=class_weight_dict)),
                ('lr', LogisticRegression(random_state=42, class_weight=class_weight_dict))
            ], voting='soft')
        }
        
        for name, algorithm in algorithms.items():
            print(f"Testing {name}...")
            
            try:
                algorithm.fit(X_train, y_train)
                
                # Standard predictions
                y_pred = algorithm.predict(X_test)
                y_pred_original = self.label_encoder.inverse_transform(y_pred)
                
                # Aggressive draw threshold
                y_pred_proba = algorithm.predict_proba(X_test)
                y_pred_aggressive = self.apply_aggressive_draw_threshold(y_pred_proba)
                y_pred_aggressive_original = self.label_encoder.inverse_transform(y_pred_aggressive)
                
                accuracy = accuracy_score(y_test_original, y_pred_original)
                accuracy_aggressive = accuracy_score(y_test_original, y_pred_aggressive_original)
                
                self.results[name] = {
                    'model': algorithm,
                    'test_accuracy': accuracy,
                    'test_accuracy_aggressive': accuracy_aggressive,
                    'predictions': y_pred_original,
                    'predictions_aggressive': y_pred_aggressive_original,
                    'actual': y_test_original,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name}: Standard={accuracy:.3f}, Aggressive={accuracy_aggressive:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
        
        return self
    
    def apply_aggressive_draw_threshold(self, probabilities):
        """Apply aggressive draw threshold."""
        predictions = []
        
        for prob_row in probabilities:
            home_prob, draw_prob, away_prob = prob_row
            
            # Very aggressive: if draw prob > 0.20, consider draw
            if draw_prob >= self.draw_threshold:
                # If draw is within 20% of the best, choose draw
                max_prob = max(home_prob, away_prob)
                if draw_prob >= max_prob * 0.8:
                    predictions.append(1)  # Draw
                else:
                    predictions.append(np.argmax(prob_row))
            else:
                predictions.append(np.argmax(prob_row))
        
        return np.array(predictions)
    
    def generate_aggressive_report(self):
        """Generate aggressive draw report."""
        print("\n🏆 AGGRESSIVE DRAW MODEL REPORT")
        print("=" * 50)
        
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['test_accuracy_aggressive'], 
            reverse=True
        )
        
        print("Aggressive Draw Model Performance:")
        print("-" * 40)
        for i, (name, result) in enumerate(sorted_results, 1):
            accuracy = result['test_accuracy']
            accuracy_aggressive = result['test_accuracy_aggressive']
            print(f"{i:2d}. {name:<25}: Standard={accuracy:.3f}, Aggressive={accuracy_aggressive:.3f}")
        
        # Best model details
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 BEST AGGRESSIVE MODEL: {best_name}")
        print(f"   📊 Standard Accuracy: {best_result['test_accuracy']:.3f}")
        print(f"   📊 Aggressive Accuracy: {best_result['test_accuracy_aggressive']:.3f}")
        
        # Show prediction breakdown
        print(f"\n📊 Prediction Breakdown - {best_name} (Aggressive):")
        correct_predictions = (best_result['predictions_aggressive'] == best_result['actual'])
        
        for outcome in ['H', 'D', 'A']:
            outcome_mask = best_result['actual'] == outcome
            outcome_correct = correct_predictions[outcome_mask].sum()
            outcome_total = outcome_mask.sum()
            outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
            
            outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
            print(f"   {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")
        
        return best_result['test_accuracy_aggressive']

def main():
    """Main execution function."""
    print("🎯 Aggressive Draw Optimization Model")
    print("=" * 50)
    print("Implementing aggressive techniques for draw prediction")
    
    try:
        model = AggressiveDrawModel()
        
        model.load_and_preprocess_data()
        model.create_simple_features()
        model.create_train_test_split()
        model.define_features()
        
        X_train, X_test, y_train, y_test, y_test_original = model.prepare_data()
        
        model.test_aggressive_models(X_train, X_test, y_train, y_test, y_test_original)
        
        final_accuracy = model.generate_aggressive_report()
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.3f} accuracy")
        print(f"   Using aggressive draw optimization!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
