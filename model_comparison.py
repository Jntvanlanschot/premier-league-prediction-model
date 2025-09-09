#!/usr/bin/env python3
"""
Premier League Model Comparison: Data-Driven vs Human-Defined Priority
=====================================================================

Compares the current data-driven feature importance vs your human-defined priority order
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    """Compare data-driven vs human-defined priority models."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        
    def load_data(self, file_path):
        """Load the processed data."""
        self.data = pd.read_csv(file_path)
        print(f"📊 Loaded {len(self.data)} matches")
        return self
    
    def create_train_test_split(self):
        """Create proper train/test splits by season."""
        print("\n📊 CREATING TRAIN/TEST SPLITS")
        print("=" * 40)
        
        # Convert Date to datetime for proper sorting
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Train: 2023-24 + 1st half 2024-25
        # Test: 2nd half 2024-25
        train_data = []
        test_data = []
        
        for season in [2324, 2425]:
            season_data = self.data[self.data['Season'] == season].copy()
            
            if season == 2324:
                train_data.append(season_data)
                print(f"  {season} season: {len(season_data)} matches → TRAIN")
                
            elif season == 2425:
                season_data = season_data.sort_values('Date').reset_index(drop=True)
                mid_point = len(season_data) // 2
                
                first_half = season_data.iloc[:mid_point]
                second_half = season_data.iloc[mid_point:]
                
                train_data.append(first_half)
                test_data.append(second_half)
                
                print(f"  {season} season: {len(first_half)} matches → TRAIN, {len(second_half)} matches → TEST")
        
        self.train_data = pd.concat(train_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)
        
        print(f"\n✅ Train set: {len(self.train_data)} matches")
        print(f"✅ Test set: {len(self.test_data)} matches")
        
        return self
    
    def get_feature_sets(self):
        """Define feature sets for both approaches."""
        
        # Current data-driven features (from focused_model.py)
        data_driven_features = [
            'avg_home_odds', 'avg_draw_odds', 'avg_away_odds',
            'home_team_goals_scored_5', 'home_team_goals_conceded_5',
            'away_team_goals_scored_5', 'away_team_goals_conceded_5',
            'home_team_goal_difference_5', 'away_team_goal_difference_5',
            'home_team_shots_5', 'home_team_shots_conceded_5',
            'away_team_shots_5', 'away_team_shots_conceded_5',
            'home_team_shots_on_target_5', 'home_team_shots_on_target_conceded_5',
            'away_team_shots_on_target_5', 'away_team_shots_on_target_conceded_5',
            'home_team_wins_5', 'home_team_draws_5', 'home_team_losses_5',
            'away_team_wins_5', 'away_team_draws_5', 'away_team_losses_5',
            'home_team_corners_5', 'home_team_corners_conceded_5',
            'away_team_corners_5', 'away_team_corners_conceded_5',
            'home_team_cards_5', 'home_team_cards_conceded_5',
            'away_team_cards_5', 'away_team_cards_conceded_5',
            'goal_diff_advantage', 'form_advantage', 'dominance_advantage'
        ]
        
        # Your human-defined priority features (in your order)
        human_priority_features = [
            # 1. Bookmaker Odds
            'avg_home_odds', 'avg_draw_odds', 'avg_away_odds',
            
            # 2. Previous Wins/Losses
            'home_team_wins_5', 'home_team_draws_5', 'home_team_losses_5',
            'away_team_wins_5', 'away_team_draws_5', 'away_team_losses_5',
            
            # 3. Goals
            'home_team_goals_scored_5', 'home_team_goals_conceded_5',
            'away_team_goals_scored_5', 'away_team_goals_conceded_5',
            'home_team_goal_difference_5', 'away_team_goal_difference_5',
            
            # 4. Shots
            'home_team_shots_5', 'home_team_shots_conceded_5',
            'away_team_shots_5', 'away_team_shots_conceded_5',
            'home_team_shots_on_target_5', 'home_team_shots_on_target_conceded_5',
            'away_team_shots_on_target_5', 'away_team_shots_on_target_conceded_5',
            
            # 5. Home/Away Performance
            'home_home_wins_5', 'home_home_draws_5', 'home_home_losses_5',
            'home_away_wins_5', 'home_away_draws_5', 'home_away_losses_5',
            'away_home_wins_5', 'away_home_draws_5', 'away_home_losses_5',
            'away_away_wins_5', 'away_away_draws_5', 'away_away_losses_5',
            
            # 6. Corners/Cards
            'home_team_corners_5', 'home_team_corners_conceded_5',
            'away_team_corners_5', 'away_team_corners_conceded_5',
            'home_team_cards_5', 'home_team_cards_conceded_5',
            'away_team_cards_5', 'away_team_cards_conceded_5'
        ]
        
        # Filter to existing features
        data_driven_features = [f for f in data_driven_features if f in self.data.columns]
        human_priority_features = [f for f in human_priority_features if f in self.data.columns]
        
        return data_driven_features, human_priority_features
    
    def train_model(self, features, model_name, apply_weights=False):
        """Train a model with given features."""
        
        # Prepare data
        X_train = self.train_data[features].fillna(0)
        y_train = self.train_data['FTR']
        X_test = self.test_data[features].fillna(0)
        y_test = self.test_data['FTR']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply human-defined weights if requested
        if apply_weights:
            X_train_scaled, X_test_scaled = self.apply_human_priority_weights(
                X_train_scaled, X_test_scaled, features
            )
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_original)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'predictions': y_pred_original,
            'actual': y_test,
            'model': model,
            'features': features
        }
        
        return accuracy
    
    def apply_human_priority_weights(self, X_train, X_test, features):
        """Apply weights based on your human-defined priority order."""
        
        # Define priority weights
        priority_weights = {
            # 1. Bookmaker Odds (weight: 1.0)
            'avg_home_odds': 1.0, 'avg_draw_odds': 1.0, 'avg_away_odds': 1.0,
            
            # 2. Previous Wins/Losses (weight: 0.8)
            'home_team_wins_5': 0.8, 'home_team_draws_5': 0.8, 'home_team_losses_5': 0.8,
            'away_team_wins_5': 0.8, 'away_team_draws_5': 0.8, 'away_team_losses_5': 0.8,
            
            # 3. Goals (weight: 0.6)
            'home_team_goals_scored_5': 0.6, 'home_team_goals_conceded_5': 0.6,
            'away_team_goals_scored_5': 0.6, 'away_team_goals_conceded_5': 0.6,
            'home_team_goal_difference_5': 0.6, 'away_team_goal_difference_5': 0.6,
            
            # 4. Shots (weight: 0.4)
            'home_team_shots_5': 0.4, 'home_team_shots_conceded_5': 0.4,
            'away_team_shots_5': 0.4, 'away_team_shots_conceded_5': 0.4,
            'home_team_shots_on_target_5': 0.4, 'home_team_shots_on_target_conceded_5': 0.4,
            'away_team_shots_on_target_5': 0.4, 'away_team_shots_on_target_conceded_5': 0.4,
            
            # 5. Home/Away Performance (weight: 0.3)
            'home_home_wins_5': 0.3, 'home_home_draws_5': 0.3, 'home_home_losses_5': 0.3,
            'home_away_wins_5': 0.3, 'home_away_draws_5': 0.3, 'home_away_losses_5': 0.3,
            'away_home_wins_5': 0.3, 'away_home_draws_5': 0.3, 'away_home_losses_5': 0.3,
            'away_away_wins_5': 0.3, 'away_away_draws_5': 0.3, 'away_away_losses_5': 0.3,
            
            # 6. Corners/Cards (weight: 0.2)
            'home_team_corners_5': 0.2, 'home_team_corners_conceded_5': 0.2,
            'away_team_corners_5': 0.2, 'away_team_corners_conceded_5': 0.2,
            'home_team_cards_5': 0.2, 'home_team_cards_conceded_5': 0.2,
            'away_team_cards_5': 0.2, 'away_team_cards_conceded_5': 0.2
        }
        
        # Apply weights
        X_train_weighted = X_train.copy()
        X_test_weighted = X_test.copy()
        
        for i, feature in enumerate(features):
            if feature in priority_weights:
                weight = priority_weights[feature]
                X_train_weighted[:, i] *= weight
                X_test_weighted[:, i] *= weight
        
        return X_train_weighted, X_test_weighted
    
    def compare_models(self):
        """Compare all model approaches."""
        print("\n🏆 MODEL COMPARISON")
        print("=" * 50)
        
        # Get feature sets
        data_driven_features, human_priority_features = self.get_feature_sets()
        
        print(f"Data-driven features: {len(data_driven_features)}")
        print(f"Human priority features: {len(human_priority_features)}")
        
        # Train models
        print("\n🤖 Training models...")
        
        # 1. Current data-driven approach
        acc1 = self.train_model(data_driven_features, "Data-Driven (Current)", apply_weights=False)
        print(f"  ✅ Data-Driven Model: {acc1:.3f}")
        
        # 2. Human priority order (no weights)
        acc2 = self.train_model(human_priority_features, "Human Priority Order", apply_weights=False)
        print(f"  ✅ Human Priority Order: {acc2:.3f}")
        
        # 3. Human priority order (with weights)
        acc3 = self.train_model(human_priority_features, "Human Priority Weighted", apply_weights=True)
        print(f"  ✅ Human Priority Weighted: {acc3:.3f}")
        
        # Compare results
        print("\n📊 COMPARISON RESULTS")
        print("=" * 40)
        
        results = [
            ("Data-Driven (Current)", acc1),
            ("Human Priority Order", acc2),
            ("Human Priority Weighted", acc3)
        ]
        
        # Sort by accuracy
        results.sort(key=lambda x: x[1], reverse=True)
        
        print("Ranking by Test Accuracy:")
        for i, (name, accuracy) in enumerate(results, 1):
            print(f"  {i}. {name:<25}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Determine winner
        best_model = results[0]
        print(f"\n🏆 WINNER: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]:.3f} ({best_model[1]*100:.1f}%)")
        
        # Show improvement
        current_accuracy = acc1
        human_weighted_accuracy = acc3
        
        if human_weighted_accuracy > current_accuracy:
            improvement = human_weighted_accuracy - current_accuracy
            print(f"\n✅ Your Human-Defined Priority is BETTER by {improvement:.3f} ({improvement*100:.1f}%)")
        elif human_weighted_accuracy < current_accuracy:
            difference = current_accuracy - human_weighted_accuracy
            print(f"\n❌ Your Human-Defined Priority is WORSE by {difference:.3f} ({difference*100:.1f}%)")
        else:
            print(f"\n🤝 Both approaches perform equally well!")
        
        return results
    
    def show_detailed_comparison(self):
        """Show detailed comparison of predictions."""
        print("\n📋 DETAILED COMPARISON")
        print("=" * 40)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            
            # Show prediction breakdown
            correct_predictions = (result['predictions'] == result['actual'])
            
            for outcome in ['H', 'D', 'A']:
                outcome_mask = result['actual'] == outcome
                outcome_correct = correct_predictions[outcome_mask].sum()
                outcome_total = outcome_mask.sum()
                outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
                
                outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
                print(f"    {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")

def main():
    """Main execution function."""
    print("🏆 Premier League Model Comparison")
    print("=" * 50)
    print("Testing: Data-Driven vs Human-Defined Priority")
    
    try:
        # Initialize comparison
        comparison = ModelComparison()
        
        # Load data
        comparison.load_data('/home/joost/Premier League Prediction Model/corrected_processed_data.csv')
        
        # Create train/test splits
        comparison.create_train_test_split()
        
        # Compare models
        results = comparison.compare_models()
        
        # Show detailed comparison
        comparison.show_detailed_comparison()
        
        print(f"\n🎯 FINAL VERDICT:")
        print(f"   Your human-defined priority order has been tested against the current data-driven approach.")
        print(f"   Check the results above to see which performs better on the test data!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
