#!/usr/bin/env python3
"""
Premier League Model - Hybrid Approach
======================================

Uses your human-defined priority order but lets the algorithm determine optimal weights
Combines domain expertise with data-driven optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class HybridPriorityModel:
    """Model that uses human-defined order but learns optimal weights."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.optimal_weights = {}
        self.feature_columns = []
        
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
    
    def define_priority_categories(self):
        """Define feature categories in your priority order."""
        print("\n🎯 DEFINING PRIORITY CATEGORIES")
        print("=" * 40)
        
        # Your human-defined priority order
        self.priority_categories = {
            # 1. Bookmaker Odds (highest priority)
            'bookmaker_odds': {
                'features': ['avg_home_odds', 'avg_draw_odds', 'avg_away_odds'],
                'description': 'Bookmaker Odds (baseline expectation)'
            },
            
            # 2. Previous Wins/Losses
            'wins_losses': {
                'features': [
                    'home_team_wins_5', 'home_team_draws_5', 'home_team_losses_5',
                    'away_team_wins_5', 'away_team_draws_5', 'away_team_losses_5'
                ],
                'description': 'Previous Wins/Losses'
            },
            
            # 3. Goals
            'goals': {
                'features': [
                    'home_team_goals_scored_5', 'home_team_goals_conceded_5',
                    'away_team_goals_scored_5', 'away_team_goals_conceded_5',
                    'home_team_goal_difference_5', 'away_team_goal_difference_5'
                ],
                'description': 'Goals Scored/Conceded'
            },
            
            # 4. Shots
            'shots': {
                'features': [
                    'home_team_shots_5', 'home_team_shots_conceded_5',
                    'away_team_shots_5', 'away_team_shots_conceded_5',
                    'home_team_shots_on_target_5', 'home_team_shots_on_target_conceded_5',
                    'away_team_shots_on_target_5', 'away_team_shots_on_target_conceded_5'
                ],
                'description': 'Shots/Shots on Target'
            },
            
            # 5. Home/Away Performance
            'home_away': {
                'features': [
                    'home_home_wins_5', 'home_home_draws_5', 'home_home_losses_5',
                    'home_away_wins_5', 'home_away_draws_5', 'home_away_losses_5',
                    'away_home_wins_5', 'away_home_draws_5', 'away_home_losses_5',
                    'away_away_wins_5', 'away_away_draws_5', 'away_away_losses_5'
                ],
                'description': 'Home/Away Performance'
            },
            
            # 6. Corners/Cards (lowest priority)
            'corners_cards': {
                'features': [
                    'home_team_corners_5', 'home_team_corners_conceded_5',
                    'away_team_corners_5', 'away_team_corners_conceded_5',
                    'home_team_cards_5', 'home_team_cards_conceded_5',
                    'away_team_cards_5', 'away_team_cards_conceded_5'
                ],
                'description': 'Corners/Cards'
            }
        }
        
        # Filter to existing features and build feature list
        self.feature_columns = []
        for category, info in self.priority_categories.items():
            existing_features = [f for f in info['features'] if f in self.data.columns]
            self.priority_categories[category]['features'] = existing_features
            self.feature_columns.extend(existing_features)
        
        print("Your Priority Order (weights to be learned):")
        for i, (category, info) in enumerate(self.priority_categories.items(), 1):
            print(f"  {i}. {info['description']:<25} - {len(info['features'])} features")
        
        print(f"\n✅ Total features: {len(self.feature_columns)}")
        
        return self
    
    def optimize_category_weights(self):
        """Fast weight optimization using greedy approach."""
        print("\n⚡ FAST WEIGHT OPTIMIZATION")
        print("=" * 40)
        
        # Prepare data
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data['FTR']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Start with your suggested weights as baseline
        baseline_weights = {
            'bookmaker_odds': 1.0,
            'wins_losses': 0.8,
            'goals': 0.6,
            'shots': 0.4,
            'home_away': 0.3,
            'corners_cards': 0.2
        }
        
        print("Starting with your suggested weights...")
        
        # Test each category individually to find optimal weight
        optimal_weights = baseline_weights.copy()
        
        for category in self.priority_categories.keys():
            print(f"  Optimizing {category}...")
            
            best_weight = optimal_weights[category]
            best_score = self.evaluate_weights(X_train_scaled, y_train_encoded, optimal_weights)
            
            # Test different weights for this category
            test_weights = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
            
            for weight in test_weights:
                test_weights_dict = optimal_weights.copy()
                test_weights_dict[category] = weight
                
                score = self.evaluate_weights(X_train_scaled, y_train_encoded, test_weights_dict)
                
                if score > best_score:
                    best_score = score
                    best_weight = weight
            
            optimal_weights[category] = best_weight
            print(f"    Best weight: {best_weight:.2f} (score: {best_score:.3f})")
        
        self.optimal_weights = optimal_weights
        
        print(f"\n✅ Fast optimization complete!")
        print("Final Optimal Weights:")
        for category, weight in optimal_weights.items():
            description = self.priority_categories[category]['description']
            print(f"  {description:<25}: {weight:.2f}")
        
        return self
    
    def evaluate_weights(self, X_scaled, y_encoded, weights):
        """Evaluate a set of weights using cross-validation."""
        # Apply weights
        X_weighted = self.apply_weights_to_features(X_scaled, weights)
        
        # Quick model for evaluation
        model = RandomForestClassifier(
            n_estimators=30,  # Faster
            random_state=42,
            max_depth=6
        )
        
        # Use cross-validation score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_weighted, y_encoded, cv=3, scoring='accuracy')
        return scores.mean()
    
    def apply_weights_to_features(self, X_scaled, weights):
        """Apply category weights to features."""
        X_weighted = X_scaled.copy()
        
        feature_idx = 0
        for category, info in self.priority_categories.items():
            category_features = info['features']
            weight = weights[category]
            
            for feature in category_features:
                if feature in self.feature_columns:
                    X_weighted[:, feature_idx] *= weight
                    feature_idx += 1
        
        return X_weighted
    
    def train_final_model(self):
        """Train the final model with optimal weights."""
        print("\n🤖 TRAINING FINAL HYBRID MODEL")
        print("=" * 40)
        
        # Prepare data
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data['FTR']
        X_test = self.test_data[self.feature_columns].fillna(0)
        y_test = self.test_data['FTR']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply optimal weights
        X_train_weighted = self.apply_weights_to_features(X_train_scaled, self.optimal_weights)
        X_test_weighted = self.apply_weights_to_features(X_test_scaled, self.optimal_weights)
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train final model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train_weighted, y_train_encoded)
        
        # Make predictions
        y_pred = self.model.predict(X_test_weighted)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_original)
        
        print(f"✅ Final model trained successfully")
        print(f"📊 Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Store results
        self.test_predictions = y_pred_original
        self.test_actual = y_test
        self.test_accuracy = accuracy
        
        return self
    
    def analyze_results(self):
        """Analyze the final results."""
        print("\n📊 FINAL RESULTS ANALYSIS")
        print("=" * 40)
        
        # Show optimal weights
        print("Learned Optimal Weights:")
        for category, weight in self.optimal_weights.items():
            description = self.priority_categories[category]['description']
            print(f"  {description:<25}: {weight:.2f}")
        
        # Show prediction breakdown
        correct_predictions = (self.test_predictions == self.test_actual)
        
        print(f"\nPrediction Breakdown:")
        for outcome in ['H', 'D', 'A']:
            outcome_mask = self.test_actual == outcome
            outcome_correct = correct_predictions[outcome_mask].sum()
            outcome_total = outcome_mask.sum()
            outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
            
            outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
            print(f"  {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")
        
        return self
    
    def generate_final_report(self):
        """Generate final report."""
        print("\n🏆 HYBRID MODEL FINAL REPORT")
        print("=" * 50)
        
        print(f"📊 FINAL TEST ACCURACY: {self.test_accuracy:.3f} ({self.test_accuracy*100:.1f}%)")
        print(f"📈 Test matches: {len(self.test_data)}")
        print(f"🎯 Features used: {len(self.feature_columns)}")
        
        print(f"\n✅ Model combines:")
        print(f"   🧠 Your domain expertise (priority order)")
        print(f"   📊 Data-driven optimization (optimal weights)")
        print(f"   🎯 Best of both worlds!")
        
        print(f"\n🎯 Your Priority Order (with learned weights):")
        for i, (category, weight) in enumerate(self.optimal_weights.items(), 1):
            description = self.priority_categories[category]['description']
            print(f"   {i}. {description:<25} (weight: {weight:.2f})")
        
        return self.test_accuracy

def main():
    """Main execution function."""
    print("🏆 Premier League Hybrid Priority Model")
    print("=" * 60)
    print("Your Priority Order + Data-Driven Optimal Weights")
    
    try:
        # Initialize model
        model = HybridPriorityModel()
        
        # Load data
        model.load_data('/home/joost/Premier League Prediction Model/corrected_processed_data.csv')
        
        # Create train/test splits
        model.create_train_test_split()
        
        # Define priority categories
        model.define_priority_categories()
        
        # Optimize category weights
        model.optimize_category_weights()
        
        # Train final model
        model.train_final_model()
        
        # Analyze results
        model.analyze_results()
        
        # Generate final report
        final_accuracy = model.generate_final_report()
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.3f} accuracy")
        print(f"   Using your priority order with data-driven optimal weights!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
