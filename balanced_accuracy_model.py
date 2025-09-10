#!/usr/bin/env python3
"""
Balanced Accuracy Model - Maximum Accuracy with Smart Draw Handling
================================================================

Implements your recommended approach:
1. Maximum accuracy as primary objective
2. Smart draw-sensitive features (not over-weighted)
3. Secondary calibration for draw probabilities
4. Probability-based evaluation (log loss, Brier score)
5. Calibration methods (isotonic regression, Platt scaling)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BalancedAccuracyModel:
    """Balanced model prioritizing maximum accuracy with smart draw handling."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.feature_columns = []
        self.calibrated_model = None
        
    def load_data(self):
        """Load data from local datasets folder."""
        print("\n📥 LOADING DATA FROM LOCAL DATASETS")
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
        print(f"✅ Loaded {len(self.data)} matches from {len(all_data)} seasons")
        return self
    
    def preprocess_data(self):
        """Enhanced preprocessing with smart draw-sensitive features."""
        print("\n🔧 SMART PREPROCESSING")
        print("=" * 30)
        
        # Handle missing values
        betting_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
                       'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 
                       'VCH', 'VCD', 'VCA']
        
        for col in betting_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Create average betting odds
        home_cols = [col for col in self.data.columns if col.endswith('H') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        draw_cols = [col for col in self.data.columns if col.endswith('D') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        away_cols = [col for col in self.data.columns if col.endswith('A') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        
        self.data['AvgH'] = self.data[home_cols].mean(axis=1, skipna=True)
        self.data['AvgD'] = self.data[draw_cols].mean(axis=1, skipna=True)
        self.data['AvgA'] = self.data[away_cols].mean(axis=1, skipna=True)
        
        # 🎯 SMART DRAW-SENSITIVE FEATURES (not over-weighted)
        print("🎯 Creating smart draw-sensitive features...")
        
        # 1. Odds balance (closer odds = more likely draw)
        self.data['odds_balance_home_away'] = abs(self.data['AvgH'] - self.data['AvgA'])
        self.data['odds_balance_home_draw'] = abs(self.data['AvgH'] - self.data['AvgD'])
        self.data['odds_balance_away_draw'] = abs(self.data['AvgA'] - self.data['AvgD'])
        
        # 2. Market draw probability
        self.data['market_draw_probability'] = 1 / self.data['AvgD']
        
        # 3. Market uncertainty (higher disagreement = more draws)
        self.data['market_disagreement_home'] = self.data[home_cols].max(axis=1) - self.data[home_cols].min(axis=1)
        self.data['market_disagreement_draw'] = self.data[draw_cols].max(axis=1) - self.data[draw_cols].min(axis=1)
        self.data['market_disagreement_away'] = self.data[away_cols].max(axis=1) - self.data[away_cols].min(axis=1)
        
        # 4. Implied probabilities
        self.data['implied_prob_home'] = 1 / self.data['AvgH']
        self.data['implied_prob_draw'] = 1 / self.data['AvgD']
        self.data['implied_prob_away'] = 1 / self.data['AvgA']
        
        # Normalize probabilities
        total_prob = self.data['implied_prob_home'] + self.data['implied_prob_draw'] + self.data['implied_prob_away']
        self.data['implied_prob_home'] = self.data['implied_prob_home'] / total_prob
        self.data['implied_prob_draw'] = self.data['implied_prob_draw'] / total_prob
        self.data['implied_prob_away'] = self.data['implied_prob_away'] / total_prob
        
        # Handle missing values in match stats
        stats_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
        for col in stats_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        print("✅ Smart preprocessing completed")
        return self
    
    def create_rolling_features(self):
        """Create rolling features with smart draw-sensitive metrics."""
        print("\n📈 CREATING SMART ROLLING FEATURES")
        print("=" * 40)
        
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        print(f"Processing {len(all_teams)} teams...")
        
        windows = [3, 5, 10]
        
        # Initialize rolling columns
        rolling_cols = []
        for window in windows:
            rolling_cols.extend([
                f'team_wins_{window}', f'team_draws_{window}', f'team_losses_{window}',
                f'team_goals_scored_{window}', f'team_goals_conceded_{window}', f'team_goal_difference_{window}',
                f'team_shots_{window}', f'team_shots_conceded_{window}', f'team_shots_on_target_{window}', f'team_shots_on_target_conceded_{window}',
                f'team_corners_{window}', f'team_corners_conceded_{window}', f'team_cards_{window}', f'team_cards_conceded_{window}',
                f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
                f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}',
                # 🎯 Smart draw-sensitive features (not over-weighted)
                f'draw_tendency_{window}', f'goal_diff_balance_{window}'
            ])
        
        # Initialize all rolling columns
        for col in rolling_cols:
            self.data[col] = 0.0
        
        # Process each team
        for team in all_teams:
            team_matches = self.data[
                (self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)
            ].copy()
            
            team_matches = team_matches.sort_values(['Season', 'Date']).reset_index(drop=True)
            
            for season in team_matches['Season'].unique():
                season_matches = team_matches[team_matches['Season'] == season].copy()
                
                for window in windows:
                    for i in range(len(season_matches)):
                        start_idx = max(0, i - window)
                        recent_matches = season_matches.iloc[start_idx:i]
                        
                        if len(recent_matches) == 0:
                            continue
                        
                        # Calculate weighted averages
                        weights = np.exp(np.linspace(-1, 0, len(recent_matches)))
                        weights = weights / weights.sum()
                        
                        # Initialize metrics
                        team_wins = team_draws = team_losses = 0
                        team_goals_scored = team_goals_conceded = 0
                        team_shots = team_shots_conceded = 0
                        team_shots_on_target = team_shots_on_target_conceded = 0
                        team_corners = team_corners_conceded = 0
                        team_cards = team_cards_conceded = 0
                        
                        home_wins = home_draws = home_losses = 0
                        away_wins = away_draws = away_losses = 0
                        
                        # Calculate weighted metrics
                        for idx, (_, match) in enumerate(recent_matches.iterrows()):
                            weight = weights[idx]
                            is_home = match['HomeTeam'] == team
                            
                            if is_home:
                                team_goals_scored += match['FTHG'] * weight
                                team_goals_conceded += match['FTAG'] * weight
                                team_shots += match['HS'] * weight
                                team_shots_conceded += match['AS'] * weight
                                team_shots_on_target += match['HST'] * weight
                                team_shots_on_target_conceded += match['AST'] * weight
                                team_corners += match['HC'] * weight
                                team_corners_conceded += match['AC'] * weight
                                team_cards += (match['HY'] + match['HR']) * weight
                                team_cards_conceded += (match['AY'] + match['AR']) * weight
                                
                                if match['FTR'] == 'H':
                                    team_wins += weight
                                    home_wins += weight
                                elif match['FTR'] == 'D':
                                    team_draws += weight
                                    home_draws += weight
                                else:
                                    team_losses += weight
                                    home_losses += weight
                            else:
                                team_goals_scored += match['FTAG'] * weight
                                team_goals_conceded += match['FTHG'] * weight
                                team_shots += match['AS'] * weight
                                team_shots_conceded += match['HS'] * weight
                                team_shots_on_target += match['AST'] * weight
                                team_shots_on_target_conceded += match['HST'] * weight
                                team_corners += match['AC'] * weight
                                team_corners_conceded += match['HC'] * weight
                                team_cards += (match['AY'] + match['AR']) * weight
                                team_cards_conceded += (match['HY'] + match['HR']) * weight
                                
                                if match['FTR'] == 'A':
                                    team_wins += weight
                                    away_wins += weight
                                elif match['FTR'] == 'D':
                                    team_draws += weight
                                    away_draws += weight
                                else:
                                    team_losses += weight
                                    away_losses += weight
                        
                        # Store rolling features
                        match_idx = season_matches.iloc[i].name
                        
                        self.data.loc[match_idx, f'team_wins_{window}'] = team_wins
                        self.data.loc[match_idx, f'team_draws_{window}'] = team_draws
                        self.data.loc[match_idx, f'team_losses_{window}'] = team_losses
                        self.data.loc[match_idx, f'team_goals_scored_{window}'] = team_goals_scored
                        self.data.loc[match_idx, f'team_goals_conceded_{window}'] = team_goals_conceded
                        self.data.loc[match_idx, f'team_goal_difference_{window}'] = team_goals_scored - team_goals_conceded
                        
                        self.data.loc[match_idx, f'team_shots_{window}'] = team_shots
                        self.data.loc[match_idx, f'team_shots_conceded_{window}'] = team_shots_conceded
                        self.data.loc[match_idx, f'team_shots_on_target_{window}'] = team_shots_on_target
                        self.data.loc[match_idx, f'team_shots_on_target_conceded_{window}'] = team_shots_on_target_conceded
                        self.data.loc[match_idx, f'team_corners_{window}'] = team_corners
                        self.data.loc[match_idx, f'team_corners_conceded_{window}'] = team_corners_conceded
                        self.data.loc[match_idx, f'team_cards_{window}'] = team_cards
                        self.data.loc[match_idx, f'team_cards_conceded_{window}'] = team_cards_conceded
                        
                        # Store home/away specific features
                        self.data.loc[match_idx, f'home_wins_{window}'] = home_wins
                        self.data.loc[match_idx, f'home_draws_{window}'] = home_draws
                        self.data.loc[match_idx, f'home_losses_{window}'] = home_losses
                        self.data.loc[match_idx, f'away_wins_{window}'] = away_wins
                        self.data.loc[match_idx, f'away_draws_{window}'] = away_draws
                        self.data.loc[match_idx, f'away_losses_{window}'] = away_losses
                        
                        # 🎯 Smart draw-sensitive features (not over-weighted)
                        # Draw tendency (percentage of draws)
                        total_matches = team_wins + team_draws + team_losses
                        draw_tendency = team_draws / total_matches if total_matches > 0 else 0
                        self.data.loc[match_idx, f'draw_tendency_{window}'] = draw_tendency
                        
                        # Goal difference balance (teams with similar goal differences)
                        goal_diff_balance = 1 / (1 + abs(team_goals_scored - team_goals_conceded))
                        self.data.loc[match_idx, f'goal_diff_balance_{window}'] = goal_diff_balance
                        
                        # Weighted form score
                        weighted_form = (team_wins - team_losses) + (team_goals_scored - team_goals_conceded) * 0.1
                        self.data.loc[match_idx, f'weighted_form_{window}'] = weighted_form
                        
                        # Form streak
                        if len(recent_matches) >= 2:
                            recent_results = []
                            for _, match in recent_matches.iterrows():
                                is_home = match['HomeTeam'] == team
                                if is_home and match['FTR'] == 'H':
                                    recent_results.append(1)
                                elif not is_home and match['FTR'] == 'A':
                                    recent_results.append(1)
                                elif match['FTR'] == 'D':
                                    recent_results.append(0)
                                else:
                                    recent_results.append(-1)
                            
                            streak = 0
                            if recent_results:
                                current_result = recent_results[-1]
                                for result in reversed(recent_results):
                                    if result == current_result:
                                        streak += 1
                                    else:
                                        break
                                streak = streak if current_result == 1 else -streak
                            
                            self.data.loc[match_idx, f'form_streak_{window}'] = streak
                        
                        # Goal trend
                        if len(recent_matches) >= 3:
                            goal_diffs = []
                            for _, match in recent_matches.iterrows():
                                is_home = match['HomeTeam'] == team
                                if is_home:
                                    goal_diff = match['FTHG'] - match['FTAG']
                                else:
                                    goal_diff = match['FTAG'] - match['FTHG']
                                goal_diffs.append(goal_diff)
                            
                            if len(goal_diffs) >= 2:
                                trend = np.polyfit(range(len(goal_diffs)), goal_diffs, 1)[0]
                                self.data.loc[match_idx, f'goal_trend_{window}'] = trend
        
        print("✅ Smart rolling features created")
        return self
    
    def create_match_features(self):
        """Create match-level features with smart draw-sensitive metrics."""
        print("\n🎯 CREATING SMART MATCH FEATURES")
        print("=" * 30)
        
        windows = [3, 5, 10]
        
        # Initialize match features
        for window in windows:
            self.data[f'goal_diff_advantage_{window}'] = 0.0
            self.data[f'form_advantage_{window}'] = 0.0
            self.data[f'shots_advantage_{window}'] = 0.0
            self.data[f'dominance_advantage_{window}'] = 0.0
            
            # 🎯 Smart draw-sensitive match features
            self.data[f'draw_tendency_balance_{window}'] = 0.0
            self.data[f'goal_diff_balance_{window}'] = 0.0
        
        for idx, row in self.data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            for window in windows:
                # Get team features
                home_goal_diff = row.get(f'team_goal_difference_{window}', 0)
                away_goal_diff = row.get(f'team_goal_difference_{window}', 0)
                
                home_wins = row.get(f'team_wins_{window}', 0)
                home_losses = row.get(f'team_losses_{window}', 0)
                away_wins = row.get(f'team_wins_{window}', 0)
                away_losses = row.get(f'team_losses_{window}', 0)
                
                home_shots = row.get(f'team_shots_{window}', 0)
                away_shots = row.get(f'team_shots_{window}', 0)
                
                home_dominance = row.get(f'weighted_form_{window}', 0)
                away_dominance = row.get(f'weighted_form_{window}', 0)
                
                # 🎯 Smart draw-sensitive features
                home_draw_tendency = row.get(f'draw_tendency_{window}', 0)
                away_draw_tendency = row.get(f'draw_tendency_{window}', 0)
                
                home_goal_diff_balance = row.get(f'goal_diff_balance_{window}', 0)
                away_goal_diff_balance = row.get(f'goal_diff_balance_{window}', 0)
                
                # Create relative features
                self.data.loc[idx, f'goal_diff_advantage_{window}'] = home_goal_diff - away_goal_diff
                self.data.loc[idx, f'form_advantage_{window}'] = (home_wins - home_losses) - (away_wins - away_losses)
                self.data.loc[idx, f'shots_advantage_{window}'] = home_shots - away_shots
                self.data.loc[idx, f'dominance_advantage_{window}'] = home_dominance - away_dominance
                
                # 🎯 Smart draw-sensitive match features
                # Draw tendency balance (teams with similar draw tendencies)
                self.data.loc[idx, f'draw_tendency_balance_{window}'] = 1 - abs(home_draw_tendency - away_draw_tendency)
                
                # Goal difference balance (teams with similar goal difference patterns)
                self.data.loc[idx, f'goal_diff_balance_{window}'] = (home_goal_diff_balance + away_goal_diff_balance) / 2
        
        print("✅ Smart match features created")
        return self
    
    def create_train_test_split(self):
        """Create proper train/test splits."""
        print("\n📊 CREATING TRAIN/TEST SPLITS")
        print("=" * 30)
        
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
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
    
    def define_balanced_features(self):
        """Define balanced feature categories prioritizing accuracy."""
        print("\n🎯 DEFINING BALANCED FEATURES")
        print("=" * 30)
        
        # Balanced feature categories (accuracy-focused with smart draw features)
        self.priority_categories = {
            # 1. Core Odds Features (highest priority for accuracy)
            'core_odds': {
                'features': [
                    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
                    'market_draw_probability'
                ],
                'description': 'Core Odds Features'
            },
            
            # 2. Smart Draw-Sensitive Features (not over-weighted)
            'smart_draw': {
                'features': [
                    'odds_balance_home_away', 'odds_balance_home_draw', 'odds_balance_away_draw',
                    'market_disagreement_home', 'market_disagreement_draw', 'market_disagreement_away'
                ],
                'description': 'Smart Draw-Sensitive Features'
            },
            
            # 3. Enhanced Rolling Features
            'enhanced_rolling': {
                'features': [],
                'description': 'Enhanced Rolling Features'
            },
            
            # 4. Smart Match Context
            'smart_context': {
                'features': [],
                'description': 'Smart Match Context'
            }
        }
        
        # Add rolling features for all windows
        windows = [3, 5, 10]
        for window in windows:
            self.priority_categories['enhanced_rolling']['features'].extend([
                f'team_wins_{window}', f'team_draws_{window}', f'team_losses_{window}',
                f'team_goals_scored_{window}', f'team_goals_conceded_{window}', f'team_goal_difference_{window}',
                f'team_shots_{window}', f'team_shots_conceded_{window}', f'team_shots_on_target_{window}', f'team_shots_on_target_conceded_{window}',
                f'team_corners_{window}', f'team_corners_conceded_{window}', f'team_cards_{window}', f'team_cards_conceded_{window}',
                f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
                f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}',
                f'draw_tendency_{window}', f'goal_diff_balance_{window}'
            ])
        
        # Add match context features for all windows
        for window in windows:
            self.priority_categories['smart_context']['features'].extend([
                f'goal_diff_advantage_{window}', f'form_advantage_{window}',
                f'shots_advantage_{window}', f'dominance_advantage_{window}',
                f'draw_tendency_balance_{window}', f'goal_diff_balance_{window}'
            ])
        
        # Filter to existing features
        self.feature_columns = []
        for category, info in self.priority_categories.items():
            existing_features = [f for f in info['features'] if f in self.data.columns]
            self.priority_categories[category]['features'] = existing_features
            self.feature_columns.extend(existing_features)
        
        print("Balanced Feature Categories:")
        for category, info in self.priority_categories.items():
            print(f"  {info['description']:<25} - {len(info['features'])} features")
        
        print(f"\n✅ Total features: {len(self.feature_columns)}")
        return self
    
    def prepare_data(self):
        """Prepare data for training."""
        print("\n🔧 PREPARING DATA")
        print("=" * 20)
        
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data['FTR']
        X_test = self.test_data[self.feature_columns].fillna(0)
        y_test = self.test_data['FTR']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"✅ Training features: {X_train_scaled.shape}")
        print(f"✅ Test features: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, y_test
    
    def handle_class_imbalance(self, y_train):
        """Handle class imbalance with balanced weights."""
        print("\n⚖️ HANDLING CLASS IMBALANCE")
        print("=" * 30)
        
        # Use balanced class weights (not over-weighted for draws)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Balanced class weights: {class_weight_dict}")
        
        # Show class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:")
        for i, count in enumerate(counts):
            result_name = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[i]
            print(f"  {result_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return class_weight_dict
    
    def test_balanced_algorithms(self, X_train, X_test, y_train, y_test, y_test_original, class_weight_dict):
        """Test balanced algorithms prioritizing accuracy."""
        print("\n🤖 TESTING BALANCED ALGORITHMS")
        print("=" * 40)
        
        # Balanced algorithms (accuracy-focused)
        algorithms = {
            'Balanced Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict
            ),
            
            'Balanced Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8
            ),
            
            'Balanced Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=0.1,
                penalty='l2',
                class_weight=class_weight_dict
            ),
            
            'Calibrated Random Forest': CalibratedClassifierCV(
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    max_depth=12,
                    class_weight=class_weight_dict
                ),
                method='isotonic',
                cv=3
            ),
            
            'Calibrated Logistic Regression': CalibratedClassifierCV(
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
        }
        
        # Test each algorithm
        for name, algorithm in algorithms.items():
            print(f"Testing {name}...")
            
            try:
                # Train model
                algorithm.fit(X_train, y_train)
                
                # Make predictions
                y_pred = algorithm.predict(X_test)
                y_pred_original = self.label_encoder.inverse_transform(y_pred)
                
                # Get probabilities
                y_pred_proba = algorithm.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_original, y_pred_original)
                logloss = log_loss(y_test, y_pred_proba)
                
                # Calculate Brier score for each class
                brier_scores = []
                for i in range(3):
                    class_mask = (y_test == i)
                    if class_mask.sum() > 0:
                        brier_score = brier_score_loss(class_mask, y_pred_proba[:, i])
                        brier_scores.append(brier_score)
                
                avg_brier_score = np.mean(brier_scores) if brier_scores else np.nan
                
                # Cross-validation
                cv_scores = cross_val_score(algorithm, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[name] = {
                    'model': algorithm,
                    'test_accuracy': accuracy,
                    'test_logloss': logloss,
                    'test_brier_score': avg_brier_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred_original,
                    'actual': y_test_original,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name}: Accuracy={accuracy:.3f}, LogLoss={logloss:.3f}, Brier={avg_brier_score:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
        
        return self
    
    def apply_secondary_calibration(self, best_model, X_test, y_test):
        """Apply secondary calibration for draw probabilities."""
        print("\n🎯 APPLYING SECONDARY CALIBRATION")
        print("=" * 40)
        
        # Get base probabilities
        base_proba = best_model.predict_proba(X_test)
        
        # Apply secondary calibration for draws
        calibrated_proba = base_proba.copy()
        
        # Find matches where draw features suggest balance
        draw_features = ['odds_balance_home_away', 'draw_tendency_balance_5', 'goal_diff_balance_5']
        draw_feature_cols = [col for col in draw_features if col in self.test_data.columns]
        
        if draw_feature_cols:
            # Calculate draw balance score
            draw_balance_scores = []
            for col in draw_feature_cols:
                if col in self.test_data.columns:
                    scores = self.test_data[col].fillna(0)
                    draw_balance_scores.append(scores)
            
            if draw_balance_scores:
                draw_balance_score = np.mean(draw_balance_scores, axis=0)
                
                # Adjust draw probabilities upward where balance suggests draws
                balance_threshold = np.percentile(draw_balance_score, 75)  # Top 25% most balanced
                balance_mask = draw_balance_score >= balance_threshold
                
                # Increase draw probability for balanced matches
                calibrated_proba[balance_mask, 1] *= 1.2  # 20% increase
                
                # Renormalize probabilities
                calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
                
                print(f"✅ Applied secondary calibration to {balance_mask.sum()} balanced matches")
        
        return calibrated_proba
    
    def generate_balanced_report(self):
        """Generate balanced accuracy report with probability evaluation."""
        print("\n🏆 BALANCED ACCURACY MODEL REPORT")
        print("=" * 70)
        
        # Sort by accuracy (primary objective)
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['test_accuracy'], 
            reverse=True
        )
        
        print("Balanced Model Performance Ranking (Accuracy-Focused):")
        print("-" * 60)
        for i, (name, result) in enumerate(sorted_results, 1):
            accuracy = result['test_accuracy']
            logloss = result.get('test_logloss', 'N/A')
            brier_score = result.get('test_brier_score', 'N/A')
            cv_mean = result.get('cv_mean', 'N/A')
            cv_std = result.get('cv_std', 'N/A')
            
            print(f"{i:2d}. {name:<30}: Accuracy={accuracy:.3f} ({accuracy*100:.1f}%)")
            if logloss != 'N/A':
                print(f"    LogLoss={logloss:.3f}")
            if brier_score != 'N/A':
                print(f"    Brier={brier_score:.3f}")
            if cv_mean != 'N/A':
                print(f"    CV={cv_mean:.3f}±{cv_std:.3f}")
        
        # Best model details
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 BEST BALANCED MODEL: {best_name}")
        print(f"   📊 Test Accuracy: {best_result['test_accuracy']:.3f} ({best_result['test_accuracy']*100:.1f}%)")
        
        if 'test_logloss' in best_result:
            print(f"   📊 Test LogLoss: {best_result['test_logloss']:.3f}")
        
        if 'test_brier_score' in best_result:
            print(f"   📊 Test Brier Score: {best_result['test_brier_score']:.3f}")
        
        if 'cv_mean' in best_result:
            print(f"   📊 Cross-Validation: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
        
        # Show prediction breakdown
        print(f"\n📊 Prediction Breakdown - {best_name}:")
        correct_predictions = (best_result['predictions'] == best_result['actual'])
        
        for outcome in ['H', 'D', 'A']:
            outcome_mask = best_result['actual'] == outcome
            outcome_correct = correct_predictions[outcome_mask].sum()
            outcome_total = outcome_mask.sum()
            outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
            
            outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
            print(f"   {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")
        
        # Show probability calibration
        print(f"\n🎯 PROBABILITY CALIBRATION ANALYSIS:")
        probabilities = best_result['probabilities']
        actual = best_result['actual']
        
        # Calculate average predicted probabilities
        avg_pred_proba = probabilities.mean(axis=0)
        actual_proba = np.array([(actual == i).mean() for i in range(3)])
        
        print(f"   Average Predicted Probabilities:")
        for i, (pred, actual) in enumerate(zip(avg_pred_proba, actual_proba)):
            outcome_name = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[i]
            print(f"     {outcome_name}: Predicted={pred:.3f}, Actual={actual:.3f}")
        
        # Draw probability analysis
        draw_pred_proba = probabilities[:, 1]
        draw_actual = (actual == 1)
        
        print(f"\n🎯 DRAW PROBABILITY ANALYSIS:")
        print(f"   Average Draw Probability: {draw_pred_proba.mean():.3f}")
        print(f"   Actual Draw Rate: {draw_actual.mean():.3f}")
        print(f"   Draw Probability Range: {draw_pred_proba.min():.3f} - {draw_pred_proba.max():.3f}")
        
        # Apply secondary calibration
        calibrated_proba = self.apply_secondary_calibration(best_result['model'], 
                                                           self.scaler.transform(self.test_data[self.feature_columns].fillna(0)), 
                                                           self.label_encoder.transform(self.test_data['FTR']))
        
        print(f"\n🎯 SECONDARY CALIBRATION RESULTS:")
        calibrated_draw_proba = calibrated_proba[:, 1]
        print(f"   Calibrated Draw Probability: {calibrated_draw_proba.mean():.3f}")
        print(f"   Calibrated Draw Range: {calibrated_draw_proba.min():.3f} - {calibrated_draw_proba.max():.3f}")
        
        return best_result['test_accuracy']

def main():
    """Main execution function."""
    print("🎯 Balanced Accuracy Model - Maximum Accuracy with Smart Draw Handling")
    print("=" * 80)
    print("Prioritizing maximum accuracy while maintaining realistic probability calibration")
    
    try:
        # Initialize balanced model
        model = BalancedAccuracyModel()
        
        # Load and preprocess data
        model.load_data()
        model.preprocess_data()
        model.create_rolling_features()
        model.create_match_features()
        
        # Create train/test splits
        model.create_train_test_split()
        
        # Define balanced features
        model.define_balanced_features()
        
        # Prepare data
        X_train, X_test, y_train, y_test, y_test_original = model.prepare_data()
        
        # Handle class imbalance
        class_weight_dict = model.handle_class_imbalance(y_train)
        
        # Test balanced algorithms
        model.test_balanced_algorithms(X_train, X_test, y_train, y_test, y_test_original, class_weight_dict)
        
        # Generate balanced report
        final_accuracy = model.generate_balanced_report()
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.3f} accuracy")
        print(f"   Maximum accuracy with smart draw handling!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
