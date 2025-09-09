#!/usr/bin/env python3
"""
Premier League Model - Algorithm Comparison & Weight Discovery
=============================================================

Tests multiple algorithms to find the best accuracy and discover optimal weights:
- Gradient Boosting
- Regularized Logistic Regression  
- Random Forest
- SVM
- Neural Networks
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class AlgorithmComparison:
    """Compare different algorithms for optimal weight discovery."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.feature_columns = []
        
    def load_data(self):
        """Load data from local datasets folder."""
        print("\n📥 LOADING DATA FROM LOCAL DATASETS")
        print("=" * 40)
        
        # Define the datasets folder path
        datasets_folder = '/home/joost/Premier League Prediction Model/datasets'
        
        # List all CSV files in the datasets folder
        import os
        csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
        csv_files.sort()  # Sort to ensure consistent order
        
        print(f"Found {len(csv_files)} season files:")
        for file in csv_files:
            print(f"  - {file}")
        
        # Load all season data
        all_data = []
        
        for file in csv_files:
            file_path = os.path.join(datasets_folder, file)
            print(f"  Loading {file}...")
            
            try:
                df = pd.read_csv(file_path)
                
                # Extract season from filename (e.g., "2023_2024_season.csv" -> "2324")
                season_name = file.replace('_season.csv', '').replace('_', '')
                if len(season_name) == 8:  # e.g., "20232024"
                    season_code = season_name[2:4] + season_name[6:8]  # "2324"
                else:
                    season_code = season_name
                
                df['Season'] = season_code
                all_data.append(df)
                print(f"    ✅ {file}: {len(df)} matches loaded")
                
            except Exception as e:
                print(f"    ❌ Error loading {file}: {e}")
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 Total dataset: {len(self.data)} matches from {len(all_data)} seasons")
            
            # Show season distribution
            print("\nSeason Distribution:")
            season_counts = self.data['Season'].value_counts().sort_index()
            for season, count in season_counts.items():
                print(f"  {season}: {count} matches")
        else:
            raise Exception("No data files could be loaded successfully")
        
        return self
    
    def create_train_test_split(self):
        """Create proper train/test splits using multiple seasons."""
        print("\n📊 CREATING TRAIN/TEST SPLITS")
        print("=" * 40)
        
        # Convert Date to datetime for proper sorting
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Get all available seasons
        available_seasons = sorted(self.data['Season'].unique())
        print(f"Available seasons: {available_seasons}")
        
        # Use more seasons for training, latest season(s) for testing
        # Train: First 4-5 seasons, Test: Last 1-2 seasons
        if len(available_seasons) >= 4:
            train_seasons = available_seasons[:-2]  # All but last 2 seasons
            test_seasons = available_seasons[-2:]   # Last 2 seasons
        else:
            train_seasons = available_seasons[:-1]  # All but last season
            test_seasons = available_seasons[-1:]   # Last season
        
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")
        
        train_data = []
        test_data = []
        
        # Process training seasons
        for season in train_seasons:
            season_data = self.data[self.data['Season'] == season].copy()
            train_data.append(season_data)
            print(f"  {season} season: {len(season_data)} matches → TRAIN")
        
        # Process testing seasons
        for season in test_seasons:
            season_data = self.data[self.data['Season'] == season].copy()
            test_data.append(season_data)
            print(f"  {season} season: {len(season_data)} matches → TEST")
        
        self.train_data = pd.concat(train_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)
        
        print(f"\n✅ Train set: {len(self.train_data)} matches")
        print(f"✅ Test set: {len(self.test_data)} matches")
        
        # Show target distribution
        print(f"\n📊 Target Distribution:")
        print(f"Train set:")
        train_ftr = self.train_data['FTR'].value_counts(normalize=True) * 100
        for result in ['H', 'D', 'A']:
            pct = train_ftr.get(result, 0)
            result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
            print(f"  {result_name}: {pct:.1f}%")
        
        print(f"Test set:")
        test_ftr = self.test_data['FTR'].value_counts(normalize=True) * 100
        for result in ['H', 'D', 'A']:
            pct = test_ftr.get(result, 0)
            result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
            print(f"  {result_name}: {pct:.1f}%")
        
        return self
    
    def preprocess_data(self):
        """Preprocess the raw data and create features."""
        print("\n🔧 PREPROCESSING DATA")
        print("=" * 30)
        
        # Handle missing values in key columns
        print("Handling missing values...")
        
        # For betting odds, use average of available bookmakers
        betting_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 
                       'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 
                       'VCH', 'VCD', 'VCA']
        
        for col in betting_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Create average betting odds
        print("Creating average betting odds...")
        home_cols = [col for col in self.data.columns if col.endswith('H') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgH'] = self.data[home_cols].mean(axis=1, skipna=True)
        
        draw_cols = [col for col in self.data.columns if col.endswith('D') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgD'] = self.data[draw_cols].mean(axis=1, skipna=True)
        
        away_cols = [col for col in self.data.columns if col.endswith('A') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgA'] = self.data[away_cols].mean(axis=1, skipna=True)
        
        # 🚀 SMART ODDS TRANSFORMATION
        print("🚀 Transforming odds to implied probabilities...")
        
        # Convert odds to implied probabilities
        self.data['implied_prob_home'] = 1 / self.data['AvgH']
        self.data['implied_prob_draw'] = 1 / self.data['AvgD']
        self.data['implied_prob_away'] = 1 / self.data['AvgA']
        
        # Normalize probabilities (they should sum to ~1)
        total_prob = self.data['implied_prob_home'] + self.data['implied_prob_draw'] + self.data['implied_prob_away']
        self.data['implied_prob_home'] = self.data['implied_prob_home'] / total_prob
        self.data['implied_prob_draw'] = self.data['implied_prob_draw'] / total_prob
        self.data['implied_prob_away'] = self.data['implied_prob_away'] / total_prob
        
        # Market disagreement features
        print("Creating market disagreement features...")
        self.data['market_disagreement_home'] = self.data[home_cols].max(axis=1) - self.data[home_cols].min(axis=1)
        self.data['market_disagreement_draw'] = self.data[draw_cols].max(axis=1) - self.data[draw_cols].min(axis=1)
        self.data['market_disagreement_away'] = self.data[away_cols].max(axis=1) - self.data[away_cols].min(axis=1)
        
        # Over/Under 2.5 odds (if available)
        if 'Avg>2.5' in self.data.columns and 'Avg<2.5' in self.data.columns:
            self.data['implied_prob_over25'] = 1 / self.data['Avg>2.5']
            self.data['implied_prob_under25'] = 1 / self.data['Avg<2.5']
            print("✅ Over/Under 2.5 odds included")
        
        # Handle missing values in match stats
        stats_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
        for col in stats_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        print("✅ Data preprocessing completed")
        return self
    
    def create_rolling_features(self):
        """🚀 Create enhanced rolling features with dynamic windows and weighted averages."""
        print("\n📈 ENHANCED ROLLING FEATURES")
        print("=" * 40)
        
        # Get all unique teams
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        print(f"Processing {len(all_teams)} teams with enhanced features...")
        
        # 🚀 Dynamic windows: 3, 5, 10 matches
        windows = [3, 5, 10]
        
        # Initialize enhanced rolling columns
        enhanced_cols = []
        for window in windows:
            enhanced_cols.extend([
                f'team_wins_{window}', f'team_draws_{window}', f'team_losses_{window}',
                f'team_goals_scored_{window}', f'team_goals_conceded_{window}', f'team_goal_difference_{window}',
                f'team_shots_{window}', f'team_shots_conceded_{window}', f'team_shots_on_target_{window}', f'team_shots_on_target_conceded_{window}',
                f'team_corners_{window}', f'team_corners_conceded_{window}', f'team_cards_{window}', f'team_cards_conceded_{window}',
                f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
                f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
                f'home_goals_scored_{window}', f'home_goals_conceded_{window}',
                f'away_goals_scored_{window}', f'away_goals_conceded_{window}',
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}'
            ])
        
        # Initialize all enhanced columns
        for col in enhanced_cols:
            self.data[col] = 0.0
        
        # Process each team separately
        for team in all_teams:
            print(f"  Processing {team}...")
            
            # Get all matches for this team
            team_matches = self.data[
                (self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)
            ].copy()
            
            # Sort by season and date
            team_matches = team_matches.sort_values(['Season', 'Date']).reset_index(drop=True)
            
            # Create enhanced rolling features for each season separately
            for season in team_matches['Season'].unique():
                season_matches = team_matches[team_matches['Season'] == season].copy()
                
                for window in windows:
                    for i in range(len(season_matches)):
                        start_idx = max(0, i - window)
                        recent_matches = season_matches.iloc[start_idx:i]
                        
                        if len(recent_matches) == 0:
                            continue
                        
                        # 🚀 Calculate weighted averages (recent matches more important)
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
                        home_goals_scored = home_goals_conceded = 0
                        away_goals_scored = away_goals_conceded = 0
                        
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
                                
                                home_goals_scored += match['FTHG'] * weight
                                home_goals_conceded += match['FTAG'] * weight
                                
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
                                
                                away_goals_scored += match['FTAG'] * weight
                                away_goals_conceded += match['FTHG'] * weight
                                
                                if match['FTR'] == 'A':
                                    team_wins += weight
                                    away_wins += weight
                                elif match['FTR'] == 'D':
                                    team_draws += weight
                                    away_draws += weight
                                else:
                                    team_losses += weight
                                    away_losses += weight
                        
                        # Store enhanced rolling features
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
                        self.data.loc[match_idx, f'home_goals_scored_{window}'] = home_goals_scored
                        self.data.loc[match_idx, f'home_goals_conceded_{window}'] = home_goals_conceded
                        self.data.loc[match_idx, f'away_goals_scored_{window}'] = away_goals_scored
                        self.data.loc[match_idx, f'away_goals_conceded_{window}'] = away_goals_conceded
                        
                        # 🚀 ENHANCED FEATURES
                        # Weighted form score
                        weighted_form = (team_wins - team_losses) + (team_goals_scored - team_goals_conceded) * 0.1
                        self.data.loc[match_idx, f'weighted_form_{window}'] = weighted_form
                        
                        # Form streak (consecutive wins/losses)
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
                            
                            # Calculate streak
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
                        
                        # Goal trend (improving/declining)
                        if len(recent_matches) >= 3:
                            goal_diffs = []
                            for _, match in recent_matches.iterrows():
                                is_home = match['HomeTeam'] == team
                                if is_home:
                                    goal_diff = match['FTHG'] - match['FTAG']
                                else:
                                    goal_diff = match['FTAG'] - match['FTHG']
                                goal_diffs.append(goal_diff)
                            
                            # Calculate trend (positive = improving, negative = declining)
                            if len(goal_diffs) >= 2:
                                trend = np.polyfit(range(len(goal_diffs)), goal_diffs, 1)[0]
                                self.data.loc[match_idx, f'goal_trend_{window}'] = trend
        
        print("✅ Rolling features created")
        return self
    
    def create_match_features(self):
        """Create enhanced match-specific features by combining team data."""
        print("\n⚽ CREATING ENHANCED MATCH FEATURES")
        print("=" * 40)
        
        # Create relative features for all windows
        windows = [3, 5, 10]
        
        for window in windows:
            self.data[f'goal_diff_advantage_{window}'] = 0.0
            self.data[f'form_advantage_{window}'] = 0.0
            self.data[f'shots_advantage_{window}'] = 0.0
            self.data[f'dominance_advantage_{window}'] = 0.0
        
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
                
                # Create relative features
                self.data.loc[idx, f'goal_diff_advantage_{window}'] = home_goal_diff - away_goal_diff
                self.data.loc[idx, f'form_advantage_{window}'] = (home_wins - home_losses) - (away_wins - away_losses)
                self.data.loc[idx, f'shots_advantage_{window}'] = home_shots - away_shots
                self.data.loc[idx, f'dominance_advantage_{window}'] = home_dominance - away_dominance
        
        print("✅ Enhanced match features created")
        return self
    
    def define_priority_categories(self):
        """Define feature categories in your priority order."""
        print("\n🎯 DEFINING PRIORITY CATEGORIES")
        print("=" * 40)
        
        # 🚀 Enhanced feature categories with all improvements
        self.priority_categories = {
            # 1. Smart Odds Features (highest priority)
            'smart_odds': {
                'features': [
                    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
                    'market_disagreement_home', 'market_disagreement_draw', 'market_disagreement_away'
                ],
                'description': 'Smart Odds Features'
            },
            
            # 2. Enhanced Rolling Features (multiple windows)
            'enhanced_rolling': {
                'features': [],
                'description': 'Enhanced Rolling Features'
            },
            
            # 3. Match Context Features
            'match_context': {
                'features': [],
                'description': 'Match Context Features'
            }
        }
        
        # Add enhanced rolling features for all windows
        windows = [3, 5, 10]
        for window in windows:
            self.priority_categories['enhanced_rolling']['features'].extend([
                f'team_wins_{window}', f'team_draws_{window}', f'team_losses_{window}',
                f'team_goals_scored_{window}', f'team_goals_conceded_{window}', f'team_goal_difference_{window}',
                f'team_shots_{window}', f'team_shots_conceded_{window}', f'team_shots_on_target_{window}', f'team_shots_on_target_conceded_{window}',
                f'team_corners_{window}', f'team_corners_conceded_{window}', f'team_cards_{window}', f'team_cards_conceded_{window}',
                f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
                f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}'
            ])
        
        # Add match context features for all windows
        for window in windows:
            self.priority_categories['match_context']['features'].extend([
                f'goal_diff_advantage_{window}', f'form_advantage_{window}',
                f'shots_advantage_{window}', f'dominance_advantage_{window}'
            ])
        
        # Add Over/Under odds if available
        if 'implied_prob_over25' in self.data.columns:
            self.priority_categories['smart_odds']['features'].extend([
                'implied_prob_over25', 'implied_prob_under25'
            ])
        
        # Filter to existing features and build feature list
        self.feature_columns = []
        for category, info in self.priority_categories.items():
            existing_features = [f for f in info['features'] if f in self.data.columns]
            self.priority_categories[category]['features'] = existing_features
            self.feature_columns.extend(existing_features)
        
        print("Your Priority Order:")
        for i, (category, info) in enumerate(self.priority_categories.items(), 1):
            print(f"  {i}. {info['description']:<25} - {len(info['features'])} features")
        
        print(f"\n✅ Total features: {len(self.feature_columns)}")
        
        return self
    
    def prepare_data(self):
        """Prepare data for all algorithms."""
        print("\n🔧 PREPARING DATA")
        print("=" * 30)
        
        # Prepare training data
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data['FTR']
        
        # Prepare test data
        X_test = self.test_data[self.feature_columns].fillna(0)
        y_test = self.test_data['FTR']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        print(f"✅ Training features: {X_train_scaled.shape}")
        print(f"✅ Test features: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, y_test
    
    def handle_class_imbalance(self, y_train):
        """🚀 Handle class imbalance with balanced class weights."""
        print("\n⚖️ HANDLING CLASS IMBALANCE")
        print("=" * 30)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Show class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:")
        for i, count in enumerate(counts):
            result_name = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[i]
            print(f"  {result_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return class_weight_dict
    
    def test_algorithms(self, X_train, X_test, y_train, y_test, y_test_original, class_weight_dict):
        """Test different algorithms to find the best one."""
        print("\n🤖 TESTING ALGORITHMS")
        print("=" * 40)
        
        # 🚀 Enhanced algorithms with automatic weight learning
        algorithms = {
            'Enhanced Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            ),
            
            'Enhanced Gradient Boosting': GradientBoostingClassifier(
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
            
            'Enhanced SVM': SVC(
                random_state=42,
                probability=True,
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced'
            ),
            
            'Enhanced Neural Network': MLPClassifier(
                random_state=42,
                hidden_layer_sizes=(200, 100, 50),
                max_iter=1000,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1
            ),
            
            'Ridge Classifier': RidgeClassifier(
                random_state=42,
                alpha=1.0,
                class_weight='balanced'
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
                
                # 🚀 Calculate enhanced metrics
                accuracy = accuracy_score(y_test_original, y_pred_original)
                
                # Calculate log loss for probability sharpness
                y_pred_proba = algorithm.predict_proba(X_test)
                logloss = log_loss(y_test, y_pred_proba)
                
                # Get stratified cross-validation score
                cv_scores = cross_val_score(algorithm, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store enhanced results
                self.results[name] = {
                    'model': algorithm,
                    'test_accuracy': accuracy,
                    'test_logloss': logloss,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred_original,
                    'actual': y_test_original,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name}: Accuracy={accuracy:.3f}, LogLoss={logloss:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
        
        return self
    
    def create_stacked_model(self, X_train, X_test, y_train, y_test, y_test_original):
        """🚀 Create stacked model combining multiple algorithms."""
        print("\n🏗️ CREATING STACKED MODEL")
        print("=" * 30)
        
        # Get predictions from base models
        base_predictions = []
        base_model_names = []
        
        for name, result in self.results.items():
            if 'probabilities' in result:
                base_predictions.append(result['probabilities'])
                base_model_names.append(name)
        
        if len(base_predictions) < 2:
            print("❌ Not enough base models for stacking")
            return self
        
        # Stack predictions
        X_stack_train = np.hstack(base_predictions)
        
        # Create meta-model
        meta_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        
        # Train meta-model
        meta_model.fit(X_stack_train, y_train)
        
        # Make stacked predictions
        y_pred_stack = meta_model.predict(X_stack_train)
        y_pred_stack_original = self.label_encoder.inverse_transform(y_pred_stack)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_original, y_pred_stack_original)
        y_pred_proba_stack = meta_model.predict_proba(X_stack_train)
        logloss = log_loss(y_test, y_pred_proba_stack)
        
        # Store stacked results
        self.results['Stacked Model'] = {
            'model': meta_model,
            'test_accuracy': accuracy,
            'test_logloss': logloss,
            'predictions': y_pred_stack_original,
            'actual': y_test_original,
            'probabilities': y_pred_proba_stack,
            'base_models': base_model_names
        }
        
        print(f"✅ Stacked Model: Accuracy={accuracy:.3f}, LogLoss={logloss:.3f}")
        print(f"   Base models: {', '.join(base_model_names)}")
        
        return self
    
    def optimize_best_algorithm(self, X_train, X_test, y_train, y_test, y_test_original):
        """Optimize the best performing algorithm."""
        print("\n🔍 OPTIMIZING BEST ALGORITHM")
        print("=" * 40)
        
        # Find best algorithm
        best_algorithm = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        best_accuracy = self.results[best_algorithm]['test_accuracy']
        
        print(f"Best algorithm: {best_algorithm} (accuracy: {best_accuracy:.3f})")
        
        # Optimize based on algorithm type
        if 'Gradient Boosting' in best_algorithm:
            self.optimize_gradient_boosting(X_train, X_test, y_train, y_test, y_test_original)
        elif 'Logistic' in best_algorithm:
            self.optimize_logistic_regression(X_train, X_test, y_train, y_test, y_test_original)
        elif 'Random Forest' in best_algorithm:
            self.optimize_random_forest(X_train, X_test, y_train, y_test, y_test_original)
        else:
            print(f"Using {best_algorithm} as-is (no optimization needed)")
        
        return self
    
    def optimize_gradient_boosting(self, X_train, X_test, y_train, y_test, y_test_original):
        """Optimize Gradient Boosting parameters."""
        print("Optimizing Gradient Boosting...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Test optimized model
        y_pred = grid_search.predict(X_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test_original, y_pred_original)
        
        print(f"  Optimized GB: {accuracy:.3f}")
        print(f"  Best params: {grid_search.best_params_}")
        
        # Store optimized results
        self.results['Gradient Boosting Optimized'] = {
            'model': grid_search.best_estimator_,
            'test_accuracy': accuracy,
            'predictions': y_pred_original,
            'actual': y_test_original,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def optimize_logistic_regression(self, X_train, X_test, y_train, y_test, y_test_original):
        """Optimize Logistic Regression parameters."""
        print("Optimizing Logistic Regression...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Test optimized model
        y_pred = grid_search.predict(X_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test_original, y_pred_original)
        
        print(f"  Optimized LR: {accuracy:.3f}")
        print(f"  Best params: {grid_search.best_params_}")
        
        # Store optimized results
        self.results['Logistic Regression Optimized'] = {
            'model': grid_search.best_estimator_,
            'test_accuracy': accuracy,
            'predictions': y_pred_original,
            'actual': y_test_original,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def optimize_random_forest(self, X_train, X_test, y_train, y_test, y_test_original):
        """Optimize Random Forest parameters."""
        print("Optimizing Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Test optimized model
        y_pred = grid_search.predict(X_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test_original, y_pred_original)
        
        print(f"  Optimized RF: {accuracy:.3f}")
        print(f"  Best params: {grid_search.best_params_}")
        
        # Store optimized results
        self.results['Random Forest Optimized'] = {
            'model': grid_search.best_estimator_,
            'test_accuracy': accuracy,
            'predictions': y_pred_original,
            'actual': y_test_original,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance from the best model."""
        print("\n🌳 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        best_model = self.results[best_model_name]['model']
        
        print(f"Feature importance from {best_model_name}:")
        
        if hasattr(best_model, 'feature_importances_'):
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 15 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<40} {row['importance']:.4f}")
            
            # Analyze by category
            print("\nImportance by Category:")
            categories = {
                'Bookmaker Odds': [f for f in importance_df['feature'] if 'odds' in f],
                'Wins/Losses': [f for f in importance_df['feature'] if 'wins' in f or 'losses' in f or 'draws' in f],
                'Goals': [f for f in importance_df['feature'] if 'goals' in f],
                'Shots': [f for f in importance_df['feature'] if 'shots' in f],
                'Home/Away': [f for f in importance_df['feature'] if 'home_' in f and ('wins' in f or 'losses' in f)],
                'Corners/Cards': [f for f in importance_df['feature'] if 'corners' in f or 'cards' in f]
            }
            
            for category, features in categories.items():
                if features:
                    category_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
                    print(f"  {category:<20}: {category_importance:.4f}")
        
        elif hasattr(best_model, 'coef_'):
            # Logistic Regression coefficients
            coef_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': best_model.coef_[0]
            }).sort_values('coefficient', key=abs, ascending=False)
            
            print("\nTop 15 Most Important Features (coefficients):")
            for i, (_, row) in enumerate(coef_df.head(15).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<40} {row['coefficient']:.4f}")
        
        else:
            print(f"Model {best_model_name} does not support feature importance analysis.")
        
        return self
    
    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n🏆 COMPREHENSIVE ALGORITHM COMPARISON REPORT")
        print("=" * 70)
        
        # Sort results by test accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['test_accuracy'], 
            reverse=True
        )
        
        print("🚀 Enhanced Model Performance Ranking:")
        print("-" * 50)
        for i, (name, result) in enumerate(sorted_results, 1):
            accuracy = result['test_accuracy']
            logloss = result.get('test_logloss', 'N/A')
            cv_mean = result.get('cv_mean', 'N/A')
            cv_std = result.get('cv_std', 'N/A')
            
            print(f"{i:2d}. {name:<30}: Accuracy={accuracy:.3f} ({accuracy*100:.1f}%)")
            if logloss != 'N/A':
                print(f"    LogLoss={logloss:.3f}")
            if cv_mean != 'N/A':
                print(f"    CV={cv_mean:.3f}±{cv_std:.3f}")
        
        # Best model details
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   📊 Test Accuracy: {best_result['test_accuracy']:.3f} ({best_result['test_accuracy']*100:.1f}%)")
        
        if 'test_logloss' in best_result:
            print(f"   📊 Test LogLoss: {best_result['test_logloss']:.3f}")
        
        if 'cv_mean' in best_result:
            print(f"   📊 Cross-Validation: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
        
        if 'best_params' in best_result:
            print(f"   ⚙️  Best Parameters: {best_result['best_params']}")
        
        if 'base_models' in best_result:
            print(f"   🏗️  Base Models: {', '.join(best_result['base_models'])}")
        
        # Show prediction breakdown for best model
        print(f"\n📊 Prediction Breakdown - {best_name}:")
        correct_predictions = (best_result['predictions'] == best_result['actual'])
        
        for outcome in ['H', 'D', 'A']:
            outcome_mask = best_result['actual'] == outcome
            outcome_correct = correct_predictions[outcome_mask].sum()
            outcome_total = outcome_mask.sum()
            outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
            
            outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
            print(f"   {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")
        
        print(f"\n✅ Your priority order with optimal algorithm achieves {best_result['test_accuracy']:.3f} accuracy!")
        
        return best_result['test_accuracy']

def main():
    """Main execution function."""
    print("🚀 Premier League Enhanced Model - Advanced ML")
    print("=" * 70)
    print("Implementing all advanced techniques for maximum accuracy")
    
    try:
        # Initialize comparison
        comparison = AlgorithmComparison()
        
        # Load data
        comparison.load_data()
        
        # Preprocess data and create features
        comparison.preprocess_data()
        comparison.create_rolling_features()
        comparison.create_match_features()
        
        # Create train/test splits
        comparison.create_train_test_split()
        
        # Define priority categories
        comparison.define_priority_categories()
        
        # Prepare data
        X_train, X_test, y_train, y_test, y_test_original = comparison.prepare_data()
        
        # 🚀 Handle class imbalance
        class_weight_dict = comparison.handle_class_imbalance(y_train)
        
        # 🚀 Test enhanced algorithms with automatic weight learning
        comparison.test_algorithms(X_train, X_test, y_train, y_test, y_test_original, class_weight_dict)
        
        # 🚀 Create stacked model (temporarily disabled due to data flow issue)
        # comparison.create_stacked_model(X_train, X_test, y_train, y_test, y_test_original)
        
        # Optimize best algorithm
        comparison.optimize_best_algorithm(X_train, X_test, y_train, y_test, y_test_original)
        
        # Analyze feature importance
        comparison.analyze_feature_importance()
        
        # Generate comprehensive report
        final_accuracy = comparison.generate_final_report()
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.3f} accuracy")
        print(f"   Using all advanced techniques!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
