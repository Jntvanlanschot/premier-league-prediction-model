#!/usr/bin/env python3
"""
Premier League Draw-Optimized Model - Enhanced Draw Prediction
=============================================================

Implements all draw-specific optimizations:
1. Adjust decision thresholding
2. Resample/reweight draws  
3. Engineer draw-specific features
4. Probability calibration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class DrawOptimizedModel:
    """Enhanced model optimized for draw prediction."""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.feature_columns = []
        self.draw_threshold = 0.28  # Minimum probability for draw prediction
        
    def load_data(self):
        """Load data from local datasets folder."""
        print("\n📥 LOADING DATA FROM LOCAL DATASETS")
        print("=" * 40)
        
        datasets_folder = '/home/joost/Premier League Prediction Model/datasets'
        import os
        csv_files = [f for f in os.listdir(datasets_folder) if f.endswith('.csv')]
        csv_files.sort()
        
        print(f"Found {len(csv_files)} season files:")
        for file in csv_files:
            print(f"  - {file}")
        
        all_data = []
        for file in csv_files:
            file_path = os.path.join(datasets_folder, file)
            print(f"  Loading {file}...")
            
            try:
                df = pd.read_csv(file_path)
                season_name = file.replace('_season.csv', '').replace('_', '')
                if len(season_name) == 8:
                    season_code = season_name[2:4] + season_name[6:8]
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
            
            season_counts = self.data['Season'].value_counts().sort_index()
            for season, count in season_counts.items():
                print(f"  {season}: {count} matches")
        else:
            raise Exception("No data files could be loaded successfully")
        
        return self
    
    def preprocess_data(self):
        """Enhanced preprocessing with draw-specific features."""
        print("\n🔧 DRAW-OPTIMIZED PREPROCESSING")
        print("=" * 40)
        
        # Handle missing values
        print("Handling missing values...")
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
        
        # Normalize probabilities
        total_prob = self.data['implied_prob_home'] + self.data['implied_prob_draw'] + self.data['implied_prob_away']
        self.data['implied_prob_home'] = self.data['implied_prob_home'] / total_prob
        self.data['implied_prob_draw'] = self.data['implied_prob_draw'] / total_prob
        self.data['implied_prob_away'] = self.data['implied_prob_away'] / total_prob
        
        # 🎯 DRAW-SPECIFIC FEATURES
        print("🎯 Creating draw-specific features...")
        
        # 1. Balanced odds (close matches more likely to draw)
        self.data['odds_balance_home_away'] = abs(self.data['implied_prob_home'] - self.data['implied_prob_away'])
        self.data['odds_balance_home_draw'] = abs(self.data['implied_prob_home'] - self.data['implied_prob_draw'])
        self.data['odds_balance_away_draw'] = abs(self.data['implied_prob_away'] - self.data['implied_prob_draw'])
        
        # 2. Market uncertainty (higher disagreement = more draws)
        self.data['market_disagreement_home'] = self.data[home_cols].max(axis=1) - self.data[home_cols].min(axis=1)
        self.data['market_disagreement_draw'] = self.data[draw_cols].max(axis=1) - self.data[draw_cols].min(axis=1)
        self.data['market_disagreement_away'] = self.data[away_cols].max(axis=1) - self.data[away_cols].min(axis=1)
        
        # 3. Draw probability from market
        self.data['market_draw_probability'] = self.data['implied_prob_draw']
        
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
        
        print("✅ Draw-optimized preprocessing completed")
        return self
    
    def create_rolling_features(self):
        """Create enhanced rolling features with draw-specific metrics."""
        print("\n📈 CREATING DRAW-OPTIMIZED ROLLING FEATURES")
        print("=" * 40)
        
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        print(f"Processing {len(all_teams)} teams with draw-optimized features...")
        
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
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}',
                # 🎯 Draw-specific rolling features
                f'draw_tendency_{window}', f'low_scoring_tendency_{window}', f'balanced_form_{window}'
            ])
        
        # Initialize all enhanced columns
        for col in enhanced_cols:
            self.data[col] = 0.0
        
        # Process each team separately
        for team in all_teams:
            print(f"  Processing {team}...")
            
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
                        
                        # 🎯 DRAW-SPECIFIC ROLLING FEATURES
                        # Draw tendency (percentage of draws in recent matches)
                        total_matches = team_wins + team_draws + team_losses
                        draw_tendency = team_draws / total_matches if total_matches > 0 else 0
                        self.data.loc[match_idx, f'draw_tendency_{window}'] = draw_tendency
                        
                        # Low scoring tendency (teams that score and concede few goals)
                        avg_goals_per_match = (team_goals_scored + team_goals_conceded) / total_matches if total_matches > 0 else 0
                        low_scoring_tendency = 1 / (1 + avg_goals_per_match)  # Higher for low-scoring teams
                        self.data.loc[match_idx, f'low_scoring_tendency_{window}'] = low_scoring_tendency
                        
                        # Balanced form (teams that don't dominate or get dominated)
                        form_balance = 1 - abs(team_wins - team_losses) / total_matches if total_matches > 0 else 0
                        self.data.loc[match_idx, f'balanced_form_{window}'] = form_balance
                        
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
        
        print("✅ Draw-optimized rolling features created")
        return self
    
    def create_draw_specific_features(self):
        """Create draw-specific match features."""
        print("\n🎯 CREATING DRAW-SPECIFIC MATCH FEATURES")
        print("=" * 40)
        
        windows = [3, 5, 10]
        
        # Initialize draw-specific features
        for window in windows:
            self.data[f'goal_diff_advantage_{window}'] = 0.0
            self.data[f'form_advantage_{window}'] = 0.0
            self.data[f'shots_advantage_{window}'] = 0.0
            self.data[f'dominance_advantage_{window}'] = 0.0
            
            # 🎯 Draw-specific relative features
            self.data[f'draw_tendency_advantage_{window}'] = 0.0
            self.data[f'low_scoring_advantage_{window}'] = 0.0
            self.data[f'balanced_form_advantage_{window}'] = 0.0
            self.data[f'goal_trend_balance_{window}'] = 0.0
        
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
                
                # 🎯 Draw-specific features
                home_draw_tendency = row.get(f'draw_tendency_{window}', 0)
                away_draw_tendency = row.get(f'draw_tendency_{window}', 0)
                
                home_low_scoring = row.get(f'low_scoring_tendency_{window}', 0)
                away_low_scoring = row.get(f'low_scoring_tendency_{window}', 0)
                
                home_balanced_form = row.get(f'balanced_form_{window}', 0)
                away_balanced_form = row.get(f'balanced_form_{window}', 0)
                
                home_goal_trend = row.get(f'goal_trend_{window}', 0)
                away_goal_trend = row.get(f'goal_trend_{window}', 0)
                
                # Create relative features
                self.data.loc[idx, f'goal_diff_advantage_{window}'] = home_goal_diff - away_goal_diff
                self.data.loc[idx, f'form_advantage_{window}'] = (home_wins - home_losses) - (away_wins - away_losses)
                self.data.loc[idx, f'shots_advantage_{window}'] = home_shots - away_shots
                self.data.loc[idx, f'dominance_advantage_{window}'] = home_dominance - away_dominance
                
                # 🎯 Draw-specific relative features
                self.data.loc[idx, f'draw_tendency_advantage_{window}'] = home_draw_tendency - away_draw_tendency
                self.data.loc[idx, f'low_scoring_advantage_{window}'] = home_low_scoring - away_low_scoring
                self.data.loc[idx, f'balanced_form_advantage_{window}'] = home_balanced_form - away_balanced_form
                
                # Goal trend balance (teams with similar trends more likely to draw)
                self.data.loc[idx, f'goal_trend_balance_{window}'] = 1 - abs(home_goal_trend - away_goal_trend)
        
        print("✅ Draw-specific match features created")
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
    
    def define_draw_optimized_features(self):
        """Define draw-optimized feature categories."""
        print("\n🎯 DEFINING DRAW-OPTIMIZED FEATURES")
        print("=" * 40)
        
        # Draw-optimized feature categories
        self.priority_categories = {
            # 1. Draw-Specific Odds Features
            'draw_odds': {
                'features': [
                    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
                    'odds_balance_home_away', 'odds_balance_home_draw', 'odds_balance_away_draw',
                    'market_draw_probability', 'market_disagreement_home', 'market_disagreement_draw', 'market_disagreement_away'
                ],
                'description': 'Draw-Specific Odds Features'
            },
            
            # 2. Draw-Specific Rolling Features
            'draw_rolling': {
                'features': [],
                'description': 'Draw-Specific Rolling Features'
            },
            
            # 3. Draw-Specific Match Context
            'draw_context': {
                'features': [],
                'description': 'Draw-Specific Match Context'
            }
        }
        
        # Add draw-specific rolling features for all windows
        windows = [3, 5, 10]
        for window in windows:
            self.priority_categories['draw_rolling']['features'].extend([
                f'team_wins_{window}', f'team_draws_{window}', f'team_losses_{window}',
                f'team_goals_scored_{window}', f'team_goals_conceded_{window}', f'team_goal_difference_{window}',
                f'team_shots_{window}', f'team_shots_conceded_{window}', f'team_shots_on_target_{window}', f'team_shots_on_target_conceded_{window}',
                f'team_corners_{window}', f'team_corners_conceded_{window}', f'team_cards_{window}', f'team_cards_conceded_{window}',
                f'home_wins_{window}', f'home_draws_{window}', f'home_losses_{window}',
                f'away_wins_{window}', f'away_draws_{window}', f'away_losses_{window}',
                f'weighted_form_{window}', f'form_streak_{window}', f'goal_trend_{window}',
                f'draw_tendency_{window}', f'low_scoring_tendency_{window}', f'balanced_form_{window}'
            ])
        
        # Add draw-specific match context features for all windows
        for window in windows:
            self.priority_categories['draw_context']['features'].extend([
                f'goal_diff_advantage_{window}', f'form_advantage_{window}',
                f'shots_advantage_{window}', f'dominance_advantage_{window}',
                f'draw_tendency_advantage_{window}', f'low_scoring_advantage_{window}',
                f'balanced_form_advantage_{window}', f'goal_trend_balance_{window}'
            ])
        
        # Add Over/Under odds if available
        if 'implied_prob_over25' in self.data.columns:
            self.priority_categories['draw_odds']['features'].extend([
                'implied_prob_over25', 'implied_prob_under25'
            ])
        
        # Filter to existing features and build feature list
        self.feature_columns = []
        for category, info in self.priority_categories.items():
            existing_features = [f for f in info['features'] if f in self.data.columns]
            self.priority_categories[category]['features'] = existing_features
            self.feature_columns.extend(existing_features)
        
        print("Draw-Optimized Feature Categories:")
        for category, info in self.priority_categories.items():
            print(f"  {info['description']:<25} - {len(info['features'])} features")
        
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
    
    def handle_draw_imbalance(self, y_train):
        """🎯 Enhanced draw imbalance handling."""
        print("\n⚖️ ENHANCED DRAW IMBALANCE HANDLING")
        print("=" * 40)
        
        # Calculate enhanced class weights (extra weight for draws)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # 🎯 Extra weight for draws
        class_weight_dict[1] = class_weight_dict[1] * 1.5  # 50% extra weight for draws
        
        print(f"Enhanced class weights: {class_weight_dict}")
        
        # Show class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution:")
        for i, count in enumerate(counts):
            result_name = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}[i]
            print(f"  {result_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return class_weight_dict
    
    def test_draw_optimized_algorithms(self, X_train, X_test, y_train, y_test, y_test_original, class_weight_dict):
        """Test draw-optimized algorithms."""
        print("\n🤖 TESTING DRAW-OPTIMIZED ALGORITHMS")
        print("=" * 40)
        
        # Draw-optimized algorithms
        algorithms = {
            'Draw-Optimized Random Forest': RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict
            ),
            
            'Draw-Optimized Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8
            ),
            
            'Draw-Optimized Logistic Regression': LogisticRegression(
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
                
                # 🎯 Apply draw threshold optimization
                y_pred_proba = algorithm.predict_proba(X_test)
                y_pred_thresholded = self.apply_draw_threshold(y_pred_proba)
                y_pred_thresholded_original = self.label_encoder.inverse_transform(y_pred_thresholded)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_original, y_pred_original)
                accuracy_thresholded = accuracy_score(y_test_original, y_pred_thresholded_original)
                
                # Calculate log loss
                logloss = log_loss(y_test, y_pred_proba)
                
                # Get cross-validation score
                cv_scores = cross_val_score(algorithm, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store enhanced results
                self.results[name] = {
                    'model': algorithm,
                    'test_accuracy': accuracy,
                    'test_accuracy_thresholded': accuracy_thresholded,
                    'test_logloss': logloss,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred_original,
                    'predictions_thresholded': y_pred_thresholded_original,
                    'actual': y_test_original,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ {name}: Accuracy={accuracy:.3f}, Thresholded={accuracy_thresholded:.3f}, LogLoss={logloss:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name}: Error - {e}")
        
        return self
    
    def apply_draw_threshold(self, probabilities):
        """🎯 Apply draw threshold optimization."""
        predictions = []
        
        for prob_row in probabilities:
            home_prob, draw_prob, away_prob = prob_row
            
            # If draw probability is above threshold, consider draw
            if draw_prob >= self.draw_threshold:
                # If draw is close to being the best, choose draw
                if draw_prob >= max(home_prob, away_prob) * 0.9:
                    predictions.append(1)  # Draw
                else:
                    predictions.append(np.argmax(prob_row))
            else:
                predictions.append(np.argmax(prob_row))
        
        return np.array(predictions)
    
    def generate_draw_optimized_report(self):
        """Generate draw-optimized final report."""
        print("\n🏆 DRAW-OPTIMIZED MODEL REPORT")
        print("=" * 70)
        
        # Sort results by thresholded accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['test_accuracy_thresholded'], 
            reverse=True
        )
        
        print("Draw-Optimized Model Performance Ranking:")
        print("-" * 50)
        for i, (name, result) in enumerate(sorted_results, 1):
            accuracy = result['test_accuracy']
            accuracy_thresholded = result['test_accuracy_thresholded']
            logloss = result.get('test_logloss', 'N/A')
            cv_mean = result.get('cv_mean', 'N/A')
            cv_std = result.get('cv_std', 'N/A')
            
            print(f"{i:2d}. {name:<30}: Accuracy={accuracy:.3f}, Thresholded={accuracy_thresholded:.3f}")
            if logloss != 'N/A':
                print(f"    LogLoss={logloss:.3f}")
            if cv_mean != 'N/A':
                print(f"    CV={cv_mean:.3f}±{cv_std:.3f}")
        
        # Best model details
        best_name, best_result = sorted_results[0]
        print(f"\n🏆 BEST DRAW-OPTIMIZED MODEL: {best_name}")
        print(f"   📊 Test Accuracy: {best_result['test_accuracy']:.3f} ({best_result['test_accuracy']*100:.1f}%)")
        print(f"   📊 Thresholded Accuracy: {best_result['test_accuracy_thresholded']:.3f} ({best_result['test_accuracy_thresholded']*100:.1f}%)")
        
        if 'test_logloss' in best_result:
            print(f"   📊 Test LogLoss: {best_result['test_logloss']:.3f}")
        
        if 'cv_mean' in best_result:
            print(f"   📊 Cross-Validation: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")
        
        # Show prediction breakdown for best model (thresholded)
        print(f"\n📊 Prediction Breakdown - {best_name} (Thresholded):")
        correct_predictions = (best_result['predictions_thresholded'] == best_result['actual'])
        
        for outcome in ['H', 'D', 'A']:
            outcome_mask = best_result['actual'] == outcome
            outcome_correct = correct_predictions[outcome_mask].sum()
            outcome_total = outcome_mask.sum()
            outcome_accuracy = outcome_correct / outcome_total if outcome_total > 0 else 0
            
            outcome_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[outcome]
            print(f"   {outcome_name}: {outcome_correct}/{outcome_total} ({outcome_accuracy:.3f})")
        
        # Show draw prediction improvement
        draw_mask = best_result['actual'] == 'D'
        draw_correct = correct_predictions[draw_mask].sum()
        draw_total = draw_mask.sum()
        draw_accuracy = draw_correct / draw_total if draw_total > 0 else 0
        
        print(f"\n🎯 DRAW PREDICTION IMPROVEMENT:")
        print(f"   Draw Accuracy: {draw_accuracy:.3f} ({draw_accuracy*100:.1f}%)")
        print(f"   Draw Threshold: {self.draw_threshold:.2f}")
        
        return best_result['test_accuracy_thresholded']

def main():
    """Main execution function."""
    print("🎯 Premier League Draw-Optimized Model")
    print("=" * 70)
    print("Optimizing specifically for draw prediction")
    
    try:
        # Initialize draw-optimized model
        model = DrawOptimizedModel()
        
        # Load data
        model.load_data()
        
        # Draw-optimized preprocessing
        model.preprocess_data()
        model.create_rolling_features()
        model.create_draw_specific_features()
        
        # Create train/test splits
        model.create_train_test_split()
        
        # Define draw-optimized features
        model.define_draw_optimized_features()
        
        # Prepare data
        X_train, X_test, y_train, y_test, y_test_original = model.prepare_data()
        
        # Enhanced draw imbalance handling
        class_weight_dict = model.handle_draw_imbalance(y_train)
        
        # Test draw-optimized algorithms
        model.test_draw_optimized_algorithms(X_train, X_test, y_train, y_test, y_test_original, class_weight_dict)
        
        # Generate draw-optimized report
        final_accuracy = model.generate_draw_optimized_report()
        
        print(f"\n🎯 FINAL RESULT: {final_accuracy:.3f} accuracy")
        print(f"   Optimized specifically for draw prediction!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
