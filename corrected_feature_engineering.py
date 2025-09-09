#!/usr/bin/env python3
"""
Premier League Match Outcome Prediction Model - Phase 2 (Fixed)
===============================================================

Corrected Feature Engineering Module
- Team-specific dataframes for each season
- Proper rolling averages using post-match data from previous games
- Exclude only the current game's post-match data when predicting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CorrectedFeatureEngineer:
    """Corrected feature engineering class for Premier League prediction model."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_column = 'FTR'
        self.team_dataframes = {}
        
    def load_data(self, data):
        """Load the raw data."""
        self.data = data.copy()
        print(f"📊 Loaded {len(self.data)} matches for feature engineering")
        return self
    
    def preprocess_basic_data(self):
        """Basic data preprocessing and cleaning."""
        print("\n🔧 BASIC DATA PREPROCESSING")
        print("=" * 40)
        
        # Convert Date column
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y')
        
        # Sort by date to ensure proper chronological order
        self.data = self.data.sort_values(['Season', 'Date']).reset_index(drop=True)
        
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
        self._create_average_odds()
        
        # Handle missing values in match stats
        stats_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC']
        for col in stats_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(0)
        
        print("✅ Basic preprocessing completed")
        return self
    
    def _create_average_odds(self):
        """Create average betting odds from all available bookmakers."""
        print("  Creating average betting odds...")
        
        # Home win odds
        home_cols = [col for col in self.data.columns if col.endswith('H') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgH'] = self.data[home_cols].mean(axis=1, skipna=True)
        
        # Draw odds
        draw_cols = [col for col in self.data.columns if col.endswith('D') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgD'] = self.data[draw_cols].mean(axis=1, skipna=True)
        
        # Away win odds
        away_cols = [col for col in self.data.columns if col.endswith('A') and col.startswith(('B365', 'BW', 'PS', 'WH', 'VC'))]
        self.data['AvgA'] = self.data[away_cols].mean(axis=1, skipna=True)
        
        print(f"    ✅ Created average odds from {len(home_cols)} bookmakers")
    
    def create_team_specific_dataframes(self):
        """Create team-specific dataframes for each season."""
        print("\n🏆 CREATING TEAM-SPECIFIC DATAFRAMES")
        print("=" * 50)
        
        # Get all unique teams
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        print(f"Processing {len(all_teams)} teams...")
        
        # Create team-specific dataframes
        for team in all_teams:
            print(f"  Processing {team}...")
            
            # Get all matches for this team (both home and away)
            team_matches = self.data[
                (self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)
            ].copy()
            
            # Sort by season and date
            team_matches = team_matches.sort_values(['Season', 'Date']).reset_index(drop=True)
            
            # Create team-specific features
            team_features = self._create_team_features(team_matches, team)
            
            # Store team dataframe
            self.team_dataframes[team] = team_features
            
            print(f"    ✅ {team}: {len(team_features)} matches processed")
        
        print(f"✅ Created {len(self.team_dataframes)} team-specific dataframes")
        return self
    
    def _create_team_features(self, team_matches, team_name):
        """Create features for a specific team's matches."""
        
        # Initialize result dataframe
        result = team_matches.copy()
        
        # Define feature columns to create
        feature_cols = [
            'team_wins_5', 'team_draws_5', 'team_losses_5',
            'team_goals_scored_5', 'team_goals_conceded_5', 'team_goal_difference_5',
            'team_shots_5', 'team_shots_conceded_5', 'team_shots_on_target_5', 'team_shots_on_target_conceded_5',
            'team_corners_5', 'team_corners_conceded_5', 'team_fouls_5', 'team_fouls_conceded_5',
            'team_cards_5', 'team_cards_conceded_5', 'team_dominance_score_5',
            'home_wins_5', 'home_draws_5', 'home_losses_5',
            'away_wins_5', 'away_draws_5', 'away_losses_5',
            'home_goals_scored_5', 'home_goals_conceded_5', 'away_goals_scored_5', 'away_goals_conceded_5'
        ]
        
        # Initialize all feature columns
        for col in feature_cols:
            result[col] = 0.0
        
        # Calculate rolling features for each match
        for i in range(len(team_matches)):
            # Get last 5 matches (or fewer if not enough data)
            start_idx = max(0, i - 5)
            recent_matches = team_matches.iloc[start_idx:i]  # Exclude current match (i)
            
            if len(recent_matches) == 0:
                continue
            
            # Calculate team performance metrics from recent matches
            team_wins = team_draws = team_losses = 0
            team_goals_scored = team_goals_conceded = 0
            team_shots = team_shots_conceded = 0
            team_shots_on_target = team_shots_on_target_conceded = 0
            team_corners = team_corners_conceded = 0
            team_fouls = team_fouls_conceded = 0
            team_cards = team_cards_conceded = 0
            
            # Home and away specific metrics
            home_wins = home_draws = home_losses = 0
            away_wins = away_draws = away_losses = 0
            home_goals_scored = home_goals_conceded = 0
            away_goals_scored = away_goals_conceded = 0
            
            for _, match in recent_matches.iterrows():
                is_home = match['HomeTeam'] == team_name
                
                if is_home:
                    # Team was home
                    team_goals_scored += match['FTHG']
                    team_goals_conceded += match['FTAG']
                    team_shots += match['HS']
                    team_shots_conceded += match['AS']
                    team_shots_on_target += match['HST']
                    team_shots_on_target_conceded += match['AST']
                    team_corners += match['HC']
                    team_corners_conceded += match['AC']
                    team_fouls += match['HF']
                    team_fouls_conceded += match['AF']
                    team_cards += match['HY'] + match['HR']
                    team_cards_conceded += match['AY'] + match['AR']
                    
                    # Home-specific metrics
                    home_goals_scored += match['FTHG']
                    home_goals_conceded += match['FTAG']
                    
                    if match['FTR'] == 'H':
                        team_wins += 1
                        home_wins += 1
                    elif match['FTR'] == 'D':
                        team_draws += 1
                        home_draws += 1
                    else:
                        team_losses += 1
                        home_losses += 1
                        
                else:
                    # Team was away
                    team_goals_scored += match['FTAG']
                    team_goals_conceded += match['FTHG']
                    team_shots += match['AS']
                    team_shots_conceded += match['HS']
                    team_shots_on_target += match['AST']
                    team_shots_on_target_conceded += match['HST']
                    team_corners += match['AC']
                    team_corners_conceded += match['HC']
                    team_fouls += match['AF']
                    team_fouls_conceded += match['HF']
                    team_cards += match['AY'] + match['AR']
                    team_cards_conceded += match['HY'] + match['HR']
                    
                    # Away-specific metrics
                    away_goals_scored += match['FTAG']
                    away_goals_conceded += match['FTHG']
                    
                    if match['FTR'] == 'A':
                        team_wins += 1
                        away_wins += 1
                    elif match['FTR'] == 'D':
                        team_draws += 1
                        away_draws += 1
                    else:
                        team_losses += 1
                        away_losses += 1
            
            # Store team performance features
            result.iloc[i, result.columns.get_loc('team_wins_5')] = team_wins
            result.iloc[i, result.columns.get_loc('team_draws_5')] = team_draws
            result.iloc[i, result.columns.get_loc('team_losses_5')] = team_losses
            result.iloc[i, result.columns.get_loc('team_goals_scored_5')] = team_goals_scored
            result.iloc[i, result.columns.get_loc('team_goals_conceded_5')] = team_goals_conceded
            result.iloc[i, result.columns.get_loc('team_goal_difference_5')] = team_goals_scored - team_goals_conceded
            
            # Store in-match stats features
            result.iloc[i, result.columns.get_loc('team_shots_5')] = team_shots
            result.iloc[i, result.columns.get_loc('team_shots_conceded_5')] = team_shots_conceded
            result.iloc[i, result.columns.get_loc('team_shots_on_target_5')] = team_shots_on_target
            result.iloc[i, result.columns.get_loc('team_shots_on_target_conceded_5')] = team_shots_on_target_conceded
            result.iloc[i, result.columns.get_loc('team_corners_5')] = team_corners
            result.iloc[i, result.columns.get_loc('team_corners_conceded_5')] = team_corners_conceded
            result.iloc[i, result.columns.get_loc('team_fouls_5')] = team_fouls
            result.iloc[i, result.columns.get_loc('team_fouls_conceded_5')] = team_fouls_conceded
            result.iloc[i, result.columns.get_loc('team_cards_5')] = team_cards
            result.iloc[i, result.columns.get_loc('team_cards_conceded_5')] = team_cards_conceded
            
            # Calculate dominance score
            dominance_score = (team_shots - team_shots_conceded) - (team_fouls - team_fouls_conceded) * 0.5
            result.iloc[i, result.columns.get_loc('team_dominance_score_5')] = dominance_score
            
            # Store home/away specific features
            result.iloc[i, result.columns.get_loc('home_wins_5')] = home_wins
            result.iloc[i, result.columns.get_loc('home_draws_5')] = home_draws
            result.iloc[i, result.columns.get_loc('home_losses_5')] = home_losses
            result.iloc[i, result.columns.get_loc('away_wins_5')] = away_wins
            result.iloc[i, result.columns.get_loc('away_draws_5')] = away_draws
            result.iloc[i, result.columns.get_loc('away_losses_5')] = away_losses
            result.iloc[i, result.columns.get_loc('home_goals_scored_5')] = home_goals_scored
            result.iloc[i, result.columns.get_loc('home_goals_conceded_5')] = home_goals_conceded
            result.iloc[i, result.columns.get_loc('away_goals_scored_5')] = away_goals_scored
            result.iloc[i, result.columns.get_loc('away_goals_conceded_5')] = away_goals_conceded
        
        return result
    
    def create_match_features(self):
        """Create match-specific features by combining team dataframes."""
        print("\n⚽ CREATING MATCH FEATURES")
        print("=" * 40)
        
        # Initialize result dataframe
        match_features = []
        
        for idx, row in self.data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get team features from their respective dataframes
            home_features = {}
            away_features = {}
            
            if home_team in self.team_dataframes:
                home_team_df = self.team_dataframes[home_team]
                # Find the matching row for this specific match
                home_match = home_team_df[
                    (home_team_df['Date'] == row['Date']) & 
                    (home_team_df['Season'] == row['Season'])
                ]
                
                if not home_match.empty:
                    home_row = home_match.iloc[0]
                    for col in home_row.index:
                        if col.startswith('team_') or col.startswith('home_') or col.startswith('away_'):
                            home_features[f'home_{col}'] = home_row[col] if pd.notna(home_row[col]) else 0
            
            if away_team in self.team_dataframes:
                away_team_df = self.team_dataframes[away_team]
                # Find the matching row for this specific match
                away_match = away_team_df[
                    (away_team_df['Date'] == row['Date']) & 
                    (away_team_df['Season'] == row['Season'])
                ]
                
                if not away_match.empty:
                    away_row = away_match.iloc[0]
                    for col in away_row.index:
                        if col.startswith('team_') or col.startswith('home_') or col.startswith('away_'):
                            away_features[f'away_{col}'] = away_row[col] if pd.notna(away_row[col]) else 0
            
            # Create relative features (home vs away)
            relative_features = {}
            
            # Goal difference advantage
            home_goal_diff = home_features.get('home_team_goal_difference_5', 0)
            away_goal_diff = away_features.get('away_team_goal_difference_5', 0)
            relative_features['goal_diff_advantage'] = home_goal_diff - away_goal_diff
            
            # Form advantage (wins - losses)
            home_wins = home_features.get('home_team_wins_5', 0)
            home_losses = home_features.get('home_team_losses_5', 0)
            away_wins = away_features.get('away_team_wins_5', 0)
            away_losses = away_features.get('away_team_losses_5', 0)
            relative_features['form_advantage'] = (home_wins - home_losses) - (away_wins - away_losses)
            
            # Dominance advantage
            home_dominance = home_features.get('home_team_dominance_score_5', 0)
            away_dominance = away_features.get('away_team_dominance_score_5', 0)
            relative_features['dominance_advantage'] = home_dominance - away_dominance
            
            # Add betting odds
            relative_features['avg_home_odds'] = row['AvgH'] if pd.notna(row['AvgH']) else 0
            relative_features['avg_draw_odds'] = row['AvgD'] if pd.notna(row['AvgD']) else 0
            relative_features['avg_away_odds'] = row['AvgA'] if pd.notna(row['AvgA']) else 0
            
            # Combine all features
            all_features = {**home_features, **away_features, **relative_features}
            match_features.append(all_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(match_features)
        
        # Add to main dataframe
        for col in features_df.columns:
            self.data[col] = features_df[col]
            if col not in self.feature_columns:
                self.feature_columns.append(col)
        
        print(f"✅ Created {len(features_df.columns)} match features")
        return self
    
    def prepare_final_dataset(self):
        """Prepare the final dataset for modeling."""
        print("\n🎯 PREPARING FINAL DATASET")
        print("=" * 40)
        
        # Remove duplicate columns and non-numeric columns
        print("Cleaning dataset...")
        
        # Remove duplicate columns
        self.data = self.data.loc[:, ~self.data.columns.duplicated()]
        
        # Remove non-numeric columns that shouldn't be features
        exclude_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR', 'Div', 'Time', 'Referee', 'HTR']
        exclude_cols = [col for col in exclude_cols if col in self.data.columns]
        
        # Get only numeric feature columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Update feature columns
        self.feature_columns = feature_cols
        
        # Select only the features we want for modeling
        model_columns = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR'] + self.feature_columns
        
        # Filter out rows with missing target
        self.processed_data = self.data[model_columns].copy()
        self.processed_data = self.processed_data.dropna(subset=['FTR'])
        
        # Handle any remaining missing values in features
        for col in self.feature_columns:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].fillna(0)
        
        print(f"✅ Final dataset: {len(self.processed_data)} matches with {len(self.feature_columns)} features")
        
        # Show feature summary
        print(f"\n📋 Feature Categories:")
        print(f"  - Team Performance Features: {len([f for f in self.feature_columns if 'team_' in f])}")
        print(f"  - Home/Away Specific Features: {len([f for f in self.feature_columns if 'home_' in f or 'away_' in f])}")
        print(f"  - Betting Odds Features: {len([f for f in self.feature_columns if 'odds' in f])}")
        print(f"  - Relative Features: {len([f for f in self.feature_columns if 'advantage' in f])}")
        
        return self
    
    def get_feature_importance_summary(self):
        """Get a summary of the created features."""
        print("\n📊 FEATURE IMPORTANCE SUMMARY")
        print("=" * 50)
        
        # Group features by category
        team_performance = [f for f in self.feature_columns if 'team_' in f]
        home_away_specific = [f for f in self.feature_columns if 'home_' in f or 'away_' in f]
        betting_odds = [f for f in self.feature_columns if 'odds' in f]
        relative_features = [f for f in self.feature_columns if 'advantage' in f]
        
        print("1. TEAM PERFORMANCE FEATURES:")
        for feature in team_performance[:5]:  # Show top 5
            print(f"   - {feature}")
        
        print("\n2. HOME/AWAY SPECIFIC FEATURES:")
        for feature in home_away_specific[:5]:  # Show top 5
            print(f"   - {feature}")
        
        print("\n3. BETTING ODDS FEATURES:")
        for feature in betting_odds:
            print(f"   - {feature}")
        
        print("\n4. RELATIVE FEATURES:")
        for feature in relative_features:
            print(f"   - {feature}")
        
        return self.processed_data

def main():
    """Main execution function for corrected feature engineering."""
    print("🏆 Premier League Corrected Feature Engineering")
    print("=" * 60)
    
    # Load the data from previous phase
    import pandas as pd
    
    # For now, let's recreate the data loading
    print("📥 Loading data...")
    try:
        # Download data again for this phase
        seasons = ['2324', '2425']
        all_data = []
        
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
            df = pd.read_csv(url)
            df['Season'] = season
            all_data.append(df)
        
        data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded {len(data)} matches")
        
        # Initialize corrected feature engineer
        fe = CorrectedFeatureEngineer()
        
        # Process the data
        fe.load_data(data)
        fe.preprocess_basic_data()
        fe.create_team_specific_dataframes()
        fe.create_match_features()
        fe.prepare_final_dataset()
        fe.get_feature_importance_summary()
        
        # Save processed data
        fe.processed_data.to_csv('/home/joost/Premier League Prediction Model/corrected_processed_data.csv', index=False)
        print(f"\n💾 Corrected processed data saved to 'corrected_processed_data.csv'")
        
        print("\n✅ Phase 2 Complete: Corrected Feature Engineering Done!")
        print("\nNext steps:")
        print("1. Review the corrected features")
        print("2. Proceed to model training")
        print("3. Implement proper train/test splits")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
