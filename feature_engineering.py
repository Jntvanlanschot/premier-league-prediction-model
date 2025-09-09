#!/usr/bin/env python3
"""
Premier League Match Outcome Prediction Model - Phase 2
=======================================================

Feature Engineering and Data Preprocessing Module
Based on user requirements:
1. Average betting odds from all bookmakers
2. Rolling averages from last 5 matches within current season
3. No referee data
4. Priority: Final Results > In-Match Stats > Bookmaker Odds
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering class for Premier League prediction model."""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_column = 'FTR'
        
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
        
        # Create average betting odds (user requirement #1)
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
    
    def create_rolling_features(self):
        """Create rolling average features based on last 5 matches within season."""
        print("\n📈 CREATING ROLLING FEATURES")
        print("=" * 40)
        
        # Initialize feature columns
        feature_cols = []
        
        # Process each team separately
        all_teams = set(self.data['HomeTeam'].unique()) | set(self.data['AwayTeam'].unique())
        
        print(f"Processing {len(all_teams)} teams...")
        
        for team in all_teams:
            print(f"  Processing {team}...")
            
            # Get all matches for this team (both home and away)
            team_matches = self.data[
                (self.data['HomeTeam'] == team) | (self.data['AwayTeam'] == team)
            ].copy()
            
            # Sort by season and date
            team_matches = team_matches.sort_values(['Season', 'Date']).reset_index(drop=True)
            
            # Create rolling features for each season separately
            for season in team_matches['Season'].unique():
                season_matches = team_matches[team_matches['Season'] == season].copy()
                
                # Create rolling features for this season
                rolling_features = self._create_team_rolling_features(season_matches, team)
                
                # Update the main dataframe
                for idx, row in rolling_features.iterrows():
                    original_idx = row['original_index']
                    for col in rolling_features.columns:
                        if col != 'original_index':
                            self.data.loc[original_idx, col] = row[col]
                            if col not in feature_cols:
                                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        print(f"✅ Created {len(feature_cols)} rolling features")
        return self
    
    def _create_team_rolling_features(self, team_matches, team_name):
        """Create rolling features for a specific team's matches in a season."""
        
        # Initialize result dataframe
        result = team_matches.copy()
        result['original_index'] = team_matches.index
        
        # Define feature columns to create
        feature_cols = [
            'home_wins_5', 'home_draws_5', 'home_losses_5',
            'away_wins_5', 'away_draws_5', 'away_losses_5',
            'goals_scored_5', 'goals_conceded_5', 'goal_difference_5',
            'shots_5', 'shots_conceded_5', 'shots_on_target_5', 'shots_on_target_conceded_5',
            'corners_5', 'corners_conceded_5', 'fouls_5', 'fouls_conceded_5',
            'cards_5', 'cards_conceded_5', 'dominance_score_5'
        ]
        
        # Initialize all feature columns
        for col in feature_cols:
            result[col] = 0.0
        
        # Calculate rolling features for each match
        for i in range(len(team_matches)):
            # Get last 5 matches (or fewer if not enough data)
            start_idx = max(0, i - 5)
            recent_matches = team_matches.iloc[start_idx:i]  # Exclude current match
            
            if len(recent_matches) == 0:
                continue
            
            # Calculate features based on user priority requirements
            
            # 1. FINAL RESULTS (Highest Priority)
            home_wins = home_draws = home_losses = 0
            away_wins = away_draws = away_losses = 0
            goals_scored = goals_conceded = 0
            
            for _, match in recent_matches.iterrows():
                is_home = match['HomeTeam'] == team_name
                
                if is_home:
                    goals_scored += match['FTHG']
                    goals_conceded += match['FTAG']
                    
                    if match['FTR'] == 'H':
                        home_wins += 1
                    elif match['FTR'] == 'D':
                        home_draws += 1
                    else:
                        home_losses += 1
                else:
                    goals_scored += match['FTAG']
                    goals_conceded += match['FTHG']
                    
                    if match['FTR'] == 'A':
                        away_wins += 1
                    elif match['FTR'] == 'D':
                        away_draws += 1
                    else:
                        away_losses += 1
            
            # Store final results features
            result.iloc[i, result.columns.get_loc('home_wins_5')] = home_wins
            result.iloc[i, result.columns.get_loc('home_draws_5')] = home_draws
            result.iloc[i, result.columns.get_loc('home_losses_5')] = home_losses
            result.iloc[i, result.columns.get_loc('away_wins_5')] = away_wins
            result.iloc[i, result.columns.get_loc('away_draws_5')] = away_draws
            result.iloc[i, result.columns.get_loc('away_losses_5')] = away_losses
            result.iloc[i, result.columns.get_loc('goals_scored_5')] = goals_scored
            result.iloc[i, result.columns.get_loc('goals_conceded_5')] = goals_conceded
            result.iloc[i, result.columns.get_loc('goal_difference_5')] = goals_scored - goals_conceded
            
            # 2. IN-MATCH STATS (Second Priority)
            shots = shots_conceded = shots_on_target = shots_on_target_conceded = 0
            corners = corners_conceded = fouls = fouls_conceded = 0
            cards = cards_conceded = 0
            
            for _, match in recent_matches.iterrows():
                is_home = match['HomeTeam'] == team_name
                
                if is_home:
                    shots += match['HS']
                    shots_conceded += match['AS']
                    shots_on_target += match['HST']
                    shots_on_target_conceded += match['AST']
                    corners += match['HC']
                    corners_conceded += match['AC']
                    fouls += match['HF']
                    fouls_conceded += match['AF']
                    cards += match['HY'] + match['HR']
                    cards_conceded += match['AY'] + match['AR']
                else:
                    shots += match['AS']
                    shots_conceded += match['HS']
                    shots_on_target += match['AST']
                    shots_on_target_conceded += match['HST']
                    corners += match['AC']
                    corners_conceded += match['HC']
                    fouls += match['AF']
                    fouls_conceded += match['HF']
                    cards += match['AY'] + match['AR']
                    cards_conceded += match['HY'] + match['HR']
            
            # Store in-match stats features
            result.iloc[i, result.columns.get_loc('shots_5')] = shots
            result.iloc[i, result.columns.get_loc('shots_conceded_5')] = shots_conceded
            result.iloc[i, result.columns.get_loc('shots_on_target_5')] = shots_on_target
            result.iloc[i, result.columns.get_loc('shots_on_target_conceded_5')] = shots_on_target_conceded
            result.iloc[i, result.columns.get_loc('corners_5')] = corners
            result.iloc[i, result.columns.get_loc('corners_conceded_5')] = corners_conceded
            result.iloc[i, result.columns.get_loc('fouls_5')] = fouls
            result.iloc[i, result.columns.get_loc('fouls_conceded_5')] = fouls_conceded
            result.iloc[i, result.columns.get_loc('cards_5')] = cards
            result.iloc[i, result.columns.get_loc('cards_conceded_5')] = cards_conceded
            
            # Calculate dominance score (shots advantage - fouls disadvantage)
            dominance_score = (shots - shots_conceded) - (fouls - fouls_conceded) * 0.5
            result.iloc[i, result.columns.get_loc('dominance_score_5')] = dominance_score
        
        return result
    
    def create_match_features(self):
        """Create match-specific features."""
        print("\n⚽ CREATING MATCH FEATURES")
        print("=" * 40)
        
        # Home/Away team features
        match_features = []
        
        for idx, row in self.data.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            
            # Get rolling features for both teams
            home_features = {}
            away_features = {}
            
            for col in self.feature_columns:
                if col.startswith('home_') or col.startswith('away_') or col.startswith('goals_') or col.startswith('shots_') or col.startswith('corners_') or col.startswith('fouls_') or col.startswith('cards_') or col.startswith('dominance_'):
                    home_features[f'home_{col}'] = row[col] if pd.notna(row[col]) else 0
                    away_features[f'away_{col}'] = row[col] if pd.notna(row[col]) else 0
            
            # Create relative features (home vs away)
            relative_features = {}
            
            # Goal difference advantage
            relative_features['goal_diff_advantage'] = home_features.get('home_goal_difference_5', 0) - away_features.get('away_goal_difference_5', 0)
            
            # Dominance advantage
            relative_features['dominance_advantage'] = home_features.get('home_dominance_score_5', 0) - away_features.get('away_dominance_score_5', 0)
            
            # Form advantage (wins - losses)
            home_form = home_features.get('home_home_wins_5', 0) + home_features.get('home_away_wins_5', 0) - home_features.get('home_home_losses_5', 0) - home_features.get('home_away_losses_5', 0)
            away_form = away_features.get('away_home_wins_5', 0) + away_features.get('away_away_wins_5', 0) - away_features.get('away_home_losses_5', 0) - away_features.get('away_away_losses_5', 0)
            relative_features['form_advantage'] = home_form - away_form
            
            # Add betting odds (Third Priority)
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
        print(f"  - Final Results Features: {len([f for f in self.feature_columns if 'wins' in f or 'losses' in f or 'goals' in f])}")
        print(f"  - In-Match Stats Features: {len([f for f in self.feature_columns if 'shots' in f or 'corners' in f or 'fouls' in f or 'dominance' in f])}")
        print(f"  - Betting Odds Features: {len([f for f in self.feature_columns if 'odds' in f])}")
        print(f"  - Relative Features: {len([f for f in self.feature_columns if 'advantage' in f])}")
        
        return self
    
    def get_feature_importance_summary(self):
        """Get a summary of the created features."""
        print("\n📊 FEATURE IMPORTANCE SUMMARY")
        print("=" * 50)
        
        # Group features by category
        final_results = [f for f in self.feature_columns if any(x in f for x in ['wins', 'losses', 'goals'])]
        match_stats = [f for f in self.feature_columns if any(x in f for x in ['shots', 'corners', 'fouls', 'cards', 'dominance'])]
        betting_odds = [f for f in self.feature_columns if 'odds' in f]
        relative_features = [f for f in self.feature_columns if 'advantage' in f]
        
        print("1. FINAL RESULTS (Highest Priority):")
        for feature in final_results[:5]:  # Show top 5
            print(f"   - {feature}")
        
        print("\n2. IN-MATCH STATS (Second Priority):")
        for feature in match_stats[:5]:  # Show top 5
            print(f"   - {feature}")
        
        print("\n3. BETTING ODDS (Third Priority):")
        for feature in betting_odds:
            print(f"   - {feature}")
        
        print("\n4. RELATIVE FEATURES:")
        for feature in relative_features:
            print(f"   - {feature}")
        
        return self.processed_data

def main():
    """Main execution function for feature engineering."""
    print("🏆 Premier League Feature Engineering")
    print("=" * 50)
    
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
        
        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Process the data
        fe.load_data(data)
        fe.preprocess_basic_data()
        fe.create_rolling_features()
        fe.create_match_features()
        fe.prepare_final_dataset()
        fe.get_feature_importance_summary()
        
        # Save processed data
        fe.processed_data.to_csv('/home/joost/Premier League Prediction Model/processed_data.csv', index=False)
        print(f"\n💾 Processed data saved to 'processed_data.csv'")
        
        print("\n✅ Phase 2 Complete: Feature Engineering Done!")
        print("\nNext steps:")
        print("1. Review the created features")
        print("2. Proceed to exploratory analysis")
        print("3. Train initial models")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
