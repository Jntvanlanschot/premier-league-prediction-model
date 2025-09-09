#!/usr/bin/env python3
"""
Premier League Match Outcome Prediction Model
============================================

This script builds a machine learning model to predict football match outcomes
(Win, Draw, Loss) using match statistics and betting odds data.

Data Source: https://www.football-data.co.uk/englandm.php
Target Variable: FTR (Full-Time Result: H=Home Win, D=Draw, A=Away Win)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PremierLeaguePredictor:
    """Main class for Premier League match outcome prediction."""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.models = {}
        self.results = {}
        
    def download_data(self, seasons=['2324', '2425']):
        """
        Download Premier League data for specified seasons.
        
        Args:
            seasons (list): List of season codes (e.g., ['2324', '2425'])
        """
        print("📥 Downloading Premier League data...")
        
        all_data = []
        
        for season in seasons:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
                print(f"  Downloading {season} season data...")
                
                df = pd.read_csv(url)
                df['Season'] = season
                all_data.append(df)
                
                print(f"  ✅ {season}: {len(df)} matches loaded")
                
            except Exception as e:
                print(f"  ❌ Error downloading {season}: {e}")
                
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 Total dataset: {len(self.data)} matches")
            return True
        else:
            print("❌ No data downloaded successfully")
            return False
    
    def explore_data(self):
        """Perform initial data exploration and summary."""
        if self.data is None:
            print("❌ No data loaded. Please download data first.")
            return
            
        print("\n🔍 DATA EXPLORATION")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Missing values
        print("\n📋 Missing Values:")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        print(missing_df[missing_df['Missing Count'] > 0].head(10))
        
        # Target variable distribution
        print("\n🎯 Target Variable (FTR) Distribution:")
        ftr_counts = self.data['FTR'].value_counts()
        ftr_pct = self.data['FTR'].value_counts(normalize=True) * 100
        
        for result in ['H', 'D', 'A']:
            count = ftr_counts.get(result, 0)
            pct = ftr_pct.get(result, 0)
            result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
            print(f"  {result_name}: {count} ({pct:.1f}%)")
        
        # Season distribution
        print("\n📅 Season Distribution:")
        season_counts = self.data['Season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"  {season}: {count} matches")
        
        # Key columns overview
        print("\n📈 Key Columns Overview:")
        key_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in self.data.columns for col in key_cols):
            print(self.data[key_cols].head())
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations for data exploration."""
        if self.data is None:
            print("❌ No data loaded.")
            return
            
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Premier League Data Exploration', fontsize=16, fontweight='bold')
        
        # 1. FTR Distribution
        ftr_counts = self.data['FTR'].value_counts()
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Gold, Red
        axes[0, 0].pie(ftr_counts.values, labels=['Home Win', 'Draw', 'Away Win'], 
                      autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Match Outcome Distribution')
        
        # 2. Goals Distribution
        if 'FTHG' in self.data.columns and 'FTAG' in self.data.columns:
            goals_data = pd.concat([self.data['FTHG'], self.data['FTAG']])
            axes[0, 1].hist(goals_data, bins=range(0, goals_data.max()+2), 
                           alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Goals per Match Distribution')
            axes[0, 1].set_xlabel('Goals')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Season-wise FTR distribution
        if 'Season' in self.data.columns:
            season_ftr = pd.crosstab(self.data['Season'], self.data['FTR'])
            season_ftr.plot(kind='bar', ax=axes[1, 0], color=colors)
            axes[1, 0].set_title('Match Outcomes by Season')
            axes[1, 0].set_xlabel('Season')
            axes[1, 0].set_ylabel('Number of Matches')
            axes[1, 0].legend(['Home Win', 'Draw', 'Away Win'])
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Goals correlation
        if 'FTHG' in self.data.columns and 'FTAG' in self.data.columns:
            axes[1, 1].scatter(self.data['FTHG'], self.data['FTAG'], alpha=0.6, color='purple')
            axes[1, 1].set_title('Home Goals vs Away Goals')
            axes[1, 1].set_xlabel('Home Goals')
            axes[1, 1].set_ylabel('Away Goals')
            
            # Add trend line
            z = np.polyfit(self.data['FTHG'], self.data['FTAG'], 1)
            p = np.poly1d(z)
            axes[1, 1].plot(self.data['FTHG'], p(self.data['FTHG']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('/home/joost/Premier League Prediction Model/data_exploration.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'data_exploration.png'")

def main():
    """Main execution function."""
    print("🏆 Premier League Match Outcome Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = PremierLeaguePredictor()
    
    # Download data
    success = predictor.download_data(['2324', '2425'])
    
    if success:
        # Explore data
        predictor.explore_data()
        
        # Create visualizations
        predictor.visualize_data()
        
        print("\n✅ Phase 1 Complete: Data downloaded and explored!")
        print("\nNext steps:")
        print("1. Review the data exploration results")
        print("2. Decide on feature selection priorities")
        print("3. Proceed to data preprocessing")
        
    else:
        print("❌ Failed to download data. Please check your internet connection.")

if __name__ == "__main__":
    main()

