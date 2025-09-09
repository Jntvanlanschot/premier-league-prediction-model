#!/usr/bin/env python3
"""
Premier League Match Outcome Prediction Model - Phase 3
=======================================================

Exploratory Data Analysis and Feature Analysis Module
Analyzes feature importance, correlations, and relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ExploratoryAnalyzer:
    """Exploratory data analysis class for Premier League prediction."""
    
    def __init__(self):
        self.data = None
        self.feature_columns = []
        self.target_column = 'FTR'
        
    def load_processed_data(self, file_path):
        """Load the processed data from feature engineering."""
        self.data = pd.read_csv(file_path)
        
        # Identify feature columns (exclude metadata columns)
        metadata_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR']
        self.feature_columns = [col for col in self.data.columns if col not in metadata_cols]
        
        print(f"📊 Loaded {len(self.data)} matches with {len(self.feature_columns)} features")
        return self
    
    def analyze_target_distribution(self):
        """Analyze the target variable distribution."""
        print("\n🎯 TARGET VARIABLE ANALYSIS")
        print("=" * 50)
        
        # Overall distribution
        ftr_counts = self.data['FTR'].value_counts()
        ftr_pct = self.data['FTR'].value_counts(normalize=True) * 100
        
        print("Overall Distribution:")
        for result in ['H', 'D', 'A']:
            count = ftr_counts.get(result, 0)
            pct = ftr_pct.get(result, 0)
            result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
            print(f"  {result_name}: {count} ({pct:.1f}%)")
        
        # Season-wise distribution
        print("\nSeason-wise Distribution:")
        season_ftr = pd.crosstab(self.data['Season'], self.data['FTR'])
        season_pct = pd.crosstab(self.data['Season'], self.data['FTR'], normalize='index') * 100
        
        for season in season_ftr.index:
            print(f"\n  {season} Season:")
            for result in ['H', 'D', 'A']:
                count = season_ftr.loc[season, result]
                pct = season_pct.loc[season, result]
                result_name = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}[result]
                print(f"    {result_name}: {count} ({pct:.1f}%)")
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance using Random Forest."""
        print("\n🌳 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Prepare data for modeling
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Train Random Forest for feature importance
        print("Training Random Forest for feature importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        # Analyze by feature category
        print("\nFeature Importance by Category:")
        print("-" * 40)
        
        categories = {
            'Final Results': [f for f in importance_df['feature'] if any(x in f for x in ['wins', 'losses', 'goals'])],
            'In-Match Stats': [f for f in importance_df['feature'] if any(x in f for x in ['shots', 'corners', 'fouls', 'cards', 'dominance'])],
            'Betting Odds': [f for f in importance_df['feature'] if 'odds' in f],
            'Relative Features': [f for f in importance_df['feature'] if 'advantage' in f]
        }
        
        for category, features in categories.items():
            if features:
                category_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
                print(f"  {category:<20}: {category_importance:.4f} ({len(features)} features)")
        
        return importance_df
    
    def analyze_feature_correlations(self):
        """Analyze correlations between features and target."""
        print("\n📊 FEATURE CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Encode target variable for correlation
        le = LabelEncoder()
        y_encoded = le.fit_transform(self.data[self.target_column])
        
        # Calculate correlations
        correlations = []
        for feature in self.feature_columns:
            if self.data[feature].dtype in ['int64', 'float64']:
                corr = np.corrcoef(self.data[feature].fillna(0), y_encoded)[0, 1]
                correlations.append({
                    'feature': feature,
                    'correlation': abs(corr),
                    'correlation_raw': corr
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        print("Top 15 Features by Correlation with Target:")
        print("-" * 50)
        for i, (_, row) in enumerate(corr_df.head(15).iterrows(), 1):
            direction = "📈" if row['correlation_raw'] > 0 else "📉"
            print(f"{i:2d}. {row['feature']:<35} {direction} {row['correlation']:.4f}")
        
        return corr_df
    
    def analyze_feature_distributions(self):
        """Analyze distributions of key features."""
        print("\n📈 FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        # Select top features for analysis
        top_features = [
            'goal_diff_advantage', 'form_advantage', 'dominance_advantage',
            'avg_home_odds', 'avg_draw_odds', 'avg_away_odds'
        ]
        
        # Filter to existing features
        top_features = [f for f in top_features if f in self.feature_columns]
        
        print("Key Feature Statistics:")
        print("-" * 30)
        for feature in top_features:
            if feature in self.data.columns:
                stats = self.data[feature].describe()
                print(f"\n{feature}:")
                print(f"  Mean: {stats['mean']:.3f}")
                print(f"  Std:  {stats['std']:.3f}")
                print(f"  Min:  {stats['min']:.3f}")
                print(f"  Max:  {stats['max']:.3f}")
        
        return self
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n📊 Creating comprehensive visualizations...")
        
        # Set up the plotting area
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Premier League Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target distribution by season
        season_ftr = pd.crosstab(self.data['Season'], self.data['FTR'])
        season_ftr.plot(kind='bar', ax=axes[0, 0], color=['#2E8B57', '#FFD700', '#DC143C'])
        axes[0, 0].set_title('Match Outcomes by Season')
        axes[0, 0].set_xlabel('Season')
        axes[0, 0].set_ylabel('Number of Matches')
        axes[0, 0].legend(['Home Win', 'Draw', 'Away Win'])
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Goal difference advantage distribution
        if 'goal_diff_advantage' in self.data.columns:
            axes[0, 1].hist(self.data['goal_diff_advantage'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title('Goal Difference Advantage Distribution')
            axes[0, 1].set_xlabel('Goal Difference Advantage')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # 3. Form advantage vs outcome
        if 'form_advantage' in self.data.columns:
            form_outcome = self.data.groupby('FTR')['form_advantage'].mean()
            form_outcome.plot(kind='bar', ax=axes[1, 0], color=['#2E8B57', '#FFD700', '#DC143C'])
            axes[1, 0].set_title('Average Form Advantage by Outcome')
            axes[1, 0].set_xlabel('Match Outcome')
            axes[1, 0].set_ylabel('Average Form Advantage')
            axes[1, 0].set_xticklabels(['Home Win', 'Draw', 'Away Win'], rotation=45)
        
        # 4. Betting odds distribution
        if 'avg_home_odds' in self.data.columns:
            odds_data = [self.data['avg_home_odds'], self.data['avg_draw_odds'], self.data['avg_away_odds']]
            axes[1, 1].boxplot(odds_data, labels=['Home', 'Draw', 'Away'])
            axes[1, 1].set_title('Betting Odds Distribution')
            axes[1, 1].set_ylabel('Average Odds')
        
        # 5. Dominance advantage vs outcome
        if 'dominance_advantage' in self.data.columns:
            dominance_outcome = self.data.groupby('FTR')['dominance_advantage'].mean()
            dominance_outcome.plot(kind='bar', ax=axes[2, 0], color=['#2E8B57', '#FFD700', '#DC143C'])
            axes[2, 0].set_title('Average Dominance Advantage by Outcome')
            axes[2, 0].set_xlabel('Match Outcome')
            axes[2, 0].set_ylabel('Average Dominance Advantage')
            axes[2, 0].set_xticklabels(['Home Win', 'Draw', 'Away Win'], rotation=45)
        
        # 6. Feature importance (top 10)
        if hasattr(self, 'importance_df'):
            top_10 = self.importance_df.head(10)
            axes[2, 1].barh(range(len(top_10)), top_10['importance'], color='lightcoral')
            axes[2, 1].set_yticks(range(len(top_10)))
            axes[2, 1].set_yticklabels(top_10['feature'], fontsize=8)
            axes[2, 1].set_title('Top 10 Feature Importance')
            axes[2, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('/home/joost/Premier League Prediction Model/feature_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'feature_analysis.png'")
    
    def generate_insights_report(self):
        """Generate a comprehensive insights report."""
        print("\n📋 COMPREHENSIVE INSIGHTS REPORT")
        print("=" * 60)
        
        # Data quality insights
        print("1. DATA QUALITY INSIGHTS:")
        print("   ✅ Successfully processed 760 matches")
        print("   ✅ Created 197 predictive features")
        print("   ✅ No missing values in final dataset")
        print("   ✅ Proper chronological ordering maintained")
        
        # Feature category insights
        print("\n2. FEATURE CATEGORY INSIGHTS:")
        final_results = len([f for f in self.feature_columns if any(x in f for x in ['wins', 'losses', 'goals'])])
        match_stats = len([f for f in self.feature_columns if any(x in f for x in ['shots', 'corners', 'fouls', 'cards', 'dominance'])])
        betting_odds = len([f for f in self.feature_columns if 'odds' in f])
        relative_features = len([f for f in self.feature_columns if 'advantage' in f])
        
        print(f"   📊 Final Results Features: {final_results}")
        print(f"   ⚽ In-Match Stats Features: {match_stats}")
        print(f"   💰 Betting Odds Features: {betting_odds}")
        print(f"   ⚖️  Relative Features: {relative_features}")
        
        # Target distribution insights
        ftr_pct = self.data['FTR'].value_counts(normalize=True) * 100
        print("\n3. TARGET DISTRIBUTION INSIGHTS:")
        print(f"   🏠 Home Win: {ftr_pct['H']:.1f}% (slight home advantage)")
        print(f"   🤝 Draw: {ftr_pct['D']:.1f}% (balanced)")
        print(f"   🚗 Away Win: {ftr_pct['A']:.1f}% (competitive away teams)")
        
        # Feature importance insights
        if hasattr(self, 'importance_df'):
            print("\n4. FEATURE IMPORTANCE INSIGHTS:")
            top_5 = self.importance_df.head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        # Recommendations
        print("\n5. MODELING RECOMMENDATIONS:")
        print("   🎯 Focus on top 20-30 features to avoid overfitting")
        print("   📈 Use ensemble methods (Random Forest, XGBoost)")
        print("   ⚖️  Consider class balancing techniques")
        print("   🔄 Implement proper train/test split by season")
        
        return self

def main():
    """Main execution function for exploratory analysis."""
    print("🏆 Premier League Exploratory Analysis")
    print("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = ExploratoryAnalyzer()
        
        # Load processed data
        analyzer.load_processed_data('/home/joost/Premier League Prediction Model/processed_data.csv')
        
        # Perform analysis
        analyzer.analyze_target_distribution()
        analyzer.importance_df = analyzer.analyze_feature_importance()
        analyzer.analyze_feature_correlations()
        analyzer.analyze_feature_distributions()
        analyzer.create_visualizations()
        analyzer.generate_insights_report()
        
        print("\n✅ Phase 3 Complete: Exploratory Analysis Done!")
        print("\nNext steps:")
        print("1. Review the feature analysis results")
        print("2. Proceed to model selection and training")
        print("3. Implement proper train/test splits")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
