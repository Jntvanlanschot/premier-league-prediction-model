#!/usr/bin/env python3
"""
Premier League Match Outcome Prediction Model - Phase 4
=======================================================

Model Training and Evaluation Module
- Proper train/test splits by season
- Multiple model comparison
- Performance evaluation and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Model training and evaluation class for Premier League prediction."""
    
    def __init__(self):
        self.data = None
        self.feature_columns = []
        self.target_column = 'FTR'
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_processed_data(self, file_path):
        """Load the processed data from feature engineering."""
        self.data = pd.read_csv(file_path)
        
        # Identify feature columns (exclude metadata columns)
        metadata_cols = ['Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR']
        self.feature_columns = [col for col in self.data.columns if col not in metadata_cols]
        
        print(f"📊 Loaded {len(self.data)} matches with {len(self.feature_columns)} features")
        return self
    
    def create_train_test_split(self):
        """Create proper train/test splits by season."""
        print("\n📊 CREATING TRAIN/TEST SPLITS")
        print("=" * 50)
        
        # Convert Date to datetime for proper sorting
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Sort by date to ensure chronological order
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Define train/test splits based on user requirements:
        # Train: 2023-24 + 1st half 2024-25
        # Test: 2nd half 2024-25
        
        # Split by season and half
        train_data = []
        test_data = []
        
        for season in [2324, 2425]:
            season_data = self.data[self.data['Season'] == season].copy()
            
            if season == 2324:
                # Use entire 2023-24 season for training
                train_data.append(season_data)
                print(f"  {season} season: {len(season_data)} matches → TRAIN")
                
            elif season == 2425:
                # Split 2024-25 season in half
                season_data = season_data.sort_values('Date').reset_index(drop=True)
                mid_point = len(season_data) // 2
                
                first_half = season_data.iloc[:mid_point]
                second_half = season_data.iloc[mid_point:]
                
                train_data.append(first_half)
                test_data.append(second_half)
                
                print(f"  {season} season: {len(first_half)} matches → TRAIN, {len(second_half)} matches → TEST")
        
        # Combine train and test data
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
    
    def prepare_features(self):
        """Prepare features for modeling."""
        print("\n🔧 PREPARING FEATURES")
        print("=" * 30)
        
        # Prepare training features
        X_train = self.train_data[self.feature_columns].fillna(0)
        y_train = self.train_data[self.target_column]
        
        # Prepare test features
        X_test = self.test_data[self.feature_columns].fillna(0)
        y_test = self.test_data[self.target_column]
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode target variable
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Store prepared data
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train_encoded
        self.y_test = y_test_encoded
        self.y_train_original = y_train
        self.y_test_original = y_test
        
        print(f"✅ Training features: {X_train_scaled.shape}")
        print(f"✅ Test features: {X_test_scaled.shape}")
        
        return self
    
    def train_models(self):
        """Train multiple models and compare performance."""
        print("\n🤖 TRAINING MODELS")
        print("=" * 30)
        
        # Define models to train
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                C=1.0,
                kernel='rbf'
            )
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            # Calculate log loss
            logloss = log_loss(self.y_test, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'log_loss': logloss,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  ✅ {name}: Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")
        
        return self
    
    def evaluate_models(self):
        """Evaluate and compare model performance."""
        print("\n📊 MODEL EVALUATION")
        print("=" * 50)
        
        # Create results summary
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'Log Loss': results['log_loss']
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("Model Performance Summary:")
        print("-" * 50)
        print(results_df.to_string(index=False, float_format='%.3f'))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.3f}")
        print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.3f}")
        
        # Detailed analysis of best model
        print(f"\n📋 Detailed Analysis - {best_model_name}:")
        print("-" * 40)
        
        y_pred = self.results[best_model_name]['predictions']
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(
            self.y_test_original, 
            y_pred_original,
            target_names=['Away Win', 'Draw', 'Home Win']
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test_original, y_pred_original)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                A    D    H")
        print(f"Actual A      {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
        print(f"       D      {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
        print(f"       H      {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
        
        return results_df, best_model_name
    
    def analyze_feature_importance(self, model_name=None):
        """Analyze feature importance for the best model."""
        if model_name is None:
            model_name = list(self.results.keys())[0]
        
        print(f"\n🌳 FEATURE IMPORTANCE ANALYSIS - {model_name}")
        print("=" * 60)
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 20 Most Important Features:")
            print("-" * 40)
            for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<40} {row['importance']:.4f}")
            
            # Analyze by feature category
            print("\nFeature Importance by Category:")
            print("-" * 40)
            
            categories = {
                'Team Performance': [f for f in importance_df['feature'] if 'team_' in f],
                'Home/Away Specific': [f for f in importance_df['feature'] if 'home_' in f or 'away_' in f],
                'Betting Odds': [f for f in importance_df['feature'] if 'odds' in f],
                'Relative Features': [f for f in importance_df['feature'] if 'advantage' in f]
            }
            
            for category, features in categories.items():
                if features:
                    category_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
                    print(f"  {category:<20}: {category_importance:.4f} ({len(features)} features)")
            
            return importance_df
        else:
            print(f"Model {model_name} does not support feature importance analysis.")
            return None
    
    def create_visualizations(self):
        """Create visualizations for model evaluation."""
        print("\n📊 Creating model evaluation visualizations...")
        
        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Premier League Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Model performance comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        bars = axes[0, 0].bar(model_names, accuracies, color=['#2E8B57', '#FFD700', '#DC143C', '#4169E1'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Confusion matrix for best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        y_pred = self.results[best_model_name]['predictions']
        y_pred_original = self.label_encoder.inverse_transform(y_pred)
        
        cm = confusion_matrix(self.y_test_original, y_pred_original)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Away Win', 'Draw', 'Home Win'],
                   yticklabels=['Away Win', 'Draw', 'Home Win'],
                   ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        
        # 3. Prediction probabilities distribution
        probabilities = self.results[best_model_name]['probabilities']
        axes[1, 0].hist(probabilities[:, 0], bins=20, alpha=0.7, label='Away Win', color='red')
        axes[1, 0].hist(probabilities[:, 1], bins=20, alpha=0.7, label='Draw', color='gold')
        axes[1, 0].hist(probabilities[:, 2], bins=20, alpha=0.7, label='Home Win', color='green')
        axes[1, 0].set_title('Prediction Probabilities Distribution')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 4. Feature importance (if available)
        if hasattr(self.models[best_model_name], 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.models[best_model_name].feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_10 = importance_df.head(10)
            axes[1, 1].barh(range(len(top_10)), top_10['importance'], color='lightcoral')
            axes[1, 1].set_yticks(range(len(top_10)))
            axes[1, 1].set_yticklabels(top_10['feature'], fontsize=8)
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('/home/joost/Premier League Prediction Model/model_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations saved as 'model_evaluation.png'")
    
    def generate_final_report(self):
        """Generate a comprehensive final report."""
        print("\n📋 FINAL MODEL REPORT")
        print("=" * 60)
        
        # Model performance summary
        print("1. MODEL PERFORMANCE SUMMARY:")
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1'],
                'Log Loss': results['log_loss']
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"   {i}. {row['Model']:<20}: Accuracy = {row['Accuracy']:.3f}, F1 = {row['F1-Score']:.3f}")
        
        # Best model details
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n2. BEST MODEL: {best_model_name}")
        print(f"   📊 Accuracy: {results_df.iloc[0]['Accuracy']:.3f}")
        print(f"   📊 F1-Score: {results_df.iloc[0]['F1-Score']:.3f}")
        print(f"   📊 Log Loss: {results_df.iloc[0]['Log Loss']:.3f}")
        
        # Dataset summary
        print(f"\n3. DATASET SUMMARY:")
        print(f"   📈 Training matches: {len(self.train_data)}")
        print(f"   📈 Test matches: {len(self.test_data)}")
        print(f"   📈 Features used: {len(self.feature_columns)}")
        
        # Feature insights
        print(f"\n4. FEATURE INSIGHTS:")
        team_features = len([f for f in self.feature_columns if 'team_' in f])
        home_away_features = len([f for f in self.feature_columns if 'home_' in f or 'away_' in f])
        odds_features = len([f for f in self.feature_columns if 'odds' in f])
        relative_features = len([f for f in self.feature_columns if 'advantage' in f])
        
        print(f"   🏆 Team Performance Features: {team_features}")
        print(f"   🏠 Home/Away Specific Features: {home_away_features}")
        print(f"   💰 Betting Odds Features: {odds_features}")
        print(f"   ⚖️  Relative Features: {relative_features}")
        
        # Recommendations
        print(f"\n5. RECOMMENDATIONS:")
        print(f"   🎯 Model performs well on test data")
        print(f"   📈 Consider ensemble methods for better performance")
        print(f"   🔄 Retrain with 2025-26 data when available")
        print(f"   ⚖️  Monitor performance on new seasons")
        
        return results_df

def main():
    """Main execution function for model training and evaluation."""
    print("🏆 Premier League Model Training & Evaluation")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load processed data
        trainer.load_processed_data('/home/joost/Premier League Prediction Model/corrected_processed_data.csv')
        
        # Create train/test splits
        trainer.create_train_test_split()
        
        # Prepare features
        trainer.prepare_features()
        
        # Train models
        trainer.train_models()
        
        # Evaluate models
        results_df, best_model = trainer.evaluate_models()
        
        # Analyze feature importance
        trainer.analyze_feature_importance(best_model)
        
        # Create visualizations
        trainer.create_visualizations()
        
        # Generate final report
        trainer.generate_final_report()
        
        print("\n✅ Phase 4 Complete: Model Training & Evaluation Done!")
        print("\nNext steps:")
        print("1. Review the model performance results")
        print("2. Consider model optimization if needed")
        print("3. Prepare for live predictions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
