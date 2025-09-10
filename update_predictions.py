#!/usr/bin/env python3
"""
Update Predictions Script
========================
Updates predictions for the next matchweek.
"""

import json
from upcoming_fixtures import get_upcoming_fixtures, generate_odds_for_fixtures
from updated_web_predictor import UpdatedWebPredictor

def update_predictions():
    """Update predictions for the next matchweek."""
    print("🔄 Updating Premier League Predictions")
    print("=" * 40)
    
    try:
        # Initialize predictor
        predictor = UpdatedWebPredictor()
        predictor.load_and_prepare_data()
        
        # Get next matchweek fixtures
        fixtures = get_upcoming_fixtures()
        next_fixtures = fixtures[0]  # Get first matchweek
        
        # Generate odds
        fixtures_with_odds = generate_odds_for_fixtures(next_fixtures)
        
        # Get predictions
        predictions = predictor.predict_upcoming_games(fixtures_with_odds)
        
        # Generate web data
        web_data = {
            'predictions': predictions,
            'model_info': {
                'accuracy': '54.4%',
                'log_loss': '0.983',
                'brier_score': '0.195',
                'cv_stability': '55.4% ± 2.5%'
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fixture_info': {
                'matchweek': 'MW1',
                'date': 'Sunday, August 31, 2025',
                'total_matches': len(predictions)
            }
        }
        
        # Save to JSON file
        with open('web_data.json', 'w') as f:
            json.dump(web_data, f, indent=2)
        
        print(f"✅ Updated predictions for {len(predictions)} games")
        print(f"📅 Matchweek: {web_data['fixture_info']['matchweek']}")
        print(f"🕐 Last updated: {web_data['last_updated']}")
        
    except Exception as e:
        print(f"❌ Error updating predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    update_predictions()
