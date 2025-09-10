#!/usr/bin/env python3
"""
Upcoming Fixtures Generator
===========================
Generates upcoming Premier League fixtures for prediction.
Current date: September 9, 2025
Next fixtures: September 13-14, 2025 (Matchweek 2)
"""

import json
from datetime import datetime, timedelta

def get_upcoming_fixtures():
    """Get upcoming fixtures starting from the next matchweek."""
    
    # Current date is September 9, 2025
    # Next fixtures are September 13-14, 2025 (Matchweek 2)
    
    upcoming_fixtures = [
        # Next weekend (September 13-14, 2025) - Matchweek 2
        {
            'date': '2025-09-13',
            'matches': [
                {'home': 'Arsenal', 'away': 'Nott\'m Forest', 'time': '12:30'},
                {'home': 'AFC Bournemouth', 'away': 'Brighton', 'time': '15:00'},
                {'home': 'Crystal Palace', 'away': 'Sunderland', 'time': '15:00'},
                {'home': 'Everton', 'away': 'Aston Villa', 'time': '15:00'},
                {'home': 'Fulham', 'away': 'Leeds United', 'time': '15:00'},
                {'home': 'Newcastle United', 'away': 'Wolves', 'time': '15:00'},
                {'home': 'West Ham United', 'away': 'Spurs', 'time': '17:30'},
                {'home': 'Brentford', 'away': 'Chelsea', 'time': '20:00'}
            ]
        },
        {
            'date': '2025-09-14',
            'matches': [
                {'home': 'Burnley', 'away': 'Liverpool', 'time': '14:00'},
                {'home': 'Man City', 'away': 'Man Utd', 'time': '16:30'}
            ]
        }
    ]
    
    return upcoming_fixtures

def get_next_matchweek_fixtures():
    """Get only the next matchweek fixtures (this weekend)."""
    fixtures = get_upcoming_fixtures()
    return fixtures[0]  # Return only the first matchweek

def generate_odds_for_fixtures(fixtures):
    """Generate realistic odds for fixtures based on team strength."""
    
    # Team strength ratings (based on recent performance)
    team_strength = {
        'Manchester City': 0.85,
        'Arsenal': 0.80,
        'Liverpool': 0.78,
        'Chelsea': 0.75,
        'Tottenham': 0.72,
        'Spurs': 0.72,
        'Manchester United': 0.70,
        'Man Utd': 0.70,
        'Newcastle United': 0.68,
        'Aston Villa': 0.65,
        'Brighton & Hove Albion': 0.62,
        'Brighton': 0.62,
        'West Ham United': 0.60,
        'Brentford': 0.58,
        'Crystal Palace': 0.55,
        'Fulham': 0.52,
        'Everton': 0.50,
        'Wolves': 0.48,
        'Wolverhampton Wanderers': 0.48,
        'Nottingham Forest': 0.45,
        'Nott\'m Forest': 0.45,
        'Burnley': 0.42,
        'AFC Bournemouth': 0.40,
        'Leeds United': 0.38,
        'Sunderland': 0.35
    }
    
    fixtures_with_odds = []
    
    for match in fixtures['matches']:
        home_team = match['home']
        away_team = match['away']
        
        # Get team strengths
        home_strength = team_strength.get(home_team, 0.50)
        away_strength = team_strength.get(away_team, 0.50)
        
        # Calculate base probabilities
        total_strength = home_strength + away_strength
        home_prob = home_strength / total_strength
        away_prob = away_strength / total_strength
        draw_prob = 0.25  # Base draw probability
        
        # Adjust probabilities to sum to 1
        total_prob = home_prob + draw_prob + away_prob
        home_prob /= total_prob
        draw_prob /= total_prob
        away_prob /= total_prob
        
        # Convert to odds
        home_odds = 1 / home_prob
        draw_odds = 1 / draw_prob
        away_odds = 1 / away_prob
        
        # Add some market variance
        import random
        home_odds *= random.uniform(0.95, 1.05)
        draw_odds *= random.uniform(0.95, 1.05)
        away_odds *= random.uniform(0.95, 1.05)
        
        fixtures_with_odds.append({
            'home_team': home_team,
            'away_team': away_team,
            'date': fixtures['date'],
            'time': match['time'],
            'odds': {
                'home': round(home_odds, 2),
                'draw': round(draw_odds, 2),
                'away': round(away_odds, 2)
            }
        })
    
    return fixtures_with_odds

def main():
    """Generate upcoming fixtures with odds."""
    print("📅 Generating upcoming Premier League fixtures...")
    print("📅 Current date: September 9, 2025")
    print("📅 Next fixtures: September 13-14, 2025 (Matchweek 2)")
    
    # Get next matchweek fixtures
    next_fixtures = get_next_matchweek_fixtures()
    
    # Generate odds for fixtures
    fixtures_with_odds = generate_odds_for_fixtures(next_fixtures)
    
    print(f"✅ Generated {len(fixtures_with_odds)} fixtures for {next_fixtures['date']}")
    
    # Save to file
    with open('upcoming_fixtures.json', 'w') as f:
        json.dump(fixtures_with_odds, f, indent=2)
    
    print("📁 Saved fixtures to upcoming_fixtures.json")
    
    return fixtures_with_odds

if __name__ == "__main__":
    main()
