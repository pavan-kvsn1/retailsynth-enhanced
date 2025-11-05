import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import date, timedelta

# ============================================================================
# SEASONALITY ENGINE (v3.2)
# ============================================================================

class SeasonalityEngine:
    """
    Models seasonal patterns in shopping behavior and product demand.
    Includes holidays, weather patterns, and cultural events.
    """
    
    def __init__(self, region: str = 'US'):
        self.region = region
        self.seasonal_events = self._define_seasonal_events()
    
    def _define_seasonal_events(self) -> Dict:
        """Define major seasonal events and their impact"""
        return {
            'US': {
                'New_Year': {'start_week': 1, 'duration': 2, 'intensity': 1.15, 'categories': ['Pantry', 'Personal_Care']},
                'Valentines': {'start_week': 7, 'duration': 1, 'intensity': 1.25, 'categories': ['Fresh']},
                'Spring_Break': {'start_week': 11, 'duration': 2, 'intensity': 1.2, 'categories': ['Pantry', 'General_Merchandise']},
                'Easter': {'start_week': 14, 'duration': 2, 'intensity': 1.3, 'categories': ['Fresh', 'Pantry']},
                'Summer_Start': {'start_week': 22, 'duration': 3, 'intensity': 1.18, 'categories': ['Fresh', 'Pantry']},
                'July_4th': {'start_week': 27, 'duration': 1, 'intensity': 1.35, 'categories': ['Fresh', 'Pantry']},
                'Back_to_School': {'start_week': 32, 'duration': 3, 'intensity': 1.25, 'categories': ['Pantry', 'Personal_Care', 'General_Merchandise']},
                'Halloween': {'start_week': 43, 'duration': 2, 'intensity': 1.4, 'categories': ['Pantry']},
                'Thanksgiving': {'start_week': 47, 'duration': 2, 'intensity': 1.8, 'categories': ['Fresh', 'Pantry']},
                'Christmas': {'start_week': 50, 'duration': 4, 'intensity': 2.0, 'categories': ['Fresh', 'Pantry', 'General_Merchandise']}
            }
        }
    
    def get_seasonality_multiplier(self, week_number: int, category: str) -> float:
        """
        Get demand multiplier for given week and category.
        Returns value typically between 0.8 and 2.0.
        """
        base_multiplier = 1.0
        events = self.seasonal_events.get(self.region, {})
        
        for event_name, event_data in events.items():
            start_week = event_data['start_week']
            end_week = start_week + event_data['duration']
            
            if start_week <= week_number < end_week:
                if category in event_data['categories']:
                    base_multiplier *= event_data['intensity']
        
        # Add weekly pattern (weekends have ~15% higher traffic)
        weekly_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * week_number / 52)
        
        return base_multiplier * weekly_pattern
    
    def visualize_seasonality_calendar(self, start_date: date, end_date: date, 
                                      categories: List[str]) -> pd.DataFrame:
        """Create visualization dataset for seasonality patterns"""
        calendar_data = []
        current_date = start_date
        week_number = 1
        
        while current_date <= end_date:
            for category in categories:
                multiplier = self.get_seasonality_multiplier(week_number, category)
                calendar_data.append({
                    'date': current_date,
                    'week_number': week_number,
                    'category': category,
                    'seasonality_multiplier': round(multiplier, 3)
                })
            
            current_date += timedelta(weeks=1)
            week_number += 1
        
        return pd.DataFrame(calendar_data)
