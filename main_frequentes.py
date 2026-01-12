import os
import json
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

#& Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-13s | %(message)s',
    datefmt='%d/%m'
)
logger = logging.getLogger(__name__)

#& load data

class DataLoader:
    def __init__(self):
        self.df_meteo = None
        self.df_events = None
        self.data_hebdo = None
        self.familles_data = None
        self.client_data = None
        
    def load_all_data(self):
        '''
        Load all data sources
        ---------------
        Input: None
        Output: Populates instance attributes
        '''
        self.df_meteo = self._charger_donnees_meteo()
        self.df_events = self._charger_donnees_events()
        self.data_hebdo = self._agregation_hebdomadaire_meteo_events()
        self.familles_data = self._charger_donnees_famille('data/families_w')
        
    def load_client_data(self, client_id):
        '''
        Load client frequent orders
        ---------------
        Input: client_id (str)
        Output: DataFrame with dates and quantities or None
        '''
        file_path = f'data/{client_id}/frequentes.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            self.client_data = df
            return df
        return None
        
    def _charger_donnees_meteo(self):
        '''
        Load and engineer weather features
        ---------------
        Input: None
        Output: DataFrame with daily weather metrics
        '''
        with open('data/weather.json', 'r') as f:
            weather_data = json.load(f)
        
        df_meteo = pd.DataFrame(weather_data['daily_data'])
        df_meteo['date'] = pd.to_datetime(df_meteo['date'])
        
        df_meteo['lunch_temp'] = df_meteo['lunch_period'].apply(lambda x: x['avg_temp_c'])
        df_meteo['lunch_precip'] = df_meteo['lunch_period'].apply(lambda x: x['total_precip_mm'])
        df_meteo['dinner_temp'] = df_meteo['dinner_period'].apply(lambda x: x['avg_temp_c'])
        df_meteo['dinner_precip'] = df_meteo['dinner_period'].apply(lambda x: x['total_precip_mm'])
        
        df_meteo['temp_range'] = df_meteo['max_temp_c'] - df_meteo['min_temp_c']
        df_meteo['is_hot_day'] = (df_meteo['avg_temp_c'] > 25).astype(int)
        df_meteo['is_cold_day'] = (df_meteo['avg_temp_c'] < 10).astype(int)
        df_meteo['high_precip'] = (df_meteo['lunch_precip'] + df_meteo['dinner_precip'] > 5).astype(int)
        df_meteo['perfect_weather'] = (
            (df_meteo['avg_temp_c'].between(18, 25)) & 
            (df_meteo['lunch_precip'] + df_meteo['dinner_precip'] < 1) & 
            (df_meteo['sun_hours'] >= 4)
        ).astype(int)
        
        return df_meteo

    def _charger_donnees_events(self):
        '''
        Load events with decay features
        ---------------
        Input: None
        Output: DataFrame with daily event impacts
        '''
        with open('data/events.json', 'r') as f:
            events_data = json.load(f)
        
        if isinstance(events_data, dict) and 'events' in events_data:
            events_list_raw = events_data['events']
        elif isinstance(events_data, list):
            events_list_raw = events_data
        else:
            return pd.DataFrame()
        
        events_list = []
        for event in events_list_raw:
            if not isinstance(event, dict):
                continue
                
            start_date = pd.to_datetime(event['date'])
            duration = event['duration_days']
            impact = event['impact']
            
            for i in range(duration):
                event_date = start_date + pd.Timedelta(days=i)
                events_list.append({
                    'date': event_date,
                    'event_name': event['name'],
                    'event_type': event['type'],
                    'event_impact': impact,
                    'event_day': i + 1,
                    'event_decay': impact * (1 - i / duration)
                })
        
        if not events_list:
            return pd.DataFrame()
        
        df_events = pd.DataFrame(events_list)
        
        df_events_agg = df_events.groupby('date').agg({
            'event_impact': 'sum',
            'event_decay': 'sum',
            'event_name': lambda x: ' + '.join(x),
            'event_type': lambda x: ' + '.join(x)
        }).reset_index()
        
        df_events_agg['has_lockdown'] = df_events_agg['event_type'].str.contains('lockdown').astype(int)
        df_events_agg['has_sport_event'] = df_events_agg['event_type'].str.contains('sport_event').astype(int)
        df_events_agg['has_cultural_event'] = df_events_agg['event_type'].str.contains('cultural_event').astype(int)
        df_events_agg['has_strike'] = df_events_agg['event_type'].str.contains('strike').astype(int)
        df_events_agg['has_reopening'] = df_events_agg['event_type'].str.contains('reopening').astype(int)
        
        return df_events_agg

    def _agregation_hebdomadaire_meteo_events(self):
        '''
        Aggregate daily data to weekly level
        ---------------
        Input: None
        Output: DataFrame with weekly features
        '''
        df_meteo = self.df_meteo.copy()
        df_events = self.df_events.copy()
        
        df_meteo['year'] = df_meteo['date'].dt.year
        df_meteo['week'] = df_meteo['date'].dt.isocalendar().week
        
        meteo_hebdo = df_meteo.groupby(['year', 'week']).agg({
            'avg_temp_c': 'mean',
            'max_temp_c': 'max',
            'min_temp_c': 'min',
            'temp_range': 'mean',
            'sun_hours': 'sum',
            'uv_index': 'mean',
            'lunch_temp': 'mean',
            'lunch_precip': 'sum',
            'dinner_temp': 'mean',
            'dinner_precip': 'sum',
            'is_rainy_day': 'sum',
            'is_windy_day': 'sum',
            'is_hot_day': 'sum',
            'is_cold_day': 'sum',
            'high_precip': 'sum',
            'perfect_weather': 'sum',
            'is_jour_ferie': 'sum',
            'is_vacances_zone_a': 'sum',
            'is_weekend': 'sum'
        }).reset_index()
        
        if df_events.empty or 'date' not in df_events.columns:
            event_columns = ['event_impact_mean', 'event_impact_max', 'event_impact_min', 
                            'event_decay_mean', 'has_lockdown', 'has_sport_event', 
                            'has_cultural_event', 'has_strike', 'has_reopening']
            for col in event_columns:
                meteo_hebdo[col] = 0
            data_hebdo = meteo_hebdo
        else:
            df_events['year'] = df_events['date'].dt.year
            df_events['week'] = df_events['date'].dt.isocalendar().week
            
            events_hebdo = df_events.groupby(['year', 'week']).agg({
                'event_impact': ['mean', 'max', 'min'],
                'event_decay': 'mean',
                'has_lockdown': 'max',
                'has_sport_event': 'max',
                'has_cultural_event': 'max',
                'has_strike': 'max',
                'has_reopening': 'max'
            }).reset_index()
            
            events_hebdo.columns = ['year', 'week', 'event_impact_mean', 'event_impact_max', 
                                   'event_impact_min', 'event_decay_mean', 'has_lockdown',
                                   'has_sport_event', 'has_cultural_event', 'has_strike', 'has_reopening']
            
            data_hebdo = meteo_hebdo.merge(events_hebdo, on=['year', 'week'], how='left')
            data_hebdo = data_hebdo.fillna(0)
        
        data_hebdo['vacation_intensity'] = data_hebdo['is_vacances_zone_a'] / 7
        data_hebdo['weather_quality'] = (
            data_hebdo['perfect_weather'] / 7 * 0.5 + 
            (7 - data_hebdo['is_rainy_day']) / 7 * 0.3 +
            data_hebdo['avg_temp_c'] / 30 * 0.2
        )
        
        return data_hebdo

    def _charger_donnees_famille(self, dossier_familles):
        '''
        Load all family-level data
        ---------------
        Input: dossier_familles (str)
        Output: dict mapping family names to DataFrames
        '''
        familles_data = {}
        
        for fichier in os.listdir(dossier_familles):
            if fichier.endswith('.csv'):
                famille_nom = fichier.replace('.csv', '')
                chemin_fichier = os.path.join(dossier_familles, fichier)
                df = pd.read_csv(chemin_fichier)
                df['date'] = pd.to_datetime(df['date'])
                familles_data[famille_nom] = df
                
        return familles_data

#& Model Part

class CoefficientModel:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def predict_global_and_coefficient(self, family_series, test_date):
        '''
        Predict global quantity and coefficient
        ---------------
        Input: family_series (pd.Series), test_date (datetime)
        Output: (global_pred, coefficient) tuple
        '''
        train = family_series[family_series.index < test_date]
        
        if len(train) < 52:
            rolling_52 = train.mean()
            return train.mean(), 1.0
        
        features = self._create_features(train)
        
        if len(features) < 10:
            rolling_52 = train.tail(52).mean()
            return train.mean(), 1.0
        
        X = features.drop('val', axis=1)
        y = features['val']
        
        if len(y[y == 0]) == 0:
            simple_model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
            )
            simple_model.fit(X, y, verbose=False)
            
            extended = pd.concat([train, pd.Series([np.nan], index=[test_date])])
            extended_features = self._create_features(extended)
            
            if len(extended_features) == 0:
                rolling_52 = train.tail(52).mean()
                return train.mean(), 1.0
            
            X_pred = extended_features.drop('val', axis=1).iloc[[-1]]
            global_pred = simple_model.predict(X_pred)[0]
            global_pred = max(0, global_pred)
        else:
            hurdle = EnhancedXGBoostHurdleModel()
            hurdle.fit(X, y, feature_names=X.columns.tolist())
            
            extended = pd.concat([train, pd.Series([np.nan], index=[test_date])])
            extended_features = self._create_features(extended)
            
            if len(extended_features) == 0:
                rolling_52 = train.tail(52).mean()
                return train.mean(), 1.0
            
            X_pred = extended_features.drop('val', axis=1).iloc[[-1]]
            global_pred = hurdle.predict(X_pred)[0]
            global_pred = max(0, global_pred)
        
        rolling_52 = train.tail(52).mean()
        coefficient = global_pred / rolling_52 if rolling_52 > 0 else 1.0
        
        return global_pred, coefficient
    
    def calculate_historical_coefficients_client(self, client_data, famille):
        '''
        Calculate historical coefficients for client-family
        ---------------
        Input: client_data (DataFrame), famille (str)
        Output: dict {week_number: {year: coefficient}}
        '''
        family_data = client_data[['date', famille]].copy()
        family_data = family_data.rename(columns={famille: 'quantity'})
        family_data['year'] = family_data['date'].dt.year
        family_data['week'] = family_data['date'].dt.isocalendar().week
        
        coefficients = {}
        
        for idx, (_, row) in enumerate(family_data.iterrows()):
            current_year = row['year']
            current_week = row['week']
            current_date = row['date']
            
            previous_data = family_data[family_data['date'] < current_date]
            
            if len(previous_data) >= 52:
                rolling_52 = previous_data['quantity'].tail(52).mean()
                
                if rolling_52 > 0:
                    coefficient = row['quantity'] / rolling_52
                    
                    week_key = current_week
                    if week_key not in coefficients:
                        coefficients[week_key] = {}
                    
                    coefficients[week_key][current_year] = coefficient
        
        return coefficients
    
    def predict_coefficient_for_2024_week(self, historical_coefficients, week_number):
        '''
        Predict 2024 coefficient from past years
        ---------------
        Input: historical_coefficients (dict), week_number (int)
        Output: coefficient (float)
        '''
        if week_number not in historical_coefficients:
            return 1.0
        
        coeffs_other_years = [
            coeff for year, coeff in historical_coefficients[week_number].items() 
            if year != 2024
        ]
        
        if len(coeffs_other_years) == 0:
            return 1.0
        
        return np.mean(coeffs_other_years)
    
    def generate_model1_coefficients(self):
        '''
        Generate Model 1 coefficients (global predictive)
        ---------------
        Input: None
        Output: DataFrame saved to CSV
        '''
        all_coefficients = []
        
        for family_name, family_data in self.data_loader.familles_data.items():
            family_series = family_data.groupby('date')[family_name].sum()
            
            test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
            
            for test_date in test_dates:
                if test_date.year == 2024:
                    global_pred, coefficient = self.predict_global_and_coefficient(family_series, test_date)
                    
                    all_coefficients.append({
                        'date': test_date,
                        'famille': family_name,
                        'coefficient': coefficient
                    })
        
        df_coefficients = pd.DataFrame(all_coefficients)
        df_coefficients.to_csv('data/families_w/coefficients.csv', index=False)
        
        return df_coefficients
    
    def generate_model3_coefficients(self):
        '''
        Generate Model 3 coefficients (historical average)
        ---------------
        Input: None
        Output: CSVs saved per client
        '''
        clients = [d for d in os.listdir('data') if d.isdigit()]
        
        for client_id in clients:
            client_data = self.data_loader.load_client_data(client_id)
            if client_data is None:
                continue
                
            families = [col for col in client_data.columns if col != 'date']
            all_coefficients = []
            
            for famille in families:
                historical_coefficients = self.calculate_historical_coefficients_client(client_data, famille)
                
                test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
                
                for test_date in test_dates:
                    if test_date.year == 2024:
                        week_number = test_date.isocalendar().week
                        coefficient = self.predict_coefficient_for_2024_week(historical_coefficients, week_number)
                        
                        all_coefficients.append({
                            'date': test_date,
                            'famille': famille,
                            'coefficient': coefficient
                        })
            
            if all_coefficients:
                df_coefficients = pd.DataFrame(all_coefficients)
                df_coefficients.to_csv(f'data/{client_id}/coef_families.csv', index=False)
    
    def _create_features(self, series):
        '''
        Create time series features
        ---------------
        Input: series (pd.Series)
        Output: DataFrame with lag, rolling, trend features
        '''
        df = pd.DataFrame({'val': series})
        df['week'] = series.index.isocalendar().week
        df['month'] = series.index.month
        df['quarter'] = series.index.quarter
        
        for lag in [1, 2, 3, 4]:
            df[f'lag{lag}'] = df['val'].shift(lag)
        
        for window in [4, 8]:
            df[f'avg{window}'] = df['val'].shift(1).rolling(window=window).mean()
            df[f'std{window}'] = df['val'].shift(1).rolling(window=window).std()
        
        df['diff'] = df['val'].shift(1) - df['val'].shift(2)
        df['accel'] = df['diff'] - (df['val'].shift(2) - df['val'].shift(3))
        df['ratio'] = df['lag1'] / (df['avg4'] + 1)
        df['trend'] = range(len(df))
        
        df = df.dropna()
        return df

class EnhancedXGBoostHurdleModel:
    def __init__(self):
        self.hurdle_model = LogisticRegression(random_state=42, max_iter=1000)
        self.count_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        self.scaler_hurdle = StandardScaler()
        self.scaler_count = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.is_hurdle_mode = True
        
    def fit(self, X, y, feature_names=None):
        '''
        Fit two-stage hurdle model
        ---------------
        Input: X (array), y (array), feature_names (list)
        Output: None (fits models)
        '''
        if len(X) == 0:
            return
        self.feature_names = feature_names
        y_binary = (y > 0).astype(int)
        
        if len(y_binary[y_binary == 0]) == 0:
            self.is_hurdle_mode = False
            self.count_model.fit(X, y, verbose=False)
            self.is_fitted = True
        else:
            self.is_hurdle_mode = True
            X_scaled_hurdle = self.scaler_hurdle.fit_transform(X)
            self.hurdle_model.fit(X_scaled_hurdle, y_binary)
            
            positive_mask = y > 0
            if positive_mask.sum() > 10:
                X_positive = X[positive_mask]
                y_positive = y[positive_mask]
                X_scaled_count = self.scaler_count.fit_transform(X_positive)
                self.count_model.fit(X_scaled_count, y_positive, verbose=False)
                self.is_fitted = True
    
    def predict(self, X):
        '''
        Predict using hurdle approach
        ---------------
        Input: X (array)
        Output: predictions (array)
        '''
        if len(X) == 0:
            return np.array([0])
            
        if not self.is_hurdle_mode:
            pred = self.count_model.predict(X)
            return np.maximum(pred, 0)
        else:
            X_scaled_hurdle = self.scaler_hurdle.transform(X)
            prob_positive = self.hurdle_model.predict_proba(X_scaled_hurdle)[:, 1]
            
            if self.is_fitted:
                X_scaled_count = self.scaler_count.transform(X)
                count_pred = self.count_model.predict(X_scaled_count)
                count_pred = np.maximum(count_pred, 0)
            else:
                count_pred = np.ones(len(X))
            
            return prob_positive * count_pred

class HurdleClientModel:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def create_comprehensive_features(self, df_client):
        '''
        Engineer comprehensive feature set
        ---------------
        Input: df_client (DataFrame)
        Output: (df_final, feature_columns) tuple
        '''
        df_client['year'] = df_client['date'].dt.year
        df_client['week'] = df_client['date'].dt.isocalendar().week
        df_client['month'] = df_client['date'].dt.month
        df_client['quarter'] = df_client['date'].dt.quarter
        
        df_client['week_sin'] = np.sin(2 * np.pi * df_client['week'] / 52)
        df_client['week_cos'] = np.cos(2 * np.pi * df_client['week'] / 52)
        df_client['month_sin'] = np.sin(2 * np.pi * df_client['month'] / 12)
        df_client['month_cos'] = np.cos(2 * np.pi * df_client['month'] / 12)
        
        df_client['is_spring'] = df_client['month'].isin([3, 4, 5]).astype(int)
        df_client['is_summer'] = df_client['month'].isin([6, 7, 8]).astype(int)
        df_client['is_autumn'] = df_client['month'].isin([9, 10, 11]).astype(int)
        df_client['is_winter'] = df_client['month'].isin([12, 1, 2]).astype(int)
        df_client['is_peak_summer'] = df_client['month'].isin([7, 8]).astype(int)
        df_client['is_holiday_season'] = df_client['month'].isin([7, 8, 12]).astype(int)
        
        df_complet = df_client.merge(self.data_loader.data_hebdo, on=['year', 'week'], how='left')
        df_complet = df_complet.fillna(0)
        
        families = [col for col in df_complet.columns if col not in ['date', 'year', 'week', 'month', 'quarter', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 'is_spring', 'is_summer', 'is_autumn', 'is_winter', 'is_peak_summer', 'is_holiday_season'] and col not in self.data_loader.data_hebdo.columns]
        
        all_features = []
        
        for family in families:
            family_data = df_complet[['date', 'year', 'week', 'month', 'quarter', 'week_sin', 'week_cos', 'month_sin', 'month_cos', 'is_spring', 'is_summer', 'is_autumn', 'is_winter', 'is_peak_summer', 'is_holiday_season', family] + [col for col in self.data_loader.data_hebdo.columns if col not in ['year', 'week']]].copy()
            family_data = family_data.rename(columns={family: 'quantity'})
            family_data['famille'] = family
            family_data = family_data.sort_values(['year', 'week'])
            
            for lag in [1, 2, 4, 8, 12]:
                family_data[f'lag_{lag}'] = family_data['quantity'].shift(lag)
            
            for window in [4, 8, 12, 26]:
                family_data[f'ma_{window}'] = family_data['quantity'].shift(1).rolling(window).mean()
                family_data[f'std_{window}'] = family_data['quantity'].shift(1).rolling(window).std()
            
            family_data['trend'] = range(len(family_data))
            family_data['volatility_4w'] = family_data['quantity'].shift(1).rolling(4).std()
            family_data['max_4w'] = family_data['quantity'].shift(1).rolling(4).max()
            family_data['min_4w'] = family_data['quantity'].shift(1).rolling(4).min()
            
            family_data['had_order_last_week'] = (family_data['lag_1'] > 0).astype(int)
            family_data['client_avg_quantity'] = family_data['quantity'].expanding().mean().shift(1)
            family_data['client_total_orders'] = (family_data['quantity'] > 0).expanding().sum().shift(1)
            family_data['client_order_frequency'] = family_data['client_total_orders'] / (family_data['trend'] + 1)
            
            for season in ['summer', 'winter']:
                mask = family_data[f'is_{season}'] == 1
                family_data[f'client_{season}_avg'] = family_data.loc[mask, 'quantity'].expanding().mean().shift(1)
            
            all_features.append(family_data)
        
        df_final = pd.concat(all_features, ignore_index=True)
        df_final = df_final.fillna(0)
        
        feature_columns = [
            'trend', 'week_sin', 'week_cos', 'month_sin', 'month_cos',
            'week', 'month', 'quarter',
            'is_spring', 'is_summer', 'is_autumn', 'is_winter',
            'is_peak_summer', 'is_holiday_season',
            'lag_1', 'lag_2', 'lag_4', 'lag_8', 'lag_12',
            'ma_4', 'ma_8', 'ma_12', 'ma_26',
            'std_4', 'std_8', 'std_12', 'std_26',
            'volatility_4w', 'max_4w', 'min_4w',
            'had_order_last_week', 'client_avg_quantity', 'client_total_orders',
            'client_order_frequency', 'client_summer_avg', 'client_winter_avg',
            'avg_temp_c', 'temp_range', 'sun_hours', 'lunch_temp', 'lunch_precip',
            'dinner_temp', 'dinner_precip', 'is_rainy_day', 'is_hot_day', 'is_cold_day',
            'high_precip', 'perfect_weather', 'weather_quality',
            'is_jour_ferie', 'vacation_intensity', 'is_weekend',
            'event_impact_mean', 'event_impact_max', 'event_impact_min', 'event_decay_mean',
            'has_lockdown', 'has_sport_event', 'has_cultural_event', 'has_strike', 'has_reopening'
        ]
        
        for col in feature_columns:
            if col not in df_final.columns:
                df_final[col] = 0
        
        return df_final, feature_columns
    
    def predict_client_families(self, client_id):
        '''
        Generate Model 2 predictions (direct hurdle)
        ---------------
        Input: client_id (str)
        Output: DataFrame saved to CSV
        '''
        client_data = self.data_loader.load_client_data(client_id)
        if client_data is None:
            return None
            
        df_complet, feature_columns = self.create_comprehensive_features(client_data)
        
        families = df_complet['famille'].unique()
        all_predictions = []
        
        for famille in families:
            family_data = df_complet[df_complet['famille'] == famille].sort_values(['year', 'week'])
            test_family = family_data[family_data['year'] == 2024]
            
            if len(test_family) == 0:
                continue
            
            for idx, (_, row) in enumerate(test_family.iterrows()):
                current_week = row['week']
                current_year = row['year']
                
                train_data = df_complet[
                    (df_complet['year'] < current_year) | 
                    ((df_complet['year'] == current_year) & (df_complet['week'] < current_week))
                ]
                
                if len(train_data) < 100:
                    continue
                    
                X_train = train_data[feature_columns]
                y_train = train_data['quantity']
                
                model = EnhancedXGBoostHurdleModel()
                model.fit(X_train.values, y_train.values, feature_names=feature_columns)
                
                X_pred = row[feature_columns].values.reshape(1, -1)
                prediction = model.predict(X_pred)[0]
                
                all_predictions.append({
                    'date': f"{current_year}-W{current_week:02d}",
                    'famille': famille,
                    'prediction': max(0, prediction)
                })
        
        df_predictions = pd.DataFrame(all_predictions)
        output_path = f'data/{client_id}/predictions.csv'
        df_predictions.to_csv(output_path, index=False)
        
        return df_predictions
    

#& Results Part


class ModelComparison:
    def __init__(self):
        self.model1_coefficients = None
        self.model3_coefficients = {}
        self.valid_families = [
            'ALCOOLS', 'APERITIF SANS ALCOOL', 'ARTE DE LA TABLE', 'BIERE BOITE', 
            'BIERE BOUTEILLE', 'BIERE FUT', 'BOUTEILLE GAZ', 'CAFE', 'CAISSE CONSIGNEE', 
            'CHAMPAGNE', 'CIDRE', 'COFFRET', 'EAUX', 'EMBALLAGE', 'EPICES', 'EQUIPEMENT', 
            'FAMILLE_NON_DEFINIE', 'FOOD', 'GLACES', 'HYGIENE', 'JUS DE FRUIT', 'KIT', 
            'LAIT', 'LOCATION', 'N/Q', 'PROSECCO', 'PUB', 'SAN BITTER', 'SIROP', 
            'SODA & LIMO', 'TABASCO', 'THE', 'VINS'
        ]
        
    def load_coefficients(self):
        '''
        Load Model 1 coefficients
        ---------------
        Input: None
        Output: Populates model1_coefficients
        '''
        if not os.path.exists('data/families_w/coefficients.csv'):
            return
        self.model1_coefficients = pd.read_csv('data/families_w/coefficients.csv')
        self.model1_coefficients['date'] = pd.to_datetime(self.model1_coefficients['date'])
        
    def load_client_coefficients(self, client_id):
        '''
        Load Model 3 client coefficients
        ---------------
        Input: client_id (str)
        Output: DataFrame or None
        '''
        file_path = f'data/{client_id}/coef_families.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            self.model3_coefficients[client_id] = df
            return df
        return None
        
    def load_client_data(self, client_id):
        '''
        Load client order history
        ---------------
        Input: client_id (str)
        Output: DataFrame or None
        '''
        file_path = f'data/{client_id}/frequentes.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
        
    def load_client_predictions(self, client_id):
        '''
        Load Model 2 predictions
        ---------------
        Input: client_id (str)
        Output: DataFrame or None
        '''
        file_path = f'data/{client_id}/predictions.csv'
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) == 0:
                return None
            df['date'] = df['date'].astype(str)
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return None
    
    def calculate_rolling_predictions(self, client_data, famille, coefficients, model_name):
        '''
        Apply coefficients to rolling mean
        ---------------
        Input: client_data (DataFrame), famille (str), coefficients (DataFrame), model_name (str)
        Output: DataFrame with predictions
        '''
        family_data = client_data[['date', famille]].copy()
        family_data = family_data.rename(columns={famille: 'quantity'})
        family_data = family_data[family_data['date'] < '2024-01-01']
        
        predictions = []
        test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
        
        for test_date in test_dates:
            coeff_row = coefficients[
                (coefficients['date'] == test_date) & 
                (coefficients['famille'] == famille)
            ]
            
            if len(coeff_row) == 0:
                coefficient = 1.0
            else:
                coefficient = coeff_row['coefficient'].iloc[0]
            
            historical_data = family_data[family_data['date'] < test_date]
            
            if len(historical_data) >= 52:
                rolling_mean = historical_data['quantity'].tail(52).mean()
            elif len(historical_data) > 0:
                rolling_mean = historical_data['quantity'].mean()
            else:
                rolling_mean = 0
            
            prediction = coefficient * rolling_mean
            predictions.append({
                'date': test_date,
                'prediction': max(0, prediction),
                'model': model_name
            })
        
        return pd.DataFrame(predictions)
    
    def get_real_values(self, client_data, famille):
        '''
        Extract 2024 actual values
        ---------------
        Input: client_data (DataFrame), famille (str)
        Output: DataFrame with real values
        '''
        family_data = client_data[['date', famille]].copy()
        family_data = family_data.rename(columns={famille: 'quantity'})
        family_data_2024 = family_data[family_data['date'].dt.year == 2024]
        
        real_values = []
        test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
        
        for test_date in test_dates:
            week_start = test_date
            week_end = test_date + pd.Timedelta(days=6)
            
            week_data = family_data_2024[
                (family_data_2024['date'] >= week_start) & 
                (family_data_2024['date'] <= week_end)
            ]
            
            if len(week_data) > 0:
                real_value = week_data['quantity'].iloc[0]
            else:
                real_value = 0
                
            real_values.append({
                'date': test_date,
                'real': real_value
            })
        
        return pd.DataFrame(real_values)
    
    def plot_available_models(self, results, client_id, nom_famille, available_models):
        '''
        Plot comparison of models vs actual
        ---------------
        Input: results (dict), client_id (str), nom_famille (str), available_models (list)
        Output: filename of saved plot
        '''
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        x = range(len(results['dates']))
        
        ax.plot(x, results['real'], 'o-', label='Réel', linewidth=3, markersize=8, color='black')
        
        if 'model1' in available_models and 'model1_pred' in results:
            ax.plot(x, results['model1_pred'], 's-', label='Modèle 1 (Hurdle Prédictif)', 
                    linewidth=2, color='red', alpha=0.8)
            
        if 'model2' in available_models and 'model2_pred' in results:
            ax.plot(x, results['model2_pred'], '^-', label='Modèle 2 (Hurdle Direct)', 
                    linewidth=2, color='blue', alpha=0.8)
            
        if 'model3' in available_models and 'model3_pred' in results:
            ax.plot(x, results['model3_pred'], 'd-', label='Modèle 3 (Coefficient Historique)', 
                    linewidth=2, color='green', alpha=0.8)
        
        models_text = " + ".join([f"M{i[-1]}" for i in available_models])
        ax.set_title(f'Comparaison ({models_text}) - Client {client_id} - {nom_famille}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Semaines 2024', fontsize=12)
        ax.set_ylabel('Quantité', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        step = max(1, len(x) // 10)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([results['dates'][i] for i in range(0, len(results['dates']), step)], 
                          rotation=45)
        
        plt.tight_layout()
        
        output_dir = f'data/{client_id}/frequente_prediction'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{output_dir}/comparison_{client_id}_{nom_famille.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    

    def generate_comparison_for_client(self, client_id):
        '''
        Generate comparison plots for client
        ---------------
        Input: client_id (str)
        Output: None (saves plots)
        '''
        client_data = self.load_client_data(client_id)
        client_predictions = self.load_client_predictions(client_id)
        client_coefficients = self.load_client_coefficients(client_id)
        
        if client_data is None:
            return None
        
        families = [col for col in client_data.columns if col in self.valid_families]
        results_summary = []
        
        for famille in families:
            available_models = []
            results = {
                'dates': [],
                'real': []
            }
            
            real_values = self.get_real_values(client_data, famille)
            test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
            
            for test_date in test_dates:
                real_row = real_values[real_values['date'] == test_date]
                if len(real_row) > 0:
                    results['dates'].append(test_date.strftime('%Y-W%W'))
                    results['real'].append(real_row['real'].iloc[0])
            
            if len(results['dates']) == 0:
                continue
            
            model1_pred = None
            if self.model1_coefficients is not None:
                model1_pred = self.calculate_rolling_predictions(
                    client_data, famille, self.model1_coefficients, 'model1'
                )
                if len(model1_pred) > 0:
                    available_models.append('model1')
                    results['model1_pred'] = []
                    for test_date in pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON'):
                        m1_row = model1_pred[model1_pred['date'] == test_date]
                        if len(m1_row) > 0:
                            results['model1_pred'].append(m1_row['prediction'].iloc[0])
                        else:
                            results['model1_pred'].append(0)
            
            if client_predictions is not None:
                model2_pred = client_predictions[client_predictions['famille'] == famille].copy()
                if len(model2_pred) > 0:
                    available_models.append('model2')
                    results['model2_pred'] = []
                    for test_date in pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON'):
                        m2_row = model2_pred[model2_pred['date'] == test_date.strftime('%Y-W%W')]
                        if len(m2_row) > 0:
                            results['model2_pred'].append(m2_row['prediction'].iloc[0])
                        else:
                            results['model2_pred'].append(0)
            
            if client_coefficients is not None:
                model3_pred = self.calculate_rolling_predictions(
                    client_data, famille, client_coefficients, 'model3'
                )
                if len(model3_pred) > 0:
                    available_models.append('model3')
                    results['model3_pred'] = []
                    for test_date in pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON'):
                        m3_row = model3_pred[model3_pred['date'] == test_date]
                        if len(m3_row) > 0:
                            results['model3_pred'].append(m3_row['prediction'].iloc[0])
                        else:
                            results['model3_pred'].append(0)
            
            if len(available_models) > 0:
                plot_filename = self.plot_available_models(results, client_id, famille, available_models)
                
    
    def run_comparison_for_all_clients(self):
        '''
        Generate plots for all clients
        ---------------
        Input: None
        Output: None (saves plots)
        '''
        self.load_coefficients()
        
        clients = [d for d in os.listdir('data') if d.isdigit()]
        
        for client_id in clients:
            self.generate_comparison_for_client(client_id)


def main():
    '''
    Main execution
    ---------------
    Input: None
    Output: None (runs full pipeline)
    '''
    data_loader = DataLoader()
    data_loader.load_all_data()
    logger.info('data loaded')
    
    coeff_model = CoefficientModel(data_loader)
    hurdle_model = HurdleClientModel(data_loader)
    
    coeff_model.generate_model1_coefficients()
    coeff_model.generate_model3_coefficients()
    
    clients = [d for d in os.listdir('data') if d.isdigit()]
    logger.info('coefficient models done')
    
    for client_id in tqdm(clients):
        predictions_file = f'data/{client_id}/predictions.csv'
        if os.path.exists(predictions_file):
            continue
        hurdle_model.predict_client_families(client_id)
    
    logger.info('all done')

    comparison = ModelComparison()
    comparison.run_comparison_for_all_clients()

if __name__ == "__main__":
    main()