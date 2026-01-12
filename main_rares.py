import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class CrostonModel:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.demand_forecast = None
        self.interval_forecast = None
        
    def fit(self, series):
        non_zero_demands = []
        intervals = []
        
        last_demand_index = -1
        
        for i, value in enumerate(series):
            if value > 0:
                non_zero_demands.append(value)
                
                if last_demand_index >= 0:
                    intervals.append(i - last_demand_index)
                
                last_demand_index = i
        
        if len(non_zero_demands) == 0:
            self.demand_forecast = 0
            self.interval_forecast = len(series)
            return
        
        if len(intervals) == 0:
            self.demand_forecast = np.mean(non_zero_demands)
            self.interval_forecast = len(series)
            return
        
        demand_forecast = non_zero_demands[0]
        interval_forecast = intervals[0] if len(intervals) > 0 else 1
        
        for i in range(1, len(non_zero_demands)):
            demand_forecast = self.alpha * non_zero_demands[i] + (1 - self.alpha) * demand_forecast
        
        for i in range(1, len(intervals)):
            interval_forecast = self.alpha * intervals[i] + (1 - self.alpha) * interval_forecast
        
        self.demand_forecast = demand_forecast
        self.interval_forecast = max(interval_forecast, 1)
    
    def predict(self):
        if self.demand_forecast is None or self.interval_forecast is None:
            return 0
        return self.demand_forecast / self.interval_forecast

class RareFamiliesModel:
    def __init__(self):
        pass
    
    def load_client_rares_data(self, client_id):
        file_path = f'data/{client_id}/rares.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def predict_same_week_last_year(self, client_data, famille, test_date):
        family_data = client_data[['date', famille]].copy()
        family_data = family_data.rename(columns={famille: 'quantity'})
        
        last_year_date = test_date - pd.Timedelta(days=365)
        week_start = last_year_date - pd.Timedelta(days=3)
        week_end = last_year_date + pd.Timedelta(days=3)
        
        last_year_data = family_data[
            (family_data['date'] >= week_start) & 
            (family_data['date'] <= week_end)
        ]
        
        if len(last_year_data) > 0:
            return last_year_data['quantity'].iloc[0]
        else:
            historical_data = family_data[family_data['date'] < test_date]
            if len(historical_data) == 0:
                return 0
            
            orders = historical_data[historical_data['quantity'] > 0]
            if len(orders) == 0:
                return 0
            
            frequency = len(orders) / len(historical_data)
            avg_quantity = orders['quantity'].mean()
            
            if np.random.random() < frequency:
                return avg_quantity
            else:
                return 0
    
    def predict_croston(self, client_data, famille, test_date):
        family_data = client_data[['date', famille]].copy()
        family_data = family_data.rename(columns={famille: 'quantity'})
        
        historical_data = family_data[family_data['date'] < test_date]
        
        if len(historical_data) < 4:
            return 0
        
        croston = CrostonModel(alpha=0.2)
        croston.fit(historical_data['quantity'].values)
        
        return croston.predict()
    
    def predict_rare_families_for_client(self, client_id):
        client_data = self.load_client_rares_data(client_id)
        if client_data is None:
            return None
        
        families = [col for col in client_data.columns if col != 'date']
        all_predictions = []
        
        test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
        
        for famille in families:
            for test_date in test_dates:
                same_week_pred = self.predict_same_week_last_year(client_data, famille, test_date)
                croston_pred = self.predict_croston(client_data, famille, test_date)
                
                all_predictions.append({
                    'date': test_date.strftime('%Y-W%W'),
                    'famille': famille,
                    'same_week_pred': max(0, same_week_pred),
                    'croston_pred': max(0, croston_pred)
                })
        
        df_predictions = pd.DataFrame(all_predictions)
        output_path = f'data/{client_id}/predictions_rares.csv'
        df_predictions.to_csv(output_path, index=False)
        
        return df_predictions
    
    def run_for_all_clients(self):
        clients = [d for d in os.listdir('data') if d.isdigit()]
        
        for client_id in tqdm(clients):
            self.predict_rare_families_for_client(client_id)

class RareModelComparison:
    def __init__(self):
        pass
    
    def load_client_rares_data(self, client_id):
        file_path = f'data/{client_id}/rares.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def load_client_rare_predictions(self, client_id):
        file_path = f'data/{client_id}/predictions_rares.csv'
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) == 0:
                return None
            return df
        except:
            return None
    
    def get_real_values_rare(self, client_data, famille):
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
            
            real_value = week_data['quantity'].iloc[0] if len(week_data) > 0 else 0
            
            real_values.append({
                'date': test_date.strftime('%Y-W%W'),
                'real': real_value
            })
        
        return pd.DataFrame(real_values)
    

    def clean_filename(self, nom_famille):
        return nom_famille.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")

    def plot_rare_models_comparison(self, results, client_id, nom_famille):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        x = range(len(results['dates']))
        
        ax.plot(x, results['real'], 'o-', label='Réel', linewidth=3, markersize=8, color='black')
        
        if len(results.get('same_week_pred', [])) > 0:
            ax.plot(x, results['same_week_pred'], 's-', label='Même Semaine Année Passée', 
                    linewidth=2, color='orange', alpha=0.8)
            
        if len(results.get('croston_pred', [])) > 0:
            ax.plot(x, results['croston_pred'], '^-', label='Croston', 
                    linewidth=2, color='purple', alpha=0.8)
        
        ax.set_title(f'Familles Rares - Client {client_id} - {nom_famille}', 
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
        
        output_dir = f'data/{client_id}/rares_prediction'
        os.makedirs(output_dir, exist_ok=True)
        
        clean_name = self.clean_filename(nom_famille)
        filename = f'{output_dir}/rare_comparison_{client_id}_{clean_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    def generate_comparison_for_client(self, client_id):
        client_data = self.load_client_rares_data(client_id)
        client_predictions = self.load_client_rare_predictions(client_id)
        
        if client_data is None or client_predictions is None:
            return None
        
        families = [col for col in client_data.columns if col != 'date']
        results_summary = []
        
        for famille in families:
            family_predictions = client_predictions[client_predictions['famille'] == famille]
            if len(family_predictions) == 0:
                continue
            
            real_values = self.get_real_values_rare(client_data, famille)
            
            results = {
                'dates': [],
                'real': [],
                'same_week_pred': [],
                'croston_pred': []
            }
            
            test_dates = pd.date_range(start='2024-01-01', end='2024-12-30', freq='W-MON')
            
            for test_date in test_dates:
                date_str = test_date.strftime('%Y-W%W')
                
                real_row = real_values[real_values['date'] == date_str]
                pred_row = family_predictions[family_predictions['date'] == date_str]
                
                if len(real_row) == 0 or len(pred_row) == 0:
                    continue
                
                results['dates'].append(date_str)
                results['real'].append(real_row['real'].iloc[0])
                results['same_week_pred'].append(pred_row['same_week_pred'].iloc[0])
                results['croston_pred'].append(pred_row['croston_pred'].iloc[0])
            
            if len(results['dates']) > 0:
                plot_filename = self.plot_rare_models_comparison(results, client_id, famille)
                
                result = {
                    'client': client_id,
                    'famille': famille,
                    'plot_file': plot_filename
                }
                
                results_summary.append(result)
        
        return results_summary
    
    def run_comparison_for_all_clients(self):
        clients = [d for d in os.listdir('data') if d.isdigit()]
        all_results = []
        
        for client_id in clients:
            client_results = self.generate_comparison_for_client(client_id)
            if client_results:
                all_results.extend(client_results)
        

def main():
    rare_model = RareFamiliesModel()
    rare_model.run_for_all_clients()
    
    comparison = RareModelComparison()
    comparison.run_comparison_for_all_clients()

if __name__ == "__main__":
    main()