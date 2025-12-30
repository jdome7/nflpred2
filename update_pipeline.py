import os
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
from sqlalchemy import create_engine, text
from data_factory import get_updated_data
from dotenv import load_dotenv

load_dotenv()


DB_URL = os.getenv('DATABASE_URL') 
engine = create_engine(DB_URL)

trained_models = joblib.load('qb_trained_models.joblib')
target_feature_map = joblib.load('qb_target_feature_map.joblib')
targets = list(trained_models.keys())


df = get_updated_data()

df = df[df['season'] == 2025]

next_week = df[df['passing_yards'].isna()]['week'].min()

print(f"Targeting Week {next_week} for new projections...")

week_proj = df[df['week'] == next_week].copy()

if week_proj.empty:
    print("No data found for the upcoming week. Exiting.")
    exit()

all_importances = []

#generate predictions
for target in targets:
    m = trained_models[target]
    selected_cols = target_feature_map[target]['top_30']
    top_features_5 = target_feature_map[target]['top_5']
    
    raw_preds = m.predict(week_proj[selected_cols])

    
    if 'count:poisson' in str(m.get_params().get('objective', '')):
        week_proj[f'{target}_pred_med'] = raw_preds
        week_proj[f'{target}_pred_low'] = poisson.ppf(0.25, raw_preds)
        week_proj[f'{target}_pred_high'] = poisson.ppf(0.75, raw_preds)
    else:
        
        sorted_preds = np.sort(raw_preds, axis=1) if raw_preds.ndim > 1 else np.tile(raw_preds, (3, 1)).T
        week_proj[f'{target}_pred_low'] = sorted_preds[:, 0]
        week_proj[f'{target}_pred_med'] = sorted_preds[:, 1]
        week_proj[f'{target}_pred_high'] = sorted_preds[:, 2]

    week_proj[f'{target}_actual'] = np.nan

    # Store feature values and importances
    for feat_name, gain_val in top_features_5:
        for _, p_row in week_proj.iterrows():
            all_importances.append({
                'week': next_week,
                'player_id': p_row['player_id'],
                'target': target,
                'feature_name': feat_name,
                'importance_gain': gain_val,
                'player_feature_value': p_row[feat_name]
            })


target_cols = []
for t in targets:
    target_cols += [f'{t}_pred_low', f'{t}_pred_med', f'{t}_pred_high', f'{t}_actual']

essential_cols = ['player_id', 'player_display_name', 'opponent', 'season', 'week']
final_projections = week_proj[essential_cols + target_cols]
final_importances = pd.DataFrame(all_importances)


with engine.begin() as conn:
    #prevent duplicates
    conn.execute(text("DELETE FROM weekly_projections2 WHERE week = :w"), {"w": next_week})
    conn.execute(text("DELETE FROM feature_importances2 WHERE week = :w"), {"w": next_week})
    conn.execute(text("TRUNCATE TABLE qb_features_raw"))

    #append new data
    df.to_sql('qb_features_raw', conn, if_exists='append', index=False)
    final_projections.to_sql('weekly_projections2', conn, if_exists='append', index=False)
    final_importances.to_sql('feature_importances2', conn, if_exists='append', index=False)
