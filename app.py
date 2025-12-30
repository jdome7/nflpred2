import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from sqlalchemy import create_engine
from scipy.stats import poisson


st.set_page_config(layout="wide", page_title="NFL QB Predictor", page_icon="🏈")


db_url = st.secrets["postgres"]["url"] if "postgres" in st.secrets else os.getenv('DB_URL')
engine = create_engine(db_url)

def calculate_fantasy_points(vals):
    score = (
        (vals['passing_yards'] * 0.04) + 
        (vals['passing_tds'] * 4.0) + 
        (vals['passing_interceptions'] * -2.0) + 
        (vals['rushing_yards'] * 0.1) + 
        (vals['rushing_tds'] * 6.0) + 
        (vals['fumbles_lost'] * -2.0)
    )
    return score


@st.cache_data()
def get_all_players():
    with engine.connect() as conn:
        players = pd.read_sql("SELECT DISTINCT player_id, player_display_name FROM player_bios ORDER BY player_display_name", conn)
    return players

def get_available_weeks(player_id):
    with engine.connect() as conn:
        query = f"SELECT DISTINCT week FROM weekly_projections2 WHERE player_id = '{player_id}' ORDER BY week desc"
        weeks = pd.read_sql(query, conn)
    return weeks['week'].tolist()

st.sidebar.header("Navigation")
players_df = get_all_players()
selected_player_name = st.sidebar.selectbox("Select Player", players_df['player_display_name'])
selected_player_id = players_df[players_df['player_display_name'] == selected_player_name]['player_id'].values[0]

available_weeks = get_available_weeks(selected_player_id)

if not available_weeks:
    st.sidebar.error("No available weeks for the selected player.")
else:
    selected_week = st.sidebar.select_slider("Select Week", options=available_weeks, value=available_weeks[0])


@st.cache_data()
def load_data(player_id, week):
    with engine.connect() as conn:
        bio = pd.read_sql(f"SELECT * FROM player_bios WHERE player_id = '{player_id}'", conn)
        proj = pd.read_sql(f"SELECT * FROM weekly_projections2 WHERE player_id = '{player_id}' AND week = {week}", conn)
        imp = pd.read_sql(f"SELECT * FROM feature_importances2 WHERE player_id = '{player_id}' AND week = {week}", conn)
        trend = pd.read_sql(f"SELECT week, passing_yards_cum_actual, passing_yards_cum_pred, rushing_yards_cum_actual, rushing_yards_cum_pred, passing_tds_cum_actual, passing_tds_cum_pred, rushing_tds_cum_pred, rushing_tds_cum_actual, passing_interceptions_cum_pred, passing_interceptions_cum_actual, fumbles_lost_cum_actual, fumbles_lost_cum_pred FROM cumulative2 WHERE player_id = '{player_id}' ORDER BY week", conn)
    return bio.iloc[0], proj.iloc[0], imp, trend


opponent = pd.read_sql(f"SELECT opponent FROM weekly_projections2 WHERE player_id = '{selected_player_id}' AND week = {selected_week}", engine).iloc[0]['opponent']

bio, proj, importance, trend = load_data(selected_player_id, selected_week)


col_photo, col_info = st.columns([1, 4])

with col_photo:
    if bio.get('headshot_url'):
        st.image(bio['headshot_url'], width='stretch')
    else:
        # Generic placeholder if URL is missing
        st.image("https://via.placeholder.com/150", witdth='stretch')

with col_info:
    st.title(f"{selected_player_name} | Week {selected_week} vs. {opponent}")
    
    # player info
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Position", f"{bio['position']}")
    m2.metric("College", bio['college_name'])
    m3.metric("Draft", f"Rd {int(bio['draft_round'])} (No. {int(bio['draft_pick'])})")
    m4.metric("Height/Weight", f"{bio['height']}\" / {bio['weight']} lbs")


st.divider()

st.subheader("Weekly Stat Projections")

score_placeholder = st.empty()

def make_gauge(target_name, low, med, high):
    options = {
        f"Low ({low})": low,
        f"Med ({med})": med,
        f"High ({high})": high if high > 0 else 1
    }

    selected_label = st.select_slider(
        label=f"Predicted {target_name.replace('_', ' ').title()}",
        options=list(options.keys()),
        value = list(options.keys())[1],
        key=f"slider_{target_name}",
        help=f"Toggle between the 25th, 50th, and 75th percentile projections for {target_name.replace('_', ' ').title()}."
    )

    current_value = options[selected_label]
   

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_value,
        gauge = {
            'axis': {'range': [0, high*1.2]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, low], 
                'color':  "#eeeeee"},
                {'range': [low, high], 'color': "#ababab"},
                {'range': [high, high*1.2], 'color': "#eeeeee"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': current_value}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width='stretch')
    return current_value

targets = ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds', 'fumbles_lost']
rows = [st.columns(3), st.columns(3)]

user_inputs = {}
targets = ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards', 'rushing_tds', 'fumbles_lost']
rows = [st.columns(3), st.columns(3)]

for i, t in enumerate(targets):
    with rows[i // 3][i % 3]:
        if(proj[f'{t}_pred_high'] < 1):
            proj[f'{t}_pred_high'] = 1  
        val = make_gauge(t, proj[f'{t}_pred_low'], proj[f'{t}_pred_med'], proj[f'{t}_pred_high'])
        user_inputs[t] = val


total_score = calculate_fantasy_points(user_inputs)


score_placeholder.metric(
    label=f"Projected Fantasy Score", 
    value=f"{total_score:.2f} pts",
)


st.divider()
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("Season Trajectory (Predicted vs. Actual)")
    
    trend_options = {
        'Passing Yards': 'passing_yards',
        'Passing TDs': 'passing_tds',
        'Rushing Yards': 'rushing_yards',
        'Rushing TDs': 'rushing_tds',
        'Interceptions': 'passing_interceptions',
        'Fumbles Lost': 'fumbles_lost'
    }
    
    selected_display = st.selectbox("Select Stat to View:", options=list(trend_options.keys()), key="trend_stat")
    stat_key = trend_options[selected_display]

    # Create the column names
    actual_col = f'{stat_key}_cum_actual'
    pred_col = f'{stat_key}_cum_pred'
    
    if actual_col in trend.columns and pred_col in trend.columns:
        fig_trend = px.line(
            trend, 
            x='week', 
            y=[actual_col, pred_col], #list of column names
            labels={'value': selected_display, 'variable': 'Type', 'week': 'Week'},
            title=f"Cumulative {selected_display} Trend",
            markers=True
        )

        fig_trend.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_trend, width='stretch')
    else:
        st.error(f"Data for {selected_display} not found in database. Expected columns: {actual_col}, {pred_col}")

with col_right:
    st.subheader("Model Explainability")
    target_exp = st.selectbox("Explain Target:", targets)
    feat_data = importance[importance['target'] == target_exp].sort_values('importance_gain', ascending=True)
    fig_imp = px.bar(feat_data, x='importance_gain', y='feature_name', orientation='h', 
                     hover_data=['player_feature_value'], title=f"Top Drivers: {target_exp}")
    st.plotly_chart(fig_imp, width='stretch')


st.divider()
st.header("About This App")
st.write("Made Josiah Domercant. Data sourced from nflverse. These predictions use an xgboost model trained on data from all weekly QB and team stats in games played between 2016 and 2025, factoring in basic stats, advanced stats, rolling averages, and more. As player performances are highly variable from week to week, these models opt for quantile regression and poisson regression, rather than simple linear regression to provide a reliable range of likely outcomes. There are six models for passing yards, passing touchdowns, rushing yards, rushing touchdowns, interceptions, and fumbles lost. Each begins with 250+ features in their first pass, then are pruned to use only the most important drivers of each respective stat. Predictions are made for the 25th, 50th, and 75th percentiles to give a range of possible outcomes rather than a single point estimate. This can be interpreted as the middle 50% range of outcomes the model predicts the player is likely to fall into; between a somewhat pessimistic and somewhat optimistic prediction. Stats with smaller and less variable distributions that tend to be either 0, 1, 2, 3.. (like interceptions and fumbles) utilize poisson regression to provide a sensible range. This app is built using Streamlit for an interactive web interface, and Plotly for the dynamic visualizations. All data and model results are stored in a cloud database for efficient querying and retrieval. This app automatically updates every Tuesday at 9:00 AM UTC throughout the regular season with up-to-date data and predictions for the upcoming week.")
st.header("Model Evaluation")
st.write("The models were evaluated using Mean Absolute Error (MAE) and skill scores on testing data from previous seasons. Overall, the models median predictions performed slightly better in most stats than the naive estimate (player's current season average), and most weekly performances are captured in the range shown. The important features for each model can be observed in the 'Model Explainability' section of the app, and many of the key drivers for each stat are very intuitive. For simplicity and proof of concept, I only provided predictions for 10 NFL QBs with a large sample of games from this season, and of varying skill levels and playstyles, to display how these models perform, both over time and across different inputs. They tend to perform very well for the average statistical quarterback (ex. Caleb Williams), but tend to underperform for players putting up very strong seasons (ex. Matthew Stafford has 10+ more passing TDs on the season than any other QB, with multiple 3+ passing TD games, yet the model will rarely predict over 2 at its median level). However,for fantasy football purposes, this app allows the user to adjust to a model's high prediction if they have a more optimistic view of the player.")
