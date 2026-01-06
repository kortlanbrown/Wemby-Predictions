#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:20:56 2024

@author: kortlanbrown
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Load the data
data = pd.read_csv('per_game_stats2.csv')
data.fillna(0, inplace=True)
data

# Filter data for Regular Season only
regular_season_data = data[data['RSorPO'] == 'Regular Season']
regular_season_data
# Career stats calculations (only regular season)
career_ppg = regular_season_data.groupby(['Player', 'RSorPO'])['PTS'].mean().unstack()
career_ppg.columns = ['Career Regular Season PPG']
career_ppg = career_ppg.round(2)
print(career_ppg)

career_ast_pg = regular_season_data.groupby(['Player', 'RSorPO'])['AST'].mean().unstack()
career_ast_pg.columns = ['Career Regular Season APG']
career_ast_pg = career_ast_pg.round(2)
print(career_ast_pg)

career_reb_pg = regular_season_data.groupby(['Player', 'RSorPO'])['TRB'].mean().unstack()
career_reb_pg.columns = ['Career Regular Season REB']
career_reb_pg = career_reb_pg.round(2)
print(career_reb_pg)

career_blk_pg = regular_season_data.groupby(['Player', 'RSorPO'])['BLK'].mean().unstack()
career_blk_pg.columns = ['Career Regular Season BLK']
career_blk_pg = career_blk_pg.round(2)
print(career_blk_pg)

career_stl_pg = regular_season_data.groupby(['Player', 'RSorPO'])['STL'].mean().unstack()
career_stl_pg.columns = ['Career Regular Season STL']
career_stl_pg = career_stl_pg.round(2)
print(career_stl_pg)

# Identify the rookie season for each player (the minimum year)
rookie_seasons = regular_season_data.groupby('Player')['Season'].min().reset_index()

# Merge rookie_seasons with the regular season data to filter only rookie stats
rookie_stats = regular_season_data.merge(rookie_seasons, on=['Player', 'Season'])

# Rookie Points per Game (PPG)
rookie_ppg = rookie_stats[['Player', 'RSorPO', 'PTS']].set_index(['Player', 'RSorPO']).round(2)
rookie_ppg.columns = ['Rookie Regular Season PPG']
print(rookie_ppg)

# Rookie Rebounds per Game (REB)
rookie_reb = rookie_stats[['Player', 'RSorPO', 'TRB']].set_index(['Player', 'RSorPO']).round(2)
rookie_reb.columns = ['Rookie Regular Season REB']
print(rookie_reb)

# Rookie Assists per Game (APG)
rookie_ast_pg = rookie_stats[['Player', 'RSorPO', 'AST']].set_index(['Player', 'RSorPO']).round(2)
rookie_ast_pg.columns = ['Rookie Regular Season APG']
print(rookie_ast_pg)

# Rookie Blocks per Game (BLK)
rookie_blk = rookie_stats[['Player', 'RSorPO', 'BLK']].set_index(['Player', 'RSorPO']).round(2)
rookie_blk.columns = ['Rookie Regular Season BLK']
print(rookie_blk)

# Rookie Steals per Game (STL)
rookie_stl = rookie_stats[['Player', 'RSorPO', 'STL']].set_index(['Player', 'RSorPO']).round(2)
rookie_stl.columns = ['Rookie Regular Season STL']
print(rookie_stl)

# Combine rookie stats into one DataFrame for model input
rookie_combined = pd.concat([rookie_ppg, rookie_reb, rookie_ast_pg, rookie_blk, rookie_stl], axis=1)
print(rookie_combined)

# Prepare data for predictive model
# Features: Rookie stats
X = rookie_combined[['Rookie Regular Season PPG', 'Rookie Regular Season REB', 'Rookie Regular Season APG', 'Rookie Regular Season BLK', 'Rookie Regular Season STL']]

# Target: Career stats (Regular Season only)
career_combined = pd.concat([career_ppg, career_reb_pg, career_ast_pg, career_blk_pg, career_stl_pg], axis=1)
y = career_combined[['Career Regular Season PPG', 'Career Regular Season REB', 'Career Regular Season APG', 'Career Regular Season BLK', 'Career Regular Season STL']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Example input data for Wembanyama (based on his rookie stats and physical attributes)
wemby_data = [[21.4, 10.6, 3.9, 3.6, 1.2]]  # Example input for Wembanyama's rookie season stats

# Predict Wembanyama's career stats
wemby_prediction = model.predict(wemby_data)
print("Predicted Career Stats (Simple Model):")
print(f"PPG: {wemby_prediction[0][0]:.2f}")
print(f"REB: {wemby_prediction[0][1]:.2f}")
print(f"APG: {wemby_prediction[0][2]:.2f}")
print(f"BLK: {wemby_prediction[0][3]:.2f}")
print(f"STL: {wemby_prediction[0][4]:.2f}")


# Calculate regression metrics
mse = mean_squared_error(y_test, wemby_prediction)
mae = mean_absolute_error(y_test, wemby_prediction)
r2 = r2_score(y_test, wemby_prediction)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Features (Rookie stats) and Targets (Career stats)
X = rookie_combined[['Rookie Regular Season PPG', 'Rookie Regular Season REB', 
                     'Rookie Regular Season APG', 'Rookie Regular Season BLK', 
                     'Rookie Regular Season STL']]
y = career_combined[['Career Regular Season PPG', 'Career Regular Season REB', 
                     'Career Regular Season APG', 'Career Regular Season BLK', 
                     'Career Regular Season STL']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Predict Wembanyama's career stats
wemby_data = [[21.4, 10.6, 3.9, 3.6, 1.2]]  # Example input for Wembanyama
wemby_data_scaled = scaler.transform(wemby_data)
wemby_prediction = rf_model.predict(wemby_data_scaled)

# Display the prediction
print("Predicted Career Stats (Random Forest Model):")
print(f"PPG: {wemby_prediction[0][0]:.2f}")
print(f"REB: {wemby_prediction[0][1]:.2f}")
print(f"APG: {wemby_prediction[0][2]:.2f}")
print(f"BLK: {wemby_prediction[0][3]:.2f}")
print(f"STL: {wemby_prediction[0][4]:.2f}")

# Create DataFrame for Wembanyama's predicted stats
wemby_stats_df = pd.DataFrame({
    'Player': ['Victor Wembanyama'],
    'Stat Type': ['Predicted Career'],  # Label for predicted stats
    'RSorPO': ['Regular Season'],  # Assuming regular season
    'PPG': [wemby_prediction[0][0]],
    'REB': [wemby_prediction[0][1]],
    'AST': [wemby_prediction[0][2]],
    'BLK': [wemby_prediction[0][3]],
    'STL': [wemby_prediction[0][4]]
})

# Random Forest Metrics
r2 = r2_score(y_test, y_pred)  # R² score
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error

print(f"Random Forest R²: {r2}")
print(f"Random Forest MSE: {mse}")
print(f"Random Forest MAE: {mae}")



final_data = pd.concat([career_combined, wemby_stats_df], ignore_index=True)

final_data.to_csv("nba_stats_comparison.csv", index=False)






# Import necessary libraries
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare the feature matrix (X) and target matrix (y)
X = rookie_combined[['Rookie Regular Season PPG', 'Rookie Regular Season REB', 
                     'Rookie Regular Season APG', 'Rookie Regular Season BLK', 
                     'Rookie Regular Season STL']]
y = career_combined[['Career Regular Season PPG', 'Career Regular Season REB', 
                     'Career Regular Season APG', 'Career Regular Season BLK', 
                     'Career Regular Season STL']]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost model (Gradient Boosting)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting (XGBoost) Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict Wembanyama's career stats (based on his rookie stats)
wemby_data = [[21.4, 10.6, 3.9, 3.6, 1.2]]  # Wembanyama's rookie season stats
wemby_data_scaled = scaler.transform(wemby_data)
wemby_prediction = model.predict(wemby_data_scaled)

# Display the prediction for Wembanyama's career stats
print("Predicted Career Stats (Gradient Boosting Model):")
print(f"PPG: {wemby_prediction[0][0]:.2f}")
print(f"REB: {wemby_prediction[0][1]:.2f}")
print(f"APG: {wemby_prediction[0][2]:.2f}")
print(f"BLK: {wemby_prediction[0][3]:.2f}")
print(f"STL: {wemby_prediction[0][4]:.2f}")

# Create a DataFrame to display the predicted stats
wemby_stats_df = pd.DataFrame({
    'Player': ['Victor Wembanyama'],
    'Stat Type': ['Predicted Career'],  # Label for predicted stats
    'RSorPO': ['Regular Season'],  # Assuming regular season
    'PPG': [wemby_prediction[0][0]],
    'REB': [wemby_prediction[0][1]],
    'AST': [wemby_prediction[0][2]],
    'BLK': [wemby_prediction[0][3]],
    'STL': [wemby_prediction[0][4]]
})

print(wemby_stats_df)
