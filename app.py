import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Ensure flask_cors is installed: pip install Flask-Cors
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define paths to your model and data files
MODEL_PATH = 'food_price_model.joblib'
X_TRAIN_COLUMNS_PATH = 'X_train_columns.joblib'
DATA_PATH = 'wfp_food_prices_npl.csv'

# --- Global variables to store loaded model and data ---
model_pipeline = None
X_train_template_columns = None
df_original_data = None # Will store the raw data for finding latest record

# --- Load model and data once when the Flask app starts ---
def load_assets():
    global model_pipeline, X_train_template_columns, df_original_data
    print("Attempting to load assets...")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found.")
            return False
        if not os.path.exists(X_TRAIN_COLUMNS_PATH):
            print(f"Error: X_train columns file '{X_TRAIN_COLUMNS_PATH}' not found.")
            return False
        if not os.path.exists(DATA_PATH):
            print(f"Error: Data file '{DATA_PATH}' not found.")
            return False

        model_pipeline = joblib.load(MODEL_PATH)
        X_train_template_columns = joblib.load(X_TRAIN_COLUMNS_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"X_train columns loaded successfully from {X_TRAIN_COLUMNS_PATH}")

        # Load the original data for `predict_future_price` to find the latest record
        df_original_data = pd.read_csv(DATA_PATH, low_memory=False)
        print(f"Original dataset loaded successfully from {DATA_PATH} (initial count: {len(df_original_data)} rows)")

        # Convert 'date' column to datetime objects, coercing errors to NaT
        df_original_data['date'] = pd.to_datetime(df_original_data['date'], errors='coerce')
        initial_rows_date_clean = len(df_original_data)
        df_original_data.dropna(subset=['date'], inplace=True)
        if len(df_original_data) < initial_rows_date_clean:
            print(f"Dropped {initial_rows_date_clean - len(df_original_data)} rows with invalid dates in original data.")

        # Ensure 'price' is present and numeric
        if 'price' not in df_original_data.columns:
            print("Error: 'price' column not found in data. Cannot proceed.")
            return False
        df_original_data['price'] = pd.to_numeric(df_original_data['price'], errors='coerce')
        initial_rows_price_clean = len(df_original_data)
        df_original_data.dropna(subset=['price'], inplace=True)
        if len(df_original_data) < initial_rows_price_clean:
            print(f"Dropped {initial_rows_price_clean - len(df_original_data)} rows with missing/invalid prices in original data.")

        print(f"Original dataset after date/price cleaning: {len(df_original_data)} rows.")

        # Ensure numeric columns are converted in df_original_data as well, matching training script
        numeric_cols_to_convert_onload = ['market_id', 'latitude', 'longitude', 'commodity_id', 'usdprice']
        for col in numeric_cols_to_convert_onload:
            if col in df_original_data.columns: # Check if column exists
                df_original_data[col] = pd.to_numeric(df_original_data[col], errors='coerce')
                if df_original_data[col].isnull().sum() > 0:
                    if col in ['latitude', 'longitude']:
                        mean_val = df_original_data[col].mean()
                        if pd.isna(mean_val): # If mean itself is NaN (e.g., all values are NaN)
                            df_original_data[col] = df_original_data[col].fillna(0) # Fallback to 0 or another default
                            print(f"Warning: Column '{col}' has all NaN values, filling with 0.")
                        else:
                            df_original_data[col] = df_original_data[col].fillna(mean_val)
                        print(f"Filled {df_original_data[col].isnull().sum()} NaNs in '{col}' with mean in original data.")
                    else:
                        initial_rows_col_clean = len(df_original_data)
                        df_original_data.dropna(subset=[col], inplace=True)
                        if len(df_original_data) < initial_rows_col_clean:
                            print(f"Dropped {initial_rows_col_clean - len(df_original_data)} rows with missing/invalid '{col}' in original data.")
            else:
                print(f"Warning: Numeric column '{col}' not found in original data. It might cause issues if expected by model.")

        # Ensure categorical columns are filled with 'Unknown' if NaN, matching training script
        categorical_features_onload = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'priceflag', 'pricetype']
        for col in categorical_features_onload:
            if col in df_original_data.columns and df_original_data[col].isnull().sum() > 0:
                df_original_data[col] = df_original_data[col].fillna('Unknown')
                print(f"Filled {df_original_data[col].isnull().sum()} NaNs in '{col}' with 'Unknown' in original data.")
            elif col not in df_original_data.columns:
                print(f"Warning: Categorical column '{col}' not found in original data. It might cause issues if expected by model.")


        print("All assets loaded and data pre-processed for Flask.")
        return True # Indicate success

    except Exception as e:
        print(f"FATAL ERROR during asset loading: {e}")
        model_pipeline = None # Set to None to indicate failure
        X_train_template_columns = None
        df_original_data = None
        return False


# --- Prediction Function (similar to the one in your training script) ---
def predict_future_price_flask(commodity, market, months_ahead):
    if model_pipeline is None or X_train_template_columns is None or df_original_data is None:
        return None, "Server assets not loaded. Please restart the server."

    # Sort the original data by date to easily find the latest record
    df_original_sorted = df_original_data.sort_values(by='date')

    # Find the most recent historical record for the specified commodity and market
    record = df_original_sorted[
        (df_original_sorted['commodity'] == commodity) &
        (df_original_sorted['market'] == market)
    ].tail(1)

    if record.empty:
        return None, f"No historical data found for '{commodity}' in '{market}'. Please select a different commodity/market pair, or try a different 'months_ahead' if the latest data is too far in the past."

    record = record.iloc[0].copy() # Get the single record as a Series

    # Calculate the future date
    future_date = record['date'] + pd.DateOffset(months=months_ahead)

    # Update date-related features in the record for the future date
    record['year'] = future_date.year
    record['month'] = future_date.month
    record['day'] = future_date.day
    record['day_of_week'] = future_date.dayofweek
    record['day_of_year'] = future_date.dayofyear
    record['week_of_year'] = future_date.isocalendar().week
    record['quarter'] = future_date.quarter

    # Create an empty DataFrame with the exact same columns as X_train_template_columns
    # This is crucial for matching the feature shape expected by the model's preprocessor.
    X_new = pd.DataFrame(columns=X_train_template_columns)

    # Populate the first (and only) row of X_new with the record's data
    for col in X_train_template_columns:
        if col in record.index:
            X_new.loc[0, col] = record[col]
        else:
            # This handles columns that might be from OneHotEncoder but not in this specific record.
            # Setting to 0 is appropriate for one-hot encoded features not present.
            # For numerical features that might be missing from the original record, this could be an issue if 0 is not a sensible default.
            X_new.loc[0, col] = 0

    # Make the prediction using the pipeline
    try:
        predicted_price = model_pipeline.predict(X_new)[0]
        return predicted_price, future_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Prediction error during model.predict(): {e}")
        return None, f"Prediction failed due to model error: {e}. Please check server logs for details."


# --- Flask Routes ---
@app.route('/')
def home():
    # Provide unique commodity and market lists for the dropdowns
    commodities_list = []
    markets_list = []
    if df_original_data is not None:
        commodities_list = sorted(df_original_data['commodity'].dropna().unique().tolist())
        markets_list = sorted(df_original_data['market'].dropna().unique().tolist())
    else:
        print("Warning: df_original_data is None. Using hardcoded fallback lists for dropdowns.")
        # Fallback if data loading failed, provide common ones
        commodities_list = ["Rice (coarse)", "Wheat flour", "Potatoes (red)", "Meat (chicken)", "Tomatoes", "Oil (vegetable)", "Salt", "Lentils (broken)", "Sugar"]
        markets_list = ["Kathmandu", "Jumla", "Birendranagar", "Dhangadhi", "Pokhara", "Biratnagar", "Butwal"]

    return render_template('index.html', commodities=commodities_list, markets=markets_list)

@app.route('/predict', methods=['POST'])
def predict():
    # --- Debugging additions ---
    print(f"\nRequest received: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    # --- End debugging additions ---

    try:
        # Ensure request is JSON
        if not request.is_json:
            print("Error: Request received is not JSON.")
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        print(f"Received JSON data: {data}")  # Debug logging

        commodity = data.get('commodity')
        market = data.get('market')
        months_ahead_str = data.get('months_ahead', '1') # Get as string, then convert

        try:
            months_ahead = int(months_ahead_str)
        except (ValueError, TypeError):
            print(f"Error: Invalid 'months_ahead' value received: {months_ahead_str}")
            return jsonify({'error': 'Invalid value for months_ahead. Must be an integer.'}), 400


        if not all([commodity, market, isinstance(months_ahead, int)]):
            print(f"Error: Missing or invalid parameters. Commodity: {commodity}, Market: {market}, Months Ahead: {months_ahead}")
            return jsonify({'error': 'Missing required parameters: commodity, market, and months_ahead (must be an integer).'}), 400

        predicted_price, message = predict_future_price_flask(commodity, market, months_ahead)

        if predicted_price is not None:
            print(f"Prediction successful for {commodity} in {market}: {predicted_price:.2f} NPR on {message}")
            return jsonify({
                'commodity': commodity,
                'market': market,
                'months_ahead': months_ahead,
                'predicted_price': f"{predicted_price:.2f} NPR",
                'prediction_date': message # message here is the date string
            })
        else:
            print(f"Prediction failed for {commodity} in {market}: {message}")
            return jsonify({'error': message}), 500 # message here is the error string

    except Exception as e:
        print(f"Unexpected error in predict route: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    print(f"404 Not Found error: {request.url}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"500 Internal Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load assets before running the app
    assets_loaded_successfully = load_assets()
    if not assets_loaded_successfully:
        print("Application will not run due to asset loading failure.")
        exit() # Exit if model loading failed

    print("\nStarting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
