import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

TARGET_COLUMNS = [
    'target_carbs_g',
    'target_daily_calories',
    'target_fat_g',
    'target_protein_g',
    'tdee',
    'bmr'
]

# Load mappings and models
activity_map = joblib.load("activity_map.joblib")
goal_map = joblib.load("goal_map.joblib")
gender_map = joblib.load("gender_map.joblib")
models = {target: joblib.load(f"lgbm_model_{target}.pkl") for target in TARGET_COLUMNS}

def prepare_features(df):
    df['activity_level'] = df['activity_level'].str.lower().str.strip()
    df['goal'] = df['goal'].str.lower().str.strip()
    df['gender'] = df['gender'].str.lower().str.strip()

    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    print("Calculating BMI...", df['bmi'].head())

    df['activity_encoded'] = df['activity_level'].map(activity_map)
    df['goal_encoded'] = df['goal'].map(goal_map)
    df['gender_encoded'] = df['gender'].map(gender_map)

    if df[['activity_encoded', 'goal_encoded', 'gender_encoded']].isnull().any().any():
        raise ValueError("Unknown categories found in input data.")

    features = ['age', 'height_cm', 'weight_kg', 'bmi', 'activity_encoded', 'gender_encoded', 'goal_encoded']
    return df[features].values

def get_user_input():
    print("Please enter the following information:")
    age = float(input("Age (years): "))
    weight_kg = float(input("Weight (kg): "))
    height_cm = float(input("Height (cm): "))
    print("Activity level options:", list(activity_map.keys()))
    activity_level = input("Activity level: ").lower().strip()
    print("Goal options:", list(goal_map.keys()))
    goal = input("Goal: ").lower().strip()
    print("Gender options:", list(gender_map.keys()))
    gender = input("Gender: ").lower().strip()

    return {
        'age': age,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'activity_level': activity_level,
        'goal': goal,
        'gender': gender
    }

if __name__ == "__main__":
    user_data = get_user_input()
    df_user = pd.DataFrame([user_data])

    try:
        X = prepare_features(df_user)
    except ValueError as e:
        logging.error(f"Input error: {e}")
        exit(1)

    logging.info("\nPrediction results:\n")
    for target in TARGET_COLUMNS:
        pred = models[target].predict(X[0].reshape(1, -1), num_iteration=models[target].best_iteration)[0]
        logging.info(f"{target}: Predicted = {pred:.2f}")
