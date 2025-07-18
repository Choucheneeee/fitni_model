import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_PATH = "user_profile.csv"

# Mapping dictionaries for categorical variables
activity_map = {"sedentary": 0, "light": 1, "moderate": 2, "active": 3, "very_active": 4}
goal_map = {"weight_loss": -1, "maintenance": 0, "performance": 1, "muscle_gain": 2}
gender_map = {"male": 0, "female": 1}

TARGET_COLUMNS = [
    'target_carbs_g',
    'target_daily_calories',
    'target_fat_g',
    'target_protein_g',
    'tdee',
    'bmr'
]

def load_and_prepare_data(path):
    df = pd.read_csv(path)

    # Normalize categorical columns
    for col in ['activity_level', 'goal', 'gender']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
        else:
            logging.error(f"Column '{col}' missing in dataset.")
            raise KeyError(f"Column '{col}' missing in dataset.")

    # Check required columns for features + targets
    required_cols = ['age', 'height_cm', 'weight_kg', 'activity_level', 'gender', 'goal'] + TARGET_COLUMNS
    df = df.dropna(subset=required_cols)

    # Add BMI feature
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

    X = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            features = [
                float(row['age']),
                float(row['height_cm']),
                float(row['weight_kg']),
                float(row['bmi']),
                activity_map[row['activity_level']],
                gender_map[row['gender']],
                goal_map[row['goal']],
            ]
            X.append(features)
        except KeyError as e:
            logging.warning(f"Skipping row due to unknown category {e}")
            skipped += 1
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping row due to invalid data: {e}")
            skipped += 1

    if skipped > 0:
        logging.info(f"Skipped {skipped} rows due to data issues.")

    X = np.array(X)

    # Extract targets as a DataFrame to keep consistent shape
    ys = df[TARGET_COLUMNS].reset_index(drop=True)
    ys = ys.iloc[:X.shape[0]]  # ensure same number of rows as X, just in case

    logging.info(f"Prepared dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, ys

def train_and_save_models(X, ys):
    models = {}
    for target in TARGET_COLUMNS:
        logging.info(f"\nTraining model for target: {target}")

        X_train, X_valid, y_train, y_valid = train_test_split(X, ys[target], test_size=0.2, random_state=42)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'seed': 42
        }

        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_valid, y_pred)
        logging.info(f"Validation MAE for {target}: {mae:.3f}")

        # Save model
        model_filename = f"lgbm_model_{target}.pkl"
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")

        models[target] = model

    # Save mapping files too
    joblib.dump(activity_map, "activity_map.joblib")
    joblib.dump(goal_map, "goal_map.joblib")
    joblib.dump(gender_map, "gender_map.joblib")
    logging.info("Mapping files saved.")

    return models

def test_models(models, X_test, y_test):
    logging.info("\nTesting saved models on test data:")
    for target, model in models.items():
        y_true = y_test[target]
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_true, y_pred)
        logging.info(f"{target} - MAE: {mae:.3f}")
        for i in range(min(5, len(y_true))):
            logging.info(f"Sample {i+1}: True={y_true.iloc[i]:.2f}, Predicted={y_pred[i]:.2f}")

if __name__ == "__main__":
    X, ys = load_and_prepare_data(DATA_PATH)
    models = train_and_save_models(X, ys)

    # Split once for testing
    X_train, X_test, y_train_dummy, y_test = train_test_split(X, ys, test_size=0.2, random_state=42)

    test_models(models, X_test, y_test)
