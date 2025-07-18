Here’s a detailed **README.md** template for your project based on the files and structure you shared. You can save it as `Readme.md` or `README.md` in your project root.

---

# Fitni\_Model: Nutrition and Fitness Prediction

## Overview

This project builds machine learning models to predict nutrition and fitness-related target values such as daily calorie needs, macronutrient targets, BMR (Basal Metabolic Rate), and TDEE (Total Daily Energy Expenditure) based on user profile data.

Models are trained using LightGBM regression and saved for later prediction. The project also supports interactive prediction by taking user inputs.

---

## Project Structure

```
fitni_model/
├── activity_map.joblib            # Mapping for activity levels to numeric codes
├── cleandataset.py                # (Optional) Script to preprocess/clean raw data
├── dataset.csv                   # Original or merged user profile dataset (CSV)
├── dataset.json                  # Original or raw dataset in JSON format
├── gender_map.joblib              # Mapping for gender to numeric codes
├── goal_map.joblib                # Mapping for user goals to numeric codes
├── lgbm_model_bmr.pkl             # Trained LightGBM model for BMR prediction
├── lgbm_model_target_carbs_g.pkl  # Model for carbs target
├── lgbm_model_target_daily_calories.pkl # Model for daily calories target
├── lgbm_model_target_fat_g.pkl    # Model for fat target
├── lgbm_model_target_protein_g.pkl# Model for protein target
├── lgbm_model_tdee.pkl            # Model for TDEE prediction
├── predict.py                    # Script to load models and predict from user input
├── Readme.md                     # This file - project explanation and usage
├── steps.md                      # Steps/notes for project usage or training
├── train_and_save.py             # Script to train models on dataset and save them
└── user_profile.csv              # User profiles with features and target labels
```

---

## Installation & Setup

1. Make sure Python 3.8+ is installed.
2. Install required Python packages:

   ```bash
   pip install pandas numpy scikit-learn lightgbm joblib
   ```
3. Place all `.joblib` mappings and model `.pkl` files in the project root as above.

---

## Usage

### 1. Train Models (If you want to retrain)

Run the training script:

```bash
python cleandataset.py
```

* This loads `user_profile.csv`, processes data, trains LightGBM models for each target variable, evaluates them, and saves the models and mappings.
* after to train and save the model run 
```bash
python train_and_save.py
```
### 2. Predict for New User Input

Run the prediction script:

```bash
python predict.py
```

* The script will ask for user input: age, weight, height, activity level, goal, and gender.
* It uses the saved models and mappings to predict all targets (`carbs`, `daily_calories`, `fat`, `protein`, `tdee`, `bmr`).
* Results are printed with predicted values.



## Data & Mappings

* `user_profile.csv`: Contains user demographic and physiological data along with true nutrition targets.
* `activity_map.joblib`, `goal_map.joblib`, `gender_map.joblib`: Encode categorical variables to numeric codes required for model input.

---

## Model Details

* Each target variable has an individual LightGBM regression model saved as `lgbm_model_<target>.pkl`.
* Models use features: age, height, weight, BMI, and encoded categorical variables.
* Early stopping and validation are used during training to avoid overfitting.

---

## Notes

* Ensure all categorical inputs during prediction are valid (matching keys in maps).
* You can extend `predict.py` for batch predictions or integration with other systems.
* The project currently supports 6 targets but can be extended.

---

## Contact / Support

For issues or questions, please open an issue or contact the maintainer.