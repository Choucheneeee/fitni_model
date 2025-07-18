import pandas as pd
import json

# Path to your JSON file
json_path = 'C:/Users/cha/Desktop/dev/java/fitni_model/dataset.json'

# Load JSON file
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract only the 'user_profile' from each record
user_profiles = [item['user_profile'] for item in data if 'user_profile' in item]

# Convert to DataFrame
df = pd.DataFrame(user_profiles)

# Save to CSV
csv_path = 'C:/Users/cha/Desktop/dev/java/fitni_model/user_profile.csv'
df.to_csv(csv_path, index=False)

print(f"✅ Exported user_profile to: {csv_path}")
