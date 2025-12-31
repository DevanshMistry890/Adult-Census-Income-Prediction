import pandas as pd
import numpy as np
import pickle
import json
import shap
import re
import types
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# --- 1. RECONSTRUCT THE DATA (Preprocessing) ---
print("🔄 Reconstructing data pipeline...")
try:
    df = pd.read_csv('adult.csv') 
except FileNotFoundError:
    df = pd.read_csv('adult.csv')

# Handle Missing Values
df = df.replace("?", np.nan)
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Discretization
ms_col = 'marital.status' if 'marital.status' in df.columns else 'marital-status'
df[ms_col] = df[ms_col].replace(
    ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
     'Never-married', 'Separated', 'Widowed'],
    ['divorced', 'married', 'married', 'married',
     'not married', 'not married', 'not married']
)

# Label Encoding
labelEncoder = preprocessing.LabelEncoder()
cols_to_encode = ['workclass', 'race', 'education', ms_col, 'occupation', 
                  'relationship', 'sex', 'native.country', 'income']

actual_cols = df.columns.tolist()
for col in cols_to_encode:
    if col in actual_cols:
        df[col] = labelEncoder.fit_transform(df[col])
    elif col == 'native.country' and 'country' in actual_cols:
        df['country'] = labelEncoder.fit_transform(df['country'])
    elif col == 'income' and 'salary' in actual_cols:
        df['salary'] = labelEncoder.fit_transform(df['salary'])

# Drop Redundant
drop_cols = [c for c in ['fnlwgt', 'education.num', 'education-num'] if c in df.columns]
df = df.drop(drop_cols, axis=1)

# Split 
X = df.iloc[:, :-1].values.astype(np.float32)
Y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print(f"✅ Data processed. Features: {X_train.shape[1]}")

# --- 2. LOAD & PROXY PATCH MODEL ---
print("📦 Loading and patching model (Runtime Proxy)...")
model = pickle.load(open('model_tuned.pkl', 'rb'))

# === THE FIX: Proxy Class to intercept save_config() ===
class BoosterProxy:
    """Wraps the XGBoost Booster to fix the JSON config on the fly."""
    def __init__(self, real_booster):
        self.real_booster = real_booster

    def save_config(self):
        # 1. Get the actual config from XGBoost (which has the brackets error)
        raw_config = self.real_booster.save_config()
        # 2. Remove brackets using regex: "[0.123]" -> "0.123"
        fixed_config = re.sub(r'"base_score":"\[(.*?)\]"', r'"base_score":"\1"', raw_config)
        return fixed_config

    def __getattr__(self, name):
        # Forward all other calls to the real booster
        return getattr(self.real_booster, name)

# Apply the proxy to the model
# When convert_xgboost calls model.get_booster(), it will get our Proxy instead
real_booster = model.get_booster()
proxy = BoosterProxy(real_booster)

def get_booster_patch(self):
    return proxy

# Monkey-patch the method on this specific instance
model.get_booster = types.MethodType(get_booster_patch, model)
print("✅ Runtime Proxy applied successfully.")
# =======================================================

# --- 3. EXPORT TO ONNX ---
print("🚀 Exporting to ONNX...")
n_features = X_train.shape[1]
initial_type = [('float_input', FloatTensorType([None, n_features]))]

# Convert (Now it uses our Proxy internally)
onx = convert_xgboost(model, initial_types=initial_type, target_opset=12)

# Save
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
print("✅ ONNX Model saved to ../frontend/public/model.onnx")

# --- 4. SHAP BACKGROUND DATA ---
print("📊 Generating SHAP background data...")
background_summary = shap.kmeans(X_train, 50)
background_data = background_summary.data.tolist()

with open("background.json", "w") as f:
    json.dump(background_data, f)
print("✅ Background data saved to ../frontend/public/background.json")

print("\n🚀 EDGE DEPLOYMENT READY!")