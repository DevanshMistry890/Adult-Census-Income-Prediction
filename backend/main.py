import pickle
import numpy as np
import shap
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load Model ---
try:
    model = pickle.load(open('model_tuned.pkl', 'rb'))
    explainer = shap.TreeExplainer(model)
    print("Model and Explainer loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# --- 2. Input Schema ---
class IncomeInput(BaseModel):
    age: float
    workclass: float
    education: float
    marital_status: float
    occupation: float
    relationship: float
    race: float
    sex: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    native_country: float

# --- 3. HELPER: Decoder Maps (To turn "4" back into "Never-Married") ---
# These must match your Frontend & LabelEncoder exactly
DECODER_MAPS = {
    "Workclass": {0: 'Federal-gov', 1: 'Local-gov', 2: 'Never-worked', 3: 'Private', 4: 'Self-emp-inc', 5: 'Self-emp-not-inc', 6: 'State-gov', 7: 'Without-pay'},
    "Education": {0: '10th', 1: '11th', 2: '12th', 3: '1st-4th', 4: '5th-6th', 5: '7th-8th', 6: '9th', 7: 'Assoc-acdm', 8: 'Assoc-voc', 9: 'Bachelors', 10: 'Doctorate', 11: 'HS-grad', 12: 'Masters', 13: 'Preschool', 14: 'Prof-school', 15: 'Some-college'},
    "Marital Status": {0: 'Divorced', 1: 'Married-AF', 2: 'Married-Civ', 3: 'Married-Absent', 4: 'Never-married', 5: 'Separated', 6: 'Widowed'},
    "Relationship": {0: 'Husband', 1: 'Not-in-family', 2: 'Other-relative', 3: 'Own-child', 4: 'Unmarried', 5: 'Wife'},
    "Race": {0: 'Amer-Indian', 1: 'Asian-Pac', 2: 'Black', 3: 'Other', 4: 'White'},
    "Sex": {0: 'Female', 1: 'Male'}
}

def get_human_reason(feature_name, feature_value, shap_score):
    """
    Translates math into English sentences.
    """
    # 1. Decode categorical values to text (e.g., 4 -> "Never-married")
    readable_value = feature_value
    if feature_name in DECODER_MAPS:
        readable_value = DECODER_MAPS[feature_name].get(int(feature_value), feature_value)
    
    # 2. Determine Direction
    direction = "increased" if shap_score > 0 else "decreased"
    strength = "significantly " if abs(shap_score) > 0.5 else ""
    
    # 3. Custom Logic for Specific Features (The "Smart" part)
    if feature_name == "Capital Gain" and feature_value == 0 and shap_score < 0:
        return "Lack of Capital Gain limits high-income potential."
    
    if feature_name == "Age" and shap_score < 0 and feature_value < 25:
        return f"Age ({int(feature_value)}) is typically too young for high income brackets."

    if feature_name == "Marital Status" and shap_score < 0:
        return f"Status '{readable_value}' is associated with lower household income."

    if feature_name == "Relationship" and shap_score < 0:
        return f"Relationship status '{readable_value}' lowers the probability."
        
    # 4. Generic Fallback
    return f"{feature_name} '{readable_value}' {strength}{direction} the score."


@app.post("/predict")
def predict_income(data: IncomeInput):
    features = [[data.age, data.workclass, data.education, data.marital_status, data.occupation, data.relationship, data.race, data.sex, data.capital_gain, data.capital_loss, data.hours_per_week, data.native_country]]
    prob = float(model.predict_proba(features)[:, 1][0])
    prediction = int(prob > 0.5)
    return {"prediction": prediction, "probability": prob, "message": ">50K" if prediction == 1 else "<=50K"}

@app.post("/explain")
def explain_prediction(data: IncomeInput):
    try:
        # 1. Prepare data
        features_list = [data.age, data.workclass, data.education, data.marital_status, data.occupation, data.relationship, data.race, data.sex, data.capital_gain, data.capital_loss, data.hours_per_week, data.native_country]
        features_array = np.array([features_list])
        
        # 2. Run SHAP
        shap_values = explainer.shap_values(features_array)[0]
        
        feature_names = ["Age", "Workclass", "Education", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours/Week", "Country"]
        
        # 3. Combine Score, Name, AND Value
        # We store (Name, SHAP Score, Input Value)
        combined = []
        for name, score, val in zip(feature_names, shap_values, features_list):
            combined.append((name, float(score), val))
            
        # 4. Sort by Impact
        combined.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3 = combined[:3]
        
        # 5. Generate Human Sentences
        human_explanations = []
        for name, score, val in top_3:
            sentence = get_human_reason(name, val, score)
            human_explanations.append([name, score, sentence])

        # Get Base Value
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray): base_val = base_val[0]

        return {
            "top_factors": human_explanations, # Now sends [Name, Score, Sentence]
            "base_value": float(base_val)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {"top_factors": [], "base_value": 0.0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)