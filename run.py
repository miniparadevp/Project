
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load datasets safely
try:
    dataset = pd.read_csv('Training.csv')
    description = pd.read_csv('description.csv')
    precautions = pd.read_csv('precautions_df.csv')
    medications = pd.read_csv('medications.csv')
    diets = pd.read_csv('diets.csv')
    workout = pd.read_csv('workout_df.csv')
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Create symptom dictionary
all_symptoms = dataset.columns[:-1]
symptoms_dict = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Map diseases to indices and reverse map
unique_diseases = dataset.iloc[:, -1].unique()
diseases_list = {disease: idx for idx, disease in enumerate(unique_diseases)}
diseases_list_reverse = {idx: disease for disease, idx in diseases_list.items()}

# Prepare training data
X_train = dataset.iloc[:, :-1].values.astype(float)
y_train = np.array([diseases_list[y] for y in dataset.iloc[:, -1]])

# Train SVM model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Helper function to get disease details
def helper(disease_name):
    desc = description.loc[description['Disease'] == disease_name, 'Description'].values
    pre = precautions.loc[precautions['Disease'] == disease_name, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications.loc[medications['Disease'] == disease_name, 'Medication'].values.tolist()
    die = diets.loc[diets['Disease'] == disease_name, 'Diet'].values.tolist()
    wrkout = workout.loc[workout['disease'] == disease_name, 'workout'].values.tolist()

    return (
        desc[0] if len(desc) > 0 else "No description available",
        pre if len(pre) > 0 else ["No precautions available"],
        med if len(med) > 0 else ["No medications available"],
        die if len(die) > 0 else ["No diet information available"],
        wrkout if len(wrkout) > 0 else ["No workout available"]
    )

# Predict disease from symptoms
def given_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    prediction_index = svc.predict([input_vector])[0]
    return diseases_list_reverse.get(prediction_index, "Unknown Disease")

# API Routes
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400
        
        # Predict disease
        predicted_disease = given_predicted_value(symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        print("data....",pre)
        
        return jsonify({
            "Disease": predicted_disease,
            "Description": desc,
            "Precautions": pre,
            "Medications": med,
            "Diet": die,
            "Workout": wrkout
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({"symptoms": list(symptoms_dict.keys())})

@app.route('/diseases', methods=['GET'])
def get_diseases():
    return jsonify({"diseases": list(diseases_list.keys())})

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {e}")
