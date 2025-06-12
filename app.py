from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load('fraud_detection_model.joblib')

def get_default_features():
    features = {}
    for i in range(1, 29):
        features[f'V{i}'] = 0.0
    features['Time'] = 0.0
    features['Amount'] = 0.0
    return features

@app.route('/')
def home():
    return render_template('card_form.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        form_data = request.form
        features = get_default_features()
        
        features.update({
            'Time': (datetime.now() - datetime(2023,1,1)).total_seconds(),
            'Amount': float(form_data['amount']),
            'V1': float(form_data.get('v1', 0)),
            'V2': float(form_data.get('v2', 0)),
            'V3': float(form_data.get('v3', 0)),
            'V4': float(form_data.get('v4', 0)),
            'V5': float(form_data.get('v5', 0)),
            'V6': float(form_data.get('v6', 0)),
            'V7': float(form_data.get('v7', 0)),
            'V8': float(form_data.get('v8', 0)),
            'V9': float(form_data.get('v9', 0)),
            'V10': float(form_data.get('v10', 0)),
            'V11': float(form_data.get('v11', 0)),
            'V12': float(form_data.get('v12', 0)),
            'V13': float(form_data.get('v13', 0)),
            'V14': float(form_data.get('v14', 0)),
            'V15': float(form_data.get('v15', 0)),
            'V16': float(form_data.get('v16', 0)),
            'V17': float(form_data.get('v17', 0)),
            'V18': float(form_data.get('v18', 0)),
            'V19': float(form_data.get('v19', 0)),
            'V20': float(form_data.get('v20', 0)),
            'V21': float(form_data.get('v21', 0)),
            'V22': float(form_data.get('v22', 0)),
            'V23': float(form_data.get('v23', 0)),
            'V24': float(form_data.get('v24', 0)),
            'V25': float(form_data.get('v25', 0)),
            'V26': float(form_data.get('v26', 0)),
            'V27': float(form_data.get('v27', 0)),
            'V28': float(form_data.get('v28', 0))
        })

        features_df = pd.DataFrame([features], columns=model.feature_names_in_)
        
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
        
        result = {
            'card_number': form_data['card_number'][-4:],
            'amount': form_data['amount'],
            'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'probability': f"{probability:.2%}",
            'risk_level': 'high' if probability > 0.6 else 'medium' if probability > 0.5 else 'low'
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)