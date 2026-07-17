from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import joblib
import numpy as np
import os
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'qmedix-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///qmedix.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)

# Database Models
class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    blood_pressure = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    smoking = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('Prediction', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'blood_pressure': self.blood_pressure,
            'cholesterol': self.cholesterol,
            'heart_rate': self.heart_rate,
            'bmi': self.bmi,
            'glucose': self.glucose,
            'smoking': self.smoking,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'))
    classical_result = db.Column(db.String(20))
    classical_confidence = db.Column(db.Float)
    quantum_result = db.Column(db.String(20))
    quantum_confidence = db.Column(db.Float)
    final_risk = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'classical_result': self.classical_result,
            'classical_confidence': self.classical_confidence,
            'quantum_result': self.quantum_result,
            'quantum_confidence': self.quantum_confidence,
            'final_risk': self.final_risk,
            'risk_level': self.risk_level,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Create tables
with app.app_context():
    db.create_all()

# Load models
print("\n" + "=" * 60)
print("🧬 QMedix - Starting Application")
print("=" * 60)

try:
    classical_model = joblib.load('models/classical_svm.pkl')
    quantum_model = joblib.load('models/quantum_qsvc.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Models loaded successfully!")
    models_loaded = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️ Please run train.py first!")
    classical_model = None
    quantum_model = None
    scaler = None
    models_loaded = False

print("=" * 60)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/patient', methods=['POST'])
def add_patient():
    try:
        data = request.get_json()
        
        required = ['name', 'age', 'gender', 'blood_pressure', 'cholesterol', 
                   'heart_rate', 'bmi', 'glucose', 'smoking']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        patient = Patient(
            name=data['name'],
            age=int(data['age']),
            gender=data['gender'],
            blood_pressure=float(data['blood_pressure']),
            cholesterol=float(data['cholesterol']),
            heart_rate=float(data['heart_rate']),
            bmi=float(data['bmi']),
            glucose=float(data['glucose']),
            smoking=bool(data['smoking'])
        )
        
        db.session.add(patient)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'id': patient.id,
            'message': 'Patient saved successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients')
def get_patients():
    try:
        patients = Patient.query.all()
        return jsonify([p.to_dict() for p in patients])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    try:
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        Prediction.query.filter_by(patient_id=patient_id).delete()
        db.session.delete(patient)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Patient {patient.name} and all associated reports deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/prediction/<int:prediction_id>', methods=['DELETE'])
def delete_prediction(prediction_id):
    try:
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        db.session.delete(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Report deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/predictions', methods=['DELETE'])
def delete_all_predictions(patient_id):
    try:
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        deleted = Prediction.query.filter_by(patient_id=patient_id).delete()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'All {deleted} reports for {patient.name} deleted successfully',
            'deleted_count': deleted
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not models_loaded:
            return jsonify({'error': 'Models not loaded. Please run train.py first.'}), 500
        
        data = request.get_json()
        patient_id = data.get('patient_id')
        
        if not patient_id:
            return jsonify({'error': 'Patient ID required'}), 400
        
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404
        
        # Prepare features
        features = np.array([[
            patient.age,
            patient.blood_pressure,
            patient.cholesterol,
            patient.heart_rate,
            patient.bmi,
            patient.glucose,
            1 if patient.smoking else 0
        ]])
        
        features_scaled = scaler.transform(features)
        
        # Classical prediction
        c_probs = classical_model.predict_proba(features_scaled)[0]
        c_pred = int(np.argmax(c_probs))
        c_conf = float(np.max(c_probs))
        c_result = 'High Risk' if c_pred == 1 else 'Low Risk'
        
        # Quantum prediction
        q_probs = quantum_model.predict_proba(features_scaled)[0]
        q_pred = int(np.argmax(q_probs))
        q_conf = float(np.max(q_probs))
        q_result = 'High Risk' if q_pred == 1 else 'Low Risk'
        
        # Calculate final risk (weighted average)
        final_risk = (c_conf * c_pred + q_conf * q_pred) / (c_conf + q_conf) if (c_conf + q_conf) > 0 else 0
        
        # Determine risk level with proper thresholds
        if final_risk >= 0.65:
            risk_level = 'High'
        elif final_risk >= 0.40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Save prediction
        prediction = Prediction(
            patient_id=patient.id,
            classical_result=c_result,
            classical_confidence=c_conf,
            quantum_result=q_result,
            quantum_confidence=q_conf,
            final_risk=final_risk,
            risk_level=risk_level
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'patient_name': patient.name,
            'patient_age': patient.age,
            'classical': {
                'result': c_result,
                'confidence': round(c_conf, 4)
            },
            'quantum': {
                'result': q_result,
                'confidence': round(q_conf, 4)
            },
            'final_risk': round(final_risk, 4),
            'risk_level': risk_level,
            'prediction_id': prediction.id
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<int:patient_id>')
def get_patient_predictions(patient_id):
    try:
        predictions = Prediction.query.filter_by(patient_id=patient_id).order_by(Prediction.created_at.desc()).all()
        return jsonify([p.to_dict() for p in predictions])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def get_analytics():
    try:
        patients = Patient.query.all()
        predictions = Prediction.query.all()
        
        risk_dist = {'High': 0, 'Medium': 0, 'Low': 0}
        for p in predictions:
            risk_dist[p.risk_level] = risk_dist.get(p.risk_level, 0) + 1
        
        avg_c_conf = np.mean([p.classical_confidence for p in predictions]) if predictions else 0
        avg_q_conf = np.mean([p.quantum_confidence for p in predictions]) if predictions else 0
        
        return jsonify({
            'total_patients': len(patients),
            'total_predictions': len(predictions),
            'avg_classical_confidence': round(avg_c_conf, 4),
            'avg_quantum_confidence': round(avg_q_conf, 4),
            'disease_distribution': risk_dist,
            'predictions_over_time': {
                'labels': [p.created_at.strftime('%Y-%m-%d') for p in predictions[-30:]],
                'data': [1 for _ in predictions[-30:]]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        patients = Patient.query.all()
        predictions = Prediction.query.all()
        
        gender_dist = {}
        for p in patients:
            gender_dist[p.gender] = gender_dist.get(p.gender, 0) + 1
        
        age_dist = {'18-30': 0, '31-45': 0, '46-60': 0, '60+': 0}
        for p in patients:
            if p.age < 30:
                age_dist['18-30'] += 1
            elif p.age < 45:
                age_dist['31-45'] += 1
            elif p.age < 60:
                age_dist['46-60'] += 1
            else:
                age_dist['60+'] += 1
        
        smokers = sum(1 for p in patients if p.smoking)
        
        return jsonify({
            'total_patients': len(patients),
            'total_predictions': len(predictions),
            'unread_alerts': 0,
            'gender_distribution': gender_dist,
            'age_distribution': age_dist,
            'smokers': smokers,
            'non_smokers': len(patients) - smokers
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    return jsonify([])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)