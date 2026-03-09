from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(100), nullable=False)
    heatmap_filename = db.Column(db.String(100), nullable=True)
    prediction_result = db.Column(db.String(20), nullable=False) # 'Normal' or 'Pneumonia'
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(20), nullable=True) # 'None', 'Mild', 'Moderate', 'Severe'
    left_lung_affected = db.Column(db.Float, nullable=True)
    right_lung_affected = db.Column(db.Float, nullable=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Clinical Patient Info
    patient_name = db.Column(db.String(100), nullable=True)
    patient_age = db.Column(db.Integer, nullable=True)
    patient_gender = db.Column(db.String(20), nullable=True)
    patient_symptoms = db.Column(db.String(255), nullable=True)
    patient_smoking_history = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f"Prediction('{self.image_filename}', '{self.prediction_result}', {self.confidence:.2f})"
