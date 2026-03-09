from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import uuid
from app import db
from app.models import Prediction

# Import inference functions which will be implemented later
# from models.inference import predict_image

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get user's past predictions
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.date_posted.desc()).all()
    
    # Calculate simple stats
    total_scans = len(predictions)
    pneumonia_count = sum(1 for p in predictions if p.prediction_result == 'Pneumonia')
    normal_count = total_scans - pneumonia_count
    
    return render_template('dashboard.html', 
                           predictions=predictions, 
                           total=total_scans, 
                           pneumonia=pneumonia_count, 
                           normal=normal_count)

@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Generate unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Extract Clinical Patient Data
            p_name = request.form.get('patient_name', 'Unknown')
            p_age = request.form.get('patient_age', type=int)
            p_gender = request.form.get('patient_gender', 'Not Specified')
            p_symptoms = request.form.get('patient_symptoms', 'None')
            p_smoking = request.form.get('patient_smoking_history', 'Not Specified')
            
            # Ensure imports
            from models.inference import process_and_predict

            # Run inference and Grad-CAM
            heatmap_filename = f"cam_{unique_filename}"
            heatmap_save_path = os.path.join(current_app.config['HEATMAP_FOLDER'], heatmap_filename)
            
            try:
                prediction_result, confidence, severity, left_affected, right_affected = process_and_predict(filepath, heatmap_save_path)
            except Exception as e:
                flash(f'Error during inference: {str(e)}', 'danger')
                return redirect(request.url)

            # Create prediction record
            new_pred = Prediction(
                patient_name=p_name,
                patient_age=p_age,
                patient_gender=p_gender,
                patient_symptoms=p_symptoms,
                patient_smoking_history=p_smoking,
                image_filename=unique_filename,
                heatmap_filename=heatmap_filename,
                prediction_result=prediction_result,
                confidence=confidence,
                severity=severity,
                left_lung_affected=left_affected,
                right_lung_affected=right_affected,
                user_id=current_user.id
            )
            db.session.add(new_pred)
            db.session.commit()
            
            # Redirect to result page
            return redirect(url_for('main.result', pred_id=new_pred.id))
            
        else:
            flash('Allowed image types are -> png, jpg, jpeg', 'danger')
            return redirect(request.url)
            
    return render_template('upload.html')

@main_bp.route('/result/<int:pred_id>')
@login_required
def result(pred_id):
    prediction = Prediction.query.get_or_404(pred_id)
    
    # Ensure a user can only view their own predictions
    if prediction.author != current_user:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('main.dashboard'))
        
    return render_template('result.html', prediction=prediction)

@main_bp.route('/performance')
@login_required
def performance():
    """Renders the static model performance and metrics dashboard."""
    return render_template('performance.html')
