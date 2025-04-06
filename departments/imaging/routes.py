from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, current_app, send_from_directory
from flask_login import login_required, current_user
from extensions import db
from datetime import datetime
import uuid
import os
from . import bp  # Import the blueprint
from sqlalchemy.orm import joinedload
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from flask_socketio import SocketIO
import pydicom
import torch
import torch.nn as nn
from joblib import dump, load
import logging
from threading import Thread
from functools import lru_cache
import warnings
from departments.models.records import Patient, PatientWaitingList
from departments.models.medicine import (RequestedImage,Imaging,ImagingResult
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app and logging
app = Flask(__name__)
socketio = SocketIO(app)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model cache directory
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

class MedicalImageModel(nn.Module):
    """Optimized medical image classification model"""
    def __init__(self, in_channels=1, num_classes=15):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.layer1(x))))
        x = self.pool(torch.relu(self.bn2(self.layer2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

class ModelManager:
    """Singleton class to manage model loading and caching"""
    _instance = None
    _models = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # Start model loading in background
            Thread(target=self._preload_models, daemon=True).start()
    
    def _preload_models(self):
        """Preload models in background"""
        self.get_image_model()
        self.get_report_model()
    
    @lru_cache(maxsize=1)
    def get_image_model(self):
        """Get image model with caching"""
        model_path = os.path.join(os.path.dirname(__file__), "models", "med_image_model.pth")
        model = MedicalImageModel().to(device)
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info("Loaded custom image model")
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                torch.save(model.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        
        model.eval()
        return model
    
    @lru_cache(maxsize=1)
    def get_report_model(self):
        """Get report generation model with caching"""
        try:
            # Try to load BioGPT first
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BioGPT-Large",
                cache_dir=MODEL_CACHE_DIR
            )
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/BioGPT-Large",
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info("Loaded BioGPT model")
            return {'tokenizer': tokenizer, 'model': model}
        except Exception as e:
            logger.warning(f"Failed to load BioGPT: {e}, falling back to DistilBERT")
            tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased",
                cache_dir=MODEL_CACHE_DIR
            )
            model = AutoModel.from_pretrained(
                "distilbert-base-uncased",
                cache_dir=MODEL_CACHE_DIR
            ).to(device)
            return {'tokenizer': tokenizer, 'model': model}

# Initialize model manager
model_manager = ModelManager()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'dcm', 'dicom'}

@lru_cache(maxsize=32)
def validate_dicom_file(filepath):
    """More robust DICOM validation with detailed error reporting"""
    try:
        # First quick check - is it a DICOM file at all?
        with open(filepath, 'rb') as f:
            preamble = f.read(128)
            prefix = f.read(4)
            if prefix != b'DICM':
                return False, "Not a valid DICOM file (missing DICM prefix)"
        
        # Now try to parse the file
        dicom = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
        
        # Check minimum required attributes
        required_tags = [
            'SOPClassUID', 'SOPInstanceUID', 'Modality',
            'StudyDate', 'SeriesInstanceUID'
        ]
        
        missing_tags = [tag for tag in required_tags if not hasattr(dicom, tag)]
        if missing_tags:
            return False, f"Missing required DICOM tags: {', '.join(missing_tags)}"
        
        # Check if image data exists
        if not hasattr(dicom, 'PixelData'):
            return False, "DICOM file contains no pixel data"
            
        return True, "Valid DICOM file"
        
    except pydicom.errors.InvalidDicomError:
        return False, "Invalid DICOM file structure"
    except IOError:
        return False, "Could not read file"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"
    
def preprocess_dicom(dicom_path, target_size=64):
    """More robust DICOM preprocessing with better error handling"""
    try:
        # Read with forced re-encoding if needed
        dicom = pydicom.dcmread(dicom_path, force=True)
        
        # Handle compressed DICOM
        if hasattr(dicom, 'file_meta') and hasattr(dicom.file_meta, 'TransferSyntaxUID'):
            if dicom.file_meta.TransferSyntaxUID.is_compressed:
                try:
                    dicom.decompress()
                except:
                    logger.warning("Could not decompress DICOM, trying raw pixel data")
        
        # Get pixel data
        if not hasattr(dicom, 'PixelData'):
            raise ValueError("DICOM file contains no pixel data")
            
        image = dicom.pixel_array.astype(np.float32)
        
        # Handle multi-frame/multi-channel images
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # RGB?
                image = np.mean(image, axis=0)  # Convert to grayscale
            else:
                image = image[image.shape[0] // 2]  # Take middle slice
        
        # Normalize
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            logger.warning("Image has no contrast, using zeros")
            image = np.zeros_like(image)
        
        # Convert to tensor and resize
        image = torch.from_numpy(np.expand_dims(image, axis=(0, 1))).float()
        image = torch.nn.functional.interpolate(
            image, size=(target_size, target_size), mode='bilinear'
        )
        
        # Get metadata
        modality = getattr(dicom, 'Modality', 'Unknown')
        body_part = getattr(dicom, 'BodyPartExamined', 
                          getattr(dicom, 'StudyDescription', 'Unknown'))
        
        return image.to(device), modality, body_part
        
    except Exception as e:
        logger.error(f"Failed to preprocess {dicom_path}: {str(e)}")
        raise ValueError(f"DICOM processing failed: {str(e)}")

def analyze_dicom(dicom_path):
    """Optimized DICOM analysis"""
    try:
        is_valid, message = validate_dicom_file(dicom_path)
        if not is_valid:
            return {"error": message, "status": "error"}
        
        image, modality, body_part = preprocess_dicom(dicom_path)
        model = model_manager.get_image_model()
        
        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            
            classes = [
                'Normal', 'Inflammation', 'Mass', 'Nodule', 'Cyst',
                'Fibrosis', 'Calcification', 'Edema', 'Tumor', 'Lesion',
                'Fracture', 'Effusion', 'Thickening', 'Infiltration', 'Abnormality'
            ]
            
            return {
                'prediction': classes[pred.item()],
                'confidence': round(confidence.item(), 3),
                'modality': modality,
                'body_part': body_part,
                'status': 'success'
            }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e), "status": "error"}

def generate_report(analysis_results, patient_id, result_id):
    """Optimized report generation"""
    successful = [r for r in analysis_results if r.get('status') == 'success']
    if not successful:
        return "No valid results"
    
    # Group results by modality and body part
    grouped = {}
    for res in successful:
        key = (res['modality'], res['body_part'])
        grouped.setdefault(key, []).append(res)
    
    # Generate findings
    findings = []
    for (modality, body_part), results in grouped.items():
        findings.append(f"{modality} of {body_part}:")
        for res in results:
            findings.append(
                f"- Found {res['prediction']} (confidence: {res['confidence']*100:.0f}%)"
            )
    
    return "\n".join([
        f"Report for {patient_id}",
        f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
        "\nFINDINGS:",
        *findings,
        "\nIMPRESSION:",
        "Clinical correlation recommended."
    ])


@bp.route('/process_imaging_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
def process_imaging_request(request_id):
    if current_user.role not in {'imaging', 'admin'}:
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    try:
        imaging_request = RequestedImage.query.get_or_404(request_id)
        imaging = Imaging.query.get_or_404(imaging_request.imaging_id)

        if request.method == 'POST':
            # Handle both single file and folder uploads
            uploaded_files = request.files.getlist('image_files')
            if not uploaded_files:
                flash('No files uploaded', 'error')
                return redirect(request.url)
            
            result_id = str(uuid.uuid4())
            dicom_upload_folder = current_app.config['DICOM_UPLOAD_FOLDER']
            os.makedirs(dicom_upload_folder, exist_ok=True)
            
            valid_files = []
            analysis_results = []
            
            for uploaded_file in uploaded_files:
                if not allowed_file(uploaded_file.filename):
                    continue
                    
                filename = secure_filename(f"{result_id}_{uploaded_file.filename}")
                file_path = os.path.join(dicom_upload_folder, filename)
                
                try:
                    uploaded_file.save(file_path)
                    is_valid, message = validate_dicom_file(file_path)
                    
                    if is_valid:
                        valid_files.append(file_path)
                        analysis_results.append(analyze_dicom(file_path))
                    else:
                        os.remove(file_path)
                        logger.warning(f"Invalid DICOM {filename}: {message}")
                        
                except Exception as e:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    logger.error(f"Error processing {filename}: {str(e)}")
            
            if not valid_files:
                flash('No valid DICOM files found', 'error')
                return redirect(request.url)
            
            # Generate combined report
            report = generate_report(analysis_results, imaging_request.patient_id, result_id)
            
            # Save results
            imaging_result = ImagingResult(
                result_id=result_id,
                patient_id=imaging_request.patient_id,
                imaging_id=imaging.id,
                test_date=datetime.utcnow(),
                result_notes=request.form.get('result_notes', ''),
                updated_by=current_user.id,
                dicom_file_path=','.join(valid_files),  # Store all file paths
                ai_findings=report
            )
            
            db.session.add(imaging_result)
            imaging_request.status = 1
            imaging_request.result_id = result_id
            db.session.commit()
            
            flash(f'Processed {len(valid_files)} DICOM files successfully', 'success')
            return redirect(url_for('imaging.index'))

        return render_template(
            'imaging/process.html',
            imaging_request=imaging_request,
            imaging=imaging
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error processing request {request_id}: {str(e)}", exc_info=True)
        flash(f'Error processing request: {str(e)}', 'error')
        return redirect(url_for('imaging.index'))

@bp.route('/view_imaging_results/<string:result_id>', methods=['GET'])
@login_required
def view_imaging_results(result_id):
    """Optimized viewer for imaging results"""
    if current_user.role not in {'medicine', 'imaging', 'admin'}:
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    try:
        imaging_result = ImagingResult.query.filter_by(result_id=result_id).first_or_404()
        imaging = Imaging.query.get(imaging_result.imaging_id) if imaging_result.imaging_id else None
        
        # Handle file download/view requests
        file_index = request.args.get('file_index', type=int)
        action = request.args.get('action', 'view')
        if file_index is not None:
            return _handle_file_request(imaging_result, file_index, action)
        
        return _render_results_view(imaging_result, imaging)

    except Exception as e:
        logger.error(f"Error viewing results {result_id}: {str(e)}", exc_info=True)
        flash(f'Error loading results: {str(e)}', 'error')
        return redirect(url_for('imaging.index'))

def _handle_file_request(imaging_result, file_index, action):
    """Handle DICOM file viewing/downloading"""
    file_paths = imaging_result.dicom_file_path.split(',') if imaging_result.dicom_file_path else []
    if not 0 <= file_index < len(file_paths):
        flash('Invalid file index', 'error')
        return redirect(url_for('imaging.view_imaging_results', result_id=imaging_result.result_id))
    
    file_path = file_paths[file_index]
    if not os.path.exists(file_path):
        flash('File not found', 'error')
        return redirect(url_for('imaging.view_imaging_results', result_id=imaging_result.result_id))
    
    return send_from_directory(
        directory=os.path.dirname(file_path),
        path=os.path.basename(file_path),
        mimetype='application/dicom',
        as_attachment=(action == 'download'),
        download_name=os.path.basename(file_path) if action == 'download' else None
    )

def _render_results_view(imaging_result, imaging):
    """Render the results view template with parsed data"""
    # Parse the AI findings report
    parsed_report = _parse_ai_findings(imaging_result.ai_findings or "")
    
    return render_template(
        'imaging/view.html',
        imaging_result=imaging_result,
        file_paths=imaging_result.dicom_file_path.split(',') if imaging_result.dicom_file_path else [],
        imaging_type=imaging.imaging_type if imaging else 'Unknown',
        **parsed_report
    )

@lru_cache(maxsize=32)
def _parse_ai_findings(ai_findings):
    """Parse the AI findings report with caching"""
    header = {}
    findings = []
    impression = []
    footer = {}
    
    if not ai_findings:
        return {
            'ai_header': header,
            'ai_findings': findings,
            'ai_impression': impression,
            'ai_footer': footer
        }
    
    lines = ai_findings.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('Radiology Report'):
            current_section = 'header'
        elif line.startswith('FINDINGS:'):
            current_section = 'findings'
        elif line.startswith('IMPRESSION:'):
            current_section = 'impression'
        elif current_section == 'header' and ':' in line:
            key, value = line.split(':', 1)
            header[key.strip()] = value.strip()
        elif current_section == 'findings':
            findings.append(line)
        elif current_section == 'impression':
            impression.append(line)
        elif ':' in line:
            key, value = line.split(':', 1)
            footer[key.strip()] = value.strip()
    
    return {
        'ai_header': header,
        'ai_findings': findings,
        'ai_impression': impression,
        'ai_footer': footer
    }

@bp.route('/imaging_results', methods=['GET'])
@login_required
def imaging_results():
    """Optimized listing of all imaging results"""
    if current_user.role not in {'medicine', 'imaging', 'admin'}:
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    try:
        results = ImagingResult.query.options(
            db.joinedload(ImagingResult.imaging)
        ).order_by(ImagingResult.test_date.desc()).all()
        
        return render_template(
            'imaging/imaging_results.html',
            results=results
        )
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}", exc_info=True)
        flash(f'Error retrieving results: {str(e)}', 'error')
        return redirect(url_for('imaging.index'))
    
 # index
@bp.route('/', methods=['GET'])
@login_required
def index():
    """Displays the imaging waiting list with pending imaging requests."""
    if current_user.role not in ['imaging', 'admin']: 
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all pending imaging requests (status=0)
        pending_requests = RequestedImage.query.filter_by(status=0).options(
            joinedload(RequestedImage.patient),
            joinedload(RequestedImage.imaging)  # Eager load imaging details
        ).all()

        # If no pending requests exist, inform the user
        if not pending_requests:
            flash('No pending imaging requests at the moment.', 'info')

        return render_template(
            'imaging/index.html',
            pending_requests=pending_requests
        )

    except Exception as e:
        flash(f'Error fetching pending imaging requests: {e}', 'error')
        print(f"Debug: Error in imaging.index: {e}")  # Debugging
        return redirect(url_for('home'))
        