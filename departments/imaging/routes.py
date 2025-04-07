import os
import uuid
import logging
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import pydicom
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from flask import  flash, redirect, render_template, request, url_for, current_app, send_from_directory
from flask_socketio import SocketIO
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from extensions import db
from departments.models.medicine import RequestedImage, Imaging, ImagingResult
from sqlalchemy.orm import joinedload
from . import bp
from app import socketio  # Import socketio from app.py
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

class MedicalImageModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=15):
        super(MedicalImageModel, self).__init__()
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
        x = self.fc1(x)
        return x

def load_models():
    models = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "med_image_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    logger.debug(f"Model path: {model_path}")

    try:
        models['image_model'] = MedicalImageModel(in_channels=1, num_classes=15).to(device)
        logger.debug("Image model initialized")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                models['image_model'].load_state_dict(state_dict, strict=False)
                logger.info(f"Custom model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load custom model: {e}")
                torch.save(models['image_model'].state_dict(), model_path)
                logger.info("Created new model weights")
        else:
            logger.info("No model found - creating new one")
            torch.save(models['image_model'].state_dict(), model_path)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        models['image_model'] = None

    try:
        models['report_tokenizer'] = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
        models['report_model'] = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large").to(device)
        logger.debug("BioGPT models loaded")
    except Exception as e:
        logger.warning(f"Failed to load BioGPT: {e}, trying ClinicalBERT")
        try:
            models['report_tokenizer'] = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            models['report_model'] = AutoModelForSeq2SeqLM.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
            logger.debug("ClinicalBERT models loaded")
        except Exception as e:
            logger.warning(f"Failed to load ClinicalBERT: {e}, using DistilBERT")
            models['report_tokenizer'] = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            models['report_model'] = AutoModelForSeq2SeqLM.from_pretrained("distilbert-base-uncased").to(device)
            logger.debug("DistilBERT models loaded")

    return models

models = load_models()

def allowed_file(filename):
    logger.debug(f"Checking if file {filename} is allowed")
    allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in {'dcm', 'dicom'}
    logger.debug(f"File {filename} allowed: {allowed}")
    return allowed

def validate_dicom_file(filepath):
    logger.debug(f"Validating DICOM file: {filepath}")
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        logger.debug(f"DICOM file read: {filepath}, SOPClassUID: {getattr(dicom, 'SOPClassUID', 'None')}")
        if not hasattr(dicom, 'PixelData'):
            logger.error(f"No PixelData in {filepath}")
            return False, "DICOM file has no pixel data"
        if dicom.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug(f"Decompressing {filepath}")
            dicom.decompress()
        _ = dicom.pixel_array
        logger.debug(f"DICOM validated: {filepath}, shape: {dicom.pixel_array.shape}")
        return True, "Valid DICOM image"
    except Exception as e:
        logger.error(f"Invalid DICOM file {filepath}: {e}")
        return False, f"Invalid DICOM file: {e}"

def get_modality_and_body_part(dicom):
    """Extract modality and body part from DICOM metadata."""
    modality = getattr(dicom, 'Modality', 'Unknown').upper()
    body_part = getattr(dicom, 'BodyPartExamined', 'Unknown').lower()
    
    modality_map = {
        'CR': 'X-ray', 'DX': 'X-ray', 'CT': 'CT Scan', 'MR': 'MRI', 
        'PT': 'PET', 'CY': 'Cytology'
    }
    modality = modality_map.get(modality, modality)
    
    if body_part == 'unknown' and hasattr(dicom, 'StudyDescription'):
        body_part = dicom.StudyDescription.lower() or 'unspecified region'
    
    return modality, body_part

def preprocess_dicom(dicom_path, target_size=64):
    logger.debug(f"Preprocessing DICOM file: {dicom_path}")
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        if dicom.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug(f"Decompressing {dicom_path}")
            dicom.decompress()
        
        image = dicom.pixel_array.astype(np.float32)
        logger.debug(f"Raw pixel array shape: {image.shape}")
        
        if len(image.shape) == 3:
            if image.shape[-1] in [3, 4]:
                logger.debug(f"Converting multi-channel to grayscale")
                image = np.mean(image, axis=-1)
            elif image.shape[0] > 1:
                logger.debug(f"Selecting middle slice for multi-slice image")
                image = image[image.shape[0] // 2]
        
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)
            logger.debug(f"No contrast in {dicom_path}, using zeros")
        
        image = np.expand_dims(image, axis=(0, 1))
        image = torch.from_numpy(image).float()
        image = torch.nn.functional.interpolate(
            image, size=(target_size, target_size), mode='bilinear', align_corners=False
        )
        logger.debug(f"Preprocessed shape: {image.shape}")
        
        modality, body_part = get_modality_and_body_part(dicom)
        return image.to(device), modality, body_part
    except Exception as e:
        logger.error(f"Preprocessing failed for {dicom_path}: {e}")
        raise

def analyze_dicom(dicom_path):
    logger.debug(f"Analyzing DICOM file: {dicom_path}")
    if not models['image_model']:
        logger.debug("Image model not available")
        return {"error": "AI analysis unavailable", "status": "error"}
    
    try:
        is_valid, message = validate_dicom_file(dicom_path)
        if not is_valid:
            logger.error(f"Validation failed for {dicom_path}: {message}")
            return {"error": message, "status": "error"}
        
        image, modality, body_part = preprocess_dicom(dicom_path)
        logger.debug(f"Image preprocessed: {dicom_path}, shape: {image.shape}, modality: {modality}, body_part: {body_part}")
        
        with torch.no_grad():
            outputs = models['image_model'](image)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            classes = [
                'Normal', 'Inflammation', 'Mass', 'Nodule', 'Cyst',
                'Fibrosis', 'Calcification', 'Edema', 'Tumor', 'Lesion',
                'Fracture', 'Effusion', 'Thickening', 'Infiltration', 'Abnormality'
            ]
            result = {
                'prediction': classes[pred.item()],
                'confidence': round(confidence.item(), 3),
                'all_probs': {c: round(p.item(), 3) for c, p in zip(classes, probs[0])},
                'status': 'success',
                'filename': os.path.basename(dicom_path),
                'modality': modality,
                'body_part': body_part
            }
            logger.debug(f"Analysis result for {dicom_path}: {result}")
            return result
    except Exception as e:
        logger.error(f"DICOM analysis error for {dicom_path}: {e}")
        return {"error": str(e), "status": "error"}

def generate_report(analysis_results, patient_id, result_id):
    """Generate a human-readable report for AI findings across all modalities and body parts."""
    logger.debug(f"Generating report from analysis results: {len(analysis_results)} items")
    successful = [r for r in analysis_results if r.get('status') == 'success']
    errors = [r for r in analysis_results if r.get('status') != 'success']
    logger.debug(f"Successful analyses: {len(successful)}, Errors: {len(errors)}")

    if not successful:
        return "No valid imaging results available."

    grouped_results = {}
    for res in successful:
        key = (res['modality'], res['body_part'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(res)

    findings = []
    for (modality, body_part), results in grouped_results.items():
        findings.append(f"{modality} Imaging of the {body_part.capitalize()}:")
        for i, res in enumerate(results, 1):
            primary = f"The primary finding is {res['prediction'].lower()}, identified with a confidence of {res['confidence']*100:.0f}%."
            secondary = [f"{condition.lower()} ({prob*100:.0f}% confidence)" 
                         for condition, prob in res['all_probs'].items() 
                         if prob > 0.1 and condition != res['prediction']]
            secondary_text = f"Other notable possibilities include {', '.join(secondary)}." if secondary else ""
            findings.append(
                f"  Image {i} ({res['filename']}): {primary} This suggests a possible {res['prediction'].lower()} in the {body_part}. {secondary_text}"
            )
        findings.append("")

    findings_text = "\n".join(findings)

    impression = []
    for (modality, body_part), results in grouped_results.items():
        max_conf = max(r['confidence'] for r in results)
        primary_finding = next(r['prediction'] for r in results if r['confidence'] == max_conf)
        impression.append(
            f"In the {modality} imaging of the {body_part}, the most significant finding is {primary_finding.lower()} "
            f"with a confidence of {max_conf*100:.0f}%, suggesting a potential abnormality in this region."
        )
    impression.append(
        "These AI-generated findings should be correlated with clinical symptoms, patient history, and additional diagnostic tests."
    )

    error_notes = ""
    if errors:
        error_notes = "\n\nProcessing Notes:\n" + "\n".join(
            f"- File {i+1}: {err.get('error', 'Unknown error')}" for i, err in enumerate(errors)
        )

    report = (
        "Radiology Report\n\n"
        f"Patient ID: {patient_id}\n"
        f"Date of Examination: {datetime.utcnow().strftime('%B %d, %Y')}\n"
        f"Examination Type: Multi-Modality Imaging (X-ray, CT Scan, MRI, PET, Cytology)\n"
        f"Result ID: {result_id}\n\n"
        "FINDINGS:\n"
        f"This report is based on the evaluation of {len(successful)} images across various modalities and body parts, "
        "analyzed using an artificial intelligence system:\n\n"
        f"{findings_text}{error_notes}\n\n"
        "IMPRESSION:\n" + "\n".join(f"{i+1}. {item}" for i, item in enumerate(impression)) + "\n\n"
        "Radiologist Review Recommended: Yes\n"
        "Reported By: AI Imaging System (Model Version 1.0)\n"
        f"Date Reported: {datetime.utcnow().strftime('%B %d, %Y')}"
    )
    logger.debug(f"Report generated: {report[:100]}...")
    return report

@bp.route('/process_imaging_request/<int:request_id>', methods=['GET', 'POST'])  # Fixed: Removed '/imaging' prefix
@login_required
def process_imaging_request(request_id):
    """Process an imaging request by uploading and analyzing DICOM files."""
    logger.debug(f"User {current_user.id} processing imaging request {request_id}")

    # Check user permissions
    if current_user.role not in ['imaging', 'admin']:
        logger.debug(f"Permission denied for user {current_user.id}")
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    # Fetch request and imaging data
    try:
        imaging_request = RequestedImage.query.get_or_404(request_id)
        imaging = Imaging.query.get_or_404(imaging_request.imaging_id)
        logger.debug(f"Loaded imaging request {request_id} and imaging {imaging.id}")
    except Exception as e:
        logger.error(f"Error fetching request {request_id}: {e}", exc_info=True)
        flash(f"Error loading request: {str(e)}", 'error')
        return redirect(url_for('imaging.index'))

    # Handle POST request (file upload and processing)
    if request.method == 'POST':
        logger.debug(f"POST request for imaging request {request_id}")

        # Validate file upload
        if 'dicom_folder' not in request.files or not request.files['dicom_folder']:
            logger.debug("No DICOM files uploaded")
            flash('No DICOM files uploaded', 'error')
            return redirect(request.url)

        # Prepare upload directory
        result_id = str(uuid.uuid4())
        upload_dir = os.path.join(current_app.config['DICOM_UPLOAD_FOLDER'], result_id)
        os.makedirs(upload_dir, exist_ok=True)
        logger.debug(f"Upload directory created: {upload_dir}")

        # Process uploaded files
        files = request.files.getlist('dicom_folder')
        total_files = len(files)
        file_paths = []
        analysis_results = []
        processed_count = 0
        logger.debug(f"Processing {total_files} files")

        for i, file in enumerate(files, 1):
            if not file or not allowed_file(file.filename):
                logger.debug(f"Skipping invalid file: {file.filename if file else 'None'}")
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            logger.debug(f"Processing file {i}/{total_files}: {filename}")

            try:
                file.save(filepath)
                result = analyze_dicom(filepath)
                result['filename'] = filename
                logger.debug(f"Analysis for {filename}: {result}")

                if result.get('status') == 'success':
                    file_paths.append(filepath)
                    processed_count += 1
                    logger.debug(f"Successfully processed {filename}")
                else:
                    os.remove(filepath)
                    logger.debug(f"Failed processing {filename}, removed")

                analysis_results.append(result)

                # Emit progress every 5 files or at the end
                if i % 5 == 0 or i == total_files:
                    try:
                        socketio.emit('progress', {
                            'current': processed_count,
                            'total': total_files,
                            'request_id': request_id
                        }, namespace='/imaging')
                        logger.debug(f"Progress emitted: {processed_count}/{total_files}")
                    except Exception as emit_error:
                        logger.error(f"Progress emit failed: {emit_error}")

            except Exception as file_error:
                logger.error(f"Error processing {filename}: {file_error}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                analysis_results.append({
                    'filename': filename,
                    'status': 'error',
                    'error': str(file_error)
                })

        # Check if any files were processed
        if not file_paths:
            logger.debug("No valid DICOM files processed")
            flash('No valid DICOM images were processed', 'error')
            return redirect(request.url)

        # Generate reports
        try:
            ai_report = generate_report(analysis_results, imaging_request.patient_id, result_id)
            final_report = request.form.get('result_notes', "AI-generated findings stored in AI Findings section.")
            logger.debug(f"AI report: {ai_report[:100]}...")
            logger.debug(f"Final report: {final_report[:100]}...")
        except Exception as report_error:
            logger.error(f"Report generation failed: {report_error}")
            ai_report = "Report generation failed. Raw AI findings available."
            final_report = "Report generation failed. Basic findings available."

        # Save results to database
        try:
            imaging_result = ImagingResult(
                result_id=result_id,
                patient_id=imaging_request.patient_id,
                imaging_id=imaging.id,
                test_date=datetime.utcnow(),
                result_notes=final_report,
                updated_by=current_user.id,
                dicom_file_path=",".join(file_paths),
                ai_findings=ai_report,
                ai_generated=True,
                files_processed=processed_count,
                files_failed=total_files - processed_count,
                processing_metadata={'file_metadata': analysis_results, 'model_version': '1.0'}
            )
            db.session.add(imaging_result)
            imaging_request.status = 'completed'
            imaging_request.result_id = result_id
            db.session.commit()
            logger.debug(f"Saved result: request_id={request_id}, result_id={result_id}")

            # Emit completion event
            try:
                socketio.emit('complete', {
                    'request_id': request_id,
                    'result_id': result_id,
                    'processed': processed_count,
                    'failed': total_files - processed_count
                }, namespace='/imaging')
                logger.debug(f"Completion emitted: processed={processed_count}, failed={total_files - processed_count}")
            except Exception as emit_error:
                logger.error(f"Completion emit failed: {emit_error}")

            flash(f"Processed {processed_count} of {total_files} files successfully", 'success')
            logger.debug(f"Redirecting to view_result: result_id={result_id}")
            return redirect(url_for('imaging.view_result', result_id=result_id))

        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database error: {db_error}", exc_info=True)
            flash('Error saving results to database', 'error')
            return redirect(request.url)

    # Handle GET request (render form)
    draft_report = ""
    if imaging_request.result_id:
        existing_result = ImagingResult.query.get(imaging_request.result_id)
        if existing_result:
            draft_report = existing_result.result_notes
            logger.debug(f"Draft report loaded: {draft_report[:100]}...")

    logger.debug(f"Rendering process.html for request_id={request_id}")
    return render_template(
        'imaging/process.html',
        imaging_request=imaging_request,
        imaging=imaging,
        draft_report=draft_report,
        ai_enabled=models['image_model'] is not None
    )
@bp.route('/view_result/<string:result_id>', methods=['GET'])
@login_required
def view_result(result_id):
    """View the imaging result for a given result_id."""
    logger.debug(f"Viewing result {result_id} for user {current_user.id}")
    if current_user.role not in ['imaging', 'admin']:
        flash('Permission denied', 'error')
        logger.debug(f"Permission denied for user {current_user.id}")
        return redirect(url_for('home'))

    try:
        imaging_result = ImagingResult.query.filter_by(result_id=result_id).first_or_404()
        imaging = Imaging.query.get_or_404(imaging_result.imaging_id)
        imaging_request = RequestedImage.query.filter_by(result_id=result_id).first_or_404()

        logger.debug(f"Rendering view.html for result_id={result_id}")
        return render_template(
            'imaging/view.html',
            imaging_result=imaging_result,
            imaging=imaging,
            imaging_request=imaging_request
        )
    except Exception as e:
        logger.error(f"Error viewing result {result_id}: {e}")
        flash(f"Error: {str(e)}", 'error')
        return redirect(url_for('imaging.index'))

@bp.route('/download/<string:result_id>/<path:filename>', methods=['GET'])
@login_required
def download_file(result_id, filename):
    """Serve a DICOM file for download."""
    logger.debug(f"Download request for result_id={result_id}, filename={filename} by user {current_user.id}")
    if current_user.role not in ['imaging', 'admin']:
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    upload_dir = os.path.join(current_app.config['DICOM_UPLOAD_FOLDER'], result_id)
    try:
        return send_from_directory(upload_dir, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        flash(f"Error downloading file: {str(e)}", 'error')
        return redirect(url_for('imaging.view_result', result_id=result_id))

@bp.route('/results', methods=['GET'])  # Changed from '/imaging_results'
@login_required
def imaging_results():
    """Display a list of all processed imaging results."""
    logger.debug(f"Accessing imaging results list for user {current_user.id}")
    
    if current_user.role not in ['medicine', 'imaging', 'admin']:
        flash('Permission denied', 'error')
        logger.debug(f"Permission denied for user {current_user.id}, role={current_user.role}")
        return redirect(url_for('home'))

    try:
        results = ImagingResult.query.order_by(ImagingResult.test_date.desc()).all()
        logger.debug(f"Retrieved {len(results)} imaging results from database")
        
        return render_template(
            'imaging/imaging_results.html',
            results=results
        )
    except Exception as e:
        logger.error(f"Error retrieving imaging results: {str(e)}", exc_info=True)
        flash(f'Error retrieving results: {str(e)}', 'error')
        return redirect(url_for('imaging.index'))

@bp.route('/view_imaging_results/<string:result_id>', methods=['GET'])
@login_required
def view_imaging_results(result_id):
    logger.debug(f"User {current_user.id} viewing results for result_id={result_id}")

    allowed_roles = {'medicine', 'imaging', 'admin'}
    if current_user.role not in allowed_roles:
        logger.debug(f"Permission denied for user {current_user.id}, role={current_user.role}")
        flash('You do not have permission to view imaging results.', 'error')
        return redirect(url_for('home'))

    try:
        imaging_result = ImagingResult.query.filter_by(result_id=result_id).first_or_404()
        imaging = Imaging.query.get(imaging_result.imaging_id) if imaging_result.imaging_id else None
        if not imaging:
            logger.warning(f"No Imaging record found for imaging_id={imaging_result.imaging_id}")
        
        logger.debug(f"Found imaging_result: result_id={imaging_result.result_id}, patient_id={imaging_result.patient_id}")

        file_paths = imaging_result.dicom_file_path.split(',') if imaging_result.dicom_file_path else []
        valid_file_paths = [path for path in file_paths if os.path.exists(path)]
        if len(valid_file_paths) < len(file_paths):
            missing_files = set(file_paths) - set(valid_file_paths)
            logger.warning(f"Missing DICOM files: {missing_files}")
            flash(f"Warning: {len(missing_files)} DICOM file(s) could not be found on the server.", 'warning')

        file_index = request.args.get('file_index', type=int)
        action = request.args.get('action', 'view')
        if file_index is not None:
            if 0 <= file_index < len(valid_file_paths):
                file_path = valid_file_paths[file_index]
                logger.debug(f"Serving DICOM file: {file_path} with action={action}")
                return send_from_directory(
                    directory=os.path.dirname(file_path),
                    path=os.path.basename(file_path),
                    mimetype='application/dicom',
                    as_attachment=(action == 'download'),
                    download_name=os.path.basename(file_path) if action == 'download' else None
                )
            else:
                logger.debug(f"Invalid file_index={file_index}, range: 0 to {len(valid_file_paths)-1}")
                flash('Invalid file index selected.', 'error')

        ai_findings = imaging_result.ai_findings or ""
        header = {}
        findings = []
        impression = []
        footer = {}

        lines = ai_findings.split('\n')
        current_section = None
        current_modality = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Radiology Report'):
                current_section = 'header'
            elif line.startswith('FINDINGS:'):
                current_section = 'findings_intro'
            elif line.startswith('IMPRESSION:'):
                current_section = 'impression'
            elif current_section == 'header' and ':' in line:
                key, value = line.split(':', 1)
                header[key.strip()] = value.strip()
            elif current_section == 'findings_intro' and not line.startswith('Image'):
                findings.append({'intro': line})
                current_section = 'findings'
            elif current_section == 'findings' and line.endswith(':') and not line.startswith('Image'):
                current_modality = line
                findings.append({'modality': current_modality})
            elif current_section == 'findings' and line.startswith('Image'):
                entry = {'text': line}
                if current_modality:
                    entry['modality'] = current_modality
                findings.append(entry)
            elif current_section == 'findings' and findings and 'text' in findings[-1]:
                findings[-1]['text'] += f"\n{line}"
            elif current_section == 'impression' and line[0].isdigit():
                impression.append(line)
            elif current_section != 'impression' and ':' in line:
                key, value = line.split(':', 1)
                footer[key.strip()] = value.strip()

        imaging_type = imaging.imaging_type if imaging else 'Unknown'
        logger.debug(f"Imaging type: {imaging_type}, Parsed AI findings: header={header}, findings={len(findings)}, impression={len(impression)}")

        logger.debug(f"Rendering view.html for result_id={result_id}")
        return render_template(
            'imaging/view.html',
            imaging_result=imaging_result,
            file_paths=valid_file_paths,
            imaging_type=imaging_type,
            ai_header=header,
            ai_findings=findings,
            ai_impression=impression,
            ai_footer=footer
        )

    except Exception as e:
        logger.error(f"Error viewing result_id={result_id}: {str(e)}", exc_info=True)
        flash(f"An error occurred while loading the imaging results: {str(e)}", 'error')
        return redirect(url_for('imaging.index'))
@bp.route('/', methods=['GET'])
@login_required
def index():
    """Display imaging waiting list"""
    if current_user.role not in ['imaging', 'admin']: 
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    try:
        pending_requests = RequestedImage.query.filter_by(status=0).options(
            joinedload(RequestedImage.patient),
            joinedload(RequestedImage.imaging)
        ).all()

        return render_template(
            'imaging/index.html',
            pending_requests=pending_requests or [],
            models_loaded=bool(models['image_model'])
        )
    except Exception as e:
        flash(f'Database error: {str(e)}', 'error')
        return redirect(url_for('home'))

