import os
import uuid
import logging
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydicom
from torchvision import models  # Explicitly import torchvision.models
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import flash, redirect, render_template, request, url_for, current_app, send_from_directory
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from extensions import db
from departments.models.medicine import RequestedImage, Imaging, ImagingResult
from sqlalchemy.orm import joinedload
from . import bp
from app import socketio

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

def load_models():
    models_dict = {}  # Renamed to avoid shadowing 'models' from torchvision
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    model_path = os.path.join(model_dir, "med_image_model.pth")
    os.makedirs(model_dir, exist_ok=True)
    logger.debug(f"Model directory: {model_dir}, Model path: {model_path}")

    # Load pre-trained ResNet18 and adapt for medical imaging
    try:
        # Load ResNet18 pre-trained on ImageNet
        resnet = models.resnet18(pretrained=True)  # Use torchvision.models explicitly
        
        # Modify the first layer to accept 1-channel input (grayscale DICOM)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final fully connected layer for 15 classes
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 15)
        
        models_dict['image_model'] = resnet.to(device)
        logger.debug("Initialized ResNet18 with modified input and output layers")

        # Load fine-tuned weights if available, otherwise use pre-trained ImageNet weights
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                state_dict = checkpoint.get('state_dict', checkpoint)
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                models_dict['image_model'].load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            except RuntimeError as e:
                logger.warning(f"Failed to load fine-tuned weights: {e}. Using pre-trained ImageNet weights.")
                torch.save({'state_dict': models_dict['image_model'].state_dict()}, model_path)
        else:
            logger.info(f"No fine-tuned weights at {model_path}. Using pre-trained ImageNet weights and saving initial state.")
            torch.save({'state_dict': models_dict['image_model'].state_dict()}, model_path)

        models_dict['image_model'].eval()
    except Exception as e:
        logger.error(f"Image model initialization failed: {e}")
        models_dict['image_model'] = None
        flash("AI image analysis unavailable due to model loading failure.", "error")

    # Load language model for report generation
    try:
        models_dict['report_tokenizer'] = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
        models_dict['report_model'] = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large").to(device)
        logger.info("Loaded microsoft/BioGPT-Large for report generation")
    except Exception as e:
        logger.warning(f"Failed to load BioGPT: {e}")
        models_dict['report_tokenizer'] = None
        models_dict['report_model'] = None

    return models_dict

models = load_models()

def allowed_file(filename):
    logger.debug(f"Checking file: {filename}")
    allowed = '.' in filename and filename.rsplit('.', 1)[1].lower() in {'dcm', 'dicom'}
    logger.debug(f"File {filename} allowed: {allowed}")
    return allowed

def validate_dicom_file(filepath):
    logger.debug(f"Validating DICOM: {filepath}")
    try:
        dicom = pydicom.dcmread(filepath, force=True)
        if not hasattr(dicom, 'PixelData'):
            logger.error(f"No PixelData in {filepath}")
            return False, "DICOM file has no pixel data"
        if dicom.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug(f"Decompressing {filepath}")
            dicom.decompress()
        _ = dicom.pixel_array
        logger.debug(f"Validated DICOM: {filepath}, shape: {dicom.pixel_array.shape}")
        return True, "Valid DICOM image"
    except Exception as e:
        logger.error(f"DICOM validation failed for {filepath}: {e}")
        return False, f"Invalid DICOM file: {e}"

def get_modality_and_body_part(dicom):
    modality = getattr(dicom, 'Modality', 'Unknown').upper()
    body_part = getattr(dicom, 'BodyPartExamined', 'Unknown').lower()
    modality_map = {'CR': 'X-ray', 'DX': 'X-ray', 'CT': 'CT Scan', 'MR': 'MRI', 'PT': 'PET', 'CY': 'Cytology'}
    modality = modality_map.get(modality, modality)
    if body_part == 'unknown' and hasattr(dicom, 'StudyDescription'):
        body_part = dicom.StudyDescription.lower() or 'unspecified region'
    
    patient_info = {
        'patient_id': getattr(dicom, 'PatientID', 'Unknown'),
        'patient_name': str(getattr(dicom, 'PatientName', 'Unknown')),
        'patient_sex': getattr(dicom, 'PatientSex', 'Unknown'),
        'patient_birth_date': getattr(dicom, 'PatientBirthDate', 'Unknown'),
        'study_date': getattr(dicom, 'StudyDate', datetime.utcnow().strftime('%Y%m%d'))
    }
    logger.debug(f"Extracted: modality={modality}, body_part={body_part}")
    return modality, body_part, patient_info

def preprocess_dicom(dicom_path, target_size=224):  # ResNet expects 224x224
    logger.debug(f"Preprocessing DICOM: {dicom_path}")
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        if dicom.file_meta.TransferSyntaxUID.is_compressed:
            logger.debug(f"Decompressing {dicom_path}")
            dicom.decompress()
        
        image = dicom.pixel_array.astype(np.float32)
        logger.debug(f"Raw image shape: {image.shape}")
        
        if len(image.shape) == 3:
            if image.shape[-1] in [3, 4]:
                image = np.mean(image, axis=-1)
            elif image.shape[0] > 1:
                image = image[image.shape[0] // 2]
        elif len(image.shape) > 3:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)
            logger.warning(f"No contrast in {dicom_path}, using zeroed image")
        
        image = np.expand_dims(image, axis=(0, 1))  # [1, 1, H, W]
        image = torch.from_numpy(image).float()
        image = F.interpolate(image, size=(target_size, target_size), mode='bilinear', align_corners=False)
        logger.debug(f"Preprocessed shape: {image.shape}")
        
        modality, body_part, patient_info = get_modality_and_body_part(dicom)
        return image.to(device), modality, body_part, patient_info
    except Exception as e:
        logger.error(f"Preprocessing failed for {dicom_path}: {e}")
        raise

def analyze_dicom(dicom_path):
    logger.debug(f"Analyzing DICOM: {dicom_path}")
    if not models['image_model']:
        logger.error("Image model unavailable")
        return {"error": "AI analysis unavailable", "status": "error"}
    
    try:
        is_valid, message = validate_dicom_file(dicom_path)
        if not is_valid:
            logger.error(f"Validation failed: {message}")
            return {"error": message, "status": "error"}
        
        image, modality, body_part, patient_info = preprocess_dicom(dicom_path)
        logger.debug(f"Preprocessed: shape={image.shape}, modality={modality}, body_part={body_part}")
        
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
                'confidence': round(confidence.item() * 100, 1),
                'all_probs': {c: round(p.item() * 100, 1) for c, p in zip(classes, probs[0]) if p.item() > 0.05},
                'status': 'success',
                'filename': os.path.basename(dicom_path),
                'modality': modality,
                'body_part': body_part,
                'patient_info': patient_info
            }
            logger.debug(f"Analysis result: {result}")
            return result
    except Exception as e:
        logger.error(f"Analysis failed for {dicom_path}: {e}")
        return {"error": str(e), "status": "error"}

def generate_report(analysis_results, patient_id, result_id):
    logger.debug(f"Generating report from {len(analysis_results)} results")
    successful = [r for r in analysis_results if r.get('status') == 'success']
    errors = [r for r in analysis_results if r.get('status') != 'success']
    logger.debug(f"Successful: {len(successful)}, Errors: {len(errors)}")

    if not successful:
        return "Radiology Report\n" + "-" * 50 + "\n\nNo valid imaging results available.\n\n" + "-" * 50

    patient_info = successful[0]['patient_info']
    patient_id = patient_info['patient_id']
    study_date = datetime.strptime(patient_info['study_date'], '%Y%m%d').strftime('%B %d, %Y') if patient_info['study_date'] != 'Unknown' else datetime.utcnow().strftime('%B %d, %Y')

    report = "Radiology Report\n" + "-" * 50 + "\n\n"
    report += "Patient Information\n"
    report += f"  Patient ID: {patient_id}\n"
    report += f"  Patient Name: {patient_info['patient_name']}\n"
    report += f"  Sex: {patient_info['patient_sex']}\n"
    report += f"  Date of Birth: {patient_info['patient_birth_date']}\n"
    report += f"  Date of Examination: {study_date}\n"
    report += "  Examination Type: Multi-Modality Imaging (X-ray, CT, MRI, PET, Cytology)\n"
    report += f"  Result ID: {result_id}\n\n"

    findings = {}
    for res in successful:
        finding = res['prediction']
        findings[finding] = findings.get(finding, []) + [(res['filename'], res['confidence'], res.get('all_probs', {}))]

    report += "Findings\n"
    total_images = len(successful)
    report += f"  This report is based on an AI-assisted evaluation of {total_images} DICOM images:\n\n"

    normal_images = findings.pop('Normal', [])
    normal_count = len(normal_images)
    report += f"  - Normal Findings: {normal_count} images ({normal_count/total_images*100:.1f}%)\n"
    if normal_images:
        confidences = [conf for _, conf, _ in normal_images]
        report += f"    - Confidence range: {min(confidences):.1f}%–{max(confidences):.1f}% (median: {sorted(confidences)[len(confidences)//2]:.1f}%)\n"

    abnormal_count = total_images - normal_count
    report += f"  - Abnormal Findings: {abnormal_count} images ({abnormal_count/total_images*100:.1f}%)\n"
    for finding, images in sorted(findings.items(), key=lambda x: len(x[1]), reverse=True):
        confidences = [conf for _, conf, _ in images]
        report += f"    - {finding}s: {len(images)} images ({len(images)/total_images*100:.1f}%)\n"
        report += f"      - Confidence range: {min(confidences):.1f}%–{max(confidences):.1f}% (median: {sorted(confidences)[len(confidences)//2]:.1f}%)\n"

    report += "\nSummary\n"
    report += f"  - Total Images Analyzed: {total_images}\n"
    report += f"  - Normal: {normal_count} ({normal_count/total_images*100:.1f}%)\n"
    report += f"  - Abnormal: {abnormal_count} ({abnormal_count/total_images*100:.1f}%)\n"

    report += "\nImpression and Recommendations\n"
    max_conf = max((r['confidence'] for r in successful), default=0)
    max_res = next((r for r in successful if r['confidence'] == max_conf), None)
    impression = [
        f"  1. Significant abnormal findings in {abnormal_count/total_images*100:.1f}% of images.",
        f"  2. Highest confidence finding: {max_res['prediction'].lower()} at {max_conf:.1f}% ({max_res['filename']}).",
        "  3. Recommendations:\n"
        "     - Clinical correlation required for all abnormal findings.\n"
        "     - Consider targeted imaging for confirmation.\n"
        "     - Radiologist review recommended."
    ]
    report += "\n".join(impression) + "\n"

    if errors:
        report += "\nProcessing Notes\n"
        report += "\n".join([f"  - Error: {e['filename']} - {e.get('error', 'Unknown error')}" for e in errors]) + "\n"

    report += "\n" + "-" * 50 + "\n"
    report += "Generated by: AI Imaging System v1.0 (ResNet18)\n"
    report += f"Date Generated: {datetime.utcnow().strftime('%B %d, %Y')}\n"

    logger.debug(f"Report generated: {report[:100]}...")
    return report

@bp.route('/process_imaging_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
def process_imaging_request(request_id):
    logger.debug(f"User {current_user.id} processing imaging request {request_id}")

    if current_user.role not in ['imaging', 'admin']:
        logger.debug(f"Permission denied for user {current_user.id}")
        flash('Permission denied', 'error')
        return redirect(url_for('home'))

    try:
        imaging_request = RequestedImage.query.get_or_404(request_id)
        imaging = Imaging.query.get_or_404(imaging_request.imaging_id)
        logger.debug(f"Loaded imaging request {request_id} and imaging {imaging.id}")
    except Exception as e:
        logger.error(f"Error fetching request {request_id}: {e}", exc_info=True)
        flash(f"Error loading request: {str(e)}", 'error')
        return redirect(url_for('imaging.index'))

    if request.method == 'POST':
        logger.debug(f"POST request for imaging request {request_id}")

        if 'dicom_folder' not in request.files or not request.files['dicom_folder']:
            logger.debug("No DICOM files uploaded")
            flash('No DICOM files uploaded', 'error')
            return redirect(request.url)

        result_id = str(uuid.uuid4())
        upload_dir = os.path.join(current_app.config['DICOM_UPLOAD_FOLDER'], result_id)
        os.makedirs(upload_dir, exist_ok=True)
        logger.debug(f"Upload directory created: {upload_dir}")

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

        if not file_paths:
            logger.debug("No valid DICOM files processed")
            flash('No valid DICOM images were processed', 'error')
            return redirect(request.url)

        try:
            ai_report = generate_report(analysis_results, imaging_request.patient_id, result_id)
            final_report = request.form.get('result_notes', "AI-generated findings stored in AI Findings section.")
            logger.debug(f"AI report: {ai_report[:100]}...")
            logger.debug(f"Final report: {final_report[:100]}...")
        except Exception as report_error:
            logger.error(f"Report generation failed: {report_error}")
            ai_report = "Report generation failed. Raw AI findings available."
            final_report = "Report generation failed. Basic findings available."

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
                processing_metadata={'file_metadata': analysis_results, 'model_version': 'ResNet18'}
            )
            db.session.add(imaging_result)
            imaging_request.status = 'completed'
            imaging_request.result_id = result_id
            db.session.commit()
            logger.debug(f"Saved result: request_id={request_id}, result_id={result_id}")

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
            return redirect(url_for('imaging.vi', result_id=result_id))

        except Exception as db_error:
            db.session.rollback()
            logger.error(f"Database error: {db_error}", exc_info=True)
            flash('Error saving results to database', 'error')
            return redirect(request.url)

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

