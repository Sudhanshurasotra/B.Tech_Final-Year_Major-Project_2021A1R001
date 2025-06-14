from flask import Flask, render_template, request, jsonify, session, redirect, url_for, make_response
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as PILImage
import logging
import sys
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.textlabels import Label

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Basic TensorFlow check
try:
    logger.info("TensorFlow is installed")
    logger.info("GPU Available: %s", tf.config.list_physical_devices('GPU'))
except Exception as e:
    logger.error("Error checking TensorFlow: %s", str(e))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autism_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    analyses = db.relationship('Analysis', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Analysis History Model
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    result = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        user = User(username=data['username'], email=data['email'])
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'Registration successful'}), 201
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(username=data['username']).first()
        if user and user.check_password(data['password']):
            login_user(user)
            return jsonify({'message': 'Login successful'}), 200
        return jsonify({'error': 'Invalid username or password'}), 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Load the model
model = None
try:
    model_path = os.path.abspath('autism.h5')
    logger.info("Attempting to load model from: %s", model_path)
    
    if not os.path.exists(model_path):
        logger.error("Model file not found at: %s", model_path)
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    logger.info("Model file found, size: %d bytes", os.path.getsize(model_path))
    
    # Try loading the model with custom_objects if needed
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully!")
        
        # Log detailed model information
        logger.info("Model Architecture:")
        model.summary(print_fn=logger.info)
        
        # Log model configuration
        logger.info("Model Configuration:")
        logger.info(f"Input Shape: {model.input_shape}")
        logger.info(f"Output Shape: {model.output_shape}")
        logger.info(f"Number of Layers: {len(model.layers)}")
        
        # Test with dummy input and log the output format
        dummy_input = np.zeros((1, 1, 128, 128, 3), dtype=np.float32)  # (batch, time, height, width, channels)
        test_output = model.predict(dummy_input, verbose=0)
        logger.info("Model test successful")
        logger.info("Model output shape: %s", test_output.shape)
        logger.info("Model output format: %s", test_output)
        
        # Log the first few layer details
        logger.info("\nFirst few layers of the model:")
        for i, layer in enumerate(model.layers[:5]):
            logger.info(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")
        
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        raise

except Exception as e:
    logger.error("Error in model setup: %s", str(e))
    raise

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for model input"""
    try:
        # Read image
        img = PILImage.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize image
        img = img.resize(target_size)
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        # Add batch and time dimensions
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add time dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def preprocess_video(video_path, target_size=(128, 128), max_frames=30):
    """Extract and preprocess frames from video"""
    try:
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame = cv2.resize(frame, target_size)
            # Normalize
            frame = frame / 255.0
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        if not frames:
            return None
            
        # Convert to numpy array and add batch dimension
        frames = np.array(frames)
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        return frames
    except Exception as e:
        print(f"Error preprocessing video: {str(e)}")
        return None

def get_prediction(file_path):
    """Get prediction from model"""
    if model is None:
        return None, None, "Model not loaded"
        
    try:
        # Determine if file is image or video
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov'))
        
        if is_video:
            # Process video
            processed_data = preprocess_video(file_path)
            if processed_data is None:
                return None, None, "Error processing video"
        else:
            # Process image
            processed_data = preprocess_image(file_path)
            if processed_data is None:
                return None, None, "Error processing image"
        
        # Get prediction
        prediction = model.predict(processed_data, verbose=0)
        # Log raw model output for debugging
        logger.info(f"Raw model output shape: {prediction.shape}")
        logger.info(f"Raw model output: {prediction}")
        
        # Get confidence score and class prediction
        if prediction.shape[-1] == 2:  # Two-class output [no_autism_prob, autism_prob]
            autism_prob = float(prediction[0][1])  # Probability of autism class
            logger.info(f"Autism probability: {autism_prob:.4f}")
            
            # Always use autism probability as confidence score
            confidence = autism_prob
            # If autism probability > 50%, it's autism detected
            is_autism = autism_prob > 0.5
            
            if is_autism:
                logger.info(f"Autism detected with confidence: {confidence:.4f}")
            else:
                logger.info(f"No autism detected with confidence: {confidence:.4f}")
            
            return confidence, is_autism, "Success"
        else:  # Single class output
            confidence = float(prediction[0][0])
            # For single class output, use same logic
            is_autism = confidence > 0.5
            logger.info(f"Single class output - Is Autism: {is_autism}, Confidence: {confidence:.4f}")
            return confidence, is_autism, "Success"
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None, None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction from model
        confidence, is_autism, message = get_prediction(filepath)
        
        if confidence is None:
            return jsonify({
                'error': f'Prediction failed: {message}',
                'file_path': f'/static/uploads/{filename}'
            }), 500
        
        # Use the same result message format as process_frame
        result_message = "Autism Detected" if is_autism else "No Autism Detected"
        
        analysis = Analysis(
            user_id=current_user.id,
            file_path=filename,
            file_type=file.content_type,
            confidence=confidence,
            result=result_message
        )
        db.session.add(analysis)
        db.session.commit()
        
        prediction = {
            'result': result_message,
            'confidence': confidence,
            'is_autism': is_autism,
            'file_path': f'/static/uploads/{filename}',
            'analysis_id': analysis.id
        }
        
        return jsonify(prediction)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
@login_required
def history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/analytics')
@login_required
def analytics():
    # Get user's analysis statistics
    total_analyses = Analysis.query.filter_by(user_id=current_user.id).count()
    successful_analyses = Analysis.query.filter_by(user_id=current_user.id).filter(Analysis.confidence > 0.5).count()
    avg_confidence = db.session.query(db.func.avg(Analysis.confidence)).filter_by(user_id=current_user.id).scalar() or 0
    
    # Get recent analyses for chart
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).limit(10).all()
    chart_data = {
        'labels': [a.timestamp.strftime('%Y-%m-%d') for a in recent_analyses],
        'confidence': [a.confidence for a in recent_analyses]
    }
    
    return render_template('analytics.html', 
                         total_analyses=total_analyses,
                         successful_analyses=successful_analyses,
                         avg_confidence=avg_confidence,
                         chart_data=json.dumps(chart_data))

def process_frame(frame):
    """Process a single frame for autism detection"""
    try:
        logger.info("Starting frame processing...")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model is not loaded!")
            return None
            
        # Validate frame
        if frame is None:
            logger.error("Received empty frame")
            return None
            
        if frame.size == 0:
            logger.error("Received frame with zero size")
            return None
            
        # Log frame details
        logger.info(f"Input frame shape: {frame.shape}")
        logger.info(f"Input frame dtype: {frame.dtype}")
        logger.info(f"Input frame min/max values: {frame.min()}/{frame.max()}")
        
        # Ensure frame is in correct format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.error(f"Invalid frame format. Expected RGB image, got shape: {frame.shape}")
            return None
            
        # Convert frame to RGB if needed
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
                logger.info("Converted frame to uint8")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.info("Successfully converted frame to RGB")
        except cv2.error as e:
            logger.error(f"Error converting frame to RGB: {str(e)}")
            return None
        
        # Convert to PIL Image
        try:
            # Convert numpy array to PIL Image with explicit mode
            pil_image = PILImage.fromarray(frame_rgb, mode='RGB')
            logger.info("Successfully converted to PIL Image")
        except Exception as e:
            logger.error(f"Error converting to PIL Image: {str(e)}")
            logger.exception("Full traceback:")
            return None
        
        # Resize to model's expected input size
        try:
            pil_image = pil_image.resize((128, 128), PILImage.Resampling.LANCZOS)
            logger.info("Successfully resized image to 128x128")
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return None
        
        # Convert to numpy array and normalize
        try:
            img_array = np.array(pil_image, dtype=np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
            logger.info(f"Successfully normalized image array. Shape: {img_array.shape}, Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        except Exception as e:
            logger.error(f"Error converting to numpy array: {str(e)}")
            return None
        
        # Add batch and time dimensions
        try:
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)  # Add time dimension
            logger.info(f"Successfully added dimensions. Final input shape: {img_array.shape}")
        except Exception as e:
            logger.error(f"Error adding dimensions: {str(e)}")
            return None
        
        # Validate input shape
        expected_shape = (1, 1, 128, 128, 3)
        if img_array.shape != expected_shape:
            logger.error(f"Invalid input shape. Expected {expected_shape}, got {img_array.shape}")
            return None
        
        # Get prediction
        try:
            logger.info("Running model prediction...")
            prediction = model.predict(img_array, verbose=0)
            logger.info(f"Raw prediction output shape: {prediction.shape}")
            logger.info(f"Raw prediction output: {prediction}")
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            return None
        
        # Handle both binary and single class outputs
        try:
            if prediction.shape[-1] == 2:  # Binary classification
                autism_prob = float(prediction[0][1])
                logger.info(f"Autism probability: {autism_prob:.4f}")
                
                # Always use autism probability as confidence score
                confidence = autism_prob
                # If autism probability > 50%, it's autism detected
                result = "Autism Detected" if autism_prob > 0.5 else "No Autism Detected"
            else:  # Single class output
                confidence = float(prediction[0][0])
                logger.info(f"Single class output - Confidence: {confidence:.4f}")
                result = "Autism Detected" if confidence > 0.5 else "No Autism Detected"
            
            logger.info(f"Final result: {result} with confidence: {confidence:.4f}")
        except Exception as e:
            logger.error(f"Error processing prediction results: {str(e)}")
            return None
        
        return {
            'result': result,
            'confidence': confidence,
            'frame': frame_rgb
        }
    except Exception as e:
        logger.error(f"Unexpected error in process_frame: {str(e)}")
        logger.exception("Full traceback:")
        return None

@app.route('/detect_frame', methods=['POST'])
@login_required
def detect_frame():
    """Handle real-time frame detection"""
    try:
        logger.info("Received frame detection request")
        
        # Get the frame data from the request
        frame_data = request.json.get('frame')
        if not frame_data:
            logger.error("No frame data provided")
            return jsonify({'error': 'No frame data provided'}), 400
            
        # Remove the data URL prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
            
        # Decode base64 image
        try:
            logger.info("Decoding base64 image...")
            frame_bytes = base64.b64decode(frame_data)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            
            # Validate frame data
            if frame_array.size == 0:
                logger.error("Empty frame data received")
                return jsonify({'error': 'Empty frame data'}), 400
                
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode image - cv2.imdecode returned None")
                return jsonify({'error': 'Failed to decode image'}), 400
                
            logger.info(f"Successfully decoded frame. Shape: {frame.shape}, Type: {frame.dtype}")
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
            
        # Process the frame
        result = process_frame(frame)
        if result is None:
            logger.error("Frame processing failed")
            return jsonify({'error': 'Error processing frame - check server logs for details'}), 500
            
        # Save the frame as an image file
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'realtime_{timestamp}.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the frame
            cv2.imwrite(filepath, cv2.cvtColor(result['frame'], cv2.COLOR_RGB2BGR))
            logger.info(f"Saved frame to: {filepath}")
            
            # Save analysis to database using the exact result from process_frame
            analysis = Analysis(
                user_id=current_user.id,
                file_path=filename,
                file_type='image/jpeg',
                confidence=result['confidence'],
                result=result['result']  # Use the exact result from process_frame
            )
            db.session.add(analysis)
            db.session.commit()
            logger.info(f"Saved analysis to database with ID: {analysis.id}")
            
        except Exception as e:
            logger.error(f"Error saving frame and analysis: {str(e)}")
            logger.exception("Full traceback:")
            # Continue with the response even if saving fails
            
        # Convert the processed frame back to base64 for display
        try:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result['frame'], cv2.COLOR_RGB2BGR))
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            logger.info("Successfully encoded processed frame")
        except Exception as e:
            logger.error(f"Error encoding processed frame: {str(e)}")
            return jsonify({'error': f'Error encoding processed frame: {str(e)}'}), 500
        
        logger.info("Successfully processed frame and returning results")
        
        return jsonify({
            'result': result['result'],
            'confidence': result['confidence'],
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'analysis_id': analysis.id if 'analysis' in locals() else None
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in detect_frame: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_pdf_report(analysis_data, user_data):
    """Generate a colorful PDF report for the analysis"""
    temp_path = None
    try:
        logger.info(f"Starting PDF generation for analysis {analysis_data['id']}")
        
        # Create a BytesIO buffer to store the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create custom styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30
        ))
        styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#4a90e2'),
            spaceAfter=12
        ))
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        ))
        
        # Build the PDF content
        story = []
        
        # Add header with logo and title
        story.append(Paragraph("Autism Detection Analysis Report", styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Add report metadata
        story.append(Paragraph("Report Details", styles['CustomHeading']))
        metadata = [
            ["Generated for:", user_data['username']],
            ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Analysis ID:", str(analysis_data['id'])],
            ["File Type:", analysis_data['file_type']]
        ]
        metadata_table = Table(metadata, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f6fa')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a90e2')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dcdde1'))
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Add analysis results
        story.append(Paragraph("Analysis Results", styles['CustomHeading']))
        
        # Create a colorful results table
        confidence = analysis_data['confidence'] * 100
        # Set result color based on detection
        if "No Autism Detected" in analysis_data['result']:
            result_color = colors.HexColor('#2ecc71')  # Green for no autism
        else:
            result_color = colors.HexColor('#e74c3c')  # Red for autism detected
        
        results_data = [
            ["Result", analysis_data['result']],
            ["Confidence", f"{confidence:.1f}%"],
            ["Detection", "No Autism Detected" if "No Autism Detected" in analysis_data['result'] else "Autism Detected"]
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f6fa')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a90e2')),
            ('TEXTCOLOR', (1, 2), (1, 2), result_color),  # Color the detection result
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dcdde1'))
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Add confidence visualization
        story.append(Paragraph("Confidence Visualization", styles['CustomHeading']))
        
        # Create a drawing for the confidence bar
        drawing = Drawing(400, 100)
        confidence_bar = VerticalBarChart()
        confidence_bar.x = 50
        confidence_bar.y = 20
        confidence_bar.height = 60
        confidence_bar.width = 300
        confidence_bar.data = [[confidence]]
        confidence_bar.categoryAxis.categoryNames = ['Confidence']
        confidence_bar.bars[0].fillColor = result_color  # Use the same color as the result
        confidence_bar.valueAxis.valueMin = 0
        confidence_bar.valueAxis.valueMax = 100
        confidence_bar.valueAxis.valueStep = 20
        
        # Add the chart to the drawing
        drawing.add(confidence_bar)
        story.append(drawing)
        story.append(Spacer(1, 20))
        
        # Add the analyzed image if available
        if 'file_path' in analysis_data and analysis_data['file_path']:
            story.append(Paragraph("Analyzed Image", styles['CustomHeading']))
            try:
                # Get absolute path and verify file exists
                abs_path = os.path.abspath(analysis_data['file_path'])
                logger.info(f"Attempting to add image to PDF: {abs_path}")
                
                if not os.path.exists(abs_path):
                    logger.error(f"Image file not found at path: {abs_path}")
                    story.append(Paragraph("Image not available", styles['CustomBody']))
                else:
                    # Use PILImage to verify and process the image
                    try:
                        pil_img = PILImage.open(abs_path)
                        logger.info(f"Successfully opened image: {abs_path}")
                        
                        # Convert to RGB if needed
                        if pil_img.mode != 'RGB':
                            logger.info(f"Converting image from {pil_img.mode} to RGB")
                            pil_img = pil_img.convert('RGB')
                        
                        # Create a temporary file in the uploads directory
                        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f'temp_{analysis_data["id"]}.jpg')
                        
                        # Save the processed image
                        logger.info(f"Saving processed image to: {temp_path}")
                        pil_img.save(temp_path, 'JPEG', quality=95)
                        
                        # Add to PDF
                        logger.info("Adding image to PDF document")
                        img = Image(temp_path, width=4*inch, height=3*inch)
                        story.append(img)
                            
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                        logger.exception("Full traceback:")
                        story.append(Paragraph("Error processing image", styles['CustomBody']))
                        
            except Exception as e:
                logger.error(f"Error adding image to PDF: {str(e)}")
                logger.exception("Full traceback:")
                story.append(Paragraph("Error loading image", styles['CustomBody']))
        
        # Add footer with disclaimer
        story.append(Spacer(1, 30))
        disclaimer = Paragraph(
            "Disclaimer: This report is generated by an AI model and should be used as a preliminary screening tool only. "
            "Please consult with healthcare professionals for a proper diagnosis.",
            ParagraphStyle(
                'Disclaimer',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#7f8c8d'),
                alignment=1
            )
        )
        story.append(disclaimer)
        
        # Build the PDF
        logger.info("Building PDF document")
        doc.build(story)
        
        # Get the PDF from the buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        logger.info(f"Successfully generated PDF for analysis {analysis_data['id']}")
        return pdf
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        logger.exception("Full traceback:")
        raise
    finally:
        # Clean up temp file after PDF is generated
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_path}: {str(e)}")

@app.route('/download_report/<int:analysis_id>')
@login_required
def download_report(analysis_id):
    """Download PDF report for a specific analysis"""
    try:
        logger.info(f"Starting report download for analysis {analysis_id}")
        
        # Get analysis data
        analysis = Analysis.query.get_or_404(analysis_id)
        logger.info(f"Found analysis record: {analysis_id}")
        logger.info(f"Analysis result from database: '{analysis.result}'")
        logger.info(f"Analysis confidence: {analysis.confidence}")
        
        # Verify user ownership
        if analysis.user_id != current_user.id:
            logger.warning(f"Unauthorized access attempt for analysis {analysis_id} by user {current_user.id}")
            return jsonify({'error': 'Unauthorized access'}), 403
        
        # Determine if autism was detected based on the result message
        is_autism_detected = "No Autism Detected" not in analysis.result
        logger.info(f"Result message contains 'No Autism Detected': {'No Autism Detected' in analysis.result}")
        logger.info(f"Determined is_autism_detected: {is_autism_detected}")
        
        # Prepare data for PDF generation
        analysis_data = {
            'id': analysis.id,
            'result': analysis.result,
            'confidence': analysis.confidence,
            'is_autism': is_autism_detected,  # Use the result message to determine detection
            'file_type': analysis.file_type,
            'file_path': os.path.join(app.config['UPLOAD_FOLDER'], analysis.file_path) if analysis.file_path else None
        }
        logger.info(f"Analysis data for PDF: {analysis_data}")
        
        user_data = {
            'username': current_user.username,
            'email': current_user.email
        }
        
        # Generate PDF
        try:
            logger.info(f"Generating PDF for analysis {analysis_id}")
            pdf = generate_pdf_report(analysis_data, user_data)
            
            # Create response
            response = make_response(pdf)
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=autism_analysis_{analysis_id}.pdf'
            
            logger.info(f"Successfully created PDF response for analysis {analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({'error': 'Error generating PDF report'}), 500
        
    except Exception as e:
        logger.error(f"Error in download_report: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': 'Error generating report'}), 500

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # Create dummy input for testing with correct shape (None, None, 128, 128, 3)
    dummy_input = np.zeros((1, 1, 128, 128, 3), dtype=np.float32)  # (batch, time, height, width, channels)
    try:
        test_output = model.predict(dummy_input, verbose=0)
        print("Model test successful!")
    except Exception as e:
        print(f"Error in model setup: {str(e)}")
        sys.exit(1)
    
    app.run(debug=True) 