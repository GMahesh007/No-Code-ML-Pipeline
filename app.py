from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import io
import base64

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Store pipeline state in memory (for demo purposes)
pipeline_state = {
    'dataset': None,
    'original_data': None,
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'model': None,
    'scaler': None,
    'preprocessing_applied': None,
    'split_ratio': 0.8,
    'target_column': None,
    'feature_columns': None
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'ML Pipeline Builder'}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file based on extension
        filename = file.filename.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel file.'}), 400
        
        # Store dataset
        pipeline_state['dataset'] = df
        pipeline_state['original_data'] = df.copy()
        
        # Get dataset information
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(5).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'info': info
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/select-target', methods=['POST'])
def select_target():
    try:
        data = request.json
        target_column = data.get('target_column')
        
        if pipeline_state['dataset'] is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = pipeline_state['dataset']
        
        if target_column not in df.columns:
            return jsonify({'error': 'Target column not found'}), 400
        
        # Store target and feature columns
        pipeline_state['target_column'] = target_column
        pipeline_state['feature_columns'] = [col for col in df.columns if col != target_column]
        
        # Check if features are numeric
        X = df[pipeline_state['feature_columns']]
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric:
            return jsonify({
                'warning': f'Non-numeric columns detected: {non_numeric}. These will be excluded.',
                'target_column': target_column,
                'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist()
            }), 200
        
        return jsonify({
            'message': 'Target column selected',
            'target_column': target_column,
            'feature_columns': pipeline_state['feature_columns']
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error selecting target: {str(e)}'}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    try:
        data = request.json
        preprocessing_type = data.get('type', 'none')
        
        if pipeline_state['dataset'] is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        if pipeline_state['target_column'] is None:
            return jsonify({'error': 'Target column not selected'}), 400
        
        df = pipeline_state['dataset']
        target_column = pipeline_state['target_column']
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Keep only numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_features]
        
        # Apply preprocessing
        if preprocessing_type == 'standardization':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_processed = pd.DataFrame(X_scaled, columns=X.columns)
            pipeline_state['scaler'] = scaler
            pipeline_state['preprocessing_applied'] = 'Standardization (StandardScaler)'
        elif preprocessing_type == 'normalization':
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            X_processed = pd.DataFrame(X_scaled, columns=X.columns)
            pipeline_state['scaler'] = scaler
            pipeline_state['preprocessing_applied'] = 'Normalization (MinMaxScaler)'
        else:
            X_processed = X
            pipeline_state['scaler'] = None
            pipeline_state['preprocessing_applied'] = 'None'
        
        # Store processed data
        pipeline_state['dataset'] = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)
        pipeline_state['feature_columns'] = numeric_features
        
        return jsonify({
            'message': f'Preprocessing applied: {pipeline_state["preprocessing_applied"]}',
            'preprocessing': pipeline_state['preprocessing_applied'],
            'features_used': numeric_features
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error preprocessing data: {str(e)}'}), 500

@app.route('/api/split', methods=['POST'])
def split_data():
    try:
        data = request.json
        split_ratio = float(data.get('split_ratio', 0.8))
        
        if pipeline_state['dataset'] is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        if pipeline_state['target_column'] is None:
            return jsonify({'error': 'Target column not selected'}), 400
        
        df = pipeline_state['dataset']
        target_column = pipeline_state['target_column']
        feature_columns = pipeline_state['feature_columns']
        
        X = df[feature_columns]
        y = df[target_column]
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split_ratio, random_state=42
        )
        
        pipeline_state['X_train'] = X_train
        pipeline_state['X_test'] = X_test
        pipeline_state['y_train'] = y_train
        pipeline_state['y_test'] = y_test
        pipeline_state['split_ratio'] = split_ratio
        
        return jsonify({
            'message': 'Data split successfully',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'split_ratio': f'{int(split_ratio*100)}-{int((1-split_ratio)*100)}'
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error splitting data: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        model_type = data.get('model_type')
        
        if pipeline_state['X_train'] is None:
            return jsonify({'error': 'Data not split. Please perform train-test split first.'}), 400
        
        X_train = pipeline_state['X_train']
        X_test = pipeline_state['X_test']
        y_train = pipeline_state['y_train']
        y_test = pipeline_state['y_test']
        
        # Train model
        if model_type == 'logistic_regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = 'Logistic Regression'
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
            model_name = 'Decision Tree Classifier'
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Store model
        pipeline_state['model'] = model
        
        # Prepare results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'confusion_matrix_plot': img_base64,
            'status': 'Training completed successfully'
        }
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'error': f'Error training model: {str(e)}'}), 500

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    try:
        status = {
            'dataset_loaded': pipeline_state['dataset'] is not None,
            'target_selected': pipeline_state['target_column'] is not None,
            'preprocessing_applied': pipeline_state['preprocessing_applied'],
            'data_split': pipeline_state['X_train'] is not None,
            'model_trained': pipeline_state['model'] is not None
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_pipeline():
    try:
        pipeline_state['dataset'] = None
        pipeline_state['original_data'] = None
        pipeline_state['X_train'] = None
        pipeline_state['X_test'] = None
        pipeline_state['y_train'] = None
        pipeline_state['y_test'] = None
        pipeline_state['model'] = None
        pipeline_state['scaler'] = None
        pipeline_state['preprocessing_applied'] = None
        pipeline_state['split_ratio'] = 0.8
        pipeline_state['target_column'] = None
        pipeline_state['feature_columns'] = None
        
        return jsonify({'message': 'Pipeline reset successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
