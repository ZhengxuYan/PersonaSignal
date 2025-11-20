import os
import json
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'supersecretkey_task2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, '../dataset_task2.jsonl')
ANNOTATIONS_FILE = os.path.join(BASE_DIR, '../annotations_task2.jsonl')

def load_dataset():
    data = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def load_annotations():
    annotations = []
    if os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, 'r') as f:
            for line in f:
                annotations.append(json.loads(line))
    return annotations

def save_annotation(annotation):
    with open(ANNOTATIONS_FILE, 'a') as f:
        f.write(json.dumps(annotation) + '\n')

@app.route('/')
def index():
    if 'username' not in session:
        return render_template('index.html', logged_in=False)
    return render_template('index.html', logged_in=True, username=session['username'])

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    if username:
        session['username'] = username
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'success': True})

@app.route('/get_question')
def get_question():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    username = session['username']
    dataset = load_dataset()
    annotations = load_annotations()
    total_count = len(dataset)
    
    req_index = request.args.get('index')
    target_index = None
    
    if req_index is not None:
        try:
            target_index = int(req_index)
            if target_index < 0 or target_index >= total_count:
                return jsonify({'error': 'Index out of bounds'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid index'}), 400
    else:
        answered_indices = set()
        for ann in annotations:
            if ann.get('annotator') == username:
                answered_indices.add(ann.get('dataset_index'))
        
        for i in range(total_count):
            if i not in answered_indices:
                target_index = i
                break
        
        if target_index is None:
            target_index = 0
            
    row = dataset[target_index]
    
    user_annotation = None
    for ann in annotations:
        if ann.get('annotator') == username and ann.get('dataset_index') == target_index:
            user_annotation = ann
            break
            
    return jsonify({
        'index': target_index,
        'total': total_count,
        'question': row.get('question'),
        'dimension_name': row.get('dimension_name'),
        'ground_truth_persona': row.get('ground_truth_persona'),
        'personalized_response': row.get('personalized_response'),
        'user_annotation': user_annotation
    })

@app.route('/submit_verification', methods=['POST'])
def submit_verification():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    data = request.json
    dataset_index = data.get('index')
    is_personalized = data.get('is_personalized') # Boolean
    
    if dataset_index is None or is_personalized is None:
        return jsonify({'error': 'Missing data'}), 400
        
    annotation = {
        'dataset_index': dataset_index,
        'annotator': session['username'],
        'is_personalized': is_personalized,
        'timestamp': os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
    }
    
    save_annotation(annotation)
    return jsonify({'success': True})

@app.route('/clear_response', methods=['POST'])
def clear_response():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    data = request.json
    dataset_index = data.get('index')
    
    if dataset_index is None:
        return jsonify({'error': 'Missing index'}), 400
        
    username = session['username']
    annotations = load_annotations()
    
    new_annotations = [ann for ann in annotations if not (ann.get('annotator') == username and ann.get('dataset_index') == dataset_index)]
    
    with open(ANNOTATIONS_FILE, 'w') as f:
        for ann in new_annotations:
            f.write(json.dumps(ann) + '\n')
            
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5003)
