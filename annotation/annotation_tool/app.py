import os
import json
import random
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this for production

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, '../sampled_dataset.jsonl')
ANNOTATIONS_FILE = os.path.join(BASE_DIR, '../annotations.jsonl')

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
    
    # Check if specific index requested
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
        # Find first unanswered question
        answered_indices = set()
        for ann in annotations:
            if ann.get('annotator') == username:
                answered_indices.add(ann.get('dataset_index'))
        
        for i in range(total_count):
            if i not in answered_indices:
                target_index = i
                break
        
        # If all answered, default to last one or 0? Let's say 0 if all done, or handle on frontend
        if target_index is None:
            target_index = 0 # Or maybe stay at last one?
            
    row = dataset[target_index]
    
    # Check if user has already answered this question
    user_annotation = None
    for ann in annotations:
        if ann.get('annotator') == username and ann.get('dataset_index') == target_index:
            user_annotation = ann
            break
            
    # Prepare options
    ground_truth = row.get('ground_truth_persona')
    distractors = row.get('distractor_personas')
    
    correct_choice_letter = row.get('correct_choice')
    correct_index = ord(correct_choice_letter) - ord('A')
    
    options = distractors[:]
    if correct_index > len(options):
            options.append(ground_truth)
    else:
        options.insert(correct_index, ground_truth)
    
    return jsonify({
        'index': target_index,
        'total': total_count,
        'question': row.get('question'),
        'dimension_name': row.get('dimension_name'),
        'dimension_description': row.get('dimension_description'),
        'dimension_values': row.get('dimension_values'),
        'why_differ': row.get('why_differ'),
        'how_subtle': row.get('how_subtle'),
        'personalized_response': row.get('personalized_response'),
        'options': options,
        'user_annotation': user_annotation
    })

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
    
    # Filter out the annotation to be deleted
    new_annotations = [ann for ann in annotations if not (ann.get('annotator') == username and ann.get('dataset_index') == dataset_index)]
    
    # Write back all annotations
    # This is inefficient for large files but fine for this scale
    with open(ANNOTATIONS_FILE, 'w') as f:
        for ann in new_annotations:
            f.write(json.dumps(ann) + '\n')
            
    return jsonify({'success': True})

@app.route('/submit_choice', methods=['POST'])
def submit_choice():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    data = request.json
    dataset_index = data.get('index')
    user_choice = data.get('choice') # Expecting 'A', 'B', etc.
    
    if dataset_index is None or not user_choice:
        return jsonify({'error': 'Missing data'}), 400
        
    dataset = load_dataset()
    if dataset_index >= len(dataset):
         return jsonify({'error': 'Invalid index'}), 400
         
    row = dataset[dataset_index]
    correct_choice = row.get('correct_choice')
    
    is_correct = (user_choice == correct_choice)
    
    if is_correct:
        # Save immediately
        annotation = {
            'dataset_index': dataset_index,
            'annotator': session['username'],
            'user_choice': user_choice,
            'is_correct': True,
            'agree_with_judge': True, # Implicitly true
            'timestamp': os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
        }
        save_annotation(annotation)
        return jsonify({'correct': True})
    else:
        # Return rationale for stage 2
        return jsonify({
            'correct': False,
            'judge_rationale': row.get('judge_rationale'),
            'correct_choice': correct_choice # Show them what was right
        })

@app.route('/submit_agreement', methods=['POST'])
def submit_agreement():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    data = request.json
    dataset_index = data.get('index')
    user_choice = data.get('choice')
    agree = data.get('agree') # Boolean
    
    if dataset_index is None or user_choice is None or agree is None:
        return jsonify({'error': 'Missing data'}), 400
        
    annotation = {
        'dataset_index': dataset_index,
        'annotator': session['username'],
        'user_choice': user_choice,
        'is_correct': False,
        'agree_with_judge': agree,
        'timestamp': os.popen('date -u +"%Y-%m-%dT%H:%M:%SZ"').read().strip()
    }
    
    save_annotation(annotation)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5002)
