from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the directory of main.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import main as text_generator_module

app = Flask(__name__)

# Load the model and necessary components from main.py
# This assumes main.py has been modified to not auto-run training/generation
# and to make its functions and variables accessible.
try:
    model = text_generator_module.tf.keras.models.load_model('nn.h5')
    char_to_index = text_generator_module.char_to_index
    index_to_char = text_generator_module.index_to_char
    SEQ_LENGTH = text_generator_module.SEQ_LENGTH
    characters = text_generator_module.characters
    text_data_for_random_start = text_generator_module.text # Used to pick a random seed
except Exception as e:
    print(f"Error loading model or components from main.py: {e}")
    # Fallback or error handling if model/components can't be loaded
    model = None


def generate_text_from_model(length, temperature):
    if not model:
        return "Error: Model not loaded."
    
    start_index = text_generator_module.random.randint(0, len(text_data_for_random_start) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text_data_for_random_start[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    for _ in range(length):
        x_predictions = text_generator_module.np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            if char in char_to_index: # Ensure char is in dictionary
                 x_predictions[0, t, char_to_index[char]] = 1
            # else: handle unknown char if necessary, e.g., skip or use a placeholder

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = text_generator_module.sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first by running main.py.'}), 500
    try:
        data = request.get_json()
        length = int(data.get('length', 100))
        temperature = float(data.get('temperature', 0.5))
        
        if not (0.1 <= temperature <= 1.0):
            return jsonify({'error': 'Temperature must be between 0.1 and 1.0'}), 400
        if not (10 <= length <= 1000):
            return jsonify({'error': 'Length must be between 10 and 1000'}), 400

        generated_text = generate_text_from_model(length, temperature)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure nn.h5 exists, otherwise guide user to run main.py for training
    if not os.path.exists('nn.h5'):
        print("Model file 'nn.h5' not found.")
        print("Please run 'python main.py' first to train and save the model.")
    else:
        print("Model 'nn.h5' loaded. Starting Flask app.")
        app.run(debug=True)

