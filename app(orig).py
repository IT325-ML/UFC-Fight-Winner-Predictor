from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load models
model_simple = pickle.load(open('hgb_model_easy_mode.pkl', 'rb'))
model_full = pickle.load(open('hgb_model.pkl', 'rb'))

def stance_to_one_hot(st, prefix):
    return [
        1 if st == 'Open Stance' else 0,
        1 if st == 'Orthodox' else 0,
        1 if st == 'Southpaw' else 0,
        1 if st == 'Switch' else 0
    ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mode = request.form['prediction_mode']
    is_title_bout = int(request.form['is_title_bout'])

    # Shared features
    r_wins_total = int(request.form['r_wins_total'])
    r_losses_total = int(request.form['r_losses_total'])
    r_age = int(request.form['r_age'])
    r_height = int(request.form['r_height'])
    r_weight = int(request.form['r_weight'])
    r_stance = request.form['r_stance']

    b_wins_total = int(request.form['b_wins_total'])
    b_losses_total = int(request.form['b_losses_total'])
    b_age = int(request.form['b_age'])
    b_height = int(request.form['b_height'])
    b_weight = int(request.form['b_weight'])
    b_stance = request.form['b_stance']

    r_stance_one_hot = stance_to_one_hot(r_stance, 'r_stance')
    b_stance_one_hot = stance_to_one_hot(b_stance, 'b_stance')

    if mode == 'simple':
        features = [
            is_title_bout,
            r_wins_total, r_losses_total, r_age, r_height, r_weight,
            *r_stance_one_hot,
            b_wins_total, b_losses_total, b_age, b_height, b_weight,
            *b_stance_one_hot
        ]
        prediction = model_simple.predict([features])[0]
    else:  # full
        r_reach = int(request.form.get('r_reach', 0))
        b_reach = int(request.form.get('b_reach', 0))
        features = [
            is_title_bout,
            r_age, r_height, r_weight, r_reach,
            *r_stance_one_hot,
            r_wins_total, r_losses_total,
            b_age, b_height, b_weight, b_reach,
            *b_stance_one_hot,
            b_wins_total, b_losses_total
        ]
        prediction = model_full.predict([features])[0]

    return f"The predicted winner is: {'Red' if prediction == 'Red' else 'Blue'}"

if __name__ == '__main__':
    app.run(debug=True)


################################################################################################################

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('hgb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # One-hot encoding for stances — must match dataset column order
    stance_mapping = {
        'Open Stance': [1, 0, 0, 0],
        'Orthodox': [0, 1, 0, 0],
        'Southpaw': [0, 0, 1, 0],
        'Switch': [0, 0, 0, 1]
    }

    is_title_bout_input = request.form.get('is_title_bout')  # 'Yes' or 'No'
    is_title_bout = 1 if is_title_bout_input == 'Yes' else 0

    # Red Fighter Inputs
    r_wins_total = int(request.form.get('r_wins_total'))
    r_losses_total = int(request.form.get('r_losses_total'))
    r_age = int(request.form.get('r_age'))
    r_height = float(request.form.get('r_height'))
    r_weight = float(request.form.get('r_weight'))
    r_reach = float(request.form.get('r_reach'))
    r_stance = request.form.get('r_stance')
    r_SLpM_total = float(request.form.get('r_SLpM_total'))
    r_SApM_total = float(request.form.get('r_SApM_total'))
    r_sig_str_acc_total = float(request.form.get('r_sig_str_acc_total'))
    r_td_acc_total = float(request.form.get('r_td_acc_total'))
    r_str_def_total = float(request.form.get('r_str_def_total'))
    r_td_def_total = float(request.form.get('r_td_def_total'))
    r_sub_avg = float(request.form.get('r_sub_avg'))
    r_td_avg = float(request.form.get('r_td_avg'))

    # Blue Fighter Inputs
    b_wins_total = int(request.form.get('b_wins_total'))
    b_losses_total = int(request.form.get('b_losses_total'))
    b_age = int(request.form.get('b_age'))
    b_height = float(request.form.get('b_height'))
    b_weight = float(request.form.get('b_weight'))
    b_reach = float(request.form.get('b_reach'))
    b_stance = request.form.get('b_stance')
    b_SLpM_total = float(request.form.get('b_SLpM_total'))
    b_SApM_total = float(request.form.get('b_SApM_total'))
    b_sig_str_acc_total = float(request.form.get('b_sig_str_acc_total'))
    b_td_acc_total = float(request.form.get('b_td_acc_total'))
    b_str_def_total = float(request.form.get('b_str_def_total'))
    b_td_def_total = float(request.form.get('b_td_def_total'))
    b_sub_avg = float(request.form.get('b_sub_avg'))
    b_td_avg = float(request.form.get('b_td_avg'))

    # One-hot encode stances
    r_stance_encoded = stance_mapping[r_stance]
    b_stance_encoded = stance_mapping[b_stance]

    # Create the feature array (order must match the training dataset)
    features = np.array([
        is_title_bout,
        r_wins_total, r_losses_total, r_age, r_height, r_weight, r_reach,
        *r_stance_encoded,
        r_SLpM_total, r_SApM_total, r_sig_str_acc_total, r_td_acc_total,
        r_str_def_total, r_td_def_total, r_sub_avg, r_td_avg,

        b_wins_total, b_losses_total, b_age, b_height, b_weight, b_reach,
        *b_stance_encoded,
        b_SLpM_total, b_SApM_total, b_sig_str_acc_total, b_td_acc_total,
        b_str_def_total, b_td_def_total, b_sub_avg, b_td_avg
    ])

    features = features.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)

    # Map prediction to outcome
    prediction_text = 'Red Fighter Wins' if prediction == 0 else 'Blue Fighter Wins'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

#####

from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load models
model_simple = pickle.load(open('hgb_model_easy_mode.pkl', 'rb'))
model_full = pickle.load(open('hgb_model.pkl', 'rb'))

def stance_to_one_hot(st, prefix):
    return [
        1 if st == 'Open Stance' else 0,
        1 if st == 'Orthodox' else 0,
        1 if st == 'Southpaw' else 0,
        1 if st == 'Switch' else 0
    ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mode = request.form['prediction_mode']
    is_title_bout = int(request.form['is_title_bout'])

    # Shared features
    r_wins_total = int(request.form['r_wins_total'])
    r_losses_total = int(request.form['r_losses_total'])
    r_age = int(request.form['r_age'])
    r_height = int(request.form['r_height'])
    r_weight = int(request.form['r_weight'])
    r_stance = request.form['r_stance']

    b_wins_total = int(request.form['b_wins_total'])
    b_losses_total = int(request.form['b_losses_total'])
    b_age = int(request.form['b_age'])
    b_height = int(request.form['b_height'])
    b_weight = int(request.form['b_weight'])
    b_stance = request.form['b_stance']

    r_stance_one_hot = stance_to_one_hot(r_stance, 'r_stance')
    b_stance_one_hot = stance_to_one_hot(b_stance, 'b_stance')

    if mode == 'simple':
        features = [
            is_title_bout,
            r_wins_total, r_losses_total, r_age, r_height, r_weight,
            *r_stance_one_hot,
            b_wins_total, b_losses_total, b_age, b_height, b_weight,
            *b_stance_one_hot
        ]
        prediction = model_simple.predict([features])[0]
    else:  # full
        r_reach = int(request.form.get('r_reach', 0))
        b_reach = int(request.form.get('b_reach', 0))
        features = [
            is_title_bout,
            r_age, r_height, r_weight, r_reach,
            *r_stance_one_hot,
            r_wins_total, r_losses_total,
            b_age, b_height, b_weight, b_reach,
            *b_stance_one_hot,
            b_wins_total, b_losses_total
        ]
        prediction = model_full.predict([features])[0]

    return f"The predicted winner is: {'Red' if prediction == 'Red' else 'Blue'}"

if __name__ == '__main__':
    app.run(debug=True)
