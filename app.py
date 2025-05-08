from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load models
model_full = joblib.load('hgb_model.pkl')
model_few = joblib.load('hgb_model_few.pkl')


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
    is_title_bout_input = request.form.get('is_title_bout')  # 'Yes' or 'No'
    is_title_bout = 1 if is_title_bout_input == 'Yes' else 0

    try:
        # Shared features
        r_wins_total = int(request.form['r_wins_total'])
        r_losses_total = int(request.form['r_losses_total'])
        r_age = float(request.form['r_age'])
        r_height = float(request.form['r_height'])
        r_weight = float(request.form['r_weight'])
        r_stance = request.form['r_stance']
        

        b_wins_total = int(request.form['b_wins_total'])
        b_losses_total = int(request.form['b_losses_total'])
        b_age = float(request.form['b_age'])
        b_height = float(request.form['b_height'])
        b_weight = float(request.form['b_weight'])
        b_stance = request.form['b_stance']

        valid_stances = ['Open Stance', 'Orthodox', 'Southpaw', 'Switch']
        if r_stance not in valid_stances or b_stance not in valid_stances:
            return render_template('index.html', error="Invalid stance value.")

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
            prediction = model_few.predict([features])[0]
        else:  # full
            r_reach = float(request.form.get('r_reach'))
            r_SLpM_total = float(request.form.get('r_SLpM_total'))
            r_SApM_total = float(request.form.get('r_SApM_total'))
            r_sig_str_acc_total = float(request.form.get('r_sig_str_acc_total'))
            r_td_acc_total = float(request.form.get('r_td_acc_total'))
            r_str_def_total = float(request.form.get('r_str_def_total'))
            r_td_def_total = float(request.form.get('r_td_def_total'))
            r_sub_avg = float(request.form.get('r_sub_avg'))
            r_td_avg = float(request.form.get('r_td_avg'))
            b_reach = float(request.form.get('b_reach'))
            b_SLpM_total = float(request.form.get('b_SLpM_total'))
            b_SApM_total = float(request.form.get('b_SApM_total'))
            b_sig_str_acc_total = float(request.form.get('b_sig_str_acc_total'))
            b_td_acc_total = float(request.form.get('b_td_acc_total'))
            b_str_def_total = float(request.form.get('b_str_def_total'))
            b_td_def_total = float(request.form.get('b_td_def_total'))
            b_sub_avg = float(request.form.get('b_sub_avg'))
            b_td_avg = float(request.form.get('b_td_avg'))

            features = [
                is_title_bout,
                r_wins_total, r_losses_total, r_age, r_height, r_weight, r_reach, *r_stance_one_hot,
                r_SLpM_total, r_SApM_total, r_sig_str_acc_total, r_td_acc_total, r_str_def_total, 
                r_td_def_total, r_sub_avg, r_td_avg,

                b_wins_total, b_losses_total, b_age, b_height, b_weight, b_reach, *b_stance_one_hot,
                b_SLpM_total, b_SApM_total, b_sig_str_acc_total, b_td_acc_total, b_str_def_total,
                b_td_def_total, b_sub_avg, b_td_avg
            ]
            
            prediction = model_full.predict([features])[0]

    except ValueError:
        return render_template('index.html', error="Invalid input: Please ensure all numeric fields have valid numbers.")
    except Exception as e:
        return render_template('index.html', error=f"Unexpected error: {str(e)}")

    winner = 'Red' if prediction == 0 else 'Blue'
    return render_template('index.html', prediction=winner)

if __name__ == '__main__':
    app.run(debug=True)