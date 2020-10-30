import flask
import pickle
import pandas as pd
import os

# Use pickle to load in the pre-trained model
with open(f'model_1.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        texture_mean = flask.request.form['texture_mean']
        perimeter_mean = flask.request.form['perimeter_mean']
        smoothness_mean = flask.request.form['smoothness_mean']
        compactness_mean = flask.request.form['compactness_mean']
        symmetry_mean = flask.request.form['symmetry_mean']
        input_variables = pd.DataFrame([[texture_mean, perimeter_mean, smoothness_mean,
                                        compactness_mean, symmetry_mean]],
                                       columns=['texture_mean',
                                                'perimeter_mean',
                                                'smoothness_mean',
                                                'compactness_mean',
                                                'symmetry_mean'])

        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Texture':texture_mean,
                                                     'Perimeter':perimeter_mean,
                                                     'Smoothness':smoothness_mean,
                                                     'Compactness':compactness_mean,
                                                     'Symmetry':symmetry_mean},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run(debug=True)