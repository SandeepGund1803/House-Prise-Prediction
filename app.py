# import libraries
from flask import Flask, render_template, request
import numpy as np
import pickle


# pickle loading
model = pickle.load(open('house_prise.pkl', 'rb'))

# creating app
app = Flask(__name__)

 
@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    waterfront = request.form['waterfront']
    view = request.form['view']
    grade = request.form['grade']
    zipcode = request.form['zipcode']
    lat = request.form['lat']
    longi = request.form['longi']

    arr = np.array([[waterfront, view, grade, zipcode, lat, longi]],dtype=float)

    pred = model.predict(arr)
    return render_template('nextpage.html', data=pred)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False) 