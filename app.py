from flask import *
import joblib
app = Flask(__name__)

# Load the trained model
model = joblib.load("Fish_Classification_Model.pkl")
# Load the LabelEncoder object
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    input_features = [[weight, length1,length2,length3,height,height]]
    predicted_species = model.predict(input_features)
    predicted_species = label_encoder.inverse_transform(predicted_species)[0]

    return render_template('result.html', predicted_species=predicted_species)
    
if __name__ == '__main__':
     app.run(debug = True)