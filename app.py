import numpy as np
from flask import Flask, request, render_template
import pickle


#create flask app name
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# prediction function
def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 12)
	loaded_model = pickle.load(open("model.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(int, to_predict_list))
		result = ValuePredictor(to_predict_list)	
		if int(result)== 1:
			prediction ='Income is >50K $'
		else:
			prediction ='Income is <=50K $'		
		return render_template("index.html", prediction_text = prediction)

if __name__ == "__main__":
    app.run()
