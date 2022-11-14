from flask import Flask
from flask import request
from flask import jsonify
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=2.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum",methods=['POST'])
def sum():
    print(request.json)
    x=request.json['x']
    y=request.json['y']
    return jsonify({'sum':x+y})

@app.route("/predict", methods=['POST'])
def predict_digit():
    image_output1 = request.json['image1']
    image_output2 = request.json['image2']
    print("done loading")
    predicted_image1 = model.predict([image_output1])
    predicted_image2 = model.predict([image_output2])
    return {"Comparison_result": True if int(predicted_image1[0])==int(predicted_image2[0]) else False}


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)