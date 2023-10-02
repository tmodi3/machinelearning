from flask import Flask, jsonify, request
import torch
import pandas as pd
from model import ReceiptsPredictor

app = Flask(__name__)

# Load model
model = ReceiptsPredictor()
model.load_state_dict(torch.load("model_state.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    month = request.json["month"]
    month_tensor = torch.tensor([[float(month)]])
    prediction = model(month_tensor).item()
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
