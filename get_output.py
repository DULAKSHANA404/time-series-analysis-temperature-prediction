from keras.models import load_model
import pandas as pd
import joblib
import numpy as np

model_path = r"C:\Users\user\Desktop\temp_predictor\RNN\model\model.keras"

model = load_model(model_path)

data = [29.9]  #enter here today temp, this will get tomorrow temp
data = np.array(data).reshape(-1,1)


scaler_path = r"C:\Users\user\Desktop\temp_predictor\RNN\data\scaler.pkl"

scaler = joblib.load(scaler_path)

data_new = scaler.transform(data)

predict = model.predict(data_new)
accuracy = np.max(predict,axis=1)[0]

result= scaler.inverse_transform(predict)[0]
print(result)

