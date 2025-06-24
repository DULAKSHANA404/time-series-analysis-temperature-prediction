import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

file_path = r"C:\Users\user\Desktop\temp_predictor\RNN\colombo_daily_temperatures_2018_2025.csv"
data_file = pd.read_csv(file_path).values

data = data_file[:,3]
print(data)

plt.plot(data)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

data = data.reshape(-1,1)

window_size = 100

scaler = MinMaxScaler(feature_range=(0,1))
data  = scaler.fit_transform(data)

data_list = []
target_list = []

for i in range(0,len(data)-window_size):
    data_list.append(data[i:i+window_size])
    target_list.append(data[i+window_size])
    
    
data = np.array(data_list)
target = np.array(target_list)

print(target.shape)
print(data.shape)

np.save("data.npy",data)
np.save("target.npy",target)
joblib.dump(scaler,"scaler.pkl")

