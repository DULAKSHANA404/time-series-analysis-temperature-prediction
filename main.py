from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

data_path = r"C:\Users\user\Desktop\temp_predictor\RNN\data\data.npy"
target_path = r"C:\Users\user\Desktop\temp_predictor\RNN\data\target.npy"
data = np.load(data_path)
target = np.load(target_path)

model = Sequential([])

model.add(LSTM(200,return_sequences=True,input_shape=(100,1)))
model.add(Dropout(0.2))
model.add(LSTM(200,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(150,activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="linear"))

model.compile(loss="mae",optimizer="adam")

model.summary()

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.2)

history = model.fit(train_data,train_target,epochs=20,validation_data=(test_data,test_target))

plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

model.save("model.keras")



