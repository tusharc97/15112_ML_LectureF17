from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Activation

# boston housing data
(x_train,y_train),(x_test,y_test) = boston_housing.load_data()

# Create model
model = Sequential()

# Input layer; hidden layer
model.add(Dense(8,input_dim=13,activation='sigmoid'))

# Output layer
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

model.fit(x_train, y_train, epochs=10000)

prediction = model.predict(x_test)

print(prediction)

loss_and_metrics = model.evaluate(x_test, y_test)

print(loss_and_metrics)
