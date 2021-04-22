from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

data_loc = 'processed'
device_id = 3
train_model = True
training_df = pd.read_csv('{}/{}/{}'.format(data_loc, device_id, 'training.csv'), index_col=0)
testing_df = pd.read_csv('{}/{}/{}'.format(data_loc, device_id, 'testing.csv'), index_col=0)

start_time = time.time()

X_train_Val = training_df.loc[:, training_df.columns != 'class'].values

X_test = testing_df.loc[:, testing_df.columns != 'class'].values
y_test = testing_df['class'].values

X_train, X_val = train_test_split(X_train_Val, random_state=1337, test_size=0.2)

t = MinMaxScaler()
X_train = t.fit_transform(X_train)
X_val = t.transform(X_val)
X_test = t.transform(X_test)

n_inputs = X_train.shape[1]

# define encoder
visible = Input(shape=(n_inputs,))
# encoder level 1
e = Dense(n_inputs)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 2
e = Dense(int(n_inputs * 0.75))(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 3
e = Dense(int(n_inputs * 0.50))(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# encoder level 4
e = Dense(int(n_inputs * 0.33))(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
# bottleneck
# n_bottleneck = round(float(n_inputs) / 2.0)
n_bottleneck = int(n_inputs * 0.25)
bottleneck = Dense(n_bottleneck)(e)
# define decoder, level 1
d = Dense(int(n_inputs * 0.33))(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(int(n_inputs * 0.50))(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 3
d = Dense(int(n_inputs * 0.75))(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 4
d = Dense(int(n_inputs))(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
# define autoencoder model
model = Model(inputs=visible, outputs=output)
# compile autoencoder model
model.compile(optimizer=Adam(lr=0.012), loss='mse')

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=7, verbose=1, factor=0.1, min_lr=1e-6),
             EarlyStopping(monitor='val_loss', patience=16),
             ModelCheckpoint(filepath='models/autoencoder_{}'.format(device_id), monitor='val_loss',
                             save_best_only=True)]

# fit the autoencoder model to reconstruct input
if train_model:
    history = model.fit(X_train, X_train, epochs=500, batch_size=60, validation_data=(X_val, X_val),
                        callbacks=callbacks)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    plt.show()

# plot loss
model.load_weights('models/autoencoder_{}'.format(device_id))

X_val_prediction = model.predict(X_val)
mse = np.mean(np.power(X_val - X_val_prediction, 2), axis=1)
print("mean is %.5f" % mse.mean())
print("min is %.5f" % mse.min())
print("max is %.5f" % mse.max())
print("std is %.5f" % mse.std())

tr = mse.mean() + mse.std()
print("tr is {}".format(tr))
X_test_prediction = model.predict(X_test)
testing_mse = np.mean(np.power(X_test - X_test_prediction, 2), axis=1)

test_labels = np.asarray([0 if testing_mse[i] <= tr else 1 for i in range(len(testing_mse))])
y_true = np.asarray([0 if label == 0 else 1 for label in y_test])

cr = classification_report(y_true, test_labels)
print(cr)

print('Execution took {:.2f}'.format((time.time() - start_time) / 60))
