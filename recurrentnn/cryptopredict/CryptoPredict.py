import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os

SEQ_LEN = 60  # Take the last 60 minutes
FUTURE_PERIOD_PREDICT = 3  # To predict the next 3 minutes
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

if not os.path.exists(f"logs\{NAME}"):
    os.makedirs(f"logs\{NAME}")

# Used to tell the network that it is good that the price increases
def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop('future', 1)

    # scale the columns
    for col in df.columns:
        if col != "target":  # Dont touch target
            df[col] = df[col].pct_change()  # Normalize the column
            df.dropna(inplace=True)  # Removes values that are not a number
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    sequential_data = []
    prev_hour = deque(maxlen=SEQ_LEN)

    for i in df.values: # Go through each row of the data
        prev_hour.append([n for n in i[:-1]]) # Dont include the target column
        if len(prev_hour) == SEQ_LEN:
            sequential_data.append([np.array(prev_hour), i[-1]])

    random.shuffle(sequential_data)

    # Balance the data
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


# We can put all the csv together with the index of time since they all have the same
main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])

    # Rename the columns
    df.rename(columns={
        "close": f"{ratio}_close",
        "volume": f"{ratio}_volume"
    }, inplace=True)

    df.set_index("time", inplace=True)
    # Only keep the close and volume columns
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    # Put the data together
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

# Add the column 'future' that contains the price 3 minutes in the future
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

# Add a column that says if it increased in the future or not
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

# Remove the last 5% to test !!Really important !!
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

# Split the data
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

model = Sequential()
model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))
filepath = "RNN_Final-{epoch:02d}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models\{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)


