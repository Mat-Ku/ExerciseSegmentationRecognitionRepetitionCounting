import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf

from configuration import Configuration
from utils.processing import Processing
from utils.plotting import Plotting
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from utils.data_loading import MMFitDataLoading


# 1. LOAD AND PREPROCESS DATA

# Load data
data_path = Configuration.Constants.MMFIT_DATA_PATH
train_weeks = MMFitDataLoading.load_data(data_path, Configuration.Constants.TRAINING_WEEK_IDS)
val_weeks = MMFitDataLoading.load_data(data_path, Configuration.Constants.VALIDATION_WEEK_IDS)

# Standardize data
train_weeks_scaled = Processing.standardize(train_weeks)
val_weeks_scaled = Processing.standardize(val_weeks)

# Retrieve exercise segments only
train_data = Processing.get_exercise_segments(train_weeks_scaled)
val_data = Processing.get_exercise_segments(val_weeks_scaled)

# Numerical encoding of exercise labels
# Fit label encoder on all training data weeks, which now do not contain 'rest' as an activity class anymore.
# Fitting it on training AND validation data is not necessary as training data already contains all exercise types.
label_enc = LabelEncoder()
label_enc.fit(pd.concat([seg for seg in train_data])['exercise'])

# Encode exercise types in training and validation data
for segment in train_data:
    segment['exercise_label'] = label_enc.transform(segment['exercise'])
for segment in val_data:
    segment['exercise_label'] = label_enc.transform(segment['exercise'])
    
# Extract accelerometer and gyroscope data (features) and activities (labels)
X_train = [segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] for segment in train_data]
y_train = [stats.mode(segment['exercise_label'])[0][0] for segment in train_data]
X_val = [segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] for segment in val_data]
y_val = [stats.mode(segment['exercise_label'])[0][0] for segment in val_data]

# Pad training and validation data with a relatively large padding 
# Purpose: Theoretically, the segmentation model in the test file might predict oddly large exercise segments, which are
# then passed down to the exercise recognition model. If the exercise recognition model has an input length, that is
# chosen too small, the code might break at that point.
pad_length = 30000  # corresponds to length of longest exercise or rest segment found in the MMFit dataset
X_train_pad = Processing.padding(X_train, pad_length)
X_val_pad = Processing.padding(X_val, pad_length)

# Reshaping
X_train_pad_rs = Processing.reshaping(X_train_pad)
X_val_pad_rs = Processing.reshaping(X_val_pad)


# 2. CONV2D EXERCISE RECOGNITION MODEL

# Set-up model
model_2 = keras.Sequential()
model_2.add(tf.keras.layers.Conv2D(filters=4,
                                   kernel_size=(6, 6),
                                   strides=6,
                                   padding='valid',
                                   activation='relu',
                                   input_shape=X_train_pad_rs[0].shape))
model_2.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1),
                                         strides=3,
                                         padding='valid'))
model_2.add(tf.keras.layers.Conv2D(filters=12,
                                   kernel_size=(3, 1),
                                   strides=3,
                                   padding='valid',
                                   activation='relu'))
model_2.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 1),
                                         strides=3,
                                         padding='valid'))
model_2.add(tf.keras.layers.SpatialDropout2D(rate=0.4))
model_2.add(tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3, 1),
                                   strides=3,
                                   padding='valid',
                                   activation='relu'))
model_2.add(tf.keras.layers.SpatialDropout2D(rate=0.6))
model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(units=10,
                                  activation='softmax'))

# Train model
model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_2.summary()
history_2 = model_2.fit(x=X_train_pad_rs,
                        y=np.array(y_train),
                        batch_size=64,
                        epochs=45,
                        validation_data=(X_val_pad_rs, np.array(y_val)),
                        verbose=1)

Plotting.plot_classification_loss(history_2)

# Save model
# model_2.save('*/exercise_recognition.h5')   # for '*', insert the path to saving location
