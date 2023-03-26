##########################
# 0. PACKAGES AND SETTINGS
##########################


# 0.1 Load libraries

import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from functions import load_data, plot_classification_loss


# 0.2 Settings

data_path = '*' # * = path to MM-Fit dataset
w_size = 300  # defines size of sliding segmentation window



##############
# 1. LOAD DATA
##############


# ID's of exercise weeks for training and validation data ('w06' and 'w17' are used for testing)
train_week_ids = ['w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w07', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w18', 
               'w19', 'w20']
val_week_ids = ['w08', 'w16']

# Extract accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise.
train_weeks = load_data(data_path, train_week_ids)
val_weeks = load_data(data_path, val_week_ids)



##################
# 2. PREPROCESSING 
##################


# 2.1 STANDARDIZE DATA

# Stack accelerometer and gyroscope data of all training weeks.
train_data_stacked = np.concatenate([item[0] for item in train_weeks], axis=0)

# Fit scaler on training data.
scaler = StandardScaler()
scaler.fit(train_data_stacked)

# Transform training data week-wise and merge it with corresponding exercise types.
train_weeks_scaled = []
for ind in range(len(train_weeks)):
    acc_gyr_scaled = scaler.transform(train_weeks[ind][0])
    df = pd.DataFrame(data = list(zip(acc_gyr_scaled[:, 0], acc_gyr_scaled[:, 1], acc_gyr_scaled[:, 2],
                                     acc_gyr_scaled[:, 3], acc_gyr_scaled[:, 4], acc_gyr_scaled[:, 5], train_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    train_weeks_scaled.append(df)

# Transform validation data week-wise and merge it with corresponding exercise types.
val_weeks_scaled = []
for ind in range(len(val_weeks)):
    acc_gyr_scaled = scaler.transform(val_weeks[ind][0])
    df = pd.DataFrame(data = list(zip(acc_gyr_scaled[:, 0], acc_gyr_scaled[:, 1], acc_gyr_scaled[:, 2],
                                     acc_gyr_scaled[:, 3], acc_gyr_scaled[:, 4], acc_gyr_scaled[:, 5], val_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    val_weeks_scaled.append(df)
    
    
# 2.2 RETRIEVE EXERCISE SEGMENTS ONLY
    
# generate dataframes from training data which only contain one exercise set each
train_data = []
for workout in train_weeks_scaled:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _,g in workout.groupby(group)]
    train_data.append(segments)
train_data = [item for sublist in train_data for item in sublist]  # flatten list of lists
train_data = [seg for seg in train_data if seg['exercise'].all() != 'none'] # eliminate 'rest' segments

# generate dataframes from validation data which only contain one exercise set each
val_data = []
for workout in val_weeks_scaled:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _,g in workout.groupby(group)]
    val_data.append(segments)
val_data = [item for sublist in val_data for item in sublist]  # flatten list of lists
val_data = [seg for seg in val_data if seg['exercise'].all() != 'none'] # eliminate 'rest' segments


# 2.3 NUMERICAL ENCODING OF EXERCISE LABELS

# fit label encoder on all training data weeks, which now do not contain 'rest' as an activity class anymore
# fitting it on training AND validation data is not necessary as training data already contains all possible exercise types
label_enc = LabelEncoder()
label_enc.fit(pd.concat([seg for seg in train_data])['exercise'])

# encode exercise types in training data
for segment in train_data:
    segment['exercise_label'] = label_enc.transform(segment['exercise']) 

# encode exercise types in validation data
for segment in val_data:
    segment['exercise_label'] = label_enc.transform(segment['exercise'])
    
    
# 2.4 DETERMINE FEATURES AND LABELS
    
# Extract acc+gyr data and exercise class label from training data
X_train = []
y_train = []
for segment in train_data:
    X_train.append(segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
    y_train.append(stats.mode(segment['exercise_label'])[0][0])

# Extract acc+gyr data and exercise class label from validation data   
X_val = []
y_val = []
for segment in val_data:
    X_val.append(segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']]) 
    y_val.append(stats.mode(segment['exercise_label'])[0][0])
    
    
# 2.6 PAD FEATURES
    
# Pad training and validation data with a relatively large padding 
# Purpose: Theoretically, the segmentation model in the test file might predict 
# oddly large exercise segments, which are then passed down to the exercise 
# recognition model. If the exercise reognition model has an input length, that 
# is chosen too small, the code might break at that point.
pad_length = 30000 # corresponds to length of longest exercise or rest segment found in the MMFit dataset
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=pad_length, dtype='float32')
X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=pad_length, dtype='float32')


# 2.7 RESHAPE FEATURES

# Purpose: 2D CNN requires input shape of (n_batch, n_rows = 30000, n_columns = 6, n_channels = 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# turn labels from lists into arrays
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)



######################################
# 3. CONV2D EXERCISE RECOGNITION MODEL
######################################


# 3.1 MODEL SET-UP

model_2 = keras.Sequential()
model_2.add(Conv2D(filters = 4, kernel_size = (6, 6), strides = 6, padding='valid', activation='relu',
                    input_shape=X_train[0].shape))
model_2.add(MaxPooling2D(pool_size = (3, 1), strides = 3, padding='valid'))
model_2.add(Conv2D(filters = 12, kernel_size = (3, 1), strides = 3, padding='valid', activation='relu'))
model_2.add(MaxPooling2D(pool_size = (3, 1), strides = 3, padding='valid'))
model_2.add(SpatialDropout2D(rate = 0.4))
model_2.add(Conv2D(filters = 128, kernel_size = (3, 1), strides = 3, padding='valid', activation='relu'))
model_2.add(SpatialDropout2D(rate = 0.6))
model_2.add(Flatten())
model_2.add(Dense(units = 10, activation='softmax'))


# 3.2 RUN MODEL AND PLOT LOSS AND ACCURACY STATISTICS

model_2.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model_2.summary()
history_2 = model_2.fit(X_train, y_train, batch_size = 64, epochs = 45, validation_data = (X_val, y_val), verbose = 1)
plot_classification_loss(history_2)


################    
# 4. SAVE MODEL
################
    
    
# Save model under a path of your choice
# model_2.save('*/exercise_recognition.h5')   # for '*', insert the path to saving location