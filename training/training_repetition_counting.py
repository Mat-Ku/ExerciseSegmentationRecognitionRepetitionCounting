from copy import deepcopy
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import warnings

from configuration import Configuration
from utils.data_loading import DataLoading
from utils.plotting import Plotting
from utils.processing import Processing

warnings.filterwarnings('ignore')


mmfit_path = Configuration.Constants.MMFIT_DATA_PATH  # path to MM-Fit dataset
recofit_path = Configuration.Constants.RECOFIT_DATA_PATH  # path to RecoFit dataset

# Load MM-Fit data
# Extract accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise.
train_weeks = DataLoading.load_data(mmfit_path, Configuration.Constants.TRAINING_WEEK_IDS)
val_weeks = DataLoading.load_data(mmfit_path, Configuration.Constants.VALIDATION_WEEK_IDS)

# Load Recofit data
# loading file 'exercise_data.50.0000_singleonly.mat' is sufficient
recofit_data = loadmat(recofit_path + '/exercise_data.50.0000_singleonly.mat')

# Retrieve MM-Fit exercise types
train_weeks_dfs = []
for ind in range(len(train_weeks)):
    df = pd.DataFrame(data=list(zip(train_weeks[ind][0][:, 0], train_weeks[ind][0][:, 1], train_weeks[ind][0][:, 2],
                                    train_weeks[ind][0][:, 3], train_weeks[ind][0][:, 4], train_weeks[ind][0][:, 5],
                                    train_weeks[ind][4])),
                      columns=['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    train_weeks_dfs.append(df)
    
val_weeks_dfs = []
for ind in range(len(val_weeks)):
    df = pd.DataFrame(data=list(zip(val_weeks[ind][0][:, 0], val_weeks[ind][0][:, 1], val_weeks[ind][0][:, 2],
                                    val_weeks[ind][0][:, 3], val_weeks[ind][0][:, 4], val_weeks[ind][0][:, 5],
                                    val_weeks[ind][4])),
                      columns=['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    val_weeks_dfs.append(df)

# Generate dataframes from training data which only contain one exercise set each
train_data_mf = []
for workout in train_weeks_dfs:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _, g in workout.groupby(group)]
    train_data_mf.append(segments)
train_data_mf = [item for sublist in train_data_mf for item in sublist]  # flatten list of lists
train_data_mf = [seg for seg in train_data_mf if seg['exercise'].all() != 'none']  # eliminate 'rest' segments
train_data_mf = [np.asarray(seg)[:, :6] for seg in train_data_mf]  # delete exercise type column
train_data_mf = deepcopy([seg[::2] for seg in train_data_mf])  # downsample training segments to 50 Hz

# Generate dataframes from validation data which only contain one exercise set each
val_data_mf = []
for workout in val_weeks_dfs:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _,g in workout.groupby(group)]
    val_data_mf.append(segments)
val_data_mf = [item for sublist in val_data_mf for item in sublist]  # flatten list of lists
val_data_mf = [seg for seg in val_data_mf if seg['exercise'].all() != 'none']  # eliminate 'rest' segments
val_data_mf = [np.asarray(seg)[:, :6] for seg in val_data_mf]  # delete exercise type column
val_data_mf = deepcopy([seg[::2] for seg in val_data_mf])  # downsample validation segments to 50 Hz

# Retrieve Recofit exercise segments
# Recofit dataset contains more than 70 exercise types. Only those matching with  MM-Fit exercises are considered.
target_ex = Configuration.Constants.RECOFIT_TARGET_EXERCISES

# Retrieve accelerometer and gyroscope data, repetition count labels and exercise names
# Nested loop runs through all participants/subjects (= outer loop) and their performed exercises (= inner loop).
# If an exercise is contained in the list 'target_ex', its subject ID, exercise index, exercise name, accelerometer data
# and gyroscope data will be appended to target_ex_data.
target_ex_data = []
subject_id = -1

for subject in recofit_data['subject_data']:  
    subject_id += 1  # assign an ID to each subject
    
    for i in range(len(subject)): 
        ex_ind = i  # get index of exercise within all exercises performed by subject
        
        try:
            ex_name = subject[i][0, 0][5][0]  # retrieve exercise name
            if ex_name in target_ex:
                reps = subject[i][0, 0][15][0, 0]  # retrieve count of repetitions performed by subject in this exercise
                ex_acc = subject[i][0, 0][14][0, 0][0]  # retrieve accelerometer data of respective exercise
                ex_gyr = subject[i][0, 0][14][0, 0][1]  # retrieve gyroscope data of respective exercise
                target_ex_data.append([subject_id, ex_ind, ex_name, reps, ex_acc, ex_gyr])
            else:  # if exercise is not a target exercise, continue and check if the next is a target exercise
                continue
        # prevent code from crashing if subject data has index lengths that are incompatible
        except IndexError:
            continue        
target_ex_data = np.array(target_ex_data)

# Filter out exercise sets with erroneous repetition counts
# e.g. 0 repetitions or negative repetition counts
target_ex_data_clean = [item for item in target_ex_data if item[3] > 0]

# Divide features from labels
data_rf = []  # stores acc+gyr data of exercise segments
labels_rf = []  # stores corresponding repetition counts
for seg in target_ex_data_clean:
    # join accelerometer and gyroscope data in one array and store them
    data_rf.append(np.concatenate((seg[4][:, 1:], seg[5][:, 1:]), axis = 1))
    # retrieve repetition count
    labels_rf.append(seg[3])
    

# Delete unsuitable instances

# Delete samples with repetition counts larger than 29
# Reason: High repetition counts are unrealistic compared to repetition counts of MM-Fit data. Not filtering them out 
# can cause test data to perform badly on CNN, since behaviour of CNN might be skewed towards predicting high counts.
del_inds_30 = list(np.where(np.asarray(labels_rf) >= 30)[0])
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_30]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_30]

# Delete samples whose length is larger than 4000 (because some outliers have length of more than 8000 timesteps)
# Reason: Those long samples tend to have much higher repetition counts than the samples in the MMFit dataset. Using
# them for training the CNN will not contribute to predicting the MMFit samples better.
del_inds_4000 = [i for i in range(len(data_rf)) if len(data_rf[i]) > 4000]
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_4000]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_4000]

# Delete samples which show misleading signal according to visual inspection. 
# Reason: They might hinder the CNN from predicting amount of peaks correctly.
# Remark: In order to create 'del_list', all time series contained in 'data_rf' were printed and visually inspected with
# respect to whether their signal is interpretable for the human eye. The indices of the time series that did not
# pass this visual test were stored in 'del_list' and are deleted from the data.
del_list = [0, 41, 42, 86, 87, 105, 118, 122, 139, 144, 155, 163, 170, 172, 182, 195, 206, 225, 253, 263, 291, 299, 310, 
           315, 319, 323, 336, 337, 338, 341, 344, 357, 365, 378, 388, 401, 432, 437, 438, 443, 453, 454, 455, 465, 475, 
           478, 498, 501, 506, 530, 536, 542, 544, 574, 582, 601, 618, 652, 668, 719]
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_list]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_list]

# Reduce the number of exercise sets whose repetition count is 20 
# Reason: Exercises with repetition counts of 20 are overrepresented in the dataset and CNN might overfit on them
# Get indices of all repetition counts == 20
del_inds_20 = [i for i in range(len(labels_rf)) if labels_rf[i] == 20]
# Select all indices, except the last 20. 
# This assures that the reduced dataset still contains 20 samples with 20 repetitions each.
del_inds_20 = del_inds_20[:-20]
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_20]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_20]

# Reduce the number of exercise sets whose repetition count is 15 
# Reason: Exercises with repetition counts of 15 are overrepresented in the dataset and CNN might overfit on them
# Get indices of all repetition counts == 15
del_inds_15 = [i for i in range(len(labels_rf)) if labels_rf[i] == 15]
# Select all indices, except the last 20 
del_inds_15 = del_inds_15[:-20]
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_15]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_15]

# 4.5 Split Recofit data into training and validation set

# determine how much % of MMFit training+validation data is validation data
val_ratio = len(val_data_mf) / (len(train_data_mf) + len(val_data_mf))

# use this value to determine number of samples of Recofit data which shall be used for validation
val_count_rf = int(len(data_rf) * val_ratio)

# randomly select as many indices from 'data_rf'
val_inds_rf = random.sample(range(0, len(data_rf)), val_count_rf)

# split training and validation data based on those indices
train_data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in sorted(val_inds_rf)]
train_labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in sorted(val_inds_rf)]
val_data_rf = [data_rf[i] for i in range(len(data_rf)) if i in sorted(val_inds_rf)]
val_labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i in sorted(val_inds_rf)]

# Standardize data
# merge training set of MM-Fit data and Recofit data
mf_rf_merged = [train_data_mf + train_data_rf]
# Stack accelerometer and gyroscope data of all training weeks.
train_data_stacked = np.concatenate([item for item in mf_rf_merged[0]], axis=0)
# Fit scaler on training data
scaler = StandardScaler()
scaler.fit(train_data_stacked)

train_data_mf = [scaler.transform(segment) for segment in train_data_mf]
val_data_mf = [scaler.transform(segment) for segment in val_data_mf]
train_data_rf = [scaler.transform(segment) for segment in train_data_rf]
val_data_rf = [scaler.transform(segment) for segment in val_data_rf]

# Retrieve MM-Fit labels
# Extract repetition count labels for training data from MMFit dataset
train_labels_mf = DataLoading.load_repetition_counts(mmfit_path, Configuration.Constants.TRAINING_WEEK_IDS)
val_labels_mf = DataLoading.load_repetition_counts(mmfit_path, Configuration.Constants.VALIDATION_WEEK_IDS)

# Savitzky-Golay-Filtering for MM-Fit data

# Delete very short segments
# Savitzky-Golay Filter is applied with window length of 75 datapoints.
# Segments shorter than that cannot be considered.
hz = 50
del_segs_train = [i for i in range(len(train_data_mf)) if len(train_data_mf[i]) < 1.5*hz]
train_data_mf = [train_data_mf[i] for i in range(len(train_data_mf)) if i not in del_segs_train]
train_labels_mf = [train_labels_mf[i] for i in range(len(train_labels_mf)) if i not in del_segs_train]

del_segs_val = [i for i in range(len(val_data_mf)) if len(val_data_mf[i]) < 1.5*hz]
val_data_mf = [val_data_mf[i] for i in range(len(val_data_mf)) if i not in del_segs_val]
val_labels_mf = [val_labels_mf[i] for i in range(len(val_labels_mf)) if i not in del_segs_val]

# Apply Savitzky-Golay filter
train_data_mf = Processing.savitzky_golay_filter(data=train_data_mf, window_length=int(1.5*hz), polyorder=3, axis=0)
val_data_mf = Processing.savitzky_golay_filter(data=val_data_mf, window_length=int(1.5*hz), polyorder=3, axis=0)

# Principal Component Analysis
X_train_mf = Processing.pca(data=train_data_mf, n_components=1)
X_val_mf = Processing.pca(data=val_data_mf, n_components=1)

# Savitzky-Golay-Filtering for Recofit data

# Delete very short segments
# Savitzky-Golay Filter is applied with window length of 75 datapoints.
# Segments shorter than that cannot be considered.
del_segs_train = [i for i in range(len(train_data_rf)) if len(train_data_rf[i]) < 1.5*hz]
train_data_rf = [train_data_rf[i] for i in range(len(train_data_rf)) if i not in del_segs_train]
train_labels_rf = [train_labels_rf[i] for i in range(len(train_labels_rf)) if i not in del_segs_train]

del_segs_val = [i for i in range(len(val_data_rf)) if len(val_data_rf[i]) < 1.5*hz]
val_data_rf = [val_data_rf[i] for i in range(len(val_data_rf)) if i not in del_segs_val]
val_labels_rf = [val_labels_rf[i] for i in range(len(val_labels_rf)) if i not in del_segs_val]

# Apply Savitzky-Golay filter
train_data_rf = Processing.savitzky_golay_filter(data=train_data_rf, window_length=int(1.5*hz), polyorder=3, axis=0)
val_data_rf = Processing.savitzky_golay_filter(data=val_data_rf, window_length=int(1.5*hz), polyorder=3, axis=0)

# Principal Component Analysis
X_train_rf = Processing.pca(data=train_data_rf, n_components=1)
X_val_rf = Processing.pca(data=val_data_rf, n_components=1)

# Merge, Pad and Reshape data

# Merge MM-Fit and RecoFit data sets
X_train = X_train_mf + X_train_rf
X_val = X_val_mf + X_val_rf
y_train = train_labels_mf + train_labels_rf
y_val = val_labels_mf + val_labels_rf

# Padding 
X_train = Processing.padding(data=X_train, window_size=30000)
X_val = Processing.padding(data=X_val, window_size=30000)

# Reshaping
X_train = Processing.reshaping(X_train)
X_val = Processing.reshaping(X_val)

# Conv1D Repetition Count Model
# Set-up model
model_3 = keras.Sequential()
# input_shape = (n_rows=30000, n_cols=1, n_channels=1)
model_3.add(tf.keras.layers.Conv1D(filters=8, kernel_size=10, strides=10, activation='relu',
                                   input_shape=(X_train.shape[1], 1)))
model_3.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))
model_3.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
model_3.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'))
model_3.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))
model_3.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu'))
model_3.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3))
model_3.add(tf.keras.layers.Flatten())
# No activation function needs to be specified. Neuron uses linear activation for regression by default
model_3.add(tf.keras.layers.Dense(units=1))

model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])
model_3.summary()
history_3 = model_3.fit(X_train, np.array(y_train), batch_size=8, epochs=25, validation_data=(X_val, np.array(y_val)),
                        verbose=1)

Plotting.plot_regression_loss(history_3)

# Save model
# model_3.save('*/repetition_counting.h5')   # for '*', insert the path to saving location
