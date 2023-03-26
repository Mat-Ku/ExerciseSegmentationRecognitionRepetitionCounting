##########################
# 0. PACKAGES AND SETTINGS
##########################


# 0.1 Load libraries

import os
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.io import loadmat
from scipy.signal import savgol_filter
from functions import load_data, plot_regression_loss

import warnings
warnings.filterwarnings('ignore')


# 0.2 Settings

mmfit_path = '*' # * = path to MM-Fit dataset
recofit_path = '*' # * = path to RecoFit dataset



#####################
# 1. LOAD MM-FIT DATA
#####################


# ID's of exercise weeks for training and validation data ('w06' and 'w17' are used for testing)
train_week_ids = ['w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w07', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 
                  'w18', 'w19', 'w20']
val_week_ids = ['w08', 'w16']

# Extract accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise.
train_weeks = load_data(mmfit_path, train_week_ids)
val_weeks = load_data(mmfit_path, val_week_ids)



######################
# 2. LOAD RECOFIT DATA
######################

# loading file 'exercise_data.50.0000_singleonly.mat' is sufficient
recofit_data = loadmat(recofit_path + '/exercise_data.50.0000_singleonly.mat')



#######################################
# 3. RETRIEVE MM-FIT EXERCISE SEGMENTS
#######################################

# create dataframes containing acc+gyr signal and corresponding exercise type
train_weeks_dfs = []
for ind in range(len(train_weeks)):
    df = pd.DataFrame(data = list(zip(train_weeks[ind][0][:, 0], train_weeks[ind][0][:, 1], train_weeks[ind][0][:, 2],
                                     train_weeks[ind][0][:, 3], train_weeks[ind][0][:, 4], train_weeks[ind][0][:, 5],
                                     train_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    train_weeks_dfs.append(df)
    
val_weeks_dfs = []
for ind in range(len(val_weeks)):
    df = pd.DataFrame(data = list(zip(val_weeks[ind][0][:, 0], val_weeks[ind][0][:, 1], val_weeks[ind][0][:, 2],
                                     val_weeks[ind][0][:, 3], val_weeks[ind][0][:, 4], val_weeks[ind][0][:, 5],
                                     val_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'exercise'])
    val_weeks_dfs.append(df)


# generate dataframes from training data which only contain one exercise set each
train_data_mf = []
for workout in train_weeks_dfs:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _,g in workout.groupby(group)]
    train_data_mf.append(segments)
train_data_mf = [item for sublist in train_data_mf for item in sublist]  # flatten list of lists
train_data_mf = [seg for seg in train_data_mf if seg['exercise'].all() != 'none'] # eliminate 'rest' segments
train_data_mf = [np.asarray(seg)[:, :6] for seg in train_data_mf] # get rid of exercise type column
train_data_mf = deepcopy([seg[::2] for seg in train_data_mf]) # downsample training segments to 50 Hz

# generate dataframes from validation data which only contain one exercise set each
val_data_mf = []
for workout in val_weeks_dfs:
    group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
    segments = [g for _,g in workout.groupby(group)]
    val_data_mf.append(segments)
val_data_mf = [item for sublist in val_data_mf for item in sublist]  # flatten list of lists
val_data_mf = [seg for seg in val_data_mf if seg['exercise'].all() != 'none'] # eliminate 'rest' segments
val_data_mf = [np.asarray(seg)[:, :6] for seg in val_data_mf] # get rid of exercise type column
val_data_mf = deepcopy([seg[::2] for seg in val_data_mf]) # downsample validation segments to 50 Hz



#######################################
# 4. RETRIEVE RECOFIT EXERCISE SEGMENTS
#######################################

# Recofit dataset contains more than 70 different exercise types. 
# The following are the ones that are similar to the MM-Fit exercises.
target_ex = [# squat variations
             'Squat (arms in front of body, parallel to ground)', 'Dumbbell Squat (hands at side)', 'Squat Jump',
             'Squat', 'Squat (hands behind head)', 'Squat (kettlebell / goblet)',
             # pushup variations 
             'Pushups', 'Pushup (knee or foot variation)', 
             # shoulder press variations
             'Shoulder Press (dumbbell)', 
             # lunge variations
             'Lunge (alternating both legs, weight optional)', 'Walking lunge', 
             # rowing variations
             'Dumbbell Deadlift Row', 'Dumbbell Row (knee on bench) (right arm)',
             # sit-up and crunch variations
             'Sit-up (hands positioned behind head)', 'Sit-ups', 'Butterfly Sit-up', 'Crunch', 'V-up',
             # triceps extension variations
             'Overhead Triceps Extension', 
             # bicep curl variations
             'Bicep Curl', 
             # later raise variations
             'Lateral Raise', 
             # jumping jack variations
             'Jumping Jacks',
             # other exercises which are different from MM-Fit exercises but were chosen for introducing more repetition 
             # counts lower than 10
             'Burpee', 'Seated Back Fly', 'Kettlebell Swing', 'Dip', 'Chest Press (rack)', 'Box Jump (on bench)',
             'Lawnmower (right arm)', 'Triceps Kickback (knee on bench) (right arm)', 
             'Two-arm Dumbbell Curl (both arms, not alternating)']


# 4.1 EXTRACT ACCELEROMETER AND GYROSCOPE DATA, REPETITION COUNT LABELS AND EXERCISE NAMES

# Nested loop runs through all participants/subjects (= outer loop) and their performed exercises (= inner loop).
# If an exercise is contained in the list 'target_ex', its subject ID, exercise index, exercise name, accelerometer data and
# gyroscope data will be appended to target_ex_data.
target_ex_data = []
subject_id = -1

for subject in recofit_data['subject_data']:  
    subject_id += 1 # assign an ID to each subject (first subject will have ID = 0)
    
    for i in range(len(subject)): 
        ex_ind = i # get index of exercise within all exercises performed by subject
        
        try:
            ex_name = subject[i][0, 0][5][0] # retrieve exercise name
            if ex_name in target_ex:
                reps = subject[i][0, 0][15][0, 0] # retrieve count of repetitions performed by subject in this exercise
                ex_acc = subject[i][0, 0][14][0, 0][0] # retrieve accelerometer data of respective exercise
                ex_gyr = subject[i][0, 0][14][0, 0][1] # retrieve gyroscope data of respective exercise
                target_ex_data.append([subject_id, ex_ind, ex_name, reps, ex_acc, ex_gyr]) # append data to target_ex_data list    
            else: # if exercise is not a target exercise, continue and check if the next is a target exercise
                continue
        except IndexError:  # except statement prevents code from crashing if subject data has index lengths that are incompatible
            continue        
target_ex_data = np.asarray(target_ex_data)


# 4.2 FILTER OUT EXERCISE SETS WITH ERRONEOUS REPETITION COUNTS 

# e.g. 0 repetitions or negative repetition counts
target_ex_data_clean = [item for item in target_ex_data if item[3] > 0]


# 4.3 DIVIDE FEATURES FROM LABELS

data_rf = [] # stores acc+gyr data of exercise segments
labels_rf = [] # stores corresponding repetition counts
for seg in target_ex_data_clean:
    # join accelerometer and gyroscope data in one array and store them
    data_rf.append(np.concatenate((seg[4][:, 1:], seg[5][:, 1:]), axis = 1))
    # retrieve repetition count
    labels_rf.append(seg[3])
    

# 4.4 DELETE UNSUITABLE INSTANCES

# Delete samples with repetition counts larger than 29
# Reason: High repetition counts are unrealistic compared to repetition counts of MM-Fit data. Not filtering them out 
#          can cause test data to perform badly on CNN, since behaviour of CNN might be skewed towards predicting high counts.
del_inds_30 = list(np.where(np.asarray(labels_rf) >= 30)[0])
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_30]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_30]


# Delete samples whose length is larger than 4000 (because some outliers have length of more than 8000 timesteps)
# Reason: Those long samples tend to have much higher repetition counts than the samples in the MMFit dataset. Using
#          them for training the CNN will not contribute to predicting the MMFit samples better. 
del_inds_4000 = [i for i in range(len(data_rf)) if len(data_rf[i]) > 4000]
data_rf = [data_rf[i] for i in range(len(data_rf)) if i not in del_inds_4000]
labels_rf = [labels_rf[i] for i in range(len(labels_rf)) if i not in del_inds_4000]


# Delete samples which show misleading signal according to visual inspection. 
# Reason: They might hinder the CNN from predicting amount of peaks correctly.
# Remark: In order to create 'del_list', all time series contained in 'data_rf' were printed and visually inspected with
#         respect to whether their signal is interpretable for the human eye. The indices of the time series that did not 
#         pass this visual test were stored in 'del_list' and are deleted from the data.
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


# 4.5 SPLIT RECOFIT DATA IN TRAINING AND VALIDATION SET

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


#####################
# 5. STANDARDIZE DATA
#####################


# 5.1 FIT STANDARDSCALER ON TRAINING SETS OF RECOFIT AND MM-FIT DATA

# merge training set of mm-fit data and recofit data
mf_rf_merged = [train_data_mf + train_data_rf]

# Stack accelerometer and gyroscope data of all training weeks.
train_data_stacked = np.concatenate([item for item in mf_rf_merged[0]], axis=0)

# Fit scaler on training data
scaler = StandardScaler()
scaler.fit(train_data_stacked)


# 5.2 TRANSFORM TRAINING AND VALIDATION DATA OF BOTH DATASETS

temp = []
for seg in train_data_mf:
    temp.append(scaler.transform(seg))
train_data_mf = temp

temp = []
for seg in val_data_mf:
    temp.append(scaler.transform(seg))
val_data_mf = temp

temp = []
for seg in train_data_rf:
    temp.append(scaler.transform(seg))
train_data_rf = temp

temp = []
for seg in val_data_rf:
    temp.append(scaler.transform(seg))
val_data_rf = temp



###########################
# 6. PREPROCESS MM-FIT DATA
###########################


# 6.1 RETRIEVE MM-FIT LABELS

# Extract repetition count labels for training data from MMFit dataset
train_labels_mf = []
for id_ in train_week_ids: 
    file = os.path.join(mmfit_path, id_ + '/', id_ + '_labels.csv')
    df_labels = pd.read_csv(file, header=None)
    list_labels = [list(row) for row in df_labels.values]
    train_labels_mf.append([l[2] for l in list_labels])
train_labels_mf = [item for sublist in train_labels_mf for item in sublist]
 
# Extract repetition count labels for validation data from MMFit dataset
val_labels_mf = []
for id_ in val_week_ids:    
    file = os.path.join(mmfit_path, id_ + '/', id_ + '_labels.csv')
    df_labels = pd.read_csv(file, header=None)
    list_labels = [list(row) for row in df_labels.values]
    val_labels_mf.append([l[2] for l in list_labels])
val_labels_mf = [item for sublist in val_labels_mf for item in sublist]


# 6.2 SAVITZKY-GOLAY FILTER

# Delete very short segments
# Savitzky-Golay Filter is applied with window length of 75 datapoints.
# Segments shorter than that cannot be considered.
hz=50
del_segs_train = []
for i in range(len(train_data_mf)):
    if len(train_data_mf[i]) < 1.5*hz:
        del_segs_train.append(i)
    else:
        continue
train_data_mf = [train_data_mf[i] for i in range(len(train_data_mf)) if i not in del_segs_train]
train_labels_mf = [train_labels_mf[i] for i in range(len(train_labels_mf)) if i not in del_segs_train]

del_segs_val = []
for i in range(len(val_data_mf)):
    if len(val_data_mf[i]) < 1.5*hz:
        del_segs_val.append(i)
    else:
        continue
val_data_mf = [val_data_mf[i] for i in range(len(val_data_mf)) if i not in del_segs_val]
val_labels_mf = [val_labels_mf[i] for i in range(len(val_labels_mf)) if i not in del_segs_val]

# Apply Savitzky-Golay filter
for seg in train_data_mf:
    seg[:, 0] = savgol_filter(seg[:, 0], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 1] = savgol_filter(seg[:, 1], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 2] = savgol_filter(seg[:, 2], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 3] = savgol_filter(seg[:, 3], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 4] = savgol_filter(seg[:, 4], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 5] = savgol_filter(seg[:, 5], window_length=int(1.5*hz), polyorder=3, axis=0)

for seg in val_data_mf:
    seg[:, 0] = savgol_filter(seg[:, 0], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 1] = savgol_filter(seg[:, 1], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 2] = savgol_filter(seg[:, 2], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 3] = savgol_filter(seg[:, 3], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 4] = savgol_filter(seg[:, 4], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 5] = savgol_filter(seg[:, 5], window_length=int(1.5*hz), polyorder=3, axis=0)


# 6.3 PRINCIPAL COMPONENT ANALYSIS

X_train_mf = []
for seg in train_data_mf:
    pca = PCA(n_components=1)
    pca.fit(seg)
    X_train_mf.append(pca.transform(seg))

X_val_mf = []
for seg in val_data_mf:
    pca = PCA(n_components=1)
    pca.fit(seg)
    X_val_mf.append(pca.transform(seg))



############################
# 7. PREPROCESS RECOFIT DATA
############################
    
    
# 7.1 SAVITZKY-GOLAY FILTER

# Delete very short segments
# Savitzky-Golay Filter is applied with window length of 75 datapoints.
# Segments shorter than that cannot be considered.
del_segs_train = []
for i in range(len(train_data_rf)):
    if len(train_data_rf[i]) < 1.5*hz:
        del_segs_train.append(i)
    else:
        continue
train_data_rf = [train_data_rf[i] for i in range(len(train_data_rf)) if i not in del_segs_train]
train_labels_rf = [train_labels_rf[i] for i in range(len(train_labels_rf)) if i not in del_segs_train]

del_segs_val = []
for i in range(len(val_data_rf)):
    if len(val_data_rf[i]) < 1.5*hz:
        del_segs_val.append(i)
    else:
        continue
val_data_rf = [val_data_rf[i] for i in range(len(val_data_rf)) if i not in del_segs_val]
val_labels_rf = [val_labels_rf[i] for i in range(len(val_labels_rf)) if i not in del_segs_val]

# Apply Savitzky-Golay filter
for seg in train_data_rf:
    seg[:, 0] = savgol_filter(seg[:, 0], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 1] = savgol_filter(seg[:, 1], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 2] = savgol_filter(seg[:, 2], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 3] = savgol_filter(seg[:, 3], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 4] = savgol_filter(seg[:, 4], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 5] = savgol_filter(seg[:, 5], window_length=int(1.5*hz), polyorder=3, axis=0)

for seg in val_data_rf:
    seg[:, 0] = savgol_filter(seg[:, 0], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 1] = savgol_filter(seg[:, 1], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 2] = savgol_filter(seg[:, 2], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 3] = savgol_filter(seg[:, 3], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 4] = savgol_filter(seg[:, 4], window_length=int(1.5*hz), polyorder=3, axis=0)
    seg[:, 5] = savgol_filter(seg[:, 5], window_length=int(1.5*hz), polyorder=3, axis=0)


# 7.2 PRINCIPAL COMPONENT ANALYSIS

X_train_rf = []
for seg in train_data_rf:
    pca = PCA(n_components=1)
    pca.fit(seg)
    X_train_rf.append(pca.transform(seg))

X_val_rf = []
for seg in val_data_rf:
    pca = PCA(n_components=1)
    pca.fit(seg)
    X_val_rf.append(pca.transform(seg))



################################
# 8. MERGE, PAD AND RESHAPE DATA
################################

# Merge MM-Fit and RecoFit data sets
X_train = X_train_mf + X_train_rf
X_val = X_val_mf + X_val_rf
y_train = train_labels_mf + train_labels_rf
y_val = val_labels_mf + val_labels_rf

# Padding 
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=30000, dtype='float32')
X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=30000, dtype='float32')

# Reshaping
# Conv1D requires input shape of (n_rows, n_columns, n_channels=1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)



#####################################
# 9. CONV1D REPETITION COUNTING MODEL
#####################################

# 9.1 MODEL SET-UP

model_3 = keras.Sequential()
model_3.add(Conv1D(filters = 8, kernel_size = 10, strides = 10, activation = 'relu', 
                   input_shape=(X_train.shape[1], 1)))  # input_shape = (n_rows=30000, n_cols=1, n_channels=1)                                                                   
model_3.add(MaxPooling1D(pool_size = 3, strides = 3)) 
model_3.add(Conv1D(filters = 32, kernel_size = 3, strides = 1, activation = 'relu')) 
model_3.add(Conv1D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu')) 
model_3.add(MaxPooling1D(pool_size = 3, strides = 3)) 
model_3.add(Conv1D(filters = 128, kernel_size = 3, strides = 1, activation = 'relu')) 
model_3.add(MaxPooling1D(pool_size = 3, strides = 3))
model_3.add(Flatten())
model_3.add(Dense(units = 1)) # No activation function needs to be specified. Neuron uses linear activation (which is necessary
                      # for regression) by default.  


# 9.2 RUN MODEL AND PLOT LOSS AND ACCURACY STATISTICS

model_3.compile(optimizer=Adam(learning_rate = 0.001), loss = 'mse', metrics = ['mae', 'mse'])
model_3.summary()
history_3 = model_3.fit(X_train, y_train, batch_size = 8, epochs = 25, validation_data=(X_val, y_val), verbose = 1)
plot_regression_loss(history_3)


# 9.3 SAVE MODEL

# Save model under a path of your choice
#model_3.save('*/repetition_counting.h5')   # for '*', insert the path to saving location
