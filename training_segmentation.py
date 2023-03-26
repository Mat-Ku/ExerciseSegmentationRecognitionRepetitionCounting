##########################
# 0. PACKAGES AND SETTINGS
##########################

# 0.1 Load libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from functions import load_data, plot_classification_loss, plot_segmentation


# 0.2 Settings

data_path = '*' # * = path to MM-Fit dataset
w_size = 300  # defines size of sliding segmentation window



##############
# 1. LOAD DATA
##############


# 1.1 ID's of exercise weeks for training and validation data ('w06' and 'w17' are used for testing)

train_week_ids = ['w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w07', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w18', 
               'w19', 'w20']
val_week_ids = ['w08', 'w16']


# 1.2 Extract accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise.

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

# Transform training data week-wise and merge it with corresponding frames, week index, activity types and exercise types.
train_weeks_scaled = []
for ind in range(len(train_weeks)):
    acc_gyr_scaled = scaler.transform(train_weeks[ind][0])
    df = pd.DataFrame(data = list(zip(acc_gyr_scaled[:, 0], acc_gyr_scaled[:, 1], acc_gyr_scaled[:, 2],
                                     acc_gyr_scaled[:, 3], acc_gyr_scaled[:, 4], acc_gyr_scaled[:, 5],
                                     train_weeks[ind][1], train_weeks[ind][2], train_weeks[ind][3], train_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'frame', 'week', 'activity', 'exercise'])
    train_weeks_scaled.append(df)

# Transform validation data week-wise and merge it with corresponding frames, week index, activity types and exercise types.
val_weeks_scaled = []
for ind in range(len(val_weeks)):
    acc_gyr_scaled = scaler.transform(val_weeks[ind][0])
    df = pd.DataFrame(data = list(zip(acc_gyr_scaled[:, 0], acc_gyr_scaled[:, 1], acc_gyr_scaled[:, 2],
                                     acc_gyr_scaled[:, 3], acc_gyr_scaled[:, 4], acc_gyr_scaled[:, 5],
                                     val_weeks[ind][1], val_weeks[ind][2], val_weeks[ind][3], val_weeks[ind][4])),
                     columns = ['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'frame', 'week', 'activity', 'exercise'])
    val_weeks_scaled.append(df)


    
# 2.2 GENERATE WINDOWS

# Retrieve segments of predefined window size from training data.
train_segs = []
for week in train_weeks_scaled:
    for window in range(0, len(week), w_size):
        if window == 0:
            continue
        else:
            if window+w_size > len(week):
                seg_1 = week[window - w_size:window]
                seg_2 = week[window:]
                train_segs.append(seg_1)
                train_segs.append(seg_2) 
            else:
                seg = week[window - w_size:window]
                train_segs.append(seg)
                
# Retrieve segments of predefined window size from validation data.
val_segs = []
for week in val_weeks_scaled:
    val_week_segs = []
    for window in range(0, len(week), w_size):
        if window == 0:
            continue
        else:
            if window + w_size > len(week):
                seg_1 = week[window - w_size:window]
                seg_2 = week[window:]
                val_week_segs.append(seg_1)
                val_week_segs.append(seg_2) 
            else:
                seg = week[window - w_size:window]
                val_week_segs.append(seg)
    val_segs.append(val_week_segs)


    
# 2.3 DETERMINE FEATURES

# Extract accelerometer and gyroscope data for training data segments as features.
X_train = []
for seg in train_segs:
    X_train.append(seg[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
    
# Extract accelerometer and gyroscope data for validation data segments as features (week-wise).
X_val = [] # stores validation data for unembedded model, but all weeks are joined later in "2.6 Pad Features"
X_val_emb = [] # stores validation data week-wise, which is needed for validation the model embedded in the segmentation loop
for week in val_segs:
    segs_week = []
    for seg in week:
        segs_week.append(seg[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
    X_val.append(segs_week)
    X_val_emb.append(segs_week)
    

    
# 2.4 DETERMINE LABELS

# Extract activity class ('exercise', 'rest' or 'mixed') for training data segments as labels.
y_train = []
for i in train_segs:
    if i['activity'].eq('exercise').all():
        y_train.append('exercise')
    elif i['activity'].eq('rest').all():
        y_train.append('rest')
    else:
        y_train.append('mixed')
y_train = [0 if item == 'exercise' else 1 if item == 'rest' else 2 for item in y_train] # encode labels numerically

# Extract activity class for validation data segments as labels.
y_val = [] # validation data label vector for unembedded model (stores labels week-wise, but all weeks are joined later in "2.7")
y_temp = []
for week in val_segs:
    segs_week = []
    for seg in week:
        if seg['activity'].eq('exercise').all():
            segs_week.append('exercise')
        elif seg['activity'].eq('rest').all():
            segs_week.append('rest')
        else:
            segs_week.append('mixed')
    y_temp.append(segs_week)
for week in y_temp:
    y_val.append([0 if item == 'exercise' else 1 if item == 'rest' else 2 for item in week]) # encode labels numerically


        
# 2.5 DELETE MIXED SEGMENTS

# Identify indices of mixed segments in training and validation data.
train_mixed_inds = np.where(np.asarray(y_train) == 2)
val_mixed_inds = []
for week in y_val:
    val_mixed_inds.append([np.where(np.asarray(week) == 2)])

# Delete mixed segments from training and validation data based on those indices.
X_train = [X_train[i] for i in range(len(X_train)) if i not in list(train_mixed_inds[0])]
y_train = [y_train[i] for i in range(len(y_train)) if i not in list(train_mixed_inds[0])]

X_val_temp = []
y_val_temp = []
for i in range(len(X_val)):
    X_val_temp.append([X_val[i][j] for j in range(len(X_val[i])) if j not in list(val_mixed_inds[i][0][0])])
    y_val_temp.append([y_val[i][j] for j in range(len(y_val[i])) if j not in list(val_mixed_inds[i][0][0])])
X_val = X_val_temp
y_val = y_val_temp
    


# 2.6 PAD FEATURES

# Purpose: Although almost all segments in the training and validation data have 300 instances, the last segments of each
#          workout do not, as the predefined window size mostly does not exactly match the length of the final segment.

# Pad all training and validation segments to length of predefined window size.
# Padding also turns features from arrays into tensors.
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=w_size, dtype='float32')
X_val = [item for week in X_val for item in week]
X_val_pad = tf.keras.preprocessing.sequence.pad_sequences(X_val, maxlen=w_size, dtype='float32')



# 2.7 RESHAPE FEATURES

# Purpose: 2D CNN requires input shape of (n_batch, n_rows, n_columns, n_channels), which requires reshaping of feature tensors.
X_train_pad_rs = X_train_pad.reshape(X_train_pad.shape[0], X_train_pad.shape[1], X_train_pad.shape[2], 1) 
X_val_pad_rs = X_val_pad.reshape(X_val_pad.shape[0], X_val_pad.shape[1], X_val_pad.shape[2], 1)

# Turn labels from lists into arrays.
y_train = np.asarray(y_train)
y_val = np.asarray([item for week in y_val for item in week])
    
    
    
#########################################
# 3. UNEMBEDDED CONV2D SEGMENTATION MODEL
#########################################


# 3.1 MODEL SET-UP

model_1 = keras.Sequential()
model_1.add(Conv2D(filters = 16, kernel_size = (6, 6), strides = 6, activation = 'relu', 
                   input_shape = (X_train_pad_rs[0].shape)))  # input_shape=(n_rows=300, n_cols=6, n_channels=1)
model_1.add(MaxPooling2D(pool_size = (6, 1), strides = 6))
model_1.add(Conv2D(filters = 32, kernel_size = (3, 1), strides = 1, activation = 'relu'))
model_1.add(SpatialDropout2D(rate = 0.1))
model_1.add(Conv2D(filters = 16, kernel_size = (3, 1), strides = 1, activation = 'relu')) 
model_1.add(SpatialDropout2D(rate = 0.2))
model_1.add(Flatten())
model_1.add(Dense(units = 2, activation = 'softmax'))



# 3.2 RUN MODEL AND PLOT LOSS AND ACCURACY STATISTICS

model_1.compile(optimizer = Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model_1.summary()
history_1 = model_1.fit(X_train_pad_rs, y_train, epochs = 10, validation_data=(X_val_pad_rs, y_val), verbose = 1)
plot_classification_loss(history_1)



#######################################
# 4. EMBEDDED CONV2D SEGMENTATION MODEL
#######################################


# 4.1 DETERMINE LABELS

# Split validation workout into 'exercise' and 'rest' segments based on activity labels, and save them in 'segments'.
segments = []
for week in val_weeks_scaled:
    group = week['activity'].ne(week['activity'].shift()).cumsum()
    segs = [g for _,g in week.groupby(group)]
    segments.append(segs)      

# Retrieve validation data labels for embedded model (embedded model requires them to be split into weeks).   
y_val_emb = []
for week in segments:
    y_week = []
    for i in week:
        if i['activity'].eq('exercise').all():
            y_week.append(['exercise', i.index[0], i.index[-1]])
        elif i['activity'].eq('rest').all():
            y_week.append(['rest', i.index[0], i.index[-1]])
        else:
            y_week.append(['mixed', i.index[0], i.index[-1]])
    y_val_emb.append(y_week)
    
    

# 4.2 HYPERPARAMETER SETTING 

# Input length of model determines to which length input segments are padded within the segmentation loop.
model_1_config = model_1.get_config()
model_1_input_length = model_1_config['layers'][0]['config']['batch_input_shape'][1] 



# 4.3 SEGMENTATION LOOP

predictions = [] # stores the predictions of all validation workout weeks
workout_ind = 0  # counting variable used for print-statements


# 4.3.1 Loop loops over all validation workout weeks.
for week in X_val_emb:
    
    week_preds = [] # stores the predictions of the validation workout week that is currently being predicted
    
    
    # 4.3.2 Nested loop classifies each validation data segment as either 'rest' or 'exercise'.
    for i in range(len(week)):
        
        # Create, pad, reshape the tensor and pass it to the trained segmentation model
        paddings = [[model_1_input_length - len(week[i]), 0], [0, 0]]
        padded_i = tf.pad(tf.convert_to_tensor(week[i]), paddings, 'CONSTANT', constant_values=0)
        reshaped_i = tf.reshape(padded_i, [1, padded_i.shape[0], padded_i.shape[1], 1])
        pred_i = model_1.predict(reshaped_i, verbose=0)
        
        # Add probabilities for each class to respective counting variable; store start and end index in respective variables
        ex_vote = pred_i[0][0]
        rest_vote = pred_i[0][1]
        start = week[i].index[0]
        end = week[i].index[-1]
        week_preds.append([ex_vote, rest_vote, 'exercise' if ex_vote > rest_vote else 'rest', start, end])

        
        # 4.3.3 Relabeling of unrealistic predictions.
        # Purpose: Correct unrealistic predictions (e.g. if a single 'e' segment is preceeded and succeeded by 'r' segments)
        
        if len(week_preds) >= 5: # if there are at least 5 previous segments ('i-4', 'i-3', ..., 'i') predicted

            # Relabels 'r, r, r, e, r' to 'r, r, r, r, r' or 'e, e, e, r, e' to 'e, e, e, e, e'
            if week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i-2][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-1][2]:
                week_preds[i-1][2] = week_preds[i-4][2]               
            
            # Relabels 'r, r, e, r, r' to 'r, r, r, r, r'  or  'e, e, r, e, e' to 'e, e, e, e, e'  
            elif week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i-1][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-2][2]:
                week_preds[i-2][2] = week_preds[i-4][2]
            
            # Relabels 'r, e, r, r, r' to 'r, r, r, r, r'  or  'e, r, e, e, e' to 'e, e, e, e, e'
            elif week_preds[i-4][2] == week_preds[i-2][2] == week_preds[i-1][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-3][2]:
                week_preds[i-3][2] = week_preds[i-4][2]
            
            # Relabels 'e, e, r, r, e' to 'e, e, e, e, e' or 'r, r, e, e, r' to 'r, r, r, r, r'
            elif week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-2][2] \
            and week_preds[i-4][2] != week_preds[i-1][2]:
                week_preds[i-2][2] = week_preds[i-4][2]
                week_preds[i-1][2] = week_preds[i-4][2]
                
            # Relabels 'e, r, r, e, e' to 'e, e, e, e, e' or 'r, e, e, r, r' to 'r, r, r, r, r'
            elif week_preds[i-4][2] == week_preds[i-1][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-3][2] \
            and week_preds[i-4][2] != week_preds[i-2][2]:
                week_preds[i-3][2] = week_preds[i-4][2]
                week_preds[i-2][2] = week_preds[i-4][2]
                
            # Relabels 'e, r, e, r, e' to 'e, e, e, e, e' or 'r, e, r, e, r' to 'r, r, r, r, r'
            elif week_preds[i-4][2] == week_preds[i-2][2] == week_preds[i][2] \
            and week_preds[i-4][2] != week_preds[i-3][2] \
            and week_preds[i-4][2] != week_preds[i-1][2]:
                week_preds[i-3][2] = week_preds[i-4][2]
                week_preds[i-1][2] = week_preds[i-4][2]
             
            else:
                pass
        
        else:  # if there are not at least 5 segments predicted yet, nothing is relabeled
            pass
        
        print('{} of {} segments of validation workout {} predicted'.format(i+1, len(week), val_week_ids[workout_ind]))
    
    
    workout_ind += 1
    predictions.append(week_preds)

    
    
# 4.4 AGGREGATION OF PREDICTED SEGMENTS

# Purpose: Aggregate predicted segments (which have a predefined window size) to consecutive 'exercise' or 'rest' segments. 
# Example: 'r','r','r','e','e','e' -> 'r', 'e' (while keeping track of start and end indices of each activity segment)

y = [] # stores predicted labels

# Loops week-wise over predicted labels
for predicted_week in predictions:
    
    y_week = [] # stores predicted labels week-wise
    
    # Loops over each segment within each validation workout week
    for i in range(len(predicted_week)):
        
        if i == 0:  # deals with very first segment of current validation workout week 
            activity = predicted_week[i][2]
            start_index = predicted_week[i][3]
            j = i
            while (predicted_week[j][2] == activity):
                end_index = predicted_week[j][4]
                j += 1
            y_week.append([activity, start_index, end_index])
        
        else:
            if predicted_week[i][2] != predicted_week[i-1][2]: # if activity of previous segment != activity of current segment
                activity = predicted_week[i][2]
                start_index = predicted_week[i][3]
                j = i
                while (predicted_week[j][2] == activity):
                    end_index = predicted_week[j][4]
                    if j == (len(predicted_week)-1):
                        break
                    else:
                        j += 1
                y_week.append([activity, start_index, end_index])
            else:
                continue
    
    y.append(y_week)
    


# 4.5 ASSESSING SEGMENTATION ACCURACY

# Measures
# a) Segment Miscount Rate: Percentage of deviation of predicted segment count from true segment count 
# b) Mean Position Error: Mean difference in seconds of start and end of predicted segment w.r.t. start and end of 
#                         the corresponding true segment 
# c) Classification Accuracy: Percentage of correctly classified segments among all classified segments 


# Loops over predicted validation workout week label vectors
for i in range(len(y)):
    
    
    # Case 1: If segmentation model predicted the number of segments for this workout correctly
    if len(y[i]) == len(y_val_emb[i]):
        
        # a) Segment Miscount Rate
        smr = abs(((len(y[i])/len(y_val_emb[i]))*100) - 100)
        
        # b) Mean Position Error
        pos_diffs = []
        for j in range(len(y[i])):
            # division by 100 i.o.t. compute result in seconds, since data has been collected at 100 Hz
            pos_diff = (abs(y[i][j][1] - y_val_emb[i][j][1]) + abs(y[i][j][2] - y_val_emb[i][j][2])) / 100 
            pos_diffs.append(pos_diff)
        mpe = (sum(pos_diffs) / len(pos_diffs))
        
        # c) Classification Accuracy
        true_classes = [k[0] for k in y_val_emb[i]]
        pred_classes = [l[0] for l in y[i]]
        ca = (len([m for m in range(len(pred_classes)) if pred_classes[m] == true_classes[m]]) / len(true_classes))*100

        
    # Case 2: If segmentation model predicted more segments for this workout than there actually are
    elif len(y[i]) > len(y_val_emb[i]):
        
        print('CAUTION: For workout {}, segmentation model predicted {} segments, while there are actually only {}.'.format(
            val_week_ids[i], len(y[i]), len(y_val_emb[i])))
        
        # a) Segment Miscount Rate
        smr = abs(((len(y[i])/len(y_val_emb[i]))*100) - 100)
        
        # b) Mean Position Error
        # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
        matches = []
        for j in y[i]:
            deltas = []
            for k in y_val_emb[i]:
                deltas.append((abs(j[1]-k[1]) + abs(j[2]-k[2])))
            matches.append(np.argmin(deltas))
        y_inds_true = [] 
        for index, val in enumerate(matches):
            if index == 0: 
                y_inds_true.append(index)
            elif matches[index] != matches[index-1]:
                y_inds_true.append(index)
            else:
                continue
        # compute mean difference in seconds of start and end indices of best-matching segments
        pos_diffs = []
        y_val_index = 0
        for l in y_inds_true:
            pos_diff = (abs(y[i][l][1] - y_val_emb[i][y_val_index][1]) + abs(y[i][l][2] - y_val_emb[i][y_val_index][2])) / 100
            pos_diffs.append(pos_diff)
            y_val_index += 1
        mpe = (sum(pos_diffs) / len(pos_diffs))                  
        
        # c) Classification Accuracy
        true_classes = [m[0] for m in y_val_emb[i]]
        pred_classes = [y[i][n][0] for n in y_inds_true]  
        ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(y[i]))*100

        
    # Case 3: If segmentation model predicted fewer segments for this workout than there actually are
    else: 
        
        print('CAUTION: For workout {}, segmentation model predicted only {} segments, while there are actually {}.'.format(
            val_week_ids[i], len(y[i]), len(y_val_emb[i])))
        
        # a) Segment Miscount Rate
        smr = abs(((len(y[i])/len(y_val_emb[i]))*100) - 100)
        
        # b) Mean Position Error
        # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
        matches = []
        for j in y[i]:
            deltas = []
            for k in y_val_emb[i]: 
                deltas.append((abs(j[1]-k[1]) + abs(j[2]-k[2])))
            matches.append(np.argmin(deltas))
        for idx in range(len(matches)-1):
            if matches[idx] == matches[idx+1]:
                matches[idx] = matches[idx+1]-1
            else:
                continue
        # compute mean difference in seconds of start and end indices of best-matching segments
        pos_diffs = []
        y_index = 0
        for l in matches:
            pos_diff = (abs(y[i][y_index][1] - y_val_emb[i][l][1]) + abs(y[i][y_index][2] - y_val_emb[i][l][2])) / 100 
            pos_diffs.append(pos_diff)
            y_index += 1
        mpe = (sum(pos_diffs) / len(pos_diffs))                   
        
        # c) Classification Accuracy
        true_classes = [y_val_emb[i][m][0] for m in matches] 
        pred_classes = [n[0] for n in y[i]]   
        ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(y[i]))*100

        
    # Print statistics
    print('Segmentation Assessement for Workout {}:\n\
    Segment Miscount Rate: {:.2f}%, Mean Position Error: {:.2f}s, Classification Accuracy: {:.2f}%\n'.format(
        val_week_ids[i], smr, mpe, ca))
    
    
    # Plot segmentation results
    plot_segmentation(y_val_emb[i], y[i], val_week_ids[i]) 
    
    
################    
# 5. SAVE MODEL
################
    
    
# Save model under a path of your choice
# model_1.save('*/segmentation.h5')   # for '*', insert the path to saving location