import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from copy import deepcopy
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

from utils.assessment import Assessment
from utils.data_loading import MMFitDataLoading
from utils.processing import Processing
from configuration import Configuration
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# MM-Fit dataset
mmfit_path = Configuration.Constants.MMFIT_DATA_PATH
test_week_ids = Configuration.Constants.TEST_WEEK_IDS

# Load models
os.chdir("..")  # move one directory up i.o.t. find models directory
cnn_1 = tf.keras.models.load_model(Configuration.Constants.SEGMENTATION_MODEL_PATH)
cnn_2 = tf.keras.models.load_model(Configuration.Constants.EXERCISE_RECOGNITION_MODEL_PATH)
cnn_3 = tf.keras.models.load_model(Configuration.Constants.REPETITION_COUNTING_MODEL_PATH)

# Extract accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise.
test_weeks = MMFitDataLoading.load_data(mmfit_path, test_week_ids)

# Standardize data
train_mean = Configuration.Constants.TRAINING_DATA_MEAN
train_sd = Configuration.Constants.TRAINING_DATA_SD
test_weeks = Processing.standardize(data=test_weeks, mean=train_mean, sd=train_sd)


##########################
# 1. ACTIVITY SEGMENTATION
##########################

# Determine features and labels
X_test = []
y_test = []

w_size = Configuration.Constants.WINDOW_SIZE
for week in test_weeks:
    
    # Determine features
    # retrieve segments of 300 instances from test data and use their acclerometer and gyroscope data as features
    week_segments = []
    for window in range(0, len(week), w_size):
        if window == 0:
            continue
        else:
            if window+w_size > len(week):
                segment_1 = week[window - w_size:window]
                segment_2 = week[window:]
                week_segments.append(segment_1[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
                week_segments.append(segment_2[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
            else:
                segment = week[window - w_size:window]
                week_segments.append(segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
    X_test.append(week_segments)  
    
    # Determine labels
    # split workout into 'exercise' and 'rest' segments based on activity labels, and save them in 'segs'
    group = week['activity'].ne(week['activity'].shift()).cumsum()
    segs = [g for _,g in week.groupby(group)]     
    y_week = []
    for seg in segs:
        if seg['activity'].eq('exercise').all():
            y_week.append(['exercise', seg.index[0], seg.index[-1]])
        elif seg['activity'].eq('rest').all():
            y_week.append(['rest', seg.index[0], seg.index[-1]])
        else: # else-clause serves just for completion, because, in contrast to 'X_test', the labels 'y_week'
              # are retrieved based on perfect segmentation, which is why no 'mixed' label is necessary
            y_week.append(['mixed', seg.index[0], seg.index[-1]])
    y_test.append(y_week)
    
# Segmentation
# Hyperparameter: Input Length of CNN
cnn_1_input_length = cnn_1.get_config()['layers'][0]['config']['batch_input_shape'][1]

# Predicting Segments
predictions = []  # 'predictions' stores the predictions of all workouts
test_week_id = 0
for week in X_test:
    week_preds = []  # stores predictions of the workout that is currently being predicted
    
    for i in range(len(week)):

        # create, pad, reshape tensor of segment and pass it to the trained segmentation model
        paddings = [[cnn_1_input_length - len(week[i]), 0], [0, 0]]
        padded_i = tf.pad(tf.convert_to_tensor(week[i]), paddings, 'CONSTANT', constant_values=0)
        # input shape: (1 tensor, 300 rows, 6 columns, 1 channel)
        reshaped_i = tf.reshape(padded_i, [1, padded_i.shape[0], padded_i.shape[1], 1])
        pred_i = cnn_1.predict(reshaped_i, verbose=0)
        
        # add probabilities for each class to respective counting variable
        ex_vote = pred_i[0][0]
        rest_vote = pred_i[0][1]
        start = week[i].index[0]
        end = week[i].index[-1]
        week_preds.append([ex_vote, rest_vote, 'exercise' if ex_vote > rest_vote else 'rest', start, end])

        # Relabeling of unrealistic predictions
        # Correct unrealistic predictions (e.g. if a single 'e' segment is preceded and succeeded by 'r' segments)
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

        # if there are not at least 5 segments predicted yet, nothing is relabelled
        else:
            pass
        print('{} of {} test segments of workout {} predicted'.format(i+1, len(week), test_week_ids[test_week_id]))
    
    test_week_id += 1
    predictions.append(week_preds)

# Aggregation
# Aggregate predicted segments of length of window size to continuous 'rest' or 'exercise' segments
y = Processing.aggregate_labels(predictions=predictions)

# Assessment
# See method documentation for meaning of returned lists
true_pred_segs, true_pred_segs_y, false_pred_segs_y, true_pred_segs_y_test, non_pred_segs_y_test = (
    Assessment.segmentation_assessment(true_labels=y_test, predicted_labels=y, week_ids=test_week_ids))


#########################
# 2. EXERCISE RECOGNITION
#########################

# Retrieve predicted exercise segments
ex_segments_list = [] # stores correctly predicted exercise segments as dataframes
false_ex_segments_list = [] # stores falsely predicted exercise segments as dataframes
unpred_ex_segments_list = [] #stores unpredicted exercise segments as dataframes  

for i in range(len(test_weeks)):
    
    # if segmentation model predicted the correct number of segments for this workout
    if len(y_test[i]) == len(y[i]):
        # retrieve predicted exercise segments for this workout
        ex_segments_week_list = []
        for seg in y[i]:
            if seg[0] == 'exercise':
                ex_segments_week_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue 
        ex_segments_list.append(ex_segments_week_list)
        # append empty list to all the other lists that were not relevant
        false_ex_segments_list.append([])
        unpred_ex_segments_list.append([])
        print('For workout {}, there are {} correctly predicted and 0 falsely or unpredicted exercise segments\n'.format(
            test_week_ids[i], len(ex_segments_list[i])))
        
    # if segmentation model predicted more segments than there actually are in this workout
    elif len(y[i]) > len(y_test[i]):
        # retrieve correctly predicted exercise segments for this workout
        true_segs = [y[i][ind] for ind in true_pred_segs_y[i]]
        ex_segments_week_list = []
        for seg in true_segs:
            if seg[0] == 'exercise':
                ex_segments_week_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue
        ex_segments_list.append(ex_segments_week_list)
        # keep track of falsely predicted exercise segments
        false_ex_segments_list.append([y[i][ind] for ind in false_pred_segs_y[i] if y[i][ind][0] == 'exercise'])
        # append empty list to all the other lists that were not relevant
        unpred_ex_segments_list.append([])
        print('For workout {}, there are {} correctly predicted and {} falsely predicted exercise segments in the test data.\n\
        In practice, the latter segments would be erroneously passed down to the exercise recognition CNN, which would classify them as one of the exercise types.\n\
        However, in order to ensure a correct assessment, the falsely predicted exercise segments will be treated as errors, diminishing the predictive accuracy of\n\
        the exercise recognition CNN.\n'.format(test_week_ids[i], len(ex_segments_list[i]), len(false_ex_segments_list[i])))
        
    # if segmentation model predicted fewer segments than there actually are in this workout
    else:
        # retrieve predicted exercise segments for this workout
        ex_segments_week_list = []
        for seg in y[i]:
            if seg[0] == 'exercise':
                ex_segments_week_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue
        ex_segments_list.append(ex_segments_week_list)
        # keep track of unpredicted exercise segments
        unpred_ex_segments_list.append([y_test[i][ind] for ind in non_pred_segs_y_test[i] if y_test[i][ind][0] == 'exercise'])
        # append empty list to all the other lists that were not relevant
        false_ex_segments_list.append([])
        print('For workout {}, there are {} correctly predicted exercise segments in the test data.\n\
        {} exercise segments remained unpredicted.\n\
        In order to ensure a correct assessment, the unpredicted exercise segments will be treated as errors, diminishing the predictive accuracy of the exercise recognition CNN.\n'.format(
            test_week_ids[i], len(ex_segments_list[i]), len(unpred_ex_segments_list[i])))


# Determine features of predicted exercise segments
X_2_test = []
for workout in ex_segments_list:
    workout_features = []
    for segment in workout:
        workout_features.append(segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']])
    X_2_test.append(workout_features)

# Numerical encoding of exercise types of predicted exercise segments
# Fit label encoder across all exercise types
ex_classes = {'Classes': ['bicep_curls', 'dumbbell_rows', 'dumbbell_shoulder_press', 'jumping_jacks',
   'lateral_shoulder_raises', 'lunges', 'pushups', 'situps', 'squats', 'tricep_extensions']}
df_classes = pd.DataFrame(data=ex_classes)
label_enc_ex_rec = LabelEncoder()
label_enc_ex_rec.fit(df_classes['Classes'])
ex_types_dict = dict(list(enumerate(label_enc_ex_rec.classes_)))

y_2_test = []

for i in range(len(test_weeks)):
    y_test_ex_labels = []
    
    # if segmentation model predicted the correct number of segments for this workout
    if len(y_test[i]) == len(y[i]):
        # retrieve 'exercise' segments from 'y_test[i]'
        ex_y_test_list = []
        for seg in y_test[i]:
            if seg[0] == 'exercise':
                ex_y_test_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue 
        # assign integer-valued labels to 'exercise' segments in 'ex_y_test_list'
        for segment in ex_y_test_list:
            segment['exercise_Label'] = label_enc_ex_rec.transform(segment['exercise']).copy()
        # create list of labels for exercise recognition, based on integer-valued exercise labels
        for segment in ex_y_test_list:
            y_test_ex_labels.append(stats.mode(segment['exercise_Label'])[0][0])

    # if segmentation model predicted more segments than there actually are for this workout
    elif len(y[i]) > len(y_test[i]):
        # retrieve 'exercise' segments from 'y_test[i]'
        ex_y_test_list = []
        for seg in y_test[i]:
            if seg[0] == 'exercise':
                ex_y_test_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue
        # assign integer-valued labels to 'exercise' segments in 'ex_y_test_list'
        for segment in ex_y_test_list:
            segment['exercise_Label'] = label_enc_ex_rec.transform(segment['exercise']).copy()
        # create list of labels for exercise recognition, based on integer-valued exercise labels
        for segment in ex_y_test_list:
            y_test_ex_labels.append(stats.mode(segment['exercise_Label'])[0][0])
            
    # if segmentation model predicted fewer segments than there actually are for this workout
    else:
        # delete unpredicted segments from 'y_test' (those unpredicted segments can be both 'rest' and/or 'exercise' segments)
        y_test_i = [y_test[i][ind] for ind in range(len(y_test[i])) if ind not in non_pred_segs_y_test[i]]
        # retrieve 'exercise' segments from 'y_test', which now only contains the matches of those segments that were predicted
        ex_y_test_list = []
        for seg in y_test_i:
            if seg[0] == 'exercise':
                ex_y_test_list.append(test_weeks[i][seg[1]:seg[2]+1])
            else:
                continue
        # assign integer-valued labels to 'exercise' segments in 'ex_y_test_list'
        for segment in ex_y_test_list:
            segment['exercise_Label'] = label_enc_ex_rec.transform(segment['exercise']).copy()
        # create list of labels for exercise recognition, based on integer-valued exercise labels
        for segment in ex_y_test_list:
            y_test_ex_labels.append(stats.mode(segment['exercise_Label'])[0][0])
    
    y_2_test.append(y_test_ex_labels)
    
    
# Padding
cnn_2_input_length = cnn_2.get_config()['layers'][0]['config']['batch_input_shape'][1]
X_2_test_pad = [Processing.padding(data=X_2_test[0], window_size=cnn_2_input_length),
                Processing.padding(data=X_2_test[1], window_size=cnn_2_input_length)]

# Rehsaping
X_2_test_pad_rs = [Processing.reshaping(X_2_test_pad[0]), Processing.reshaping(X_2_test_pad[1])]
y_2_test = np.asarray(y_2_test)
for i in range(len(X_2_test_pad_rs)):
    print('After padding and reshaping, the shape of the predicted exercise segments of workout {} is {}.'.format(
        test_week_ids[i], X_2_test_pad_rs[i].shape))
    print('This means\n{} tensors à\n{} rows,\n{} columns,\n{} channel\neach.\n'.format(X_2_test_pad_rs[i].shape[0],
                                                                                        X_2_test_pad_rs[i].shape[1],
                                                                                        X_2_test_pad_rs[i].shape[2],
                                                                                        X_2_test_pad_rs[i].shape[3]))

# Prediction and assessment
y_2_preds = []
for i in range(len(test_weeks)):

    # if segmentation model predicted the correct number of segments for this workout
    if len(y_test[i]) == len(y[i]):
        
        # Confusion marix
        y_2_pred = np.argmax(cnn_2.predict(X_2_test_pad_rs[i]), axis=-1)
        y_2_preds.append(y_2_pred)
        cm_2 = confusion_matrix(y_2_test[i], y_2_pred, labels=np.arange(len(df_classes)))
        display_2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=label_enc_ex_rec.classes_)
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        display_2.plot(ax=ax, xticks_rotation='vertical')
        plt.title('Workout {}'.format(test_week_ids[i]))
        plt.show()
        
        # Accuracy score
        score_2 = len([j for j, k in zip(y_2_pred, y_2_test[i]) if j == k])
        acc_2 = (score_2/len(y_2_test[i]))*100
        print('For workout {}, the test accuracy for exercise recognition is {:.2f} %.'.format(test_week_ids[i], acc_2))

    # if segmentation model predicted more segments than there actually are for this workout
    elif len(y[i]) > len(y_test[i]):
        
        # Confusion matrix
        y_2_pred = np.argmax(cnn_2.predict(X_2_test_pad_rs[i]), axis=-1)
        y_2_preds.append(y_2_pred)
        cm_2 = confusion_matrix(y_2_test[i], y_2_pred, labels=np.arange(len(df_classes)))
        display_2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=label_enc_ex_rec.classes_)
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        display_2.plot(ax=ax, xticks_rotation='vertical')
        plt.title('Workout {}'.format(test_week_ids[i]))
        plt.show()
        
        # Accuracy score
        score_2 = len([j for j, k in zip(y_2_pred, y_2_test[i]) if j == k])
        acc_2 = (score_2/(len(y_2_test[i])+len(false_ex_segments_list[i])))*100
        print('For workout {}, the test accuracy for exercise recognition is {:.2f} %.'.format(test_week_ids[i], acc_2))
        print('This result already includes the error penalty imposed due to the identification of {} non-existing exercise segments by the segmentation model.'.format(
            len(false_ex_segments_list[i])))

    # if segmentation model predicted fewer segments than there actually are for this workout
    else:
        
        # Confusion matrix
        y_2_pred = np.argmax(cnn_2.predict(X_2_test_pad_rs[i]), axis=-1)
        y_2_preds.append(y_2_pred)
        cm_2 = confusion_matrix(y_2_test[i], y_2_pred, labels=np.arange(len(df_classes)))
        display_2 = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=label_enc_ex_rec.classes_)
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        display_2.plot(ax=ax, xticks_rotation='vertical')
        plt.title('Workout {}'.format(test_week_ids[i]))
        plt.show()
        
        # Accuracy score
        score_2 = len([j for j, k in zip(y_2_pred, y_2_test[i]) if j == k])
        acc_2 = (score_2/(len(y_2_test[i])+len(unpred_ex_segments_list[i])))*100
        print('For workout {}, the test accuracy for exercise recognition is {:.2f} %.'.format(test_week_ids[i], acc_2))
        print('This result already includes the error penalty imposed due to the segmentation model failing to identify {} exercise segments.'.format(
            len(unpred_ex_segments_list[i])))
        

########################
# 3. REPETITION COUNTING
########################

# Downsample features of predicted exercise segments to 50 Hz
# Purpose: CNN for repetition counting was trained with 50 Hz data, because Recofit dataset was collected at 50 Hz
test_data_rep_count = []
for workout in X_2_test:
    test_data_rep_count.append(deepcopy([seg.iloc[::2] for seg in workout]))  

# Savitzky-Golay-Filter and PCA
hz = 50
for i in range(len(test_data_rep_count)):
    
    # Savitzky-Golay-Filter
    for seg in test_data_rep_count[i]:
        seg['x_acc'] = savgol_filter(seg['x_acc'], window_length=int(1.5*hz), polyorder=3, axis=0)
        seg['y_acc'] = savgol_filter(seg['y_acc'], window_length=int(1.5*hz), polyorder=3, axis=0)
        seg['z_acc'] = savgol_filter(seg['z_acc'], window_length=int(1.5*hz), polyorder=3, axis=0)
        seg['x_gyr'] = savgol_filter(seg['x_gyr'], window_length=int(1.5*hz), polyorder=3, axis=0)
        seg['y_gyr'] = savgol_filter(seg['y_gyr'], window_length=int(1.5*hz), polyorder=3, axis=0)
        seg['z_gyr'] = savgol_filter(seg['z_gyr'], window_length=int(1.5*hz), polyorder=3, axis=0)  
    
    # Principal Component Analysis
    for seg in test_data_rep_count[i]:
        scaled_data = seg[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] 
        pca = PCA(n_components=1)
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        seg['pca']=pca_data        

# Determine PCA signal as feature
X_3_test = []
for i in range(len(test_data_rep_count)):
    workout_pca = []
    for seg in test_data_rep_count[i]:
        workout_pca.append(seg['pca'])
    X_3_test.append(workout_pca)

# Determine repetition counts as labels
y_3_test = []

for i in range(len(test_weeks)):
    rep_counts_workout_i = [] 
    
    # if segmentation model predicted the correct number of segments for this workout
    if len(y_test[i]) == len(y[i]):
        week = test_week_ids[i]    
        file = os.path.join(mmfit_path, week + '/', week + '_labels.csv')
        df_labels = pd.read_csv(file, header=None)
        list_labels = [list(row) for row in df_labels.values]
        rep_counts_workout_i.append([l[2] for l in list_labels])
        rep_counts_workout_i = [item for sublist in rep_counts_workout_i for item in sublist]
        y_3_test.append(rep_counts_workout_i)            
    
    # if segmentation model predicted more segments than there actually are in this workout
    elif len(y[i]) > len(y_test[i]):
        week = test_week_ids[i]
        file = os.path.join(mmfit_path, week + '/', week + '_labels.csv')
        df_labels = pd.read_csv(file, header=None)
        list_labels = [list(row) for row in df_labels.values]
        rep_counts_workout_i.append([l[2] for l in list_labels])
        rep_counts_workout_i = [item for sublist in rep_counts_workout_i for item in sublist]
        y_3_test.append(rep_counts_workout_i)               
    
    # if segmentation model predicted fewer segments than there are in this workout
    else:
        week = test_week_ids[i]
        file = os.path.join(mmfit_path, week + '/', week + '_labels.csv')
        df_labels = pd.read_csv(file, header=None)
        list_labels = [list(row) for row in df_labels.values]
        rep_counts_workout_i.append([l[2] for l in list_labels])
        rep_counts_workout_i = [item for sublist in rep_counts_workout_i for item in sublist]           
        # Delete exercise segments from 'y_3_test', which were not predicted by segmentation model.
        # Otherwise, 'X_3_test' and 'y_3_test' do not have same length
        del_inds_y_3_test_i = []
        # start and end of unpredicted exercise segment are given as instance indices, but the labels are defined i.t.o. frames 
        for unpred_ex_seg in unpred_ex_segments_list[i]:
            # retrieve start and end frames of corresponding unpredicted exercise segment
            start_frame = test_weeks[i][unpred_ex_seg[1]:unpred_ex_seg[2]+1]['frame'].iloc[0]
            end_frame = test_weeks[i][unpred_ex_seg[1]:unpred_ex_seg[2]+1]['frame'].iloc[-1]
            # use start and end frames for identifying the index location of the unpredicted exercise segment among y_3_test
            ind = df_labels.index[(df_labels[0] == start_frame) & (df_labels[1] == end_frame)]
            del_inds_y_3_test_i.append(ind[0])
        # delete unpredicted exercise segments from 'y_3_test' by using their location index
        rep_counts_workout_i = [rep_counts_workout_i[idx] for idx in range(len(rep_counts_workout_i)) if idx not in del_inds_y_3_test_i]
        # delete unpredicted exercise segments from 'y_3_test' by using their location index
        y_3_test.append(rep_counts_workout_i)
        
# Padding
cnn_3_input_length = cnn_3.get_config()['layers'][0]['config']['batch_input_shape'][1]
X_3_test_pad = [Processing.padding(data=X_3_test[0], window_size=cnn_3_input_length),
                Processing.padding(data=X_3_test[1], window_size=cnn_3_input_length)]
    
# Reshaping
X_3_test_pad_rs = [Processing.reshaping(X_3_test_pad[0]), Processing.reshaping(X_3_test_pad[1])]
y_3_test = np.asarray(y_3_test)
for i in range(len(X_3_test_pad_rs)):
    print('After padding and reshaping, the shape of the predicted exercise segments of workout {} is {}.'.format(test_week_ids[i], X_3_test_pad_rs[i].shape))
    print('This means\n{} tensors à\n{} rows,\n{} channel\neach.\n'.format(X_3_test_pad_rs[i].shape[0], X_3_test_pad_rs[i].shape[1], X_3_test_pad_rs[i].shape[2]))                                                

# Prediction and assessment
# Calculate and print error statistics
exact = []; delta_one = []; delta_two = []; mae_list = []; mse_list = []; mae_dict_list = []; mse_dict_list = []
y_3_preds = []

for i in range(len(test_weeks)):

    # if segmentation model predicted the correct number of segments
    if len(y_test[i]) == len(y[i]):
        
        # make predictions for test data samples
        loss, _, __ = cnn_3.evaluate(X_3_test_pad_rs[i], np.asarray(y_3_test[i]), verbose=0)
        y_3_pred = np.round_(cnn_3.predict(X_3_test_pad_rs[i]))
        y_3_pred_flat = [item[0] for item in y_3_pred]
        y_3_preds.append(y_3_pred_flat)
        
        # calculate how many predictions deviate by 0, +-1 or +-2 repetitions from the labels
        delta = np.asarray(y_3_pred_flat) - np.asarray(y_3_test[i])
        exact_ = 0; delta_one_ = 0; delta_two_ = 0
        for j in delta:
            if j == 0:
                exact_ +=1
            elif j == 1 or j == -1:
                delta_one_ += 1
            elif j == 2 or j == -2:
                delta_two_ += 1
            else:
                continue
        exact.append(((exact_ / len(delta))*100))
        delta_one.append((((exact_ + delta_one_)/len(delta))*100))
        delta_two.append((((exact_ + delta_one_ + delta_two_)/len(delta))*100))
        
        # calculate MAE and MSE across all predictions
        mae = sum([abs(val) for val in delta]) / len(delta)
        mse = sum([val**2 for val in delta]) / len(delta)
        mae_list.append(mae)
        mse_list.append(mse)
        
        # calculate mean absolute error and mean squared error for each predicted exercise type
        y_2_test_ex_names = []
        for elm in y_2_test[i]:
            for class_ in list(enumerate(label_enc_ex_rec.classes_)):
                if elm == class_[0]:
                    y_2_test_ex_names.append(class_[1])
                else:
                    continue
        delta_per_ex = {}
        for entry in list(zip(y_2_test_ex_names, delta)):
            if entry[0] not in delta_per_ex.keys():
                delta_per_ex[entry[0]] = [entry[1]]
            else:
                delta_per_ex[entry[0]].extend([entry[1]])
        mae_dict = {}; mse_dict = {}
        for key, val in delta_per_ex.items():
            mae_dict[key] = sum([abs(v) for v in val])/len(val)
            mse_dict[key] = (sum([(abs(v))**2 for v in val]))/len(val)   
        mae_dict_list.append(mae_dict)
        mse_dict_list.append(mse_dict)

    # if segmentation model predicted more segments than there actually are
    elif len(y[i]) > len(y_test[i]):
        
        # make predictions for test data samples
        loss, _, __ = cnn_3.evaluate(X_3_test_pad_rs[i], np.asarray(y_3_test[i]), verbose=0)
        y_3_pred = np.round_(cnn_3.predict(X_3_test_pad_rs[i]))
        y_3_pred_flat = [item[0] for item in y_3_pred]
        y_3_preds.append(y_3_pred_flat)
        
        # calculate how many predictions deviate by 0, +-1 or +-2 repetitions from the labels
        delta = np.asarray(y_3_pred_flat) - np.asarray(y_3_test[i])
        exact_ = 0; delta_one_ = 0; delta_two_ = 0
        for j in delta:
            if j == 0:
                exact_ +=1
            elif j == 1 or j == -1:
                delta_one_ += 1
            elif j == 2 or j == -2:
                delta_two_ += 1
            else:
                continue
        exact.append((((exact_ / (len(delta) + len(false_ex_segments_list[i])))*100)))
        delta_one.append((((exact_ + delta_one_) / (len(delta) + len(false_ex_segments_list[i])))*100))
        delta_two.append((((exact_ + delta_one_ + delta_two_) / (len(delta) + len(false_ex_segments_list[i])))*100))        
        
        # calculate MAE and MSE across all predictions, including the penalty due to falsely predicted exercise segments
        mae = sum([abs(val) for val in delta]) / (len(delta) - len(false_ex_segments_list[i]))
        mse = sum([(val**2) for val in delta]) / (len(delta) - len(false_ex_segments_list[i]))
        mae_list.append(mae)
        mse_list.append(mse)
        
        # calculate mean absolute error and mean squared error for each predicted exercise type
        y_2_test_ex_names = []
        for elm in y_2_test[i]:
            for class_ in list(enumerate(label_enc_ex_rec.classes_)):
                if elm == class_[0]:
                    y_2_test_ex_names.append(class_[1])
                else:
                    continue
        delta_per_ex = {}
        for entry in list(zip(y_2_test_ex_names, delta)):
            if entry[0] not in delta_per_ex.keys():
                delta_per_ex[entry[0]] = [entry[1]]
            else:
                delta_per_ex[entry[0]].extend([entry[1]])
        mae_dict = {}; mse_dict = {}
        for key, val in delta_per_ex.items():
            mae_dict[key] = sum([abs(v) for v in val])/len(val)
            mse_dict[key] = (sum([(abs(v))**2 for v in val]))/len(val)   
        mae_dict_list.append(mae_dict)
        mse_dict_list.append(mse_dict)

    # if segmentation model predicted fewer segments than there actually are
    else:
        
        # make predictions for test data samples
        loss, _, __ = cnn_3.evaluate(X_3_test_pad_rs[i], np.asarray(y_3_test[i]), verbose=0)
        y_3_pred = np.round_(cnn_3.predict(X_3_test_pad_rs[i]))
        y_3_pred_flat = [item[0] for item in y_3_pred]
        y_3_preds.append(y_3_pred_flat)
        
        # calculate how many predictions deviate by 0, +-1 or +-2 from the labels, including the penalty due to 
        # unpredicted exercise segments
        delta = np.asarray(y_3_pred_flat) - np.asarray(y_3_test[i])
        exact_ = 0; delta_one_ = 0; delta_two_ = 0
        for j in delta:
            if j == 0:
                exact_ +=1
            elif j == 1 or j == -1:
                delta_one_ += 1
            elif j == 2 or j == -2:
                delta_two_ += 1
            else:
                continue
        exact.append((((exact_ / (len(delta) + len(unpred_ex_segments_list[i])))*100)))
        delta_one.append((((exact_ + delta_one_) / (len(delta) + len(unpred_ex_segments_list[i])))*100))
        delta_two.append((((exact_ + delta_one_ + delta_two_) / (len(delta) + len(unpred_ex_segments_list[i])))*100))
        
        # calculate MAE and MSE across all predictions, including the penalty due to unpredicted exercise segments
        mae = sum([abs(val) for val in delta]) / (len(delta) - len(unpred_ex_segments_list[i]))
        mse = sum([(val**2) for val in delta]) / (len(delta) - len(unpred_ex_segments_list[i]))
        mae_list.append(mae)
        mse_list.append(mse)
        
        # calculate mean absolute error and mean squared error for each predicted exercise type
        y_2_test_ex_names = []
        for elm in y_2_test[i]:
            for class_ in list(enumerate(label_enc_ex_rec.classes_)):
                if elm == class_[0]:
                    y_2_test_ex_names.append(class_[1])
                else:
                    continue
        delta_per_ex = {}
        for entry in list(zip(y_2_test_ex_names, delta)):
            if entry[0] not in delta_per_ex.keys():
                delta_per_ex[entry[0]] = [entry[1]]
            else:
                delta_per_ex[entry[0]].extend([entry[1]])
        mae_dict = {}; mse_dict = {}
        for key, val in delta_per_ex.items():
            mae_dict[key] = sum([abs(v) for v in val])/len(val)
            mse_dict[key] = (sum([(abs(v))**2 for v in val]))/len(val)   
        mae_dict_list.append(mae_dict)
        mse_dict_list.append(mse_dict)            
        
    print('REPETITION COUNTING STATISTICS FOR WORKOUT {}:'.format(test_week_ids[i]))
    print('Mean Absolute Error across all predicted exercise segments: {:.2f} repetitions.'.format(mae_list[i]))
    print('Mean Squared Error across all predicted exercise segments: {:.2f} repetitions.'.format(mse_list[i]))
    print('Percentage of exercise segments whose predicted repetition count is equal to the true repetition count: {:.2f}%.'.format(exact[i]))
    print('Percentage of exercise segments whose predicted repetition count is within +-1 repetitions of the true repetition count: {:.2f}%.'.format(delta_one[i]))
    print('Percentage of exercise segments whose predicted repetition count is within +-2 repetitions of the true repetition count: {:.2f}%.'.format(delta_two[i])) 
    print('Mean Absolute Error with respect to exercise types:\n {}'.format(mae_dict_list[i]))
    print('Mean Squared Error with respect to exercise types:\n {}\n'.format(mse_dict_list[i]))                     

# Plot results
# create numerated exercise type dictionary
exercise_dict = {}
for ind, ex in list(enumerate(label_enc_ex_rec.classes_)):
    exercise_dict[ind] = ex
    
workout_ind = 0
for i in range(len(X_3_test_pad_rs)):
    for j in range(len(X_3_test_pad_rs[i])):
        print('Workout {}, Exercise Segment {}'.format(test_week_ids[workout_ind], j))
        print('True Exercise Type: {}'.format(exercise_dict[y_2_test[i][j]]))
        print('Predicted Exercise Type: {}'.format(exercise_dict[y_2_preds[i][j]]))
        print('True Repetition Count: {}'.format(y_3_test[i][j]))
        print('Predicted Repetition Count: {}'.format(int(y_3_preds[i][j])))
        plt.figure(figsize=(8, 2.5))
        plt.plot(np.linspace(0, len(X_2_test[i][j])/100, len(test_data_rep_count[i][j]['pca'])), test_data_rep_count[i][j]['pca'])
        plt.xlabel('Time (s)')
        plt.ylabel('PCA Signal')
        plt.show()
    workout_ind += 1
