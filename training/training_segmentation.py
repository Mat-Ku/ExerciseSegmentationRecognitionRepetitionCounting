import numpy as np
import tensorflow as tf

from utils.assessment import Assessment
from utils.plotting import Plotting
from utils.processing import Processing
from utils.data_loading import DataLoading
from configuration import Configuration


# 1. LOAD AND PREPROCESS DATA

# Load accelerometer and gyroscope data of left smartwatch, as well as frames, week index, activity and exercise
data_path = Configuration.Constants.MMFIT_DATA_PATH  # path to MM-Fit dataset
train_weeks = DataLoading.load_data(data_path, Configuration.Constants.TRAINING_WEEK_IDS)
val_weeks = DataLoading.load_data(data_path, Configuration.Constants.VALIDATION_WEEK_IDS)

# Standardize data
train_weeks_scaled = Processing.standardize(train_weeks)
val_weeks_scaled = Processing.standardize(val_weeks)

# Cut data into segments according to window size
training_segments = Processing.segmenting(train_weeks_scaled, Configuration.Constants.WINDOW_SIZE)
training_segments = [segment for list_ in training_segments for segment in list_]
validation_segments = Processing.segmenting(val_weeks_scaled, Configuration.Constants.WINDOW_SIZE)

# Extract accelerometer and gyroscope data (features) and activities (labels)
X_train = [segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] for segment in training_segments]
X_val = [segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] for week in validation_segments for segment in week]
y_train = [0 if segment['activity'].eq('exercise').all() else 1 if segment['activity'].eq('rest').all() else 2 for segment in training_segments]
y_val = [0 if segment['activity'].eq('exercise').all() else 1 if segment['activity'].eq('rest').all() else 2 for week in validation_segments for segment in week]

# Delete mixed segments, as they are not intended to be predicted
train_mixed_inds = np.where(np.array(y_train) == 2)
val_mixed_inds = np.where(np.array(y_val) == 2)
X_train = [X_train[i] for i in range(len(X_train)) if i not in list(train_mixed_inds[0])]
y_train = [y_train[i] for i in range(len(y_train)) if i not in list(train_mixed_inds[0])]
X_val = [X_val[i] for i in range(len(X_val)) if i not in list(val_mixed_inds[0])]
y_val = [y_val[i] for i in range(len(y_val)) if i not in list(val_mixed_inds[0])]

# Padding of segments
X_train_pad = Processing.padding(X_train, Configuration.Constants.WINDOW_SIZE)
X_val_pad = Processing.padding(X_val, Configuration.Constants.WINDOW_SIZE)

# Reshaping of segments
X_train_pad_rs = Processing.reshaping(X_train_pad)
X_val_pad_rs = Processing.reshaping(X_val_pad)


# 2. UNEMBEDDED CONV2D SEGMENTATION MODEL

# Set-up model
model_1 = tf.keras.Sequential()
model_1.add(tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(6, 6),
                                   strides=6,
                                   activation='relu',
                                   input_shape=X_train_pad_rs[0].shape))
model_1.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 1),
                                         strides=6))
model_1.add(tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 1),
                                   strides=1,
                                   activation='relu'))
model_1.add(tf.keras.layers.SpatialDropout2D(rate=0.1))
model_1.add(tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(3, 1),
                                   strides=1,
                                   activation='relu'))
model_1.add(tf.keras.layers.SpatialDropout2D(rate=0.2))
model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(units=2,
                                  activation='softmax'))

# Train model
model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
model_1.summary()
history_1 = model_1.fit(x=X_train_pad_rs,
                        y=np.array(y_train),
                        epochs=10,
                        validation_data=(X_val_pad_rs, np.array(y_val)),
                        verbose=1)

Plotting.plot_classification_loss(history_1)


# 3. LOOP-EMBEDDED CONV2D SEGMENTATION MODEL

# Validation features stored week-wise, which is needed for validation the model embedded in the segmentation loop
X_val_emb = [[segment[['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr']] for segment in week]
             for week in validation_segments]

# Split validation workout into 'exercise' and 'rest' segments based on activity labels
segments = []
for week in val_weeks_scaled:
    group = week['activity'].ne(week['activity'].shift()).cumsum()
    segs = [g for _, g in week.groupby(group)]
    segments.append(segs)      

# Retrieve validation data labels for embedded model week-wise
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

# Input length of model determines to which length input segments are padded within the segmentation loop
model_1_input_length = model_1.get_config()['layers'][0]['config']['batch_input_shape'][1]

# Validate trained model in segmentation loop
predictions = []  # stores the predictions of all validation workout weeks
workout_ind = 0  # counting variable used for print-statements

for week in X_val_emb:
    week_preds = []  # stores the predictions of the validation workout week that is currently being predicted

    # Nested loop classifies each validation data segment as either 'rest' or 'exercise'
    for i in range(len(week)):
        
        # Create, pad, reshape the tensor and pass it to the trained segmentation model
        paddings = [[model_1_input_length - len(week[i]), 0], [0, 0]]
        padded_i = tf.pad(tf.convert_to_tensor(week[i]), paddings, 'CONSTANT', constant_values=0)
        reshaped_i = tf.reshape(padded_i, [1, padded_i.shape[0], padded_i.shape[1], 1])
        pred_i = model_1.predict(reshaped_i, verbose=0)
        
        # Add probabilities for each class to respective counting variable;
        # store start and end index in respective variables
        ex_vote = pred_i[0][0]
        rest_vote = pred_i[0][1]
        start = week[i].index[0]
        end = week[i].index[-1]
        week_preds.append([ex_vote, rest_vote, 'exercise' if ex_vote > rest_vote else 'rest', start, end])

        # Correct unrealistic predictions (e.g. if a single 'e' segment is preceded and succeeded by 'r' segments)
        if len(week_preds) >= 5:  # if there are at least 5 previous segments ('i-4', 'i-3', ..., 'i') predicted

            # Relabels 'r, r, r, e, r' to 'r, r, r, r, r' or 'e, e, e, r, e' to 'e, e, e, e, e'
            if (week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i-2][2] == week_preds[i][2]
                    and week_preds[i-4][2] != week_preds[i-1][2]):
                week_preds[i-1][2] = week_preds[i-4][2]
            
            # Relabels 'r, r, e, r, r' to 'r, r, r, r, r'  or  'e, e, r, e, e' to 'e, e, e, e, e'  
            elif (week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i-1][2] == week_preds[i][2]
                  and week_preds[i-4][2] != week_preds[i-2][2]):
                week_preds[i-2][2] = week_preds[i-4][2]
            
            # Relabels 'r, e, r, r, r' to 'r, r, r, r, r'  or  'e, r, e, e, e' to 'e, e, e, e, e'
            elif (week_preds[i-4][2] == week_preds[i-2][2] == week_preds[i-1][2] == week_preds[i][2]
                  and week_preds[i-4][2] != week_preds[i-3][2]):
                week_preds[i-3][2] = week_preds[i-4][2]
            
            # Relabels 'e, e, r, r, e' to 'e, e, e, e, e' or 'r, r, e, e, r' to 'r, r, r, r, r'
            elif (week_preds[i-4][2] == week_preds[i-3][2] == week_preds[i][2]
                  and week_preds[i-4][2] != week_preds[i-2][2]
                  and week_preds[i-4][2] != week_preds[i-1][2]):
                week_preds[i-2][2] = week_preds[i-4][2]
                week_preds[i-1][2] = week_preds[i-4][2]
                
            # Relabels 'e, r, r, e, e' to 'e, e, e, e, e' or 'r, e, e, r, r' to 'r, r, r, r, r'
            elif (week_preds[i-4][2] == week_preds[i-1][2] == week_preds[i][2]
                  and week_preds[i-4][2] != week_preds[i-3][2]
                  and week_preds[i-4][2] != week_preds[i-2][2]):
                week_preds[i-3][2] = week_preds[i-4][2]
                week_preds[i-2][2] = week_preds[i-4][2]
                
            # Relabels 'e, r, e, r, e' to 'e, e, e, e, e' or 'r, e, r, e, r' to 'r, r, r, r, r'
            elif (week_preds[i-4][2] == week_preds[i-2][2] == week_preds[i][2]
                  and week_preds[i-4][2] != week_preds[i-3][2]
                  and week_preds[i-4][2] != week_preds[i-1][2]):
                week_preds[i-3][2] = week_preds[i-4][2]
                week_preds[i-1][2] = week_preds[i-4][2]
             
            else:
                pass
        # if there are not at least 5 segments predicted yet, nothing is relabeled
        else:
            pass
        
        print('{} of {} segments of validation workout {} predicted'.format(i+1, len(week), Configuration.Constants.VALIDATION_WEEK_IDS[workout_ind]))

    workout_ind += 1
    predictions.append(week_preds)

# Aggregate predicted segments of length of window size to continuous 'rest' or 'exercise' segments
y = Processing.aggregate_labels(predictions=predictions)

# Assess segmentation result by printing segmentation statistics and segmentation plot
Assessment.segmentation_assessment(true_labels=y_val_emb,
                                   predicted_labels=y,
                                   week_ids=Configuration.Constants.VALIDATION_WEEK_IDS)

# Save model
# model_1.save('*/segmentation.h5')   # for '*', insert the path to saving location
