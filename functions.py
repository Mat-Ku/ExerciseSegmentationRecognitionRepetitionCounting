import os
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt



def plot_classification_loss(history):
    """
    Plots loss and accuracy curve of training process of classification CNN
    'history' = History of the trained CNN model
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Training Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Validation Loss')
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Training Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    
    
def plot_regression_loss(history):
    """
    Plots loss and accuracy curve of training process of regression CNN
    'history' = History of the trained CNN model
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(hist['epoch'], hist['mae'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Validation Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.plot(hist['epoch'], hist['mse'], label='Training Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Validation Error')
    plt.legend()
    
    
    
def plot_segmentation(true, pred, workout_week):
    """
    Plots segmentation bar plots per workout week, based on true and predicted segmentation
    'true' = List of segments and segment boundaries of true workout
    'pred' = List of segments and segment boundaries of predicted workout
    'workout_week' = ID of workout week
    """
    fig, axs = plt.subplots(figsize=(12, 2), nrows=2, ncols=1, sharex=True, sharey=True)
    fig.suptitle('Workout {}'.format(workout_week))

    # print true segmentation
    colors1 = [] # stores color scheme
    # loop defines sequence of colors
    for i in true:
        if i[0] == 'rest':
            colors1.append('orange')
        else:
            colors1.append('blue') 
    cmap1 = mpl.colors.ListedColormap(colors1) # color map
    bounds1 = [0] + [i[2] for i in true] # boundaries
    norm1 = mpl.colors.BoundaryNorm(bounds1, len(colors1))
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap1, norm=norm1), 
                 cax=axs[0], 
                 ticks=[[0], [true[-1][2]]], # bar boundaries 
                 spacing='proportional', 
                 orientation='horizontal')
    axs[0].yaxis.set_label_position('right')
    axs[0].set_ylabel('True')

    # print predicted segmentation result 
    colors2 = [] # stores color scheme
    # loop defines sequence of colors
    for i in pred:
        if i[0] == 'rest':
            colors2.append('orange')
        else:
            colors2.append('blue') 
    cmap2 = mpl.colors.ListedColormap(colors2) # color map
    bounds2 = [0] + [i[2] for i in pred] # boundaries
    norm2 = mpl.colors.BoundaryNorm(bounds2, len(colors2))
    plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap2, norm=norm2), 
                 cax=axs[1], 
                 ticks=[[0], [pred[-1][2]]], # bar boundaries 
                 spacing='proportional',
                 orientation='horizontal')
    axs[1].yaxis.set_label_position('right')
    axs[1].set_ylabel('Pred')
    axs[1].set_xlabel('Instances', labelpad=-10)
    
    # set legend
    lines = [mpl.lines.Line2D([0], [0], color='orange', lw=4),
                mpl.lines.Line2D([0], [0], color='blue', lw=4)]
    fig.legend(lines, ['rest', 'exercise'], loc='upper right', bbox_to_anchor=(0.8, 1.25))

    # plot x-ticks only for lower bar
    for ax in axs:
        ax.label_outer()
    plt.show()



def load_lab(data_path):
    """
    Loads data labels.
    'data_path' = path to csv labels file
    Returns nested list, one for each exercise set, composed of [first frame, last frame, repetitions, exercise type] 
    """
    label_list = []
    with open(data_path, 'r') as csv_:
        reader = csv.reader(csv_)
        for row in reader:
            row.append([int(row[0]), int(row[1]), int(row[2]), row[3]])
    return label_list



def load_mod(data_path):
    """
    Loads data modality from indicated path.
    'data_path' = path to modality
    Returns data modality in an array; if not found, returns error.
    """
    try:
        m = np.load(data_path)
    except FileNotFoundError:
        m = None
        print('FileNotFoundError')
    return m



def load_data(data_path, week_ids):
    """
    Loads data from left smartwatch, filters out none-values,  extracts coordinate values of acc and gyr,
    and assigns activity and exercise class labels.
    'data_path' = File path to MM-Fit data
    'week_ids' = List of IDs of workout weeks
    Returns nested list containing [acc+gyr data, frames, week index, activity classes, exercise types]
    for each workout week named in week_ids.
    """
    week_list = []
    
    print('Progress:')
    for week_id in week_ids:
        print('Week {} is currently preprocessed...'.format(week_id))

        # Select accelerometer and gyroscope data from left smartwatch
        data = os.path.join(data_path, week_id) 
        mods = {}
        mods['sw_l_acc'] = load_mod(os.path.join(data, week_id + '_sw_l_acc.npy'))
        mods['sw_l_gyr'] = load_mod(os.path.join(data, week_id + '_sw_l_gyr.npy'))

        # Filter out modalities that contain none-values
        mods = {a: b for a, b in mods.items() if b is not None}

        # Extract x, y and z-values of accelerometer and gyroscope data
        frame = [item[0] for item in mods['sw_l_acc']]
        x_acc = [item[2] for item in mods['sw_l_acc']]
        y_acc = [item[3] for item in mods['sw_l_acc']]
        z_acc = [item[4] for item in mods['sw_l_acc']]
        x_gyr = [item[2] for item in mods['sw_l_gyr']]
        y_gyr = [item[3] for item in mods['sw_l_gyr']]
        z_gyr = [item[4] for item in mods['sw_l_gyr']]
        week_index = [week_id] * len(frame)
        acc_gyr = np.array(list(zip(x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr)))
        
        # Assign labels for activity classes (= 'exercise' or 'rest') and exercise classes (= 'squats', 'pushups', etc.)
        labels = load_lab(os.path.join(data + '/' + week_id + '_labels.csv'))
        activities = []
        exercises = []
        for f in frame:
            for label in labels:
                if  f >= label[0] and f <= label[1]:
                    activities.append('exercise')
                    exercises.append(label[3])
                    break
                else:
                    if labels.index(label) < (len(labels)-1):
                        continue
                    else:
                        activities.append('rest')
                        exercises.append('none')
                        break
        
        week_list.append([acc_gyr, frame, week_index, activities, exercises])
        
    return week_list