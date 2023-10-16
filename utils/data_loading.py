import os
import csv
import numpy as np
import pandas as pd


class DataLoading:
    """
    Class containing all functions used for loading MM-Fit and Recofit data.
    """
    @staticmethod
    def load_labels(data_path):
        """
        Loads data labels.

        Parameters:
            data_path: Path to csv labels file.

        Returns:
            Nested list, one for each exercise set, composed of [first frame, last frame, repetitions, exercise type].
        """
        labels = []
        with open(data_path, 'r') as csv_:
            reader = csv.reader(csv_)
            for row in reader:
                labels.append([int(row[0]), int(row[1]), int(row[2]), row[3]])
        return labels

    @staticmethod
    def load_mods(data_path):
        """
        Loads data modality from indicated path.

        Parameters:
            data_path: Path to modality.

        Returns:
            Data modality in an array; if not found, returns error.
        """
        try:
            m = np.load(data_path)
        except FileNotFoundError:
            m = None
            print('FileNotFoundError')
        return m

    @staticmethod
    def load_data(data_path, week_ids):
        """
        Loads data from left smartwatch by completing the following steps:
            1. Select accelerometer and gyroscope data of left smartwatch.
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
            mods['sw_l_acc'] = DataLoading.load_mods(os.path.join(data, week_id + '_sw_l_acc.npy'))
            mods['sw_l_gyr'] = DataLoading.load_mods(os.path.join(data, week_id + '_sw_l_gyr.npy'))

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

            # Assign labels for activity classes (= 'exercise' or 'rest') and exercise classes (= 'squats' etc.)
            labels = DataLoading.load_labels(os.path.join(data + '/' + week_id + '_labels.csv'))
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

    @staticmethod
    def load_repetition_counts(data_path, week_ids):
        """
        Loads repetition counts of exercise segments of specified workout week(s).

        Parameters:
            data_path: File path to MM-Fit data.
            week_ids: List of IDs of workout weeks.

        Returns:
            List of labels.
        """
        labels = []
        for id_ in week_ids:
            file = os.path.join(data_path, id_ + '/', id_ + '_labels.csv')
            df_labels = pd.read_csv(file, header=None)
            list_labels = [list(row) for row in df_labels.values]
            labels.append([l[2] for l in list_labels])
        return [item for sublist in labels for item in sublist]
