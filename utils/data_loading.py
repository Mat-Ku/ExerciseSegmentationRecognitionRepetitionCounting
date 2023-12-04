from abc import ABCMeta, abstractmethod
from scipy.io import loadmat
from typing import List, Union

import os
import csv
import numpy as np
import pandas as pd


class DataLoading(metaclass=ABCMeta):
    """
    Abstract class enforcing a method for data loading to be implemented in all child classes.
    """

    @staticmethod
    @abstractmethod
    def load_data(data_path: str, *args, **kwargs) -> List:
        """
        Load data from disk.

        :param data_path: Path to data set

        :return: List of workouts (for MMFit) of list of segments (for RecoFit)
        """
        pass


class MMFitDataLoading(DataLoading):
    """
    Class containing all methods used for loading MM-Fit data.
    """

    @staticmethod
    def load_exercise_labels(data_path: str) -> List[List[Union[int, int, str]]]:
        """
        Loads data labels.

        :param data_path: Path to csv labels file.

        :return: Nested list, one for each exercise set, composed of [first frame, last frame, exercise type].
        """
        labels = []
        with open(data_path, 'r') as csv_:
            reader = csv.reader(csv_)
            for row in reader:
                labels.append([int(row[0]), int(row[1]), row[3]])
        return labels

    @staticmethod
    def load_mods(data_path: str) -> np.ndarray:
        """
        Loads data modality from indicated path.

        :param data_path: Path to data modality

        :return: Data from modality
        """
        try:
            m = np.load(data_path)
        except FileNotFoundError:
            m = None
            print('FileNotFoundError')
        return m

    @staticmethod
    def load_data(data_path: str, week_ids: List[str]) -> List[List[Union[np.ndarray, str]]]:
        """
        Loads data from left smartwatch, filters out none-values, extracts coordinate values of accelerometer and
        gyroscope, and assigns activity and exercise class labels.

        :param data_path: File path to MM-Fit data
        :param week_ids: Selection of workout weeks that shall be loaded

        :return: Nested list containing [acc+gyr data, week ID, activity classes, exercise types] for each workout week
        named in week_ids.
        """
        week_list = []

        print('Progress:')
        for week_id in week_ids:
            print('Week {} is currently processed...'.format(week_id))

            # Select accelerometer and gyroscope data from left smartwatch
            data = os.path.join(data_path, week_id)
            mods = {}
            mods['sw_l_acc'] = MMFitDataLoading.load_mods(os.path.join(data, week_id + '_sw_l_acc.npy'))
            mods['sw_l_gyr'] = MMFitDataLoading.load_mods(os.path.join(data, week_id + '_sw_l_gyr.npy'))

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
            labels = MMFitDataLoading.load_exercise_labels(os.path.join(data + '/' + week_id + '_labels.csv'))
            activities = []
            exercises = []
            for f in frame:
                for label in labels:
                    if f >= label[0] and f <= label[1]:
                        activities.append('exercise')
                        exercises.append(label[2])
                        break
                    else:
                        if labels.index(label) < (len(labels)-1):
                            continue
                        else:
                            activities.append('rest')
                            exercises.append('none')
                            break

            week_list.append([acc_gyr, week_index, activities, exercises, frame])

        return week_list

    @staticmethod
    def load_repetition_counts(data_path: str, week_ids: List[str]) -> List[int]:
        """
        Loads repetition counts of exercise segments of specified workout week(s).

        :param data_path: File path to MM-Fit data
        :param week_ids: List of IDs of workout weeks

        :return: List of repetition counts
        """
        repetition_counts = []
        for id_ in week_ids:
            file = os.path.join(data_path, id_ + '/', id_ + '_labels.csv')
            df_labels = pd.read_csv(file, header=None)
            list_labels = [list(row) for row in df_labels.values]
            repetition_counts.append([l[2] for l in list_labels])
        return [item for sublist in repetition_counts for item in sublist]


class RecoFitDataLoading(DataLoading):
    """
    Class containing all methods used for loading RecoFit data.
    """

    @staticmethod
    def load_data(data_path: str, exercise_types: List[str]) -> List[List[Union[np.ndarray, int, str]]]:
        """
        Loads accelerometer and gyroscope data, repetition counts and exercise types of those Recofit data segments, that
        have the specified exercise type.

        :param data_path: Path to RecoFit data set
        :param exercise_types: Selection of exercise types that shall be considered

        :return: Nested list, containing acc+gyr data, repetition count and exercise type for each segment
        """

        data = []
        recofit_data = loadmat(data_path)

        for subject in recofit_data['subject_data']:
            for i in range(len(subject)):
                try:
                    exercise_name = subject[i][0, 0][5][0]
                    if exercise_name in exercise_types:
                        acc = subject[i][0, 0][14][0, 0][0]
                        gyr = subject[i][0, 0][14][0, 0][1]
                        acc_gyr = np.concatenate((acc[:, 1:], gyr[:, 1:]), axis=1)  # first column is time
                        repetitions = subject[i][0, 0][15][0, 0]
                        data.append([acc_gyr, repetitions, exercise_name])
                    else:
                        continue

                except IndexError:
                    continue

        return data
