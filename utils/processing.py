import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Processing:
    """
    Class containing all processing steps applied to the data in advance of passing it to a model.
    """
    @staticmethod
    def standardize(data, mean=None, sd=None):
        """
        Standardizes input data, so that output data has mean=0 and std=1

        Parameters:
            data: Input data
            mean: Predefined mean, given as 6-dimensional array
            sd: Predefined standard deviation, given as 6-dimensional array

        Returns:
            List of Dataframes. Each Dataframe contains standardized data of one workout
        """
        if mean and sd:
            for segment in data:
                segment[0][:, 0] = (segment[0][:, 0] - mean[0]) / sd[0]
                segment[0][:, 1] = (segment[0][:, 1] - mean[1]) / sd[1]
                segment[0][:, 2] = (segment[0][:, 2] - mean[2]) / sd[2]
                segment[0][:, 3] = (segment[0][:, 3] - mean[3]) / sd[3]
                segment[0][:, 4] = (segment[0][:, 4] - mean[4]) / sd[4]
                segment[0][:, 5] = (segment[0][:, 5] - mean[5]) / sd[5]

            weeks_scaled = []
            for workout in data:
                df = pd.DataFrame(data=list(zip(workout[0][:, 0], workout[0][:, 1], workout[0][:, 2],
                                                workout[0][:, 3], workout[0][:, 4], workout[0][:, 5],
                                                workout[1], workout[2], workout[3], workout[4])),
                                  columns=['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'week',
                                           'activity', 'exercise', 'frame'])
                weeks_scaled.append(df)

            return weeks_scaled
        else:
            # Stack accelerometer and gyroscope data of all weeks
            data_stacked = np.concatenate([item[0] for item in data], axis=0)

            # Fit scaler on data
            scaler = StandardScaler()
            scaler.fit(data_stacked)

            # Transform data week-wise and merge with corresponding frames, week index, activity types and exercise types
            weeks_scaled = []
            for ind in range(len(data)):
                acc_gyr_scaled = scaler.transform(data[ind][0])
                df = pd.DataFrame(data=list(zip(acc_gyr_scaled[:, 0], acc_gyr_scaled[:, 1], acc_gyr_scaled[:, 2],
                                                acc_gyr_scaled[:, 3], acc_gyr_scaled[:, 4], acc_gyr_scaled[:, 5],
                                                data[ind][1], data[ind][2], data[ind][3], data[ind][4])),
                                  columns=['x_acc', 'y_acc', 'z_acc', 'x_gyr', 'y_gyr', 'z_gyr', 'week',
                                           'activity', 'exercise', 'frame'])
                weeks_scaled.append(df)

            return weeks_scaled

    @staticmethod
    def segmenting(data, window_size):
        """
        Slices input data into segments of length of given window size.

        Parameters:
            data: Input data.
            window_size: Number of instances per slice.

        Returns:
            List of segments of length of window size.
        """
        segments = []
        for week in data:
            week_segments = []
            for window in range(0, len(week), window_size):
                if window == 0:
                    continue
                else:
                    if window + window_size > len(week):
                        seg_1 = week[window - window_size:window]
                        seg_2 = week[window:]
                        week_segments.append(seg_1)
                        week_segments.append(seg_2)
                    else:
                        seg = week[window - window_size:window]
                        week_segments.append(seg)
            segments.append(week_segments)

        return segments

    @staticmethod
    def padding(data, window_size):
        """
        Pads all segments in the data to a common length of size window_size.

        Parameters:
            data: Input data segments.
            window_size: Length to which segments are padded.

        Returns:
            Padded data segments as tensors.
        """
        padded_data = []
        if type(data[0]) is pd.Series:
            for series in data:
                padded_data.append(tf.pad(tensor=series.values, paddings=tf.constant([[window_size - len(series), 0]])))
        if type(data[0]) is pd.DataFrame:
            for df in data:
                padded_data.append(tf.pad(tensor=df.values, paddings=tf.constant([[window_size - len(df), 0], [0, 0]])))
        if type(data[0]) is np.ndarray:
            for array in data:
                padded_data.append(tf.pad(tensor=array, paddings=tf.constant([[window_size - len(array), 0], [0, 0]])))

        return padded_data

    @staticmethod
    def reshaping(data):
        """
        Reshapes tensors to format required by 2D-CNN, being (n_rows=300, n_columns=6, n_channels=1)

        Parameters:
            data: Data segments passed as tensors.

        Returns:
            Reshaped segments as ndarray.
        """
        reshaped_data = []
        if len(data[0].shape) == 1:
            for tensor in data:
                reshaped_data.append(np.array(tensor).reshape(tensor.shape[0], 1))
        else:
            for tensor in data:
                reshaped_data.append(np.array(tensor).reshape(tensor.shape[0], tensor.shape[1], 1))

        return np.array(reshaped_data)

    @staticmethod
    def aggregate_labels(predictions):
        """
        Aggregate predicted segment labels, having length of a predefined window size, to consecutive 'exercise'
        or 'rest' segments.
        Example: 'r','r','r','e','e','e' -> 'r', 'e' (while keeping track of start and end indices of each
        activity segment)

        Parameters:
            predictions: List of predictions per segment made by segmentation loop.

        Returns:
            List of aggregated predictions.
        """
        y = []
        for predicted_week in predictions:
            y_week = []  # stores predicted labels week-wise
            for i in range(len(predicted_week)):
                if i == 0:  # deals with very first segment of current validation workout week
                    activity = predicted_week[i][2]
                    start_index = predicted_week[i][3]
                    j = i
                    while predicted_week[j][2] == activity:
                        end_index = predicted_week[j][4]
                        j += 1
                    y_week.append([activity, start_index, end_index])
                else:
                    if predicted_week[i][2] != predicted_week[i - 1][2]:
                        activity = predicted_week[i][2]
                        start_index = predicted_week[i][3]
                        j = i
                        while (predicted_week[j][2] == activity):
                            end_index = predicted_week[j][4]
                            if j == (len(predicted_week) - 1):
                                break
                            else:
                                j += 1
                        y_week.append([activity, start_index, end_index])
                    else:
                        continue
            y.append(y_week)

        return y

    @staticmethod
    def get_exercise_segments(data):
        """
        Retrieves only exercise segments from workout data.

        Parameters:
            data: Input data containing both rest and exercise segments.

        Returns:
            List containing only exercise segments.
        """
        exercise_data = []
        for workout in data:
            group = workout['exercise'].ne(workout['exercise'].shift()).cumsum()
            segments = [g for _, g in workout.groupby(group)]
            exercise_data.append(segments)

        # Flatten list and eliminate rest segments
        return [segment for sublist in exercise_data for segment in sublist if segment['exercise'].all() != 'none']

    @staticmethod
    def savitzky_golay_filter(data, window_length, polyorder, axis):
        """
        Applies Savitzky-Golay-Filter to input data.

        Parameters:
            data: Input data.
            window_length: Step size at which filter is applied.
            polyorder: Order of polynomial, which is fit to each window.
            axis: Axis along which data is filtered.

        Returns:
            Data filtered along given axis.
        """
        for segment in data:
            segment[:, 0] = savgol_filter(segment[:, 0], window_length=window_length, polyorder=polyorder, axis=axis)
            segment[:, 1] = savgol_filter(segment[:, 1], window_length=window_length, polyorder=polyorder, axis=axis)
            segment[:, 2] = savgol_filter(segment[:, 2], window_length=window_length, polyorder=polyorder, axis=axis)
            segment[:, 3] = savgol_filter(segment[:, 3], window_length=window_length, polyorder=polyorder, axis=axis)
            segment[:, 4] = savgol_filter(segment[:, 4], window_length=window_length, polyorder=polyorder, axis=axis)
            segment[:, 5] = savgol_filter(segment[:, 5], window_length=window_length, polyorder=polyorder, axis=axis)

        return data

    @staticmethod
    def pca(data, n_components):
        """
        Conducts Principal Component Analysis on data.

        Parameters:
            data: Input data.
            n_components: Number of dimensions to which input data shall be reduced.

        Returns:
            List of dimensionality-reduced segments.
        """
        segments = []
        for segment in data:
            pca = PCA(n_components=n_components)
            pca.fit(segment)
            segments.append(pca.transform(segment))

        return segments
