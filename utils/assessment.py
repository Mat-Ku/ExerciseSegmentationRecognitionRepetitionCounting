import numpy as np

from utils.plotting import Plotting


class Assessment:

    @staticmethod
    def segmentation_assessment_old(true_labels, predicted_labels, validation_week_ids):
        """
        Measures the overall accuracy of the segmentation step by three metrics:
        a) Segment Miscount Rate: Percentage of deviation of predicted segment count from true segment count
        b) Mean Position Error: Mean difference in seconds of start and end of predicted segment w.r.t. start and end of
                                the corresponding true segment
        c) Classification Accuracy: Percentage of correctly classified segments among all classified segments

        Parameters:
            true_labels: Ground truth.
            predicted_labels: Segment labels predicted by segmentation loop.
            validation_week_ids: IDs of weeks used as validation data.

        Prints:
            Segmentation assessment statistics.
        """
        # Loops over predicted validation workout week label vectors
        for i in range(len(predicted_labels)):

            # Case 1: If segmentation model predicted the number of segments for this workout correctly
            if len(predicted_labels[i]) == len(true_labels[i]):

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                pos_diffs = []
                for j in range(len(predicted_labels[i])):
                    # division by 100 i.o.t. compute result in seconds, since data has been collected at 100 Hz
                    pos_diff = (abs(predicted_labels[i][j][1] - true_labels[i][j][1]) + abs(predicted_labels[i][j][2] - true_labels[i][j][2])) / 100
                    pos_diffs.append(pos_diff)
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [k[0] for k in true_labels[i]]
                pred_classes = [l[0] for l in predicted_labels[i]]
                ca = (len([m for m in range(len(pred_classes)) if pred_classes[m] == true_classes[m]]) / len(
                    true_classes)) * 100

            # Case 2: If segmentation model predicted more segments for this workout than there actually are
            elif len(predicted_labels[i]) > len(true_labels[i]):

                print('CAUTION: For workout {}, segmentation model predicted {} segments, while there are actually only {}.'.format(
                        validation_week_ids[i], len(predicted_labels[i]), len(true_labels[i])))

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
                matches = []
                for j in predicted_labels[i]:
                    deltas = []
                    for k in true_labels[i]:
                        deltas.append((abs(j[1] - k[1]) + abs(j[2] - k[2])))
                    matches.append(np.argmin(deltas))
                y_inds_true = []
                for index, val in enumerate(matches):
                    if index == 0:
                        y_inds_true.append(index)
                    elif matches[index] != matches[index - 1]:
                        y_inds_true.append(index)
                    else:
                        continue
                # compute mean difference in seconds of start and end indices of best-matching segments
                pos_diffs = []
                y_val_index = 0
                for l in y_inds_true:
                    pos_diff = (abs(predicted_labels[i][l][1] - true_labels[i][y_val_index][1]) + abs(
                        predicted_labels[i][l][2] - true_labels[i][y_val_index][2])) / 100
                    pos_diffs.append(pos_diff)
                    y_val_index += 1
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [m[0] for m in true_labels[i]]
                pred_classes = [predicted_labels[i][n][0] for n in y_inds_true]
                ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(
                    predicted_labels[i])) * 100

            # Case 3: If segmentation model predicted fewer segments for this workout than there actually are
            else:
                print('CAUTION: For workout {}, segmentation model predicted only {} segments, while there are actually {}.'.format(
                        validation_week_ids[i], len(predicted_labels[i]), len(true_labels[i])))

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
                matches = []
                for j in predicted_labels[i]:
                    deltas = []
                    for k in true_labels[i]:
                        deltas.append((abs(j[1] - k[1]) + abs(j[2] - k[2])))
                    matches.append(np.argmin(deltas))
                for idx in range(len(matches) - 1):
                    if matches[idx] == matches[idx + 1]:
                        matches[idx] = matches[idx + 1] - 1
                    else:
                        continue
                # compute mean difference in seconds of start and end indices of best-matching segments
                pos_diffs = []
                y_index = 0
                for l in matches:
                    pos_diff = (abs(predicted_labels[i][y_index][1] - true_labels[i][l][1]) + abs(
                        predicted_labels[i][y_index][2] - true_labels[i][l][2])) / 100
                    pos_diffs.append(pos_diff)
                    y_index += 1
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [true_labels[i][m][0] for m in matches]
                pred_classes = [n[0] for n in predicted_labels[i]]
                ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(
                    predicted_labels[i])) * 100

            # Print statistics
            print('Segmentation Assessment for Workout {}:\n\
            Segment Miscount Rate: {:.2f}%, Mean Position Error: {:.2f}s, Classification Accuracy: {:.2f}%\n'.format(
                validation_week_ids[i], smr, mpe, ca))

            # Plot comparison of true and predicted segmentation
            Plotting.plot_segmentation(true_labels[i], predicted_labels[i], validation_week_ids[i])

    @staticmethod
    def segmentation_assessment(true_labels, predicted_labels, week_ids):

        # Lists that apply, if segmentation model predicted the correct number of segments:
        true_pred_segs = []  # stores index of correctly identified segments i.t.o. index order of both 'y' and 'y_test'
        # (which is the same in this case)

        # Lists that apply, if segmentation model predicted more segments than there actually are:
        true_pred_segs_y = []  # stores index of correctly identified segments i.t.o. index order of 'y'
        false_pred_segs_y = []  # stores index of falsely identified segments i.t.o. index order of 'y'

        # Lists that apply, if segmentation model predicted fewer segments than there actually are:
        true_pred_segs_y_test = []  # stores index of correctly identified segments i.t.o. index order of 'y_test'
        non_pred_segs_y_test = []  # stores index of unidentified segments i.t.o. index order of 'y_test'

        for i in range(len(predicted_labels)):

            # Case 1: If segmentation model predicted the number of segments for this workout correctly
            if len(predicted_labels[i]) == len(true_labels[i]):

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                pos_diffs = []
                for j in range(len(predicted_labels[i])):
                    # division by 100 i.o.t. compute result in seconds, since data has been collected at 100 Hz
                    pos_diff = (abs(predicted_labels[i][j][1] - true_labels[i][j][1]) + abs(
                        predicted_labels[i][j][2] - true_labels[i][j][2])) / 100  # differences are computed in seconds
                    pos_diffs.append(pos_diff)
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [k[0] for k in true_labels[i]]
                pred_classes = [l[0] for l in predicted_labels[i]]
                ca = (len([m for m in range(len(pred_classes)) if pred_classes[m] == true_classes[m]]) / len(
                    true_classes)) * 100

                # safe index of each segment of 'y[i]', as each of them was identified correctly
                true_pred_segs.append([ind for ind in range(len(predicted_labels[i]))])
                # append empty list to all the other lists that were not relevant
                true_pred_segs_y.append([])
                false_pred_segs_y.append([])
                true_pred_segs_y_test.append([])
                non_pred_segs_y_test.append([])

            # Case 2: If segmentation model predicted more segments for this workout than there actually are
            elif len(predicted_labels[i]) > len(true_labels[i]):

                print('CAUTION: For workout {}, the segmentation model predicted {} segments, while there are actually only {} segments.\n\
                This divergence will have a negative impact on the error statistics.\n\
                Falsely detected segments are treated as errors in the Classification and Repetition Counting steps.'.format(
                    week_ids[i], len(predicted_labels[i]), len(true_labels[i])))

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
                matches = []
                for j in predicted_labels[i]:
                    deltas = []
                    for k in true_labels[i]:
                        deltas.append((abs(j[1] - k[1]) + abs(j[2] - k[2])))
                    matches.append(np.argmin(deltas))
                y_inds_true = []
                for index, val in enumerate(matches):
                    if index == 0:  # if matches[index] == 0 and index == 0:
                        y_inds_true.append(index)
                    elif matches[index] != matches[index - 1]:
                        y_inds_true.append(index)
                    else:
                        continue
                # compute mean difference in seconds of start and end indices of best-matching segments
                pos_diffs = []
                y_test_index = 0
                for l in y_inds_true:
                    pos_diff = (abs(predicted_labels[i][l][1] - true_labels[i][y_test_index][1]) + abs(
                        predicted_labels[i][l][2] - true_labels[i][y_test_index][2])) / 100
                    pos_diffs.append(pos_diff)
                    y_test_index += 1
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [m[0] for m in true_labels[i]]
                pred_classes = [predicted_labels[i][n][0] for n in y_inds_true]
                ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(
                    predicted_labels[i])) * 100

                # Safe index of those segments of 'y_test[i]', which are also contained in 'y[i]', and were therefore correctly identified
                true_pred_segs_y.append(y_inds_true)
                # Safe index of those segments of 'y[i]', which are not contained in 'y_test[i]', and were therefore falsely identified
                false_pred_segs_list = []
                for p in range(len(matches) - 1):
                    if matches[p] == matches[p + 1]:
                        false_pred_segs_list.append(p + 1)
                    else:
                        continue
                false_pred_segs_y.append(false_pred_segs_list)
                # append empty list to all the other lists that were not relevant
                true_pred_segs.append([])
                true_pred_segs_y_test.append([])
                non_pred_segs_y_test.append([])

            # Case 3: If segmentation model predicted fewer segments for this workout than there actually are
            else:

                print('CAUTION: For workout {}, the segmentation model predicted only {} segments, while there are actually {} segments.\n\
                This divergence will have a negative impact on the error statistics.\n\
                Undetected segments are treated as errors in the Classification and Repetition Counting steps.'.format(
                    week_ids[i], len(predicted_labels[i]), len(true_labels[i])))

                # a) Segment Miscount Rate
                smr = abs(((len(predicted_labels[i]) / len(true_labels[i])) * 100) - 100)

                # b) Mean Position Error
                # find best-matching segments between predicted segments and true segments i.t.o. segment boundaries
                matches = []
                for j in predicted_labels[i]:
                    deltas = []
                    for k in true_labels[i]:
                        deltas.append((abs(j[1] - k[1]) + abs(j[2] - k[2])))
                    matches.append(np.argmin(deltas))
                for idx in range(len(matches) - 1):
                    if matches[idx] == matches[idx + 1]:
                        matches[idx] = matches[idx + 1] - 1
                    else:
                        continue
                # compute mean difference in seconds of start and end indices of best-matching segments
                pos_diffs = []
                y_index = 0
                for l in matches:
                    pos_diff = (abs(predicted_labels[i][y_index][1] - true_labels[i][l][1]) + abs(
                        predicted_labels[i][y_index][2] - true_labels[i][l][2])) / 100  # differences are computed in seconds
                    pos_diffs.append(pos_diff)
                    y_index += 1
                mpe = (sum(pos_diffs) / len(pos_diffs))

                # c) Classification Accuracy
                true_classes = [true_labels[i][m][0] for m in matches]
                pred_classes = [n[0] for n in predicted_labels[i]]
                ca = (len([o for o in range(len(pred_classes)) if pred_classes[o] == true_classes[o]]) / len(
                    predicted_labels[i])) * 100

                # Safe index of those segments of 'y_test[i]', which are also contained in 'y[i]', and were therefore correctly identified
                true_pred_segs_y_test.append(matches)
                # Safe index of those segments of 'y_test[i]', which were not identified segmentation model
                non_pred_segs_y_test.append([ind for ind in range(len(true_labels[i])) if ind not in matches])
                # append empty list to all the other lists that were not relevant
                true_pred_segs.append([])
                true_pred_segs_y.append([])
                false_pred_segs_y.append([])

                # Print Segment Miscount Rate, Mean Position Error and Classification Accuracy
            print('Segmentation Assessement for Workout {}:\n\
            Segment Miscount Rate: {:.2f}%, Mean Position Error: {:.2f}s, Classification Accuracy: {:.2f}%\n'.format(
                week_ids[i], smr, mpe, ca))

            Plotting.plot_segmentation(true_labels[i], predicted_labels[i], week_ids[i])

        return true_pred_segs, true_pred_segs_y, false_pred_segs_y, true_pred_segs_y_test, non_pred_segs_y_test
