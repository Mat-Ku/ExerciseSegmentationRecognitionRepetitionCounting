import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class Plotting:
    """
    Class containing all functions for plotting results.
    """
    @staticmethod
    def plot_classification_loss(history):
        """
        Plots loss and accuracy of training process of classification CNN.

        Parameters:
            history: History of the trained CNN model.
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

    @staticmethod
    def plot_regression_loss(history):
        """
        Plots loss and accuracy of training process of regression CNN.

        Parameters:
            history: History of the trained CNN model.
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

    @staticmethod
    def plot_segmentation(true_labels, predicted_labels, validation_week_ids):
        """
        Plots segmentation bar plots per workout week, based on true and predicted segmentation.

        Parameters:
            true_labels: Ground truth.
            predicted_labels: Segment labels predicted by segmentation loop.
            validation_week_ids: IDs of weeks used as validation data.

        Returns:
            Plot containing true and predicted segmentation
        """
        fig, axs = plt.subplots(figsize=(12, 2), nrows=2, ncols=1, sharex=True, sharey=True)
        fig.suptitle('Workout {}'.format(validation_week_ids))

        # print true segmentation
        colors1 = [] # stores color scheme
        # loop defines sequence of colors
        for i in true_labels:
            if i[0] == 'rest':
                colors1.append('orange')
            else:
                colors1.append('blue')
        cmap1 = mpl.colors.ListedColormap(colors1)  # color map
        bounds1 = [0] + [i[2] for i in true_labels]  # boundaries
        norm1 = mpl.colors.BoundaryNorm(bounds1, len(colors1))
        plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap1, norm=norm1),
                     cax=axs[0],
                     ticks=[[0], [true_labels[-1][2]]],  # bar boundaries
                     spacing='proportional',
                     orientation='horizontal')
        axs[0].yaxis.set_label_position('right')
        axs[0].set_ylabel('True')

        # print predicted segmentation result
        colors2 = [] # stores color scheme
        # loop defines sequence of colors
        for i in predicted_labels:
            if i[0] == 'rest':
                colors2.append('orange')
            else:
                colors2.append('blue')
        cmap2 = mpl.colors.ListedColormap(colors2)  # color map
        bounds2 = [0] + [i[2] for i in predicted_labels]  # boundaries
        norm2 = mpl.colors.BoundaryNorm(bounds2, len(colors2))
        plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap2, norm=norm2),
                     cax=axs[1],
                     ticks=[[0], [predicted_labels[-1][2]]],  # bar boundaries
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
