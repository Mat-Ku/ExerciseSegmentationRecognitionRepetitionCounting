class Configuration:

    class Constants:
        # Paths
        MMFIT_DATA_PATH = "G:/2_studium/data_science/4th_term/master_thesis/python_code/data/mm-fit"
        RECOFIT_DATA_PATH = "G:/2_studium/data_science/4th_term/master_thesis/python_code/data/recofit/exercise_data.50.0000_singleonly.mat"
        SEGMENTATION_MODEL_PATH = "models/segmentation.h5"
        EXERCISE_RECOGNITION_MODEL_PATH = "models/exercise_recognition.h5"
        REPETITION_COUNTING_MODEL_PATH = "models/repetition_counting.h5"

        # Week IDs
        ALL_WEEK_IDS = ['w00', 'w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14',
                        'w15', 'w16', 'w17', 'w18', 'w19', 'w20']
        TRAINING_WEEK_IDS = ['w00', 'w01'] #'w02', 'w03', 'w04', 'w05', 'w07', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15',
                             #'w18', 'w19', 'w20']
        VALIDATION_WEEK_IDS = ['w08', 'w16']
        TEST_WEEK_IDS = ['w06', 'w17']

        # Constant values

        # Means and standard deviations of StandardScaler fitted on training data
        # The six values each correspond to the six feature columns (x_acc, y_acc, z_acc, x_gyr, y_gyr, z_gyr)
        TRAINING_DATA_MEAN = [-4.03832065, -3.30813777, 1.83333368, -0.01333664, 0.00769117, -0.02414823]
        TRAINING_DATA_SD = [6.29026018, 5.13139172, 4.51634952, 1.1398996, 0.78557017, 1.07878059]
        WINDOW_SIZE = 300
        RECOFIT_TARGET_EXERCISES = [
            # squat variations
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
