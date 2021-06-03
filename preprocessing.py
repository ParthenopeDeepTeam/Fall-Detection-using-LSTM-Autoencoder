import json
import os
import cv2
import glob
import numpy as np
# skeleton_keypoints è l'intero dataset

_NUM_KEYPOINTS = 19
_X_SIZE = 2304
_Y_SIZE = 1296
_Q_FACTOR = 5       # quantization factor

def load_data(path):
    skeletons_keypoints = []

    if path=='Test-Set':
        directories = ['Mixed-Falls', 'Mixed-Falls-Mirrored', 'Mixed-Falls-2', 'Mixed-Falls-2-Mirrored']
    else:
        directories = ['Random-Walk', 'Random-Walk-Mirrored', 'Slow-Walk-2', 'Slow-Walk-2-Mirrored', 'Slow-Walk', 'Slow-Walk-Mirrored', 'Still', 'Still-Mirrored',
                    'Mom-Walk', 'Mom-Walk-Mirrored', 'Running-Exercises', 'Running-Exercises-Mirrored','Obstacle', 'Obstacle-Mirrored']
        #'Mixed-Walk', 'Mixed-Walk-Mirrored','Exercise-Obstacles', 'Exercise-Obstacles-Mirrored', 'Lifting-Objects', 'Lifting-Objects-Mirrored'] 
        #'Mixed-Walk', 'Mixed-Walk-Mirrored','Exercise-Obstacles', 'Exercise-Obstacles-Mirrored', 'Lifting-Objects', 'Lifting-Objects-Mirrored',]
        #directories = ['Random-Walk', 'Slow-Walk-2', 'Slow-Walk', 'Still', 'Mom-Walk', 'Mixed-Walk', 'Obstacle', 'Running-Exercises', 'Exercise-Obstacles', 'Lifting-Objects']
    
    for dir in directories:    
        sub_dir = (os.path.join(path,dir)) # Full path of each directory
        print("Loading data from " + sub_dir)
        for file in sorted(glob.glob(sub_dir+'/*.json')):
            with open(file) as file_json:
                data = json.load(file_json)
                if data["people"]:  # If any person has been detected by openpose in that frame
                    keypoints = data["people"][0]["pose_keypoints_2d"]              #TO-DO sostituire 0 con l'indice degli n scheletri nel json

                    # We remove  keypoints 15 16 23 24 20 21 because they bring no informations about skeleton position
                    # considering them just as noise
                    for i in [24, 23, 21, 20, 16, 15]:
                        del keypoints[i*3:(i*3)+3]

                    skeletons_keypoints.append(keypoints)  # contiene gli scheletri di tutti i files
    
    print("Loaded " + str(len(skeletons_keypoints)) + " skeletons [" + str(len(skeletons_keypoints)) + " frames]" )
    return skeletons_keypoints


def remove_low_scoring_keypoints(skeleton_keypoints, tolerance=0.15): #original 0.1
    # we list through all keypoints confidences, and if they are lower than a delta are set to zero
    # (and the coordinates? they remain the same or...?)

    preprocessed_ds = []
    for skeleton in skeleton_keypoints:
        # solution 1: this solution can't be used with normalized coordinates
        # we obtain index of confidences of the i-th skeleton that are lower than tolerance
        # zero_confidence_index = [i for i in skeleton if skeleton[i]<tolerance]
        # than we set to zero that elements
        # skeleton[zero_confidence_index] = 0

        # solution 2:
        new_skeleton = skeleton
        for i in range(len(new_skeleton)):
            if (i + 1) % 3 == 0:  # ogni volta al 3 elemento (confidence) controlliamo se supera la tolleranza
                if new_skeleton[i] < tolerance:
                    new_skeleton[i] = 0

        preprocessed_ds.append(new_skeleton)

    return preprocessed_ds


def remove_null_skeleton(skeleton_keypoints, mean_confidence_tolerance=0.35): #original 0.25
    # the first part of the preprocessing needs to be the removal of all null skeletons 
    # in order to have always "full" keypoints

    new_skeletons = []

    for skeleton in skeleton_keypoints:
        
        actual_mean = 0
        for j in range(len(skeleton)):
            if (j + 1) % 3 == 0:  # ogni volta al 3 elemento (confidence) 
                actual_mean += skeleton[j]

        actual_mean /= _NUM_KEYPOINTS

        if actual_mean >= mean_confidence_tolerance:
            del skeleton[2::3]  # Get rid of the scores
            new_skeletons.append(skeleton)
            
    print(len(new_skeletons), "skeletons remaining after removing irrilevant frames")
    return new_skeletons


def downsample_dataset(skeleton_keypoints):
    # the video is recorded at 60 frame per second, we apply a downsampling reducing the frames at 30 per second
    # we just halve the dataset holding even frames

    return [skeleton_keypoints[i] for i in range(len(skeleton_keypoints)) if i % 2 == 0]


def normalization(skeleton_keypoints):
    
    new_skeletons = skeleton_keypoints

    for skeleton in new_skeletons:
        keypoints_x = []
        keypoints_y = []

        for i in range(len(skeleton)):
            if i%2==0:
                keypoints_x.append(int(skeleton[i]))
            else:
                keypoints_y.append(int(skeleton[i]))

        # Normalization based on image coordinates
        # for i in range(len(skeleton)):
        #     if i%2==0:  #if it's the x coord
        #         skeleton[i] = skeleton[i]/_X_SIZE
        #     else:       # if it's the y coord
        #         skeleton[i] = skeleton[i]/_Y_SIZE

        # min-max normalization
        # 0 values must be removed for the min-max normalization
        x = np.array(keypoints_x)
        x = x[ x != 0]
        y = np.array(keypoints_y)
        y = y[ y != 0]

        # We also round float values holding just N decimal places
        for i in range(len(skeleton)):
            if i%2==0: #if it's the x coord
                skeleton[i] = (keypoints_x[int(i/2)] - x.min())/(x.max()-x.min())
            else: # if it's the y coord
                skeleton[i] = (keypoints_y[int(i/2)] - y.min())/(y.max()-y.min())
    	
    return _quantization(new_skeletons)

def _quantization(skeleton_keypoints):

    numpy_skeletons = np.array(skeleton_keypoints).flatten()
    print("Valori unici PRIMA:",np.unique(numpy_skeletons).shape[0])

    new_skeletons = skeleton_keypoints
    
    # Define bins of quantization
    x_bins = np.linspace(0,1,int(_X_SIZE/_Q_FACTOR))
    y_bins = np.linspace(0,1,int(_Y_SIZE/_Q_FACTOR))
    
    for skeleton in new_skeletons:
        for i in range(len(skeleton)):
            if i%2==0:
                #if it's x coord
                skeleton[i] = _find_bin(skeleton[i], x_bins)
            else:
                #if it's y coord
                skeleton[i] = _find_bin(skeleton[i], y_bins)

    numpy_skeletons = np.array(new_skeletons).flatten()
    print("Valori unici DOPO:",np.unique(numpy_skeletons).shape[0])

    return new_skeletons


def _find_bin(value, bins):
    for i in range(len(bins)):
        if value <= bins[i]:
            return bins[i]


def data_windowing(skeleton_keypoints, window_lenght=3, overlap=1.5):
    # our assumption is that the video is recorded at 60fps, than downsampled at 30fps
    # so 1 second has 25 frames

    batched_skeletons = []

    frames_in_window = 25 * window_lenght #75
    overlap_frames = int(overlap * 25) #38

    batched_skeletons.append(skeleton_keypoints[0:frames_in_window])
    # manually creating the first windows to avoid overlapping with a non-existing previous windows

    for i in range(frames_in_window, len(skeleton_keypoints), frames_in_window): # this range gives us an index like this... (0, 30, 60, 90, 120...)
        if len(skeleton_keypoints[i-overlap_frames:i+frames_in_window-overlap_frames]) == frames_in_window: #if the window has the right number of frames...
            batched_skeletons.append(skeleton_keypoints[i-overlap_frames:i+frames_in_window-overlap_frames])

    return batched_skeletons


if __name__ == "__main__":
    skeletons_keypoints = load_data(path="Train-Set")
    skeletons_keypoints = remove_low_scoring_keypoints(skeletons_keypoints)
    skeletons_keypoints = remove_null_skeleton(skeletons_keypoints)
    # skeletons_keypoints = downsample_dataset(skeletons_keypoints)
    # the dataset returned by the transformation has not confidence scores for each keypoint
    # we don't need them anymore
    skeletons_keypoints = normalization(skeletons_keypoints)
    batched_scheletons = data_windowing(skeletons_keypoints, overlap=1.5)
    
    print(len(batched_scheletons), "total windows")
    print(np.array(batched_scheletons).shape)
