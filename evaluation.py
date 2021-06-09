import numpy as np
from numpy import linalg as LA
from model_and_training import load_model, get_preprocessed_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import seaborn as sns
import pandas as pd
import datetime
from draw_skeletons import draw_skeleton, generate_video

_NUM_WINDOWS = 234 #39


# Windows from class 0 which manifest strange behaviours with keypoints
#problematic_windows del kid set: np.array([4, 22, 24])
#problematic_windows = []
problematic_windows = np.array([23, 39, 40, 77, 99, 114, 157]) # del test set


# Class 1 = anomaly
#fall_windows =  np.array([0, 1, 2, 6, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 26, 29, 33, 34, 35, 36, 37]) #KID_TESTSET
fall_windows = np.array([1, 2, 10, 15, 19, 20, 25, 31, 36, 37, 41, 42, 46, 47, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63,
64, 65, 66, 67, 74, 78, 79, 86, 87, 91, 95, 96, 111, 116, 122, 128, 129, 133, 134, 135, 136, 137, 138, 139, 140, 141, 
142, 143, 147, 150, 151, 155, 156, 161, 162, 166, 167, 170, 174, 177, 178, 182, 183, 188, 191, 192, 195, 196, 202, 203, 
207, 211, 215, 218, 219, 222, 223, 224, 229, 231, 232, 5, 6, 11, 14, 21, 24, 26, 35, 51, 68, 71, 76, 82, 97, 106, 110, 
112, 115, 127, 144, 152, 154, 163, 190, 197, 198, 201, 208, 214, 233])


# Class 0 = normality   
walk_windows = np.array(range(0,_NUM_WINDOWS))                                                                        
walk_windows = np.delete(walk_windows, np.flip(np.sort(fall_windows)))


def __f_score(precision, recall, beta=2):
    # The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
    # where an F-beta score reaches its best value at 1 and worst score at 0.
    return (((beta**2) + 1)*(recall*precision)) / (((beta**2) * precision) + recall)


def __precision(confusion_matrix):
    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    # So it's all the falls recognized as falls, divided by all the examples recognized by the model as falls
    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])


def __recall(confusion_matrix):
    # The recall is intuitively the ability of the classifier to find all the positive samples.
    # So it's all the falls recognized as falls, divided by all the falls examples
    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])


def __generate_confusion_matrix(true_labels, predicted_labels, model_type, prefix, path="images/"): #Kids/images
    # Generate confusion matrix
    plt.figure()
    cm = confusion_matrix(true_labels, predicted_labels)
    # This division is needed to obatin percentage values
    cm_perc = np.empty([2,2])
    cm_perc[0,:] = cm[0,:]/np.sum(cm[0,:])
    cm_perc[1,:] = cm[1,:]/np.sum(cm[1,:])
    #cm_perc = cm/cm.sum(axis=0) # this doesnt work... we don't know why lol
    labels = np.array([f'{c}\n{cp*100:.2f}%' for c, cp in zip(cm.flatten(), cm_perc.flatten())])
    labels = labels.reshape(2,2)
    cm_df = pd.DataFrame(cm_perc, index=["Fall", "Walk"], columns=["Fall", "Walk"])

    sns.heatmap(cm_df, annot=labels, fmt='')
    plt.title('Confusion Matrix of model ' + model_type)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig(path + model_type + "/" + prefix + 'confusion_matrix_'+'.jpg')

    return cm


def __generate_PR_curve(recalls, precisions, AUC, model_type, path='images/'): 
    plt.figure()
    plt.plot(recalls, precisions, marker='.', label="PR Curve")
    plt.title("PR curve of model " + model_type + " with " + "AUC: " + str("{:.4f}".format(AUC)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(path + model_type + "/" + 'PR_curve_'+'.jpg')


def predict_on_test_set(data, model, verbose=True):

    mean_norm_fall = 0
    mean_norm_walk = 0

    norms = np.zeros(_NUM_WINDOWS)
    predicted_data = np.empty([_NUM_WINDOWS,75,38])

    # Initialize model internal variables by calling it on dummy data
    _ = model.predict(np.reshape(data[0], [1,75,38]))
    avg_time = 0

    # Calculate difference in norm for falling windows
    if verbose:
        print("\nNorms of falling windows:")

    for i in fall_windows:
        start_time = datetime.datetime.utcnow()
        x = data[i]
        x = np.reshape(x, [1,75,38])
        y = model.predict(x)
        predicted_data[i] = y
        norms[i] = LA.norm(x-y)
        mean_norm_fall += norms[i]
        time_delta = datetime.datetime.utcnow() - start_time
        avg_time += time_delta.total_seconds()
        if verbose:
            print("Window "+str(i)+": "+str(LA.norm(x-y)))
        

    # Calculate difference in norm for walking windows
    if verbose:
        print("\nNorms of walk windows:")
    for i in walk_windows: # 112 elements
        start_time = datetime.datetime.utcnow()
        x = data[i]
        x = np.reshape(x,[1,75,38])
        y = model.predict(x)
        predicted_data[i] = y
        if i not in problematic_windows:
            norms[i] = LA.norm(x-y)
            mean_norm_walk += norms[i]
        time_delta = datetime.datetime.utcnow() - start_time
        avg_time += time_delta.total_seconds()
        if verbose:
            print("Window "+str(i)+": "+ str(norms[i]))
    
    print("Average execution time for inference:", avg_time/_NUM_WINDOWS)

    mean_norm_fall /= len(fall_windows)
    mean_norm_walk /= (len(walk_windows)-len(problematic_windows))
    if verbose:
        print("|-----------------------|\nMean norm on fall windows:",mean_norm_fall)
        print("Mean norm on walk windows:",mean_norm_walk,"\n|-----------------------|")

    return predicted_data, norms, mean_norm_fall, mean_norm_walk


def __generate_labels(norms, threshold):

    # Assign labels for input data
    true_labels = ["Walk"]*_NUM_WINDOWS
    for i in range(0,_NUM_WINDOWS): 
        if i in fall_windows:
            true_labels[i] = "Fall"

    # Assign labels for predicitons chcking if the norms are above the threshold
    predicted_labels = ["Walk"]*_NUM_WINDOWS
    for i in range(0, _NUM_WINDOWS):
        if norms[i] >= threshold:
            predicted_labels[i] = "Fall"

    # Delete problematic windows labels
    for i in np.flip(problematic_windows):
        del true_labels[i]
        del predicted_labels[i]
        
    return true_labels, predicted_labels


def __calculate_best_threshold(recalls, precisions, thresholds, norms):

    max_f_score = 0
    max_f_score_index = 0
    for i, t in enumerate(thresholds):
        actual_f_score = __f_score(precision=precisions[i], recall=recalls[i])
        if actual_f_score > max_f_score:
            max_f_score = actual_f_score
            max_f_score_index = i

    best_threshold = (thresholds[max_f_score_index] * (np.max(norms) - np.min(norms))) + np.min(norms)
    print("Best threshold:",best_threshold," with F-score:", max_f_score)

    return best_threshold


def draw_reconstruction(predicted_data):
    save_path = "Batches-Test-Set-MAE-Reconstructed"
    # Write each batch of frames in a separate folder
    for i in range(len(predicted_data)):            #index on batches
	    print("Writing batch ", i)
	    for j in range(len(predicted_data[i])):     #index on frames
		    draw_skeleton(predicted_data[i][j], i, j, save_path)

    generate_video(save_path)



# 1 is FALL
# 0 is WALK
if __name__ == "__main__":
    
    model_type="64-32"
    # Load data manually
    data = get_preprocessed_data(path="Test-Set")
    # np.save(numpy_folder_path+"test-data", data)

    # Load data from numpy savefile
    #numpy_folder_path = "processed-data/"
    #data = np.load(numpy_folder_path+"test-data.npy")

    print(data.shape)# the right shape is (Windows, Frames, Keypoints) = (Batch, Timesteps, N_features)
    data_size = data.shape[0]
    timesteps = data.shape[1]
    n_features = data.shape[2]

    # Predict
    model = load_model(timesteps=timesteps, n_features=n_features, architecture='shallow')
    model.load_weights("models/best.h5")
    predicted_data, norms, mean_norm_fall, mean_norm_walk = predict_on_test_set(data, model, verbose=True)

    # draw_reconstruction(predicted_data)

    # Generate labels using as threshold the mean between the average of walk norms and the average of fall norms
    threshold = (mean_norm_fall+mean_norm_walk)/2
    print("|---------> Threshold:", threshold,"\n")
    true_labels, predicted_labels = __generate_labels(norms, threshold)

    # Generating confusion matrix
    cm = __generate_confusion_matrix(true_labels, predicted_labels, prefix="NORMAL_", model_type=model_type)

    # Getting precision, recall and F-score for our threshold
    our_recall = __recall(cm)
    our_precision = __precision(cm)                                    
    our_f_score = __f_score(our_precision, our_recall)
    print("Precision:",our_precision,"\nRecall:",our_recall,"\nF-score:",our_f_score)

    # We create a new array of norms without problematic windows (norms = 0)
    filtered_norms = norms[norms != 0]
    # A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds
    normalized_prediction_norms = (filtered_norms - np.min(filtered_norms)) / (np.max(filtered_norms)-np.min(filtered_norms))

    precisions, recalls, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=normalized_prediction_norms, pos_label=["Fall"])
    
    # Calculate the AUC (Area under the curve), this value will be used for evaluate performance of the model
    AUC = auc(recalls, precisions)
    print("AUC:", AUC)

    # Plot Precision-Recall curve
    __generate_PR_curve(recalls=recalls, precisions=precisions, AUC=AUC, model_type=model_type)
    
    best_threshold = __calculate_best_threshold(recalls=recalls, precisions=precisions, thresholds=thresholds, norms=filtered_norms)

    # Generate confusion matrix for best threshold
    true_labels, predicted_labels = __generate_labels(norms, best_threshold)
    cm = __generate_confusion_matrix(true_labels, predicted_labels, prefix="BEST_", model_type=model_type)