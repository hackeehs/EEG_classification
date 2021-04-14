import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
'''
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    

    tp = np.sum((prediction[i] == True and ground_truth[i] == True) for i in range(prediction.shape[0]))
    tn = np.sum((prediction[i] == False and ground_truth[i] == False) for i in range(prediction.shape[0]))
    fp = np.sum((prediction[i] == True and ground_truth[i] == False) for i in range(prediction.shape[0]))
    fn = np.sum((prediction[i] == False and ground_truth[i] == True) for i in range(prediction.shape[0]))
            
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn )
    
    return precision, recall, f1, accuracy
