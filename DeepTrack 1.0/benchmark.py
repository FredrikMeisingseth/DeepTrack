def get_operating_characteristics(labels, predictions):
    """
    Method that returns the operating characteristics of a prediction.
    Input:
        labels: the batch_labels
        predictions: the batch_predictions. Sigmoid and cutoff should be applied BEFORE applying this method.
    Outputs:
        P:  condition positive - number of real positives in the label (pixels that are 1 in label)
        N:  condition negative - number of real negatives in the label (pixels that are 0 in label)
        TP: true positive - number of correct positive predictions (pixels that are 1 in both label and prediction)
        TN: true negative - number of correct negative predictions (pixels that are 0 in both label and prediction)
        FP: false positive - number of incorrect positive predictions (pixels that are 0 in label, but 1 in prediction)
        FN: false negative - number of incorrect negative predictions (pixels that are 1 in label, but 0 in prediction)
    """
    import numpy as np

    label_first_feature = np.ndarray.flatten(labels[:, :, :, 0])
    prediction_first_feature = np.ndarray.flatten(predictions[:, :, :, 0])

    P = sum(label_first_feature)
    N = sum(1 - label_first_feature)

    TP = sum(label_first_feature * prediction_first_feature)
    TN = sum((1 - label_first_feature) * (1 - prediction_first_feature))

    FP = sum((1 - label_first_feature) * prediction_first_feature)
    FN = sum(label_first_feature * (1 - prediction_first_feature))

    return P, N, TP, TN, FP, FN


def distance_from_upper_left_corner_ROC(operating_characteristics, FPR_weight = 1.0):
    """
    Method that calculates the distance from the upper left corner of the ROC space for a given TPR and FPR
    Inputs:
        TPR - True Positive Rate
        FPR - False Positive Rate
        FPR_weight - instead of just FPR, FPR*FPR_weight is used in the calculation. This is usually a value between 0
                     and 1. A small value means that changes in FPR affect the distance less.
    """
    P, N, TP, TN, FP, FN = operating_characteristics

    TPR = TP/P
    FPR = FP/N

    return ((1-TPR)**2 + (FPR*FPR_weight)**2) ** (1/2)