import sklearn.metrics


def tp(Y_pred, Y_test) -> int:
    """Use the confusion matrix to calculate values used in other accuracy metrics. (True Positive)"""
    conf_mat = sklearn.metrics.confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    return conf_mat[1, 1]  # Since this is binary classification, we can just use the confusion matrix directly


def fp(Y_pred, Y_test) -> int:
    """Use the confusion matrix to calculate values used in other accuracy metrics. (False Positive)"""
    conf_mat = sklearn.metrics.confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    return conf_mat[0, 1]


def fn(Y_pred, Y_test) -> int:
    """Use the confusion matrix to calculate values used in other accuracy metrics. (False Negative)"""
    conf_mat = sklearn.metrics.confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    return conf_mat[1, 0]


def tn(Y_pred, Y_test) -> int:
    """Use the confusion matrix to calculate values used in other accuracy metrics. (True Negative)"""
    conf_mat = sklearn.metrics.confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    return conf_mat[0, 0]


def tpr(Y_pred, Y_test) -> float:
    """Calculate the true positive rate (Recall)"""
    return tp(Y_pred, Y_test) / max(1, tp(Y_pred, Y_test) + fn(Y_pred, Y_test))


def fpr(Y_pred, Y_test) -> float:
    """Calculate the false positive rate"""
    return fp(Y_pred, Y_test) / max(1, fp(Y_pred, Y_test) + tn(Y_pred, Y_test))
