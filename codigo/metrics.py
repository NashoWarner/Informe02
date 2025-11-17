import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

class Metrics(object):
    def __init__(self):
        pass

    def accuracy(self, y, y_hat):
        '''
        Sin utilizar sklearn, implemente la métrica de precisión como la relación entre la 
        cantidad de puntos de datos predichos correctamente y la cantidad total de puntos de datos.
        
        Without using sklearn, implement the accuracy metric as the ratio between the number of
        correctly predicted datapoints against total number of datapoints
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
        
        Return:
            accuracy: scalar
        '''
        # Convertir y a numpy array si es necesario
        y = np.array(y)
        y_hat = np.array(y_hat)
        
        # Calcular la precisión como (TP + TN) / Total
        correct_predictions = np.sum(y == y_hat)
        total_predictions = len(y)
        
        accuracy = correct_predictions / total_predictions
        
        return accuracy

    def recall(self, y, y_hat, average='macro'):
        '''
        Use sklearn's recall_score function to calculate the recall score
            
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
            average: None or string 'macro'
        
        Return:
            recall: scalar or list of scalars with length of # of unique labels
        '''
        return recall_score(y, y_hat, average=average)

    def precision(self, y, y_hat, average='macro'):
        '''
        Use sklearn's precision_score function to calculate the precision score
            
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
            average: None or string 'macro'
        
        Return:
            precision: scalar or list of scalars with length of # of unique labels
        '''
        return precision_score(y, y_hat, average=average)

    def f1_score(self, y, y_hat, average='macro'):
        '''
        Use sklearn's f1_score function to calculate the f1 score
            
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
            average: None or string 'macro'
        
        Return:
            f1_score: scalar or list of scalars with length of # of unique labels
        '''
        return f1_score(y, y_hat, average=average)

    def roc_auc_score(self, y, y_hat, average='macro'):
        '''
        Use sklearn's roc_auc_score function to calculate the ROC_AUC score
            
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
            average: None or string 'macro'
        
        Return:
            roc_auc_score: scalar or list of scalars with length of # of unique labels
        '''
        return roc_auc_score(y, y_hat, average=average, multi_class='ovr')
    
    def confusion_matrix(self, y, y_hat):
        '''
        Use sklearn's confusion_matrix function to calculate the Confusion Matrix
            
        Args:
            y_hat: (N,) numpy vector of predicted labels
            y: list of N labels
        
        Return:
            confusion_matrix: numpy array of the predictions vs ground truth counts.
        '''
        return confusion_matrix(y, y_hat)