from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



def calc_confusion_matrix(y_test, y_pred):
    '''Calculate true positives, true negatives, false negatives, true negatives
    @param y_test: the label
    @param y_pred: the class predicted by the model
    '''
    return confusion_matrix(y_test, y_pred).ravel() 
                    
    
def evaluate_model(y_test, y_pred, pos_label_scores):
    '''Computes the confusion matrix, precision, recall F1-Score and AUC 
    
    @param y_test: the label
    @param y_pred: the class predicted by the model
    @param pos_label_scores_: prediction scores for the class where customers leave the bank 
    '''
    
    tn, fp, fn, tp=calc_confusion_matrix(y_test, y_pred)
    
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*precision*recall/(precision+recall)
    
    plot_roc_curve(y_test, pos_label_scores)
    
    auc=roc_auc_score(y_test, pos_label_scores)
    
    print('Precision: %.5f: '%precision)
    print('Recall: %.5f: '%recall)
    print('F1: %.5f: '%f1)
    print('AUC: %f'%auc)
    print(confusion_matrix(y_test, y_pred))  
    

def plot_roc_curve(y_test, pos_label_scores):
    '''Computes the AUC. 
    
    @param y_test: the label
    @param pos_label_scores_: prediction scores for the class where customers leave the bank
    '''
    
    fpr, tpr, thresholds = roc_curve(y_test, pos_label_scores)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Recall')
    plt.title('ROC curve')
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   