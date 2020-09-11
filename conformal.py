import numpy as np

def MICP(train,test,by,epsilon):
    ''' Mondrian inductive conformal predictor with a chosen threshold e'''
    dev_data = {"labels":[1 if item>=1 else 0 for sublist in train['ave_annot_score'] for item in sublist],
                'pred_probs':[item for sublist in train[by+'_score'] for item in sublist]}

    test_data = {"labels":[1 if item>=1 else 0  for sublist in test['ave_annot_score'] for item in sublist],
                'pred_probs':[item for sublist in test[by+'_score'] for item in sublist]}
              
    dev_labels = dev_data['labels']
    dev_labels = np.array(dev_labels)
    dev_pred_probs = dev_data['pred_probs']
    dev_pred_probs = np.array(dev_pred_probs,dtype='float32')

    test_labels = test_data['labels']
    test_labels = np.array(test_labels)
    test_pred_probs = test_data['pred_probs']
    test_pred_probs = np.array(test_pred_probs,dtype='float32')
    
    dev_postive_scores = [dev_pred_probs[index][1] for index,value in enumerate(list(dev_labels)) if value==1]
    dev_negative_scores = [dev_pred_probs[index][0] for index,value in enumerate(list(dev_labels)) if value==0]
    dev_ncm_positive_scores = np.array([-item for item in dev_postive_scores])
    dev_ncm_negative_scores = np.array([-item for item in dev_negative_scores])

    test_ncm_positive_scores = np.array([-item[1] for item in test_pred_probs])
    test_ncm_negative_scores = np.array([-item[0] for item in test_pred_probs])
    test_postive_quantiles = np.sum(test_ncm_positive_scores.reshape((len(test_ncm_positive_scores),1)) < 
       dev_ncm_positive_scores.reshape((1,len(dev_ncm_positive_scores))),axis=1) / float(len(dev_ncm_positive_scores)+1)

    test_negative_quantiles = np.sum(test_ncm_negative_scores.reshape((len(test_ncm_negative_scores),1)) < 
       dev_ncm_negative_scores.reshape((1,len(dev_ncm_negative_scores))),axis=1) / float(len(dev_ncm_negative_scores)+1)
    test_quantiles = np.array(list(zip(test_negative_quantiles,test_postive_quantiles)))
    
    test_preds = np.array(test_quantiles > epsilon, dtype=int)
    test_preds = test_preds.argmax(axis=1)
    
    largest_p = [max(item) for item in test_quantiles]
    test_preds = np.multiply(test_preds,largest_p)

    return test_preds
