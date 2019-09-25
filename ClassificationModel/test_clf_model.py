# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:38:04 2019

@author: PC
"""

import os
import time
import numpy as np
from net_classify_test import *
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate
import glob
import pandas as pd
import SimpleITK as sitk
import scipy
import scipy.ndimage
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.metrics import accuracy_score,roc_curve,recall_score,roc_auc_score,auc,confusion_matrix,cohen_kappa_score, f1_score, precision_score,matthews_corrcoef 
from tqdm import tqdm
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFromModel, SelectKBest
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

if __name__ == "__main__":
    Pretrained_path = r'.\TaiZhouHospital\model\clf_IA_VS_nonIA'
    model = ClassifyNet()#.cuda()
    classify_path = os.path.join(Pretrained_path, '020.ckpt')
    modelCheckpoint = torch.load(classify_path)
    pretrained_dict = modelCheckpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#filter out unnecessary keys
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    img_path = r'.\GGO_DataSet\test_data\test_Img'
    list_path = r'.\GGO_DataSet\test_data\test.csv'
    
    f = open(list_path)
    GGO_list = pd.read_csv(f)
    List_Num = np.array(GGO_list['Num'].tolist())
    Type = GGO_list['Type'].tolist()
    Class = np.array(GGO_list['Class'].tolist()) 
    List_Num = List_Num[[i for i,x in enumerate(Type) if x!='Solid']]
    Class= Class[[i for i,x in enumerate(Type) if x!='Solid']]
    List_Num = List_Num[[i for i,x in enumerate(Class) if x!=0]]
    Histopathology = np.array(GGO_list['Histopathology'].tolist())
    Histopathology = Histopathology[[i for i,x in enumerate(Type) if x!='Solid']]
    Histopathology = Histopathology[[i for i,x in enumerate(Class) if x!=0]]

    prob = []
    prob_label = []
    real_class = []
    test_result=[]
    for i in tqdm(range(len(List_Num))): 
        roi_path = os.path.join(img_path, List_Num[i]+'_roi.npy')
        label_path = os.path.join(img_path, List_Num[i]+'_label.npy')
        data = np.load(roi_path)
        data = data[np.newaxis,...]
        data = torch.from_numpy(data.astype(np.float32))
        GGO_Class = np.load(label_path)
        with torch.no_grad():
            input_data = Variable(data)#.cuda()
            predict = model(input_data)            
            result = predict.data.cpu().numpy()
            prob.append(result[0][1])
            real_class.append(GGO_Class)
            prob_label.append(np.argmax(result[0]))
            test = {}
            test['Num'] = List_Num[i]
            test['Class'] = GGO_Class
            test['Prob'] = result[0][1]
            test['Histopathology'] = Histopathology[i]
            test_result.append(test)
    df = pd.DataFrame(test_result).fillna('null')
    df.to_csv('./test_result.csv',index=False,sep=',')
    print('Our Model ACC:',accuracy_score(real_class,prob_label)*100)      
    fpr,tpr,threshold = roc_curve(np.array(real_class),prob)
    auc = auc(fpr,tpr)
    auc_fl_cnn, auc_fh_cnn, auc_fstd_cnn = confindence_interval_compute(np.array(prob), np.array(real_class))
    print('AUC:%.2f'%auc,'+/-%.2f'%auc_fstd_cnn,'  95% CI:[','%.2f,'%auc_fl_cnn,'%.2f'%auc_fh_cnn,']')
    F1 = f1_score(np.array(real_class),prob_label)
    print('F1:',F1)
    F1_w = f1_score(np.array(real_class),prob_label,average='weighted')
    print('F1_weight:',F1_w)
    MCC = matthews_corrcoef(np.array(real_class),prob_label)
    print('MCC:',MCC)


    training_csv = r'.\Radiomics_Feature.csv'
    testing_csv = r'.\testing_Radiomics_Feature.csv'  
    f_training = open(training_csv)
    train_list = pd.read_csv(f_training)
    train_x = np.array(train_list.values[:,3:])
    train_y = np.array(train_list['Class'].tolist())
    
    f_testing = open(testing_csv)
    test_list = pd.read_csv(f_testing)
   
    test_x = np.array(test_list.values[:,3:])
    test_y = np.array(test_list['Class'].tolist())
    

    # Feature normalization
    min_max_scaler = MinMaxScaler()
    train_x = min_max_scaler.fit_transform(np.array(train_x,dtype=np.float64))
    test_x = min_max_scaler.transform(test_x)

    selector = SelectKBest(f_classif, 20)
    train_x = selector.fit_transform(train_x, train_y)
    test_x = selector.transform(test_x)
    
    clf = SVC(kernel='rbf',  probability=True, random_state=0, gamma='scale')
    clf.fit(train_x, train_y)
    test_prob = clf.predict_proba(test_x)[:,1]
    test_label = clf.predict(test_x)
    print('Radiomics:',accuracy_score(test_y,test_label))
    fpr3,tpr3,threshold3 = roc_curve(np.array(test_y),test_prob)
    auc3 = roc_auc_score(np.array(test_y),test_prob)
    auc_fl_ra, auc_fh_ra, auc_fstd_ra = confindence_interval_compute(np.array(test_prob), np.array(test_y))
    print('AUC:%.2f'%auc3,'+/-%.2f'%auc_fstd_ra,'  95% CI:[','%.2f,'%auc_fl_ra,'%.2f'%auc_fh_ra,']')
    F1_3 = f1_score(np.array(test_y),test_label)
    print('F1:', F1_3)
    F1_w3 = f1_score(np.array(test_y),test_label,average='weighted')
    print('F1_weight:',F1_w3)
    MCC3 = matthews_corrcoef(np.array(test_y),test_label)
    print('MCC:',MCC3)
    
    radiology1_path = r'.\GGO_DataSet\test_data\radiology_HW.csv'
    radiology1 = open(radiology1_path)
    radiology1_List = pd.read_csv(radiology1)
    radiology1_result = radiology1_List['Diagnosis'].tolist()
    print('HW:',accuracy_score(real_class,radiology1_result))
    fpr1,tpr1,threshold1 = roc_curve(np.array(real_class),radiology1_result)
    F1_1 = f1_score(np.array(real_class),radiology1_result)
    print('F1:',F1_1)
    F1_w1 = f1_score(np.array(real_class),radiology1_result,average='weighted')
    print('F1_weight:',F1_w1)
    MCC1 = matthews_corrcoef(np.array(real_class),radiology1_result)
    print('MCC:',MCC1)
#    auc1 = auc(fpr1,tpr1)
#    print(auc1)
    TN1, FP1, FN1, TP1 = confusion_matrix(real_class,radiology1_result).ravel()

    ACC1 = (TP1+TN1)/(TP1+FP1+FN1+TN1)
    print('ACC:%0.4f'%ACC1)
    
    radiology2_path = r'.\GGO_DataSet\test_data\radiology_WSP.csv'
    radiology2 = open(radiology2_path)
    radiology2_List = pd.read_csv(radiology2)
    radiology2_result = radiology2_List['Diagnosis'].tolist()
    print('WSP:',accuracy_score(real_class,radiology2_result))
    fpr2,tpr2,threshold2 = roc_curve(np.array(real_class),radiology2_result)
    F1_2 = f1_score(np.array(real_class),radiology2_result)
    print('F1:',F1_2)
    F1_w2 = f1_score(np.array(real_class),radiology2_result,average='weighted')
    print('F1_weight:',F1_w2)
    MCC2 = matthews_corrcoef(np.array(real_class),radiology2_result)
    print('MCC:',MCC2)
    TN2, FP2, FN2, TP2 = confusion_matrix(real_class,radiology2_result).ravel()

    ACC2 = (TP2+TN2)/(TP2+FP2+FN2+TN2)
    print('ACC:%0.4f'%ACC2)
    
    kappa = cohen_kappa_score(radiology1_result,radiology2_result)
    print('kappa:%0.4f'%kappa)
        
    kappa1 = cohen_kappa_score(radiology1_result,prob_label)
    print('kappa1:%0.4f'%kappa1)
    
    kappa2 = cohen_kappa_score(radiology2_result,prob_label)
    print('kappa1:%0.4f'%kappa2)
    
    ## Fusion
    scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for scale in scales:
        prob_fusion = scale*np.array(prob)+(1-scale)*np.array(test_prob)
#        fpr,tpr,threshold = roc_curve(np.array(real_class),prob_fusion)
#        auc_value = auc(fpr,tpr)
        auc_value = roc_auc_score(np.array(real_class),prob_fusion)
        auc_fl, auc_fh, auc_fstd = confindence_interval_compute(np.array(prob_fusion), np.array(real_class))
        print('Fusion Scale',scale,'AUC:%.2f'%auc_value,'+/-%.2f'%auc_fstd,
              '  95% CI:[','%.2f,'%auc_fl,'%.2f'%auc_fh,']')      
    Fusion = np.zeros([len(prob),2])
    Fusion[:,0] = np.array(prob)
    Fusion[:,1] = np.array(test_prob)
    Fusion_min = Fusion.min(1)
    Fusion_max = Fusion.max(1)

    auc_min = roc_auc_score(np.array(real_class),Fusion_min)
    auc_fl_min, auc_fh_min, auc_fstd_min = confindence_interval_compute(np.array(Fusion_min), np.array(real_class))
    print('Min Fusion AUC:%.2f'%auc_min,'+/-%.2f'%auc_fstd_min,'  95% CI:[','%.2f,'%auc_fl_min,'%.2f'%auc_fh_min,']')
#    fpr,tpr,threshold = roc_curve(np.array(y),Fusion_max)

    auc_max = roc_auc_score(np.array(real_class),Fusion_max)
    auc_fl_max, auc_fh_max,auc_fstd_max = confindence_interval_compute(np.array(Fusion_max), np.array(real_class))
    print('Max Fusion AUC:%.2f'%auc_max,'+/-%.2f'%auc_fstd_max,'  95% CI:[','%.2f,'%auc_fl_max,'%.2f'%auc_fh_max,']')
    

    Fusion_new = Fusion_max
    fpr4,tpr4,threshold4 = roc_curve(np.array(real_class),Fusion_new)
    F1_0 = f1_score(np.array(real_class),Fusion_new>=0.6)
    print('ACC:',accuracy_score(real_class,Fusion_new>=0.6))
    print('F1:', F1_0)
    F1_w0 = f1_score(np.array(real_class),Fusion_new>=0.6,average='weighted')
    print('F1_weight:',F1_w0)
    MCC0 = matthews_corrcoef(np.array(real_class),Fusion_new>=0.6)
    print('MCC:',MCC0)
    
    stat_val3, p_val3 = stats.ttest_ind(prob, Fusion_new, equal_var=False)
    print('Seg_transfer and Fusion p:%.5f'%p_val3)
    stat_val4, p_val4 = stats.ttest_ind(test_prob, Fusion_new, equal_var=False)
    print('Radiomics and Fusion p:%.5f'%p_val4)
    stat_val5, p_val5 = stats.ttest_ind(test_prob, prob, equal_var=False)
    print('Seg_transfer and Radiomics p:%.5f'%p_val5)
    
    font = {'family' : 'Times New Roman',
			'weight' : 'normal',
			'size'   : 10,}
    plt.rc('font', **font)

    lw = 1
    plt.figure(figsize=(5,5))
    plt.plot(fpr4, tpr4, color='red',
             lw=lw, label='Fusion model AUC:%.2f'%roc_auc_score(np.array(real_class),Fusion_new))
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='DL model AUC:%.2f'%roc_auc_score(np.array(real_class),prob))

    plt.plot(fpr3, tpr3, color='g',
             lw=lw, label='Radiomics model AUC:%.2f'%roc_auc_score(np.array(test_y),test_prob))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.plot(fpr2[1], tpr2[1], color='orange',marker = '^',
             label='Senior Radiologist') #

    plt.plot(fpr1[1], tpr1[1], color='c',marker = 'o',
             label='Junior Radiologist') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
        
        
        