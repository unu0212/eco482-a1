#!/usr/bin/env python
# coding: utf-8

# # Algorithmic implementation

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# ## 

# In[2]:


df = pd.read_csv("mobile_money.csv")


df = df.apply(lambda x: x.replace({'yes': True, 'no': False}) if x.dtype == 'object' else x)

param = ['mpesa_user','cellphone', 'totexppc', 'wkexppc', 'wealth', 'size', 'education_years', 
                'pos', 'user_neg', 'ag', 'user_sick', 'sendd', 'recdd', 'bank_acct', 
                'mattress', 'sacco', 'merry', 'occ_farmer', 'occ_public', 'occ_prof', 
                 'occ_help', 'occ_bus', 'occ_sales', 'occ_ind', 'occ_other', 'occ_ue']

features_name = ['cellphone', 'totexppc', 'wkexppc', 'wealth', 'size', 'education_years', 
                'pos', 'user_neg', 'ag', 'user_sick', 'sendd', 'recdd', 'bank_acct', 
                'mattress', 'sacco', 'merry', 'occ_farmer', 'occ_public', 'occ_prof', 
                 'occ_help', 'occ_bus', 'occ_sales', 'occ_ind', 'occ_other', 'occ_ue']
df = df[param].dropna()
outcome = df['mpesa_user']
df_feat = df[features_name]

df


# ## 

# In[3]:


outcome.describe()


# ##

# In[20]:


grouped_stats = df.groupby('mpesa_user')[features_name].describe()
grouped_stats.T.head(50)
for var in features_name: print(df.groupby('mpesa_user')[var].describe())


# ## 

# In[5]:


#Split the data set into training and test 80-20

x_train, x_test, y_train, y_test = train_test_split(df_feat, outcome, 
                                                    test_size = 0.2, random_state = 21)

#Standardizing the data set
#stage 1
scaler = preprocessing.StandardScaler()
#stage2
scaler.fit(x_train)
#stage 3
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#Logistic regression
log_reg = LogisticRegression()
log_reg.fit(x_train_scaled, y_train)

#Random Forest  Classifier
rforest = RandomForestClassifier()
rforest.fit(x_train_scaled, y_train)

#LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train_scaled, y_train)



# ## 
# The best classifier is the Logistic Regression, as can be seen in the following.

# In[6]:


#Getting accuracies for the classifiers
accuracy = dict()

log_score = log_reg.score(x_test_scaled, y_test)

rforest_score = rforest.score(x_test_scaled, y_test)

lda_score = lda.score(x_test_scaled, y_test)

accuracy['Logistic regression accuracy'] = log_score
accuracy['Random forest accuracy'] = rforest_score
accuracy['LDA accuracy'] = lda_score

accuracy_df = pd.DataFrame(accuracy, index=[0])
print(accuracy_df)


#Getting the AUC scores for the classifiers

log_auc = roc_auc_score(y_test, log_reg.predict_proba(x_test_scaled)[:, 1])
fpr_log, tpr_log, _ = roc_curve(y_test, log_reg.predict_proba(x_test_scaled)[:, 1])

rf_auc = roc_auc_score(y_test, rforest.predict_proba(x_test_scaled)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rforest.predict_proba(x_test_scaled)[:, 1])

lda_auc = roc_auc_score(y_test, lda.predict_proba(x_test_scaled)[:, 1])
fpr_lda, tpr_lda, _ = roc_curve(y_test, lda.predict_proba(x_test_scaled)[:, 1])

#Plotting the curves
plt.figure()
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {log_auc:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {lda_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


# ## 

# In[7]:


snames, simportances = zip(*sorted(zip(features_name, rforest.feature_importances_), 
                                  key=lambda pair: pair[1]))
plt.barh(snames, simportances)


# ## 

# In[12]:


k_value = range(1, 11)
accuracies = []

for k in k_value:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, x_train_scaled, y_train, cv=5)
    accuracies.append(cv_scores.mean())
    
for k, acc in zip(k_value, accuracies):
    print(f"K={k}: Mean Accuracy={acc:.4f}")


# ## 

# In[21]:


log_regl = LogisticRegression(solver='liblinear')
param_grid = {'penalty':['l1', 'l2'], 
             'C':[0.1, 1, 10, 100]}

grid_search = GridSearchCV(estimator = log_regl, param_grid = param_grid,
                          cv = 5)

grid_search.fit(x_train_scaled, y_train)

for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
    print(f"C={params['C']}, Penalty={params['penalty']}: Mean CV Score={mean_score:.4f}")


# ## 
# In our result, it is observed that when C increases from 0.1 to 100 (meaning regularization becomes weaker), the Mean CV Score stabilizes at approximately 0.8766. This suggests that when C reaches a specific point, additional decrease in regularization does not enhance the model's effectiveness. The optimal result is achieved when C=0.1 with l1 regularization, indicating that a moderate amount of regularization assists the model in managing bias and variance, resulting in improved generalization. Nevertheless, as the C rises, there is a possibility of slight overfitting in the model, leading to a plateau in the score without notable improvements. Regularization helps make sure the model is less complex and reduces the chances of overfitting to noise in the training data, resulting in improved cross-validation performance with optimal regularization (moderate C).

# ##
# The Logistic Regression is the best classifier for our data. It has the highest cross-validation accuracy of 0.8794 for C=0.1, L1 penalty and the highest AUC of 0.93 compared to the other classifiers. While KNN, Random Forest and LDA has good accuracy, their accuracy and AUC values are lower compared to the Logistic Regression's.

# ##
# The main findings from the research on M-Pesa adoption include: 
# 1.Cell phone possession was identified as the most important factor, with having a bank account, overall wealth, sending money transfers, and receiving money transfers following closely behind.
# 2.Logistic Regression stood out among the classifiers, delivering top performance with a 0.879 cross-validation accuracy and a 0.93 AUC, establishing it as the most dependable model for forecasting M-Pesa adoption. Additional classifiers such as Random Forest, K-Nearest Neighbors (KNN), and Linear Discriminant Analysis (LDA) had strong performances, but Logistic Regression consistently outshined them in terms of accuracy and AUC. 
# 3.These results indicate that having access to communication technology and being financially included are essential for the adoption of M-Pesa in Kenya.
