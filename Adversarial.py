import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference, average_odds_difference, \
    disparate_impact_ratio, theil_index
from aif360.sklearn.datasets import standardize_dataset
from aif360.sklearn.inprocessing import AdversarialDebiasing
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from tabulate import tabulate

# Loading the dataset
Dataset_name = "Heart"  # "Heart", "Diabetes", "Credit", "CAMH"

if Dataset_name == "Heart":
    df = pd.read_csv("heart_failure_pre.csv")
    numerical_features = ['age', 'time', 'ejection_fraction', 'platelets',
                          'serum_creatinine', 'serum_sodium']
    # fairness mapping
    prot_attr = 'sex'
    label = 'DEATH_EVENT'
    priv_group = 1  # man
    pos_label = 1  # dead

if Dataset_name == "Diabetes":
    df = pd.read_csv("diabetes_resample_SMOTE.csv")
    numerical_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                          'number_outpatient',
                          'number_emergency', 'number_inpatient', 'number_diagnoses', 'age', 'diag_1', 'diag_2',
                          'diag_3', 'max_glu_serum',
                          'A1Cresult', 'metformin',
                          'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
                          'glipizide',
                          'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
                          'troglitazone',
                          'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                          'glimepiride-pioglitazone', 'metformin-pioglitazone', 'change',
                          'diabetesMed'
                          ]
    prot_attr = 'race'
    label = 'readmitted'
    priv_group = 1  # white
    pos_label = 1  # readmitted

if Dataset_name == "Credit":
    df = pd.read_csv("credit_pre.csv")
    numerical_features = ['month', 'credit_amount', 'investment_as_income_percentage', 'residence_since',
                          'number_of_credits', 'people_liable_for']
    # fairness mapping
    prot_attr = 'age'
    label = 'credit'
    priv_group = 1  # old (aged)
    pos_label = 1  # good credit

if Dataset_name == "CAMH":
    df = pd.read_csv("CAMH_pre.csv")

    numerical_features = ['Wave', 'Region', 'Age', 'Gender', 'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4',
                          'Q4_5', 'Q4_6', 'Q4_99', 'Q5', 'Q6 ', 'Q7', 'Q15 ', 'Q18', 'Q20x1 ',
                          'Q20x2', 'Q20x3', 'HouseHold', 'Children', 'Q25', 'Q26', 'Income', 'Q29']
    # fairness mapping
    prot_attr = 'race'
    label = 'anxiety'
    priv_group = 1  # white
    pos_label = 1  # severe anxiety

for feature in numerical_features:
    val = df[feature].values[:, np.newaxis]
    scaler = MaxAbsScaler().fit(val)
    df[feature] = scaler.transform(val)


dataset = standardize_dataset(df, prot_attr=prot_attr, target=label, numeric_only=False, dropna=True)
X, Y = dataset

clf = AdversarialDebiasing(prot_attr=prot_attr, debias=True, classifier_num_hidden_units=200, batch_size=27,
                           num_epochs=100, random_state=8, verbose=True)

cv = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
acc_list = []
f1_list = []
prec_list = []
recal_list = []
AUC_list = []
conf_matrix_list = []
sta_parity = []
equ_opp = []
avg_odd = []
dis_imp = []
theil_inx = []
for train_index, test_index in cv.split(X, Y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    clf.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = clf.predict(X_test)
    b = 1 + y_pred - y_test

    conf_matrix_list.append(confusion_matrix(y_test, y_pred))
    acc_list.append(accuracy_score(y_test, y_pred))
    prec_list.append(precision_score(y_test, y_pred))
    recal_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))
    AUC_list.append(balanced_accuracy_score(y_test, y_pred))
    # fairness metrics
    sta_parity.append(
        statistical_parity_difference(y_test, y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label))
    equ_opp.append(
        equal_opportunity_difference(y_test, y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label))
    avg_odd.append(
        average_odds_difference(y_test, y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label))
    dis_imp.append(
        disparate_impact_ratio(y_test, y_pred, prot_attr=prot_attr, priv_group=priv_group, pos_label=pos_label))
    theil_inx.append(theil_index(b))


mean_conf_matrix = np.sum(conf_matrix_list, axis=0)
print("Mean Confusion matrix:\n%s" % mean_conf_matrix)
mean_acc = np.mean(acc_list)
mean_prec = np.mean(prec_list)
mean_recal = np.mean(recal_list)
mean_f1 = np.mean(f1_list)
mean_AUC = np.mean(AUC_list)
mean_sta_parity = np.mean(sta_parity, axis=0)
mean_equ_opp = np.mean(equ_opp, axis=0)
mean_avg_odd = np.mean(avg_odd, axis=0)
mean_dis_imp = np.mean(dis_imp, axis=0)
mean_theil_inx = np.mean(theil_inx, axis=0)

result = [[mean_acc, mean_prec, mean_recal, mean_f1, mean_AUC, mean_sta_parity, mean_equ_opp, mean_avg_odd,
           mean_dis_imp, mean_theil_inx]]
headers = ["Accuracy", "Precision", "Recall", "F1", "AUC", "statistical parity", "Equality of Opportunity",
           "Average odds", "disparate impact", "Theil index"]

print(tabulate(result, headers=headers))
