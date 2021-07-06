# Load all necessary packages
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    balanced_accuracy_score
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from aif360.datasets import StandardDataset
from tabulate import tabulate
np.random.seed(0)
# Loading the dataset
Dataset_name = "Heart"  # "Heart", "Diabetes", "Credit", "CAMH"

if Dataset_name == "Heart":
    df = pd.read_csv("Datasets/Heart_failure_pre.csv")
    numerical_features = ['age', 'time', 'ejection_fraction', 'platelets',
                          'serum_creatinine', 'serum_sodium']
    # fairness mapping
    prot_attr = ['sex']
    label = 'DEATH_EVENT'
    priv_group = [[1]]  # man
    pos_label = [1]  # dead
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

if Dataset_name == "Diabetes":
    df = pd.read_csv("Datasets/Diabetes_pre.csv")
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
    prot_attr = ['race']
    label = 'readmitted'
    priv_group = [[1]]  # white
    pos_label = [1]  # dead
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]

if Dataset_name == "Credit":
    df = pd.read_csv("Datasets/Credit_pre.csv")
    numerical_features = ['month', 'credit_amount', 'investment_as_income_percentage', 'residence_since',
                          'number_of_credits', 'people_liable_for']
    # fairness mapping
    prot_attr = ['age']
    label = 'credit'
    priv_group = [[1]]  # old (aged)
    pos_label = [1]  # good credit
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

if Dataset_name == "CAMH":
    df = pd.read_csv("Datasets/CAMH_pre.cs")

    numerical_features = ['Wave', 'Region', 'Age', 'Gender', 'Q4_1', 'Q4_2', 'Q4_3', 'Q4_4',
                          'Q4_5', 'Q4_6', 'Q4_99', 'Q5', 'Q6 ', 'Q7', 'Q15 ', 'Q18', 'Q20x1 ',
                          'Q20x2', 'Q20x3', 'HouseHold', 'Children', 'Q25', 'Q26', 'Income', 'Q29']
    # fairness mapping
    prot_attr = ['race']
    label = 'anxiety'
    priv_group = [[1]]  # man
    pos_label = [1]  # dead
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]


dataset_orig = StandardDataset(df, label_name=label, favorable_classes=pos_label,
                               protected_attribute_names=prot_attr, privileged_classes=priv_group, )
scale = StandardScaler()

clf = [LogisticRegression(C=0.009, solver='liblinear', max_iter=500, random_state=0),
       DecisionTreeClassifier(max_depth=2, random_state=0),
       GaussianNB(),
       SVC(C=100, gamma=0.001, probability=True)
       ]

for clf in clf:
    print("Classifier is:\n", clf)
    fold = dataset_orig.split(10, shuffle=True)
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

    for idx, val in enumerate(fold):
        test_features = val.features
        test_labels = val.labels
        temp = fold.copy()
        temp.remove(val)
        train_features = []
        train_labels = []
        for obj in temp:
            train_features.append(obj.features)
            train_labels.append(obj.labels)

        train_features = np.concatenate(tuple(train_features), axis=0)
        train_labels = np.concatenate(tuple(train_labels), axis=0)
        train = np.concatenate((train_features, train_labels), axis=1)
        df_train = pd.DataFrame(train, columns=dataset_orig.feature_names + dataset_orig.label_names)
        dataset_train = StandardDataset(df_train, label_name=label, favorable_classes=pos_label,
                                        protected_attribute_names=prot_attr, privileged_classes=priv_group, )
        test = np.concatenate((test_features, test_labels), axis=1)
        df_test = pd.DataFrame(test, columns=dataset_orig.feature_names + dataset_orig.label_names)
        dataset_test = StandardDataset(df_test, label_name=label, favorable_classes=pos_label,
                                       protected_attribute_names=prot_attr, privileged_classes=priv_group, )

        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        RW.fit(dataset_train)
        df_reweighted = RW.transform(dataset_train)
        new_wights = df_reweighted.instance_weights

        X_train = scale.fit_transform(df_reweighted.features)
        y_train = df_reweighted.labels.ravel()
        X_test = scale.fit_transform(dataset_test.features)
        y_test = dataset_test.labels.ravel()

        clf.fit(X_train, y_train, sample_weight=new_wights)

        # Predicting the test set results
        y_pred = clf.predict(X_test)
        dataset_pred = dataset_test.copy()
        dataset_pred.labels = y_pred

        conf_matrix_list.append(confusion_matrix(y_test, y_pred))
        acc_list.append(accuracy_score(y_test, y_pred))
        prec_list.append(precision_score(y_test, y_pred))
        recal_list.append(recall_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred))
        AUC_list.append(balanced_accuracy_score(y_test, y_pred))

        fair_metric = ClassificationMetric(dataset_test, dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
        sta_parity.append(fair_metric.statistical_parity_difference())
        equ_opp.append(fair_metric.equal_opportunity_difference())
        avg_odd.append(fair_metric.average_odds_difference())
        dis_imp.append(fair_metric.disparate_impact())
        theil_inx.append(fair_metric.theil_index())

    mean_conf_matrix = np.sum(conf_matrix_list, axis=0)
    print("Mean Cofusion matrix:\n%s" % mean_conf_matrix)
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
