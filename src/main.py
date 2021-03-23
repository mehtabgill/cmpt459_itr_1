import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.metrics import plot_confusion_matrix



def convert_string_numerical_categorical(dataframe, attributes):
    le = LabelEncoder()
    df = dataframe.copy()
    for attribute in attributes:
        le.fit(df[attribute].unique())
        df[attribute] = le.transform(df[attribute].to_list())
    return df


def train_with_estimator(n):
    df = pd.read_csv('../data/joined_cases_train.csv')
    df = df.drop(["latitude", "longitude", "Combined_Key", "country"], axis=1)
    y = df.pop('outcome')
    x = df

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    rf1 = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_split=None,
        bootstrap=True,
    )

    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit_transform(df[['sex']])

    column_trans = make_column_transformer(
        (OneHotEncoder(), ['sex']),
        remainder='passthrough'
    )

    column_trans.fit_transform(df)

    pipe1 = make_pipeline(column_trans, rf1)

    pipe1.fit(X_train, Y_train)
    file_path = "../models/random_forest_classifier.pkl"
    with open(file_path, 'wb') as fid:
        pickle.dump(pipe1, fid)

    pipe1 = pickle.load(open(file_path, 'rb'))

    rocs = {label: [] for label in y.unique()}

    # for label in y.unique():
    #     pipe.fit(X_train, Y_train)
    #     rf_probs = pipe.predict_proba(X_test)

    rf_probs = pipe1.predict_proba(X_test)
    rf_pred = pipe1.predict(X_test)

    rf_pred_train = pipe1.predict(X_train)

    rf_auc = roc_auc_score(Y_test, rf_probs, multi_class='ovr')
    print('Random Forest AUC:  %.3f' % (rf_auc))

    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    L = ["deceased", "hospitalized", "nonhospitalized", "recovered"]
    binarized_y = label_binarize(Y_test, classes=["deceased", "hospitalized", "nonhospitalized", "recovered"])
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binarized_y[:, i], rf_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.legend(L)

    print("Result on the test data")
    print(confusion_matrix(Y_test, rf_pred))
    print(classification_report(Y_test, rf_pred, digits=3))

    print("Result on the training data")
    print(confusion_matrix(Y_train, rf_pred_train))
    print(classification_report(Y_train, rf_pred_train, digits=3))

    # cross-validation

    cross_validation_accuracy = cross_val_score(pipe1, x, y).mean()
    print("The Cross Validation accuracy is %.3f" % (cross_validation_accuracy))


def detect_overfit():
    df = pd.read_csv('../data/joined_cases_train.csv')
    df = df.drop(["latitude", "longitude", "Combined_Key", "country"], axis=1)
    y = df.pop('outcome')
    x = df
    #
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit_transform(df[['sex']])

    column_trans = make_column_transformer(
        (OneHotEncoder(), ['sex']),
        remainder='passthrough'
    )

    x = column_trans.fit_transform(df)

    parameter_range = np.arange(1, 1000, 100)

    train_score, test_score = validation_curve(RandomForestClassifier(), x, y,
                                               param_name="n_estimators",
                                               param_range=parameter_range,
                                               cv=5, scoring="accuracy")

    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)
    plt.plot(parameter_range, mean_train_score,
             label="Training Score", color='b')
    plt.plot(parameter_range, mean_test_score,
             label="Cross Validation Score", color='g')


def run_random_forest():
    train_with_estimator(100)
    # train_with_estimator(200)
    # train_with_estimator(500)
    # train_with_estimator(1000)


if __name__ == '__main__':

    # RANDOM FOREST :::: 
    print('--- Starting  RANDOM FOREST CLASSIFIER --------')
    run_random_forest()

    # GB MODEL :::
    # load data
    df = pd.read_csv('../data/joined_cases_train.csv')
    # Convert string data to numeric data
    string_categorical_attributes = ['sex', 'Combined_Key', 'country', 'outcome']
    df = convert_string_numerical_categorical(dataframe=df, attributes=string_categorical_attributes)

    outcomes = df['outcome']
    df = df.drop(columns=['outcome'])

    # 2.1 split 
    x_train, x_test, y_train, y_test = train_test_split(df, outcomes, train_size=0.80, test_size=0.20, random_state=42)

    #2.2
    # LightGBM
    print('--- Starting GB Classifier --------')
    start_time = time.time()
    GBM_clsf = GradientBoostingClassifier(learning_rate=0.9, max_depth=8, n_estimators=10, random_state=42)
    
    GBM_clsf.fit(x_train, y_train)
    print('------- GB model ready! Now Saving... ------')
    pickle.dump(GBM_clsf, open('../models/GBM.pkl', 'wb'))
    
    # 2.3 
    # Evaluation
    loaded_GBM = pickle.load(open('../models/GBM.pkl', 'rb'))
    print('------- GB Model loaded! -------------------')
    
    print('------ GB Model Classification report - VALIDATION Data -----------')
    loaded_GBM_result_test = loaded_GBM.score(x_test, y_test)
    print('------- GB Model VALIDATION data accuracy ----->  ', loaded_GBM_result_test)
    print('------- GB Model VALIDATION data Confusion Matrix--------')
    matrix = plot_confusion_matrix(loaded_GBM, x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.savefig('../plots/GBM_validation_c_matrix.png')
    plt.show()
    print('------- GB Model VALIDATION data Classification Report --------')
    report_loaded_gbm_test = classification_report(y_test, loaded_GBM.predict(x_test), target_names=['deceased', 'hospitalized', 'nonhospitalized', 'recovered'])
    print(report_loaded_gbm_test)


    print('------ GB Model Classification report - TRAIN Data -----------')
    loaded_GBM_result_train = loaded_GBM.score(x_train, y_train)
    print('------- GB Model TRAIN data accuracy ----->  ', loaded_GBM_result_train)
    print('------- GB Model TRAIN data Confusion Matrix --------')
    matrix = plot_confusion_matrix(loaded_GBM, x_train, y_train,
                                 normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.savefig('../plots/GBM_train_c_matrix.png')
    plt.show()
    print('------- GB Model TRAIN data Classification Report --------')
    report_loaded_gbm_train = classification_report(y_train, loaded_GBM.predict(x_train), target_names=['deceased', 'hospitalized', 'nonhospitalized', 'recovered'])
    print(report_loaded_gbm_train)


    # !!!! IMPORTANT !!!!! VALIDATION CURVE TAKES AROUND 3HRS TO BUILD, therefore the code for it commented out. Produced result is saved in ./plots/GBM_validation_curve.png

    # 2.4
    # plot validation curve for GB model
    # n_estimators_range = list(range(100, 1000, 250))

    # # n_jobs = -1 means use all cores of the computer
    # train_scores, test_scores = validation_curve(GradientBoostingClassifier(), df, outcomes, param_name="n_estimators", param_range=n_estimators_range,
    # scoring="accuracy", n_jobs=-1)
    # mean_train = np.mean(train_scores, axis=1)
    # std_train = np.std(train_scores, axis=1)
    # mean_test = np.mean(test_scores, axis=1)
    # std_test = np.std(test_scores, axis=1)

    # plt.title("Validation Curve with GBM")
    # plt.xlabel("n_estimators")
    # plt.ylabel("Score")
    # plt.plot(n_estimators_range, mean_train, label="Training score")
    # plt.fill_between(n_estimators_range, mean_train - std_train,
    #                 mean_train + std_train,
    #                 color="yellow")
    # plt.plot(n_estimators_range, mean_test, label="Cross-validation score")
    # plt.fill_between(n_estimators_range, mean_test - std_test,
    #                 mean_test + std_test,
    #                 color="red")
    # plt.legend(loc="best")
    # plt.savefig('../plots/GBM_validation_curve.png')
    # plt.show()

    



