
"""Techniques applied:
    1. Random Forests
    2. Support Vector Machine (linear and rbf kernel)
    3. Logistic Regression
    4. Neural Network
"""

# %% Load modules

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold


# %% Function to calculate sensitivity ans specificty
def calculate_diagnostic_performance(actual_predicted):
    """ Calculate sensitivty and specificty.
    Takes a Numpy array of 1 and zero, two columns: actual and predicted
    Returns a tuple of results:
    1) accuracy: proportion of test results that are correct    
    2) sensitivity: proportion of true +ve identified
    3) specificity: proportion of true -ve identified
    4) positive likelihood: increased probability of true +ve if test +ve
    5) negative likelihood: reduced probability of true +ve if test -ve
    6) false positive rate: proportion of false +ves in true -ve patients
    7) false negative rate:  proportion of false -ves in true +ve patients
    8) positive predictive value: chance of true +ve if test +ve
    9) negative predictive value: chance of true -ve if test -ve
    10) Count of test positives

    *false positive rate is the percentage of healthy individuals who 
    incorrectly receive a positive test result
    * alse neagtive rate is the percentage of diseased individuals who 
    incorrectly receive a negative test result
    
    """
    actual_predicted = test_results.values
    actual_positives = actual_predicted[:, 0] == 1
    actual_negatives = actual_predicted[:, 0] == 0
    test_positives = actual_predicted[:, 1] == 1
    test_negatives = actual_predicted[:, 1] == 0
    test_correct = actual_predicted[:, 0] == actual_predicted[:, 1]
    accuracy = np.average(test_correct)
    true_positives = actual_positives & test_positives
    true_negatives = actual_negatives & test_negatives
    sensitivity = np.sum(true_positives) / np.sum(actual_positives)
    specificity = np.sum(true_negatives) / np.sum(actual_negatives)
    positive_likelihood = sensitivity / (1 - specificity)
    negative_likelihood = (1 - sensitivity) / specificity
    false_postive_rate = 1 - specificity
    false_negative_rate = 1 - sensitivity
    positive_predictive_value = np.sum(true_positives) / np.sum(test_positives)
    negative_predicitive_value = np.sum(true_negatives) / np.sum(test_negatives)
    positive_rate = np.mean(actual_predicted[:,1])
    return (accuracy, sensitivity, specificity, positive_likelihood,
            negative_likelihood, false_postive_rate, false_negative_rate,
            positive_predictive_value, negative_predicitive_value, 
            positive_rate)


# %% Print diagnostics results
def print_diagnostic_results(results):
    # format all results to three decimal places
    three_decimals = ["%.3f" % v for v in results]
    print()
    print('Diagnostic results')
    print('  accuracy:\t\t\t', three_decimals[0])
    print('  sensitivity:\t\t\t', three_decimals[1])
    print('  specificity:\t\t\t', three_decimals[2])
    print('  positive likelyhood:\t\t', three_decimals[3])
    print('  negative likelyhood:\t\t', three_decimals[4])
    print('  false positive rate:\t\t', three_decimals[5])
    print('  false negative rate:\t\t', three_decimals[6])
    print('  positive predictive value:\t', three_decimals[7])
    print('  negative predicitve value:\t', three_decimals[8])
    print()

#%% Create results folder if needed
# (Not used in this demo)   
# OUTPUT_LOCATION = 'results'
# if not os.path.exists(OUTPUT_LOCATION):
#    os.makedirs(OUTPUT_LOCATION)
    
def load_data():
    data = pd.read_csv('data/data_for_ml_clin_only.csv')
    # shuffle data:
    data = data.sample(frac=1, random_state=0)
    feature_labels = list (data.columns[1:])
    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
    return X, y, feature_labels
    
# %% Import data
# Load data
X, y , feature_labels = load_data()
number_of_features = len(feature_labels)

# Set up strtified k-fold
splits = 10
skf = StratifiedKFold(n_splits = splits)
skf.get_n_splits(X, y)

# %% Set up results dataframes
forest_results = np.zeros((splits, 10))
forest_importance = np.zeros((splits, number_of_features))
svm_results_linear = np.zeros((splits, 10))
svm_results_rbf = np.zeros((splits, 10))
lr_results = np.zeros((splits, 10))
nn_results = np.zeros((splits, 10))
learning_summary = pd.DataFrame()

training_set_size_list = ([10,20,30,40,50,60,70,80,90,100,
                           125,150,175,200,225,250,
                           300,350,400,450,500,
                           600,700,800,900,1000,1100,1200,1300,1400,1500,1600,
                           1700,1800])
    


for training_set_size in training_set_size_list:
    # %% Loop through the k splits of training/test data
    loop_count = 0
    
    for train_index, test_index in skf.split(X, y):
        
        print ('Split', loop_count + 1, 'out of', splits)
        print ('Training set size:', training_set_size)
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Take random sample from training set
        
        selection = np.zeros(X_train.shape[0])
        selection[0:training_set_size] = 1
        selection = selection.astype(bool)
        np.random_seed=0
        np.random.shuffle(selection)
        X_train = X_train[selection,:]
        y_train = y_train[selection]
    
        sc = StandardScaler()  # new Standard Scalar object
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        combined_results = pd.DataFrame()
    
        # %% Random forests
        forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, 
                                        class_weight='balanced', 
                                        random_state=0)
        forest.fit(X_train, y_train)
        forest_importance[loop_count, :] = forest.feature_importances_
        y_pred = forest.predict(X_test)
        test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
        diagnostic_performance = (calculate_diagnostic_performance
                                  (test_results.values))
        forest_results[loop_count, :] = diagnostic_performance
        combined_results['Forest'] = y_pred
    
        # %% SVM (Support Vector Machine) Linear
        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train_std, y_train)
        y_pred = svm.predict(X_test_std)
        test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
        diagnostic_performance = (calculate_diagnostic_performance
                                  (test_results.values))
        svm_results_linear[loop_count, :] = diagnostic_performance
        combined_results['SVM_linear'] = y_pred
    
        # %% SVM (Support Vector Machine) RBF
        svm = SVC(kernel='rbf', C=1.0, random_state=0)
        svm.fit(X_train_std, y_train)
        y_pred = svm.predict(X_test_std)
        test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
        diagnostic_performance = (calculate_diagnostic_performance
                                  (test_results.values))
        svm_results_rbf[loop_count, :] = diagnostic_performance
        combined_results['SVM_rbf'] = y_pred
    
        # %% Logistic Regression
        lr = LogisticRegression(C=100)
        lr.fit(X_train_std, y_train)
        y_pred = lr.predict(X_test_std)
        test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
        diagnostic_performance = (calculate_diagnostic_performance
                                  (test_results.values))
        lr_results[loop_count, :] = diagnostic_performance
        combined_results['LR'] = y_pred
    
        # %% Neural Network
        clf = MLPClassifier(solver='adam', alpha=1e-8, hidden_layer_sizes=(500,50),
                            max_iter=100000, shuffle=True, learning_rate_init=0.0005,
                            activation='logistic', learning_rate='adaptive', tol=1e-7,
                            random_state=0)
        clf.fit(X_train_std, y_train)
        y_pred = clf.predict(X_test_std)
        test_results = pd.DataFrame(np.vstack((y_test, y_pred)).T)
        diagnostic_performance = (calculate_diagnostic_performance
                                  (test_results.values))
        nn_results[loop_count, :] = diagnostic_performance
        combined_results['NN'] = y_pred
        
        # Increment loop count
        loop_count += 1
    
    # %% Transfer results to Pandas arrays
    results_summary = pd.DataFrame()
    
    results_column_names = (['accuracy', 'sensitivity', 
                             'specificity',
                             'positive likelihood', 
                             'negative likelihood', 
                             'false positive rate', 
                             'false negative rate',
                             'positive predictive value',
                             'negative predictive value', 
                             'positive rate'])
    
    forest_results_df = pd.DataFrame(forest_results)
    forest_results_df.columns = results_column_names
    forest_importance_df = pd.DataFrame(forest_importance)
    forest_importance_df.columns = feature_labels
    results_summary['Forest'] = forest_results_df.mean()
    
    svm_results_lin_df = pd.DataFrame(svm_results_linear)
    svm_results_lin_df.columns = results_column_names
    results_summary['SVM_lin'] = svm_results_lin_df.mean()
    
    svm_results_rbf_df = pd.DataFrame(svm_results_rbf)
    svm_results_rbf_df.columns = results_column_names
    results_summary['SVM_rbf'] = svm_results_rbf_df.mean()
    
    lr_results_df = pd.DataFrame(lr_results)
    lr_results_df.columns = results_column_names
    results_summary['LR'] = lr_results_df.mean()
    
    nn_results_df = pd.DataFrame(nn_results)
    nn_results_df.columns = results_column_names
    results_summary['Neural'] = nn_results_df.mean()
    
    learning_summary[training_set_size] = results_summary.loc['accuracy']


# %% Print summary results
print()
print('Results Summary:')
print(learning_summary)
learning_summary.to_csv('learning_rate_results.csv')

# %% Save files
#forest_results_df.to_csv('results/forest_results.csv')
#forest_importance_df.to_csv('results/forest_importance.csv')
#svm_results_lin_df.to_csv('results/svm_lin_results.csv')
#svm_results_rbf_df.to_csv('results/svm_rbf_results.csv')
#lr_results_df.to_csv('results/logistic_results.csv')
#nn_results_df.to_csv('results/neural_network_results.csv')
#results_summary.to_csv('results/results_summary.csv')
