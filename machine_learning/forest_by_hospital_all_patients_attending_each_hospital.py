import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

# %% Set up results numpy array
forest_results_predicted = np.zeros((7, 1))

# %% Loop through training hospitals
# Select data and decision only from training hospital

loop_count = 0
for training_hospital in range(7):
    
    print ('hospital', loop_count + 1)
    
    mask = np.argmax(X[:, 0:7],axis=1) == training_hospital
    X_train = X[mask, 7:]
    y_train = y[mask]

    # %% Train random forest classifier on data from selected hospital
    forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, 
                                    class_weight='balanced', random_state=0)
    forest.fit(X_train, y_train)
    
    # %% Loop through each hospitals patients and test using classifier
    X_test = X[:,7:]
    y_pred = forest.predict(X_test)
    forest_results_predicted[training_hospital,0] = (
            np.mean(y_pred))
        
    # Increment loop count
    loop_count += 1
    
forest_results_predicted = pd.DataFrame(forest_results_predicted)
forest_results_predicted.to_csv('forest_hospital_predicted_all patients.csv')


    



