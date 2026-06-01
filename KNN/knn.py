import pandas as pd
import numpy as np
import kagglehub
import os

# load dataset
path = kagglehub.dataset_download("wenruliu/adult-income-dataset")

files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)

# read csv and store as dataframe
df = pd.read_csv(full_path)

# separate target column (a series)
target = 'income'
X = df.drop(columns = [target])     # features dataframe
Y = df[target]                      # target column (a series)

#preprocess to remove empty values
X = X.replace(' ?', np.nan)
X = X.dropna()
Y = Y[X.index]   # to keep same no. of rows as X

''' Because this section was not working, used the pandas get dummies instead
# Do one-hot encoding to calculate distances using numerical values later
# for any categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# I don't think we need to do this for target, as we'll work directly with labels
# Y = Y.astype('category').cat.codes
'''
# Encoding categorical data
X = pd.get_dummies(X)
# convert the series/df into numpy arrays to enable vector operations later
X = X.to_numpy()
Y = Y.to_numpy()

#ensure all data types of floats
X = X.astype(float)

# Normalize the feature values to give all features equal weight
# axis = 0 means go column by column (feature-wise)
fmin = X.min(axis=0)
fmax = X.max(axis=0)
diff = fmax - fmin
diff[diff==0] = 1  # to avoid div. by zero in cases where one column as constant value
X = (X - fmin)/diff

print(X.shape)
print(Y.shape)

# Randomize the rows to get actual model performance
permute = np.random.permutation(len(X))
X = X[permute]
Y = Y[permute]

# Split dataset into 2 - one is for training, one for testing
n = len(df) // 2

X_train = X[n:]
X_test = X[:n]

Y_train = Y[n:]
Y_test = Y[:n]

# Done data pre-processing, now let's do KNN

# KNN
def knn(X_train, X_test, Y_train, Y_test, k):
    pred = []

    for point in X_test:
        #calculate distance of test point from all points in training data
        # returns 1D array of distances
        # we do X_train - point to get x1-x2 for all training points per column
        # and then get distance by calculating row wise (axis = 1)
        dst = np.linalg.norm(X_train - point, axis = 1)
        indices = np.argsort(dst)   # returns indices sorted based on the values
        top_k = indices[:k]         # get indices of k nearest neighbours of that test point
        k_labels = Y_train[top_k]   # get labels of the kNN
        # get the individual distinct labels and their counts in the k_labels Series
        label, count = np.unique(k_labels, return_counts = True)
        pred_label = label[np.argmax(count)]    # argmax returns index of element with max count
        pred.append(pred_label)                 # Store the predicted label in a list

    # convert the pred list to pred array to enable vector operations
    pred = np.array(pred)
    accuracy = np.sum(pred == Y_test) / len(Y_test)

    return pred, accuracy

k = int(input("Enter value for K: "))

# for running the code, reducing the dataset size
X_train = X_train[:5000]
X_test = X_test[:5000]

Y_train = Y_train[:5000]
Y_test = Y_test[:5000]

# running the code
predicted, accuracy = knn(X_train, X_test, Y_train, Y_test, k)

# showing the results for rows 20 to 40
df_compare = pd.DataFrame({
    "Predicted": predicted,
    "Actual": Y_test,
    "Match": predicted == Y_test
})

print(df_compare.iloc[20:40])

print(f"\nAccuracy of model: {accuracy}")





    
