# Masoud Karimi 
# This code extracts the first 500 principal components (On the metapeak region features)
# Saves the first 5 to a variable 

# Must run groupings.py first 

import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

# Load data from the pickle file
with open('arrays_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Tensor is the binary matrix
# Label is the list of 0's and 1's corresponding to cancer and healthy 
tensor = loaded_data['noDiseaseTensor']
labels = loaded_data['noDiseaseLabels']

# This dataframe stores the names of each metapeak region 
dataframe = loaded_data['dataframe']

torch.manual_seed(2)

# Create the training and test splits
X = tensor
row_indices = np.arange(X.shape[0])
row_indices_train, row_indices_test = train_test_split(row_indices, test_size=0.2, random_state=3904)
X_train = X[row_indices_train]
X_test = X[row_indices_test]

# Create the y train of the cancer/healthy labels 
y_train = [labels[i] for i in row_indices_train]
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = [labels[i] for i in row_indices_test]
y_test = torch.tensor(y_test, dtype=torch.float32)


# Perform PCA with 500 components
final_pca = PCA(n_components=500)

# Only x_train data is fitted as x_test will be evaluated 
X_train_final_pca = final_pca.fit_transform(X_train)
X_test_final_pca = final_pca.transform(X_test)

# Display the first few principal components
num_components_to_display = 5  
print("First {} Principal Components:".format(num_components_to_display))
for i in range(num_components_to_display):
    print("PC{}: {}".format(i + 1, final_pca.components_[i]))
    print(len(final_pca.components_[i]))


# Now we order by weight abs value
print("Highest Weights by Index:")

# Display the features within each principal component ordered by their highest absolute weight
num_features_to_display = 30 
num_components_to_display = 5  

print("Features Ordered by Highest Absolute Weight in Each Principal Component:")
for i in range(num_components_to_display):
    pc = final_pca.components_[i]
    absolute_weights = np.abs(pc)
    sorted_feature_indices = np.argsort(absolute_weights)[::-1][:num_features_to_display]

    print("\nPrincipal Component {}: ".format(i + 1))
    for j in range(num_features_to_display):

        # This code retrieves which metapeak region it was from the dataframe
        feature_index = sorted_feature_indices[j]
        weight_value = pc[feature_index]
        metapeak_value = dataframe.loc[feature_index, "Metapeak Region"]
        print("Index {}: Weight {:.4f} Region: {}".format(feature_index, weight_value, metapeak_value))


# Saving the first 5 principal components to see the distribution of the weights
# I saved these to make histograms in histograms.py
pc1 = final_pca.components_[0]
pc2 = final_pca.components_[1]
pc3 = final_pca.components_[2]
pc4 = final_pca.components_[3]
pc5 = final_pca.components_[4]


# Define the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)  # Increase the number as needed

# Cross-validation using StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate mean accuracy
cross_val_accuracy = cross_val_score(logistic_model, X_train_final_pca, y_train, cv=cv, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f}%".format(np.mean(cross_val_accuracy) * 100))

# Fit the model on the entire training set
logistic_model.fit(X_train_final_pca, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test_final_pca)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(X_test_final_pca)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
'''
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
'''



# Exporting the variables to be used in other files 
data = {
    'tensor': tensor,
    'pc1':pc1,
    'pc2':pc2,
    'pc3':pc3,
    'pc4':pc4,
    'pc5':pc5,
    'pcaList':final_pca.components_
}

with open('arrays_data.pkl', 'wb') as file:
    # Saved!
    pickle.dump(data, file)