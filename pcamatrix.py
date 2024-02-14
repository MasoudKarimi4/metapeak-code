# Masoud Karimi 
# This code extracts the first 500 principal components 
# Saves the matrix of principal components into a matrix that is tranposed 

# Must run groupings.py first 

import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Load data from the pickle file
with open('arrays_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

tensor=loaded_data['noDiseaseTensor']

print(tensor.shape)

# Retrieve the principal components
final_pca = PCA(n_components=500)
weights_matrix = final_pca.fit_transform(tensor)

print(weights_matrix.shape)

# Transpose the matrix of principal components
weights_matrix_transposed = np.transpose(weights_matrix)

# Create a DataFrame to store the transposed weights matrix
weights_df = pd.DataFrame(weights_matrix_transposed)

print(weights_df)

# Save the DataFrame to a CSV file
# When you uncomment this line, it will generate a CSV in the directory 
#weights_df.to_csv('principal_components.csv', index=False)
