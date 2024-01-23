# Masoud Karimi 
# This code is mainly for cleaning and preparing the data to be used in other python files for ML 

import pandas as pd
import torch as torch 
import pickle 
import numpy as np

# Reading the full name
pd.set_option('display.max_colwidth', None)

# Read the CSV file containing sample data such as healthy, cancer
data = pd.read_csv('MD.csv')

# This sets the data frame of the metapeaks into a variable from H3K27ac
binaryMatrices = pd.read_table('H3K27ac_normal_normal_metapeak_binary.tsv')

# Group the selected rows by the "Cancer/No cancer" column
grouped_data = data.groupby('harmonized_sample_disease_high')

cancer_fnames = []
healthy_fnames = []
disease_fnames = []
duplicate_fnames = []

columns = binaryMatrices.columns.tolist()

print(columns[0])
print(columns[1])

print(len(columns))

# Iterate over grouped_data
for group_name, group_data in grouped_data:
    print("Group:", group_name)
    ihec_chipseq_lines = group_data[group_data['FName'].str.contains('ihec.chipseq') & group_data['FName'].isin(columns)]['FName']
    print(ihec_chipseq_lines)
    print("\n")

    # Assigning FNames to groups
    if ihec_chipseq_lines.empty:
        continue  # Skip the group if no matching filenames are found

    if group_name == "Cancer":
        cancer_fnames.extend(ihec_chipseq_lines.tolist())
    elif group_name == "Disease":
        disease_fnames.extend(ihec_chipseq_lines.tolist())
    else:
        healthy_fnames.extend(ihec_chipseq_lines.tolist())

# Print the counts of each group
print(len(cancer_fnames))
print(len(disease_fnames))
print(len(healthy_fnames))


# Assigning labels to each row of the matrix
tensorLabels = []
for column_name, column_data in binaryMatrices.items():
  
    if column_name in cancer_fnames:
        tensorLabels.append(0)
    elif column_name in healthy_fnames:
        tensorLabels.append(1)
    elif column_name in disease_fnames:
        tensorLabels.append(2)

print(tensorLabels)

# Converting the binary matrix into a numpy array, then a torch tensor
matrix = binaryMatrices.to_numpy()
tensor = torch.from_numpy(matrix).float()

# Transposing the tensor/ dataframe now and saving it after
tensor = torch.transpose(tensor, 0, 1)
binaryMatrices = binaryMatrices.transpose()

# This code removes the disease samples to do binary logistic regression with cancer 0 vs healthy 1 
indexRemove = []
noDiseaseLabels = []
for i in range(len(tensorLabels)):

    if tensorLabels[i] == 2:
        indexRemove.append(i)
    else:
        noDiseaseLabels.append(tensorLabels[i])

# Creating a mask tensor to select all indices except the ones to remove
# In this case, the disease samples 
mask = torch.ones_like(tensor, dtype=torch.bool)
mask[indexRemove] = False

# Selecting elements from the original tensor based on the mask
new_tensor = torch.masked_select(tensor, mask)
new_tensor = torch.reshape(new_tensor, (-1, tensor.shape[1]))

# Creating a DataFrame that has feature index with the corresponding Metapeak region
columnDF = pd.DataFrame({'Metapeak Region': binaryMatrices.columns})

print(columnDF)



# Saves the data into variables with Pickle Package
# This file needs to be run first in order to use the saved data in other files 
data = {
    'healthy' : healthy_fnames,
    'cancer' : cancer_fnames,
    'disease': disease_fnames,
    'tensor': tensor,
    'tsv': binaryMatrices,
    'labels': tensorLabels,
    'noDiseaseLabels': noDiseaseLabels,
    'noDiseaseTensor': new_tensor,
    'dataframe': columnDF
}

with open('arrays_data.pkl', 'wb') as file:
    # Saved!
    pickle.dump(data, file)



# Here I just counted the ratio of cancer and healthy 
    
# Cancer: 284
# Healthy: 1119
# Total: 1403

# Healthy Ratio: 0.79757
# Cancer Ratio: 0.2017