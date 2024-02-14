Python code for PCA/Logistic Regression of IHEC Enhancer data 

Since the pickle package was used to save variables, you need to run groupings.py first and then run other files

groupings.py: Cleans the data and generates variables that will be saved with the pickle library. Run this first

pc500: Generates the weights of the first 5 principal components with metapeak regions, evaluates logistic regression with a roc curve

pcmatrix: Generates a matrix of the 500 principal components 
