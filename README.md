# Loan-Eligibility-Prediction

# Importing necessary libraries: pandas for data manipulation, numpy for numerical operations, and randint from random for generating random integers.

import pandas as pd
import numpy as np
from random import randint

# Reading the training data from the specified CSV file

train=pd.read_csv(r'/content/train_u6lujuX_CVtuZ9i.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.isnull().sum()

# Extracting the 'Loan_Status' column, dropping it from the training data, reading the test data, extracting 'Loan_ID' from the test data,
# and creating a combined dataset by appending the training and test data. Displaying the first few rows of the combined dataset.

Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv(r'/content/train_u6lujuX_CVtuZ9i.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()

# Generating descriptive statistics for the combined dataset using the describe() function.
data.describe()

# Checking and displaying the count of missing values in each column of the combined dataset using the isnull() function.
data.isnull().sum()

# Checking the data type of the 'Dependents' column in the combined dataset using the dtypes attribute.
data.Dependents.dtypes

# Importing necessary libraries for visualization.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Calculating the correlation matrix for the combined dataset.
corrmat = data.corr()

# Creating a histogram for the 'LoanAmount' column.
plt.figure(figsize=(5, 5))
sns.histplot(data['LoanAmount'], bins=20, kde=True, color='blue')
plt.title('Histogram of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.show()


# Mapping 'Male' to 1 and 'Female' to 0 in the 'Gender' column of the combined dataset.
# Displaying the count of each unique value in the 'Gender' column using value_counts().

data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()

# Mapping 'Male' to 1 and 'Female' to 0 in the 'Gender' column of the combined dataset.
# Displaying the count of each unique value in the 'Gender' column using value_counts().

data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()

# Calculating the correlation matrix for the combined dataset.
corrmat = data.corr()

# Creating a histogram for the 'LoanAmount' column.
plt.figure(figsize=(5, 5))
sns.histplot(data['LoanAmount'], bins=20, kde=True, color='blue')
plt.title('Histogram of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.show()

# Mapping 'Yes' to 1 and 'No' to 0 in the 'Married' column of the combined dataset.
data.Married=data.Married.map({'Yes':1,'No':0})

# Displaying the count of each unique value in the 'Married' column of the combined dataset using value_counts().
data.Married.value_counts()

# Mapping '0' to 0, '1' to 1, '2' to 2, and '3+' to 3 in the 'Dependents' column of the combined dataset.
# Displaying the count of each unique value in the 'Dependents' column using value_counts().
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
data.Dependents.value_counts()

# Calculating the correlation matrix for the combined dataset.
corrmat = data.corr()

# Creating a histogram for the 'LoanAmount' column.
plt.figure(figsize=(5, 5))
sns.histplot(data['LoanAmount'], bins=20, kde=True, color='blue')
plt.title('Histogram of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.show()

# Mapping 'Graduate' to 1 and 'Not Graduate' to 0 in the 'Education' column of the combined dataset.
# Displaying the count of each unique value in the 'Education' column using value_counts().
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
data.Education.value_counts()

# Mapping 'Yes' to 1 and 'No' to 0 in the 'Self_Employed' column of the combined dataset.
# Displaying the count of each unique value in the 'Self_Employed' column using value_counts().
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
data.Self_Employed.value_counts()

# Displaying the count of each unique value in the 'Property_Area' column of the combined dataset using value_counts().
data.Property_Area.value_counts()

# Mapping 'Urban' to 2, 'Rural' to 0, and 'Semiurban' to 1 in the 'Property_Area' column of the combined dataset.
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})

# Displaying the count of each unique value in the 'Property_Area' column of the combined dataset after mapping using value_counts().
data.Property_Area.value_counts()

# Calculating the correlation matrix for the combined dataset.
corrmat = data.corr()

# Creating a histogram for the 'LoanAmount' column.
plt.figure(figsize=(5, 5))
sns.histplot(data['LoanAmount'], bins=20, kde=True, color='blue')
plt.title('Histogram of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.show()

# Displaying the first few rows of the modified combined dataset.
data.head()

# Displaying the total number of elements in the 'Credit_History' column of the combined dataset using the size attribute.
data.Credit_History.size

# Filling the missing values in the 'Credit_History' column of the combined dataset with a random integer (0 or 1) using numpy's randint.
data.Credit_History.fillna(np.random.randint(0,2),inplace=True)

# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values.
data.isnull().sum()

# Filling the missing values in the 'Married' column of the combined dataset with a random integer (0 or 1) using numpy's randint.
data.Married.fillna(np.random.randint(0,2),inplace=True)

# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values in the 'Married' column.
data.isnull().sum()

# Filling the missing values in the 'LoanAmount' column with the median and in the 'Loan_Amount_Term' column with the mean.
# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values.

data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)
data.isnull().sum()


# Filling the missing values in the 'LoanAmount' column with the median and in the 'Loan_Amount_Term' column with the mean.
# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values.

data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)
data.isnull().sum()

# Displaying the count of each unique value in the 'Gender' column of the combined dataset using value_counts().

data.Gender.value_counts()

# Filling the missing values in the 'Gender' column of the combined dataset with a random integer (0 or 1) using numpy's randint.
# Displaying the count of each unique value in the 'Gender' column after filling missing values.

from random import randint
data.Gender.fillna(np.random.randint(0,2),inplace=True)
data.Gender.value_counts()

# Filling the missing values in the 'Dependents' column with the median.
# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values.

data.Dependents.fillna(data.Dependents.median(),inplace=True)
data.isnull().sum()

# Calculating the correlation matrix for the combined dataset.
corrmat = data.corr()

# Creating a histogram for the 'LoanAmount' column.
plt.figure(figsize=(5, 5))
sns.histplot(data['LoanAmount'], bins=20, kde=True, color='blue')
plt.title('Histogram of LoanAmount')
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.show()

# Filling the missing values in the 'Self_Employed' column with a random integer (0 or 1) using numpy's randint.
# Checking and displaying the count of missing values in each column of the combined dataset after filling missing values.

data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)
data.isnull().sum()

data.head()

# Dropping the 'Loan_ID' column from the combined dataset.
# Checking and displaying the count of missing values in each column of the dataset after dropping 'Loan_ID'.

data.drop('Loan_ID',inplace=True,axis=1)
data.isnull().sum()

# Creating training features (train_X) and labels (train_y) from the first 614 rows of the combined dataset.
# Creating testing features (X_test) from the remaining rows of the combined dataset.
# Setting a seed value for reproducibility.

train_X=data.iloc[:614,]
train_y=Loan_status
X_test=data.iloc[614:,]
seed=7

# Splitting the training data into training and testing sets using train_test_split from sklearn.
# Setting the random state to the specified seed value for reproducibility.

from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create the HistGradientBoostingClassifier model
svc = HistGradientBoostingClassifier()

# Fit the model to the data
svc.fit(train_X, train_y)

# Predict the labels for the test data
pred = svc.predict(test_X)

# Evaluate the model
print(accuracy_score(test_y, pred))
print(confusion_matrix(test_y, pred))
print(classification_report(test_y, pred))

# Importing pandas and creating an empty DataFrame named df_output.

import pandas as pd
df_output=pd.DataFrame()

# Making predictions on the test data using the Support Vector Classifier (svc) and converting the output to integers.

outp=svc.predict(test_X).astype(int)
outp

# Adding 'Loan_ID' and 'Loan_Status' columns to the DataFrame df_output using Loan_ID and the predicted values (outp).

df_output['Loan_ID'] = Loan_ID
df_output['Loan_Status'] = pd.Series(outp)

# Displaying the first few rows of the DataFrame df_output.

df_output.head()

