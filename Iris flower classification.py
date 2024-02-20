# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 00:54:44 2024

@author: Dell
"""

############################# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


###############################Importing Data and description
df = pd.read_csv("IRIS.csv")

pd.set_option('display.max_columns', None)

df.head(5)
df.info()
print(df.shape)
print(df.columns)
df.describe()


#Distribution of columns one by one
###########################sepal_length columnnnnnnnnnnnn

# draw a histogram to see the distribution of sepal_length column
sns.histplot(data=df, x='sepal_length',  kde=True)
# plot the mean, median and mode of sepal_length column using sns
sns.histplot(df['sepal_length'], kde=True)
plt.axvline(df['sepal_length'].mean(), color='red')
plt.axvline(df['sepal_length'].median(), color='green')
plt.axvline(df['sepal_length'].mode()[0], color='blue')

# print the value of mean, median and mode of sepal_length column
print('Mean:', df['sepal_length'].mean())
print('Median:', df['sepal_length'].median())
print('Mode:', df['sepal_length'].mode()[0])

# draw a histogram if see the distribution of sepal_length column as species
sns.histplot(data=df, x='sepal_length', hue='species', multiple='stack')



########################### sepal_width columnnnnnnnnnnnn
sns.histplot(data=df, x='sepal_width', kde=True)
sns.histplot(df['sepal_width'], kde=True)
plt.axvline(df['sepal_width'].mean(), color='red')
plt.axvline(df['sepal_width'].median(), color='green')
plt.axvline(df['sepal_width'].mode()[0], color='blue')

print('Mean:', df['sepal_width'].mean())
print('Median:', df['sepal_width'].median())
print('Mode:', df['sepal_width'].mode()[0])

sns.histplot(data=df, x='sepal_width', hue= 'species', multiple='stack')


############################# petal_length columnnnnnnnnnnnn
sns.histplot(data=df, x='petal_length', kde=True)
sns.histplot(df['petal_length'], kde=True)
plt.axvline(df['petal_length'].mean(), color='red')
plt.axvline(df['petal_length'].median(), color='green')
plt.axvline(df['petal_length'].mode()[0], color='blue')

print('Mean:', df['petal_length'].mean())
print('Median:', df['petal_length'].median())
print('Mode:', df['petal_length'].mode()[0])

sns.histplot(data=df, x='petal_length', hue= 'species', multiple='stack')


############################# petal_width columnnnnnnnnnnnn
sns.histplot(data=df, x='petal_width', kde=True)
sns.histplot(df['petal_width'], kde=True)
plt.axvline(df['petal_width'].mean(), color='red')
plt.axvline(df['petal_width'].median(), color='green')
plt.axvline(df['petal_width'].mode()[0], color='blue')

print('Mean:', df['petal_width'].mean())
print('Median:', df['petal_width'].median())
print('Mode:', df['petal_width'].mode()[0])

sns.histplot(data=df, x='petal_width', hue= 'species', multiple='stack')



###############################species columnnnnnnnnnnnn


df['species'].unique()
df['species'].nunique()
# value count of species column
df['species'].value_counts()

# Encode the object species column
le_species = LabelEncoder()
df['species'] = le_species.fit_transform(df['species'])

# we done feature scaling to get in same range values
# import the scalar
scalar = MinMaxScaler()
# fit the scalar on data
scaled_df = scalar.fit_transform(df)
# convert this data into a pandas dataframe
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
scaled_df.head()

# # decode the object column
# df['species'] = le_species.inverse_transform(df['species'])


#############################Check Missing Values###############

df.isnull().sum().sort_values(ascending = False)


#############################Dealing with Outliers##############

# make box plots of all the numeric columns one by one
import seaborn as sns
import matplotlib.pyplot as plt
numeric_columns = df.select_dtypes(include=['int', 'float']).columns

for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[col])
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.show()

# make box plots of all the numeric columns combine
import seaborn as sns
import matplotlib.pyplot as plt
numeric_columns = df.select_dtypes(include=['int', 'float']).columns

plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numeric_columns], orient='h')
plt.title('Box Plots of Numeric Columns')
plt.xlabel('Value')
plt.show()




###########################Splitting data

X = df.drop('species', axis=1)
y = df['species']

####################### Method 1 k fold cross validation


# Define number of folds for cross-validation
num_folds = 5

# Initialize KFold cross-validation splitter
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize classifiers
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier(random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)

# Perform cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X, y, cv=kf)

# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=kf)

# Perform cross-validation for Decision Tree
dt_cv_scores = cross_val_score(dt_model, X, y, cv=kf)

# Print cross-validation scores
print("KNN Cross-validation scores:", knn_cv_scores)
print("Random Forest Cross-validation scores:", rf_cv_scores)
print("Decision Tree Cross-validation scores:", dt_cv_scores)

# Print mean and standard deviation of cross-validation scores
print("Mean KNN CV score:", knn_cv_scores.mean())
print("Standard deviation of KNN CV scores:", knn_cv_scores.std())

print("Mean Random Forest CV score:", rf_cv_scores.mean())
print("Standard deviation of Random Forest CV scores:", rf_cv_scores.std())

print("Mean Decision Tree CV score:", dt_cv_scores.mean())
print("Standard deviation of Decision Tree CV scores:", dt_cv_scores.std())


##########################Method 2 train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state=42)


from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier(n_neighbors=5, weights='uniform', leaf_size=30,
                              p=20, metric='minkowski', metric_params=None, n_jobs=None)
model1.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model1.score(X_train, y_train)))

pred1 = model1.predict(X_test)
model1.score(X_test, y_test)

cm1 = confusion_matrix(y_test, pred1)
print(confusion_matrix(y_test,pred1)) 
print(classification_report(y_test,pred1))


from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier(max_depth=2)
model2.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model2.score(X_train, y_train)))

pred2 = model2.predict(X_test)
model2.score(X_test, y_test)

cm2 = confusion_matrix(y_test, pred2)
print(confusion_matrix(y_test,pred2)) 
print(classification_report(y_test,pred2))



from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=2, random_state=42)
model3.fit(X_train, y_train)
print('Accuracy training : {:.3f}'.format(model3.score(X_train, y_train)))

pred3 = model3.predict(X_test)
model3.score(X_test, y_test)

cm3 = confusion_matrix(y_test, pred3)
print(confusion_matrix(y_test,pred3)) 
print(classification_report(y_test,pred3))



