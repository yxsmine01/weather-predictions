# import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# upload csv file

from google.colab import files

uploaded = files.upload()

for filename in uploaded.keys():
    print('Uploaded file "{name}" with length {length} bytes'.format(
        name=filename, length=len(uploaded[filename])))
    
# data to be read into a dataframe

file_path = 'seattle-weather.csv'
df = pd.read_csv(file_path)
print(df)

# Dimensions of the dataset
print("Dimensions of the data:", df.shape)

# Discriptive statistics
print(df.describe())

# Summary of the features in the dataset
print(df.info())

# Checking for missing values in the dataset
print("\nAre there any missing points in the dataset?:", df.isnull().values.any())

print("Number of Duplicated Rows:", df.duplicated().sum())

# Display first five rows
print("First 5 rows of the dataset:")
df.head()

# Display last five rows
print("First 5 rows of the dataset:")
df.tail()

# Create histogram to visualize the distribution of numerical values
numerical_features = ['precipitation', 'temp_max', 'temp_min', 'wind']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, feature in enumerate(numerical_features):
    ax = axes[i // 2, i % 2]
    df[feature].hist(bins=30, ax=ax)
    ax.set_title(f'Histogram of {feature.capitalize()}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualize the distribution of categorical features
categorical_features = ['weather']
for feature in categorical_features:
    plt.figure(figsize=(6, 4))
    df[feature].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()
    
# Create a heatmap to visualize the correlation between numerical features
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), cmap='Blues', annot=True);

# Create subplots to visualize relationship between features

# Create a figure with four subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))

# Plot maximum temperature vs. weather on the second subplot
sns.lineplot(x='temp_max', y='weather', data=df, ax=ax1)

# Plot minimum temperature vs. weather on the third subplot
sns.lineplot(x='temp_min', y='weather', data=df, ax=ax2)

# Plot wind vs. weather on the fourth subplot
sns.lineplot(x='wind', y='weather', data=df, ax=ax3)

# Display the plots
plt.show()

# Convert date strings to datetime objects
df['date'] = pd.to_datetime(df['date'])

# Extract year from the date
df['year'] = df['date'].dt.year

# Count the occurrences of each weather category for each year
yearly_weather_counts = df.groupby(['year', 'weather']).size().unstack(fill_value=0)

# Plotting
yearly_weather_counts.plot(kind='bar', stacked=True)
plt.xlabel('Year')
plt.ylabel('Occurrences')
plt.title('Weather Occurrences by Year')
plt.legend(title='Weather')  # Add legend with weather categories
plt.show()

# Visualize the distribution of weather based on max. temperature
plt.figure(figsize=(10, 6))
sns.boxplot(x='temp_max', y='weather', data=df, palette='Set3')
plt.title('Weather Distribution by maximum temperature')
plt.show()

# Visualize the distribution of weather based on min. temperature
plt.figure(figsize=(10, 6))
sns.boxplot(x='temp_min', y='weather', data=df, palette='viridis')
plt.title('Weather Distribution by minimum temperature')
plt.show()

# Visualize the distribution of weather based on wind
plt.figure(figsize=(10, 6))
sns.boxplot(x='wind', y='weather', data=df, palette='coolwarm')
plt.title('Weather Distribution by wind')
plt.show()

# Visualize the distribution of weather based on year
plt.figure(figsize=(10, 6))
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
sns.boxplot(x='year', y='weather', data=df, palette='magma')
plt.title('Weather Distribution by year')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract features from the 'date' column
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6

# Encode 'weather' columns
df['weather_encode'] = LabelEncoder().fit_transform(df['weather'])

df.head()

# Define the features (X) and target variable (y) for Random forest classification
X = df[['month','precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather_encode']
X.head()

# Split the dataset into training and testing sets for Random forest classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

param_grid = {'n_estimators': randint(10,1000),
              'max_features': ('sqrt', 'log2', None),
              'max_depth': [None] + list(range(1, 31)),
              'min_samples_split': randint(0, 50),
              'min_samples_leaf': randint(1, 20),
              'bootstrap': (True, False)}

# Create a random forest classifier
rf_random = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf_random,
                                 param_distributions = param_grid,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Instantiate a Random forest classifier
rf_model = RandomForestClassifier(n_estimators = 733, max_depth = None, max_features = 'sqrt', min_samples_split = 11, min_samples_leaf = 10, bootstrap = False)

# Fit the Random forest model using the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score

print('Accuracy (rf): ', accuracy_score(y_test, y_pred_rf)*100)
print('Precision (rf): ', precision_score(y_test, y_pred_rf, average = 'weighted')*100)
print('Recall (rf): ', recall_score(y_test, y_pred_rf, average = 'weighted')*100)

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# Export the first three decision trees from the forest
for i in range(3):
    tree = rf_model.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
    
# View a few samples of actual and predicted smoking status
num_samples_to_view = 5

for i in range(num_samples_to_view):
    print('===================================================')
    print(f'Test Sample #{i+1}')
    print()
    print(X_test.iloc[[i]])
    print()
    print(f'Actual Weather Variable (0 for drizzle, 1 for fog, 2 for rain, 3 for snow, 4 for sun): {y_test.iloc[i]}')
    print(f'Predicted Weather Variable (0 for drizzle, 1 for fog, 2 for rain, 3 for snow, 4 for sun): {y_pred_rf[i]}')

    print()
    
# Generate the Confusion Matrix for the Random forest model
cm_rf = metrics.confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)
disp.plot(cmap='Blues')
plt.show();

print(metrics.classification_report(y_test, y_pred_rf, target_names=['drizzle', 'fog', 'rain', 'snow', 'sun']))
