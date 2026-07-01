import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('heart.csv')
#----------------EDA with data-cleaning----------------#

# print(df.shape) #918cols and 12 rows
# print(df.info()) #no null values
# print(df.describe()) #statistical summary of the data
# df.duplicated().sum() #no duplicate values
# print(df['HeartDisease'].value_counts()) #checking the target variable distribution

# sns.countplot(x='HeartDisease',data=df)
# print(df.isnull().sum()) #checking for null values

def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)
# plt.show()

#now we replace the 0 values in Cholesterol and RestingBP with their respective mean values as they are not possible to be 0
mean_ch=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol'] = df['Cholesterol'].replace(0, mean_ch)
df['Cholesterol']=df['Cholesterol'].round(2)

mean_rb=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP'] = df['RestingBP'].replace(0, mean_rb)
df['RestingBP']=df['RestingBP'].round(2)

# plotting('Age',1)
# plotting('RestingBP',2)
# plotting('Cholesterol',3)
# plotting('MaxHR',4)
# plt.show()

#we analyze categorical variables with countplots
# plt.figure(figsize=(12,10))
# sns.countplot(x='ChestPainType',hue='HeartDisease',data=df)
# sns.countplot(x='FastingBS',hue='HeartDisease',data=df)
# plt.show()        

# sns.boxplot(x='HeartDisease',y='cholestrol',data=df)
# sns.violinplot(x='HeartDisease',y='Age',data=df)
# plt.show()  

#let create heat map
# sns.heatmap(df.corr(numeric_only= True), annot=True)
# plt.show()

#now we do data preprocessing and cleaning
df_encode=pd.get_dummies(df,drop_first=1)
df_encode=df_encode.astype(int)

#standard scaler.
from sklearn.preprocessing import StandardScaler
numeric_cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scaler = StandardScaler()
df_encode[numeric_cols] = scaler.fit_transform(df_encode[numeric_cols])
# print(df_encode.head())



#finally we choose model which is best for our data.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

x=df_encode.drop('HeartDisease',axis=1)
y=df_encode['HeartDisease']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC()
}
test = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    test.append({'Model': name, 'Accuracy': accuracy, 'F1 Score': f1})

# print(test)             #here i show logistic regression is best model for this data with 86% accuracy and 87% f1 score. 


import joblib

# Save the best model
best_model = models['Logistic Regression']
joblib.dump(best_model, 'best_heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future use
joblib.dump(x.columns.to_list(), 'model_columns.pkl')  # Save the model columns for future use 