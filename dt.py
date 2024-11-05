import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Create a sample dataset
data = {
    'Age': [25, 45, 35, 50, 23, 31, 40, 36, 22, 48],
    'Income': [50, 80, 60, 90, 35, 55, 75, 65, 40, 85],
    'MaritalStatus': ['Single', 'Married', 'Single', 'Married', 'Single', 'Single', 'Divorced', 'Married', 'Single', 'Married'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes']
}


df = pd.DataFrame(data)

labelencoder = LabelEncoder()
df['MaritalStatus'] = labelencoder.fit_transform(df['MaritalStatus'])
df['Purchased'] = labelencoder.fit_transform(df['Purchased'])

X = df[['Age', 'Income', 'MaritalStatus']]
y = df['Purchased']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
