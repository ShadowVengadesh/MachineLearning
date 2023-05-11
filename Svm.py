import pandas as pd 

fromsklearn.model_selection import train_test_split 

fromsklearn.svm import SVC 

fromsklearn.metrics import accuracy_score

put is verified. 

# Load the dataset 

data = pd.read_csv('data.csv') 

# Split the data into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], 

test_size=0.3, random_state=42) 
# Train an SVM model with a linear kernel 

svm_linear = SVC(kernel='linear') 

svm_linear.fit(X_train, y_train) 

# Predict the test set labels 

y_pred = svm_linear.predict(X_test) 

# Evaluate the model's accuracy 

accuracy = accuracy_score(y_test, y_pred) 

print(f'Linear SVM accuracy: {accuracy:.2f}') 

# Train an SVM model with a polynomial kernel 

svm_poly = SVC(kernel='poly', degree=3) 

svm_poly.fit(X_train, y_train) 

# Predict the test set labels 

y_pred = svm_poly.predict(X_test) 

# Evaluate the model's accuracy 

accuracy = accuracy_score(y_test, y_pred) 

print(f'Polynomial SVM accuracy: {accuracy:.2f}') 
