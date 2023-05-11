from sklearn import datasets 

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier, VotingClassifier 

from sklearn.svm import SVC 

from sklearn.linear_model import LogisticRegression 

# load sample dataset 

iris = datasets.load_iris() 

# split dataset into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3) 

# build individual models 

svc_model = SVC(kernel='linear', probability=True) 

rf_model = RandomForestClassifier(n_estimators=10) 

lr_model = LogisticRegression() 

# create ensemble model 

ensemble = VotingClassifier(estimators=[('svc', svc_model), ('rf', rf_model), ('lr', lr_model)], 

voting='soft') 

# train ensemble model 

ensemble.fit(X_train, y_train) 

# make predictions on test set 

y_pred = ensemble.predict(X_test) 

# print ensemble model accuracy 

print("Ensemble Accuracy:", ensemble.score(X_test, y_test))
