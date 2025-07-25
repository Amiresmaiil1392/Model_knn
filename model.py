#We have to load the libraries we need first, we use the Sklear library.
import sklearn 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#loading iris dataset
iris = load_iris()
#We divide our data into two categories of features and tags.
X = iris.data
y = iris.target
#In this step, you need to prepare your data for training and testing the model, and we use the famous method train_test_split to share the data.
X_train,X_test,y_train,y_test= train_test_split(
    X,y, #We use the X and Y variables to split the data, which are the same attributes and tags.
    test_size = 0.3, #That is, we load 0.3 out of 150 columns for testing, ie 45 columns.
    random_state = 1 #It doesn't matter if the number is correct!
)
#In this step, we have to load our model. We use the KNN algorithm, which is also one of the famous algorithms for categorizing.
model_knn = KNeighborsClassifier(
    n_neighbors = 3 #In this variable, specify the number of close neighbors.
)
#At this point we need to train our model with the data.
model_knn.fit(X_train,y_train)
#At this point we need to predict the accuracy of the model with the test data.
y_pread = model_knn.predict(X_test)
#Now we need to know the accuracy of our model, so we find our model accuracy based on test labels and print it.
print("Accuracy:", metrics.accuracy_score(y_test,y_pread))
#We can display the performance of the model on the chart using the Matplotlib framework:
import matplotlib.pyplot as plt
plt.title("Model prediction:(green)")
plt.plot(y_pread,c = "g")
plt.show()
plt.title("Real prediction:(red)")
plt.plot(y_test,c = "r")
plt.show()
#You can also save or even load your model with the joblib framework.
from joblib import dump
dump(model_knn ,'Model_KNN_for_iris.joblib')
#To load:
#from joblib import load
#model_loading = load("Model.joblib")