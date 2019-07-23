from sklearn import metrics
import numpy as np
from sklearn.neural_network import MLPClassifier

class train_mlp:
    def __init__(self,X_train,X_test,Y_train,Y_test,params=None):
        self.X_train,self.X_test,self.Y_train,self.Y_test = X_train,X_test,Y_train,Y_test

    def fit(self):
        self.classifier = MLPClassifier(hidden_layer_sizes=(512,256,64),random_state=0,verbose=True,learning_rate='adaptive',activation='relu',solver='sgd',
                                        learning_rate_init=0.1,momentum=0.95,nesterovs_momentum=True)
        self.classifier.fit(self.X_train,self.Y_train)

    def fit_transform(self):
        self.classifier = MLPClassifier(hidden_layer_sizes=(512,256,64),random_state=0,verbose=True,learning_rate='adaptive',activation='relu',solver='sgd',
                                        learning_rate_init=0.1,momentum=0.95,nesterovs_momentum=True)
        self.classifier.fit(self.X_train, self.Y_train)
        return self.classifier.predict(self.X_train)

    def validation_result(self):
        return self.classifier.predict(self.X_test)

    def accuracy(self):
        train_prediction = self.classifier.predict(self.X_train)
        accuracy = metrics.accuracy_score(self.Y_train, train_prediction)
        print('Training accuracy = ', accuracy)
        conf_matrix = metrics.confusion_matrix(self.Y_train, train_prediction)
        print('Confusion matrix =  \n', conf_matrix)

        test_prediction = self.classifier.predict(self.X_test)
        accuracy = metrics.accuracy_score(self.Y_test, test_prediction)
        print('Test accuracy = ', accuracy)
        conf_matrix = metrics.confusion_matrix(self.Y_test, test_prediction)
        print('Confusion matrix =  \n', conf_matrix)



    def transform(self, X_inference):
        return self.classifier.predict(X_inference)