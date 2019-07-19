import lightgbm as lgbm
from sklearn import metrics
import numpy as np

class train_lgbm:
    def __init__(self,X_train,X_test,Y_train,Y_test,params=None):
        self.X_train,self.X_test,self.Y_train,self.Y_test = X_train,X_test,Y_train,Y_test
        self.train_data, self.valid_data = self.prepare_data(X_train,X_test,Y_train,Y_test)
        self.lgbm_params = {}


    def prepare_data(self,X_train,X_test,Y_train,Y_test):
        train_data = lgbm.Dataset(X_train,label=Y_train)
        valid_data = train_data.create_valid(X_test,label=Y_test)

        print('Training and validation files generated')

        return train_data, valid_data

    def fit(self):
        self.lgbm_classifier = lgbm.train(self.lgbm_params,self.train_data,valid_sets=[self.valid_data])

    def fit_transform(self):
        self.lgbm_classifier = lgbm.train(self.lgbm_params, self.train_data, valid_sets=[self.valid_data])
        return self.lgbm_classifier.predict(self.X_train)

    def validation_result(self):
        return self.lgbm_classifier.predict(self.X_test)

    def accuracy(self):
        train_prediction = np.round(self.lgbm_classifier.predict(self.X_train))
        accuracy = metrics.accuracy_score(self.Y_train,train_prediction)
        print('Training accuracy = ',accuracy)
        conf_matrix = metrics.confusion_matrix(self.Y_train,train_prediction)
        print('Confusion matrix =  \n',conf_matrix)

        test_prediction = np.round(self.lgbm_classifier.predict(self.X_test))
        accuracy = metrics.accuracy_score(self.Y_test,test_prediction)
        print('Validation accuracy = ',accuracy)
        conf_matrix = metrics.confusion_matrix(self.Y_test, test_prediction)
        print('Confusion matrix =  \n', conf_matrix)


    def transform(self,X_inference):
        return self.lgbm_classifier.predict(X_inference)

    def set_params(self,params=None):
        learning_rate = 0.1
        objective = 'multiclass'
        metric = 'logloss'
        boosting = 'dart'
        num_iterations = 1000
        verbosity = 2
        self.lgbm_params = {'learning_rate':learning_rate,
                            'objective':objective,
                            'metric':metric,
                            'boosting':boosting,
                            'num_iterations':num_iterations,
                            'verbosity':verbosity,
                            'xgboost_dart_mode':True}