import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.train_lgbm import train_lgbm
from src.train_gbm import train_gbm
from src.train_svc import train_svc
from src.train_rf import train_rf
from src.train_sgd import train_sgd
from src.train_mlp import train_mlp
from joblib import dump, load
import os
import time
import numpy as np

class trainer_main:
    def __init__(self,args):
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.X_train,self.X_test,self.Y_train,self.Y_test = self.prepare_data()
        self.method = args.method
        self.export_folder = args.export_folder
        self.mode = args.mode
        self.model = args.model
        if self.mode == 'train':
            self.train()
            self.save_model()
        else:
            self.test()



    def prepare_data(self):
        """
        Prepares the input and validation data from the json file
        :return:
        """
        df_train_file = pd.read_json(self.train_file)
        df_test_file = pd.read_json(self.test_file)

        labels = df_train_file['cuisine']
        train_set = df_train_file['ingredients']
        test_set = df_test_file['ingredients']

        # get vocabulary
        train_set_words = [words for words in train_set]
        test_set_words = [words for words in test_set]

        train_text = [' '.join(sentence) for sentence in train_set_words]
        test_text = [' '.join(sentence) for sentence in test_set_words]

        self.vectorizer = TfidfVectorizer()

        Train_set = self.vectorizer.fit_transform(train_text)
        Test_set = self.vectorizer.transform(test_text)

        self.encoder = LabelEncoder()
        label_set = self.encoder.fit_transform(labels)

        # Generate training and validation files
        X_train,X_test,Y_train,Y_test =train_test_split(Train_set,label_set,test_size=0.2,shuffle=True,stratify=label_set)

        return X_train,X_test,Y_train,Y_test

    def train(self):
        self.init_trainer()

        self.trainer.fit()
        self.trainer.accuracy()
        #self.evaluate()

    def init_trainer(self):
        if self.method == 'lgbm':
            self.trainer = train_lgbm(X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
        elif self.method == 'gbm_sklearn':
            self.trainer = train_gbm(X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
        elif self.method == 'SVC':
            self.trainer = train_svc(X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
        elif self.method == 'RF':
            self.trainer = train_rf(X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
        elif self.method == 'SGD':
            self.trainer = train_sgd(X_train=self.X_train, X_test=self.X_test, Y_train=self.Y_train, Y_test=self.Y_test)
        elif self.method == 'MLP':
            self.trainer = train_mlp(X_train=self.X_train, X_test=self.X_test, Y_train=self.Y_train, Y_test=self.Y_test)
    def save_model(self):
        if self.method == 'lgbm':
            pass
        else:

            if os.path.isdir(self.export_folder) is False:
                os.mkdir(self.export_folder)

            st = str(int(time.time()))
            export_folder = os.path.join(self.export_folder,st)
            os.mkdir(export_folder)

            export_file = os.path.join(export_folder, self.method+'.joblib')

            dump(self.trainer.classifier,export_file)
            print('Model saved in ',export_file)

    def test(self):
        if self.method == 'lgbm':
            pass
        else:
            classifier = load(self.model)

    def evaluate(self):
        print('Training data analyse: ')
        self.print_errors(self.X_train,self.Y_train)

        print('Test data analyse: ')
        self.print_errors(self.X_test, self.Y_test)

    def print_errors(self, input_data, input_label):
        prediction = self.trainer.classifier.predict(input_data)
        probability_prediction = None
        try:
            probability_prediction = self.trainer.classifier.predict_proba(input_data)
        except:
            pass
        print('Errors : ')

        difference = np.where((prediction - input_label) != 0)[0]
        for i, diff_index in enumerate(difference):
            input_instance = self.vectorizer.inverse_transform(input_data[diff_index])
            target_instance = self.encoder.inverse_transform([input_label[diff_index]])
            predicted_instance = self.encoder.inverse_transform([prediction[diff_index]])
            train_text = [' '.join(word) for word in input_instance]
            if probability_prediction is not None:
                probabilities = probability_prediction[diff_index]
                top3_indices = (-probabilities).argsort()[:3]
                top3 = self.encoder.inverse_transform(top3_indices)
                top3_prob = probabilities[top3_indices]
                probs = [str(top_label) + '(' + str(np.round(top_percent,2)) + '%),' for top_label, top_percent in zip(top3, top3_prob)]
                prob_text = ''
                for word in probs:
                    prob_text = prob_text + word

                print('{}: {} : Target {} / Predicted {} '.format(i+1, train_text[0], target_instance[0],prob_text))
            else:
                print('{}: {} : Target {} / Predicted {} '.format(i + 1, train_text[0], target_instance[0],predicted_instance[0]))