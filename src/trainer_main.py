import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src.train_lgbm import train_lgbm


class trainer_main:
    def __init__(self,args):
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.X_train,self.X_test,self.Y_train,self.Y_test = self.prepare_data()
        self.method = args.method
        self.train()


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
        if self.method == 'lgbm':
            self.trainer = train_lgbm(X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
            self.trainer.fit()
            self.trainer.accuracy()
