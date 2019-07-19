import json
import os

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Whats_cooking/Dataset/train.json')

train_data = None
with open(train_file) as json_file:
    train_data = json.load(json_file)

import pandas as pd
df_train_file = pd.read_json(train_file)

cuisines = pd.unique(df_train_file['cuisine'])
print('Cuisines found: \n ',cuisines)

for cuisine in cuisines:
    selective_cuisine = df_train_file[df_train_file['cuisine']==cuisine]
    print('Number of {} cuisine = {} \n'.format(cuisine,selective_cuisine.shape[0]))


test_file = os.path.normpath('/home/naraya01/AEN/GIT/Whats_cooking/Dataset/test.json')
df_test_file = pd.read_json(test_json)
print('Test file looks the following..\n')
for line in df_test_file.iterrows():
    print(line['ingredients'])
