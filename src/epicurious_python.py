import os
import pandas as pd


file_path = os.path.normpath('../Dataset/full_format_recipes.json')
df_recipes = pd.read_json(file_path)

categories = df_recipes['categories']
#unique_categories = pd.unique(all_categories)[0]

all_categories = [cat for cat in categories]
print(len(all_categories))
category_string = []
for i,cat in enumerate(all_categories):
    if pd.isna(cat) is True:
        continue

    for cat_string in cat:
        category_string.append(cat_string)

unique_strings = pd.unique(category_string)
print(unique_strings)