import os

LOCAL = False

features = ['fixed_acidity',
            'volatile_acidity',
            'citric_acid',
            'residual_sugar',
            'chlorides',
            'free_sulfur_dioxide',
            'total_sulfur_dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol',
            'is_white']
labels = ['Low', 'Medium', 'High']

def generate_wine(label: str):
    import pandas as pd
    import random

    wine_df = pd.read_csv(
        "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv")
    wine_df['is_white'] = wine_df['type'] == 'white'
    wine_df['is_white'].fillna(False)
    wine_df['is_white'] = wine_df['is_white'].astype('int')
    wine_df = wine_df.drop('type', axis=1)
    wine_df = wine_df.dropna()
    wine_df['quality'] = wine_df['quality'].apply(lambda q: 0 if q <= 5 else 1 if q < 7 else 2)
    wine_df['quality'].value_counts().sort_index().plot.bar()
    wine_df.columns = wine_df.columns.str.replace(' ', '_')

    quality = labels.index(label)
    wine_df = wine_df[wine_df['quality'] == quality]
    new_entry = {}

    for feature in features:
        new_entry[feature] = [random.uniform(wine_df[feature].min(), wine_df[feature].max())]

    df = pd.DataFrame(new_entry)
    df['is_white'] = df['is_white'].apply(lambda x: 1 if x > 0.5 else 0)
    df['quality'] = quality

    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random
    import math

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = math.floor(random.uniform(0, 3))

    wine_df = generate_wine(labels[int(pick_random)])
    print(f"{labels[int(pick_random)]} added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="salary", version=3)
    wine_fg.insert(wine_df)


if __name__ == "__main__":
    if LOCAL:
        from dotenv import load_dotenv
        load_dotenv()
        g()
    else:
        g()
