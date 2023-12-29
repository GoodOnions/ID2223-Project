import os

LOCAL = False

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    labels = ["Low", "Medium", 'High']

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    print("downloading model")
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    print("model loaded successfully")

    feature_view = fs.get_feature_view(name="salary", version=3)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    # print(y_pred)
    offset = 1
    wine = y_pred[y_pred.size - offset]
    print("Wine predicted: " + labels[wine] + ' quality')

    wine_fg = fs.get_feature_group(name="salary", version=3)
    df = wine_fg.read()
    label = df.iloc[-offset]["quality"]

    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine quality Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now],
    }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job": False})

    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it -
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, 'wine_df_recent.png', table_conversion='matplotlib')
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./wine_df_recent.png", "Resources/images", overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    print("Number of different salary predictions to date: " + str(predictions.value_counts().count()))
    results = confusion_matrix(labels, predictions)

    df_cm = pd.DataFrame(results, ['True Low', 'True Medium', 'True High'],
                         ['Pred Low', 'Pred Medium', 'Pred High'])

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./wine_confusion_matrix.png")
    dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True:
        from dotenv import load_dotenv
        load_dotenv()
        g()
    else:
        g()
