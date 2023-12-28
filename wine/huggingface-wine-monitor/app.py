import gradio as gr
import hopsworks

labels = ['Low', 'Medium', 'High']

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/images/wine_df_recent.png")
dataset_api.download("Resources/images/wine_confusion_matrix.png")

monitor_fg = fs.get_or_create_feature_group(name="wine_predictions", version=1, primary_key=["datetime"],
                                            description="Wine quality Prediction/Outcome Monitoring")

history_df = monitor_fg.read()
last_prediction = history_df.tail(1)
last_prediction = last_prediction.to_dict(orient='records')[0]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predicted")
            gr.Label(f"{labels[last_prediction['prediction']] + ' quality' if last_prediction is not None else 'No predictions yet'}")
        with gr.Column():
            gr.Label("Today's Actual quality")
            gr.Label(f"{labels[int(last_prediction['label'])] + ' quality' if last_prediction is not None else 'No predictions yet'}")
    with gr.Row():
        with gr.Column():
            gr.Label("Recent Prediction History")
            gr.Image("wine_df_recent.png", elem_id="recent-predictions")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            gr.Image("wine_confusion_matrix.png", elem_id="confusion-matrix")

demo.launch()
