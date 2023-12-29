import gradio as gr
import hopsworks
import joblib
import pandas as pd

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
labels = ["Low", "Medium", "High"]

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")


def wine(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
         total_sulfur_dioxide, density, pH, sulphates, alcohol, white) -> str:
    print("Calling function")
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
         total_sulfur_dioxide, density, pH, sulphates, alcohol, white]], columns=features)
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    #     print("Res: {0}").format(res)
    print(res)

    return f"{labels[res[0]]} quality"


demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with salary characteristics to get the salary quality (low, medium, high)",
    allow_flagging="never",
    inputs=[
        gr.components.Number(label='fixed acidity'),
        gr.components.Number(label='volatile acidity'),
        gr.components.Number(label='citric acid'),
        gr.components.Number(label='residual sugar'),
        gr.components.Number(label='chlorides'),
        gr.components.Number(label='free sulfur dioxide'),
        gr.components.Number(label='total sulfur dioxide'),
        gr.components.Number(label='density'),
        gr.components.Number(label='pH'),
        gr.components.Number(label='sulphates'),
        gr.components.Number(label='alcohol'),
        gr.components.Checkbox(label='is white'),
    ],
    outputs=gr.Text())

demo.launch(debug=True)