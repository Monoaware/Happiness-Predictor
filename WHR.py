import gradio as gr
import pickle
import pandas as pd

# Load model:
with open("WHR_LifeLadder_Regression_Model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler:
with open("WHR_KNN_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def predict(logGDP,
            healthyLifeExp,
            deliveryQuality,
            socialSupport,
            democraticQuality,
            freedom,
            perceptionsCorr,
            positiveAffect,
            generosity,
            giniHousehold,
            negativeAffect):

    input_df = pd.DataFrame([[
            deliveryQuality,
            democraticQuality,
            freedom,
            generosity,
            healthyLifeExp,
            logGDP,
            negativeAffect,
            perceptionsCorr,
            positiveAffect,
            socialSupport,
            giniHousehold
    ]],
            columns=[
            "Delivery Quality",
            "Democratic Quality",
            "Freedom to make life choices",
            "Generosity",
            "Healthy life expectancy at birth",
            "Log GDP per capita",
            "Negative affect",
            "Perceptions of corruption",
            "Positive affect",
            "Social support",
            "gini of household income reported in Gallup, by wp5-year"])

    scaled_array = scaler.transform(input_df)
    scaled_inputs = pd.DataFrame(scaled_array, columns=input_df.columns)

    prediction = model.predict(scaled_inputs)

    return round(prediction[0], 3)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(label="Log GDP per capital", value=1, minimum=0, maximum=13, step=0.5),
        gr.Slider(label="Healthy life expectancy at birth", value=60, minimum=40, maximum=80, step=1),
        gr.Slider(label="Delivery Quality", value=0, minimum=-1.9, maximum=2.18, step=0.01),
        gr.Slider(label="Social support", value=0.5, minimum=0, maximum=1, step=0.1),
        gr.Slider(label="Democratic Quality", value=0, minimum=-2, maximum=1.5, step=0.01),
        gr.Slider(label="Freedom to make life choices", value=0.5, minimum=0, maximum=1, step=0.1),
        gr.Slider(label="Perceptions of corruption", value=0.5, minimum=0, maximum=1, step=0.1),
        gr.Slider(label="Positive affect", value=0.7, minimum=0, maximum=1, step=0.1),
        gr.Slider(label="Generosity", value=0, minimum=-0.35, maximum=0.5, step=0.01),
        gr.Slider(label="gini of household income reported in Gallup, by wp5-year", value=0.5, minimum=0.2, maximum=0.8, step=0.1),
        gr.Slider(label="Negative affect", value=0.7, minimum=0, maximum=1, step=0.1),
    ],
    outputs=gr.Textbox(label="Life Ladder (Happiness Score)"),
    title="KNN WHR Life Ladder Predictor",
    description="Drag the sliders to see how different feature values affect the happiness score!"
)

demo.launch(share=True)
