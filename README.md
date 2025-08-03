# Happiness-Predictor
My first independent attempt at working through the entire ML workflow. The models in this project were trained on the 2018 World Happiness Report dataset. A GUI is included for the KNN mode, created using Gradio.

**ML PIPELINE:**

Check out "Happiness Predictor Report.pdf" or "DefineAndSolveMLProblem.ipynb" (for the actual Jupyter notebook file) to see how I prepared my data and trained my models!

For a user-friendly experience, make sure to run "WHR.py" with the two pickle files ("WHR_KNN_scaler.pkl" and "WHR_LifeLadder_Regression_Model.pkl") under the same project folder to make predictions with my KNN model.

**IMPORTANT INFORMATION & CONSIDERATIONS:**

This model was trained on the 2018 World Happiness Report dataset (https://www.worldhappiness.report/ed/2018/). The raw data can be found in the CSV file.

I have not yet evaluated the model's performance on more recent data, the implication being that the model may not make relevant predictions for Life Ladder scores for later years. Considering the rapidly evolving world around us, this is highly likely to be the case. In order to resolve this issue, a simple solution would be to retrain the models on newer datasets, experimentally determining how far back in time we want our datasets to go.

Notice that when running the KNN model, if you make a prediction on a feature with negative correlation to the label (i.e. perceptions of corruption), the Life Ladder score may actually increase as opposed to decrease, which would be the expected behavior. Or, you may experience that incrementing certain sliders have little to no effect on the overall Life Ladder score. I hypothesize that this is due to the fact that KNN models are instance-based ML algorithms: because they don't rely on weights for training, but rather proximity to training examples, they are unable to capture the full extent of direct feature-label relationships. 

A way to improve the KNN model would have been to retrain the model on the training + validation data splits after testing it in order to ensure that the model has as many examples available to it as possible during prediction time. In contrast, I expect that the Random Forest Regressor would not exhibit this issue compared to the KNN model because it uses decision trees, which are weight-based, and can therefore capture more complex feature interactions. However, this is something that I need to determine experimentally.


