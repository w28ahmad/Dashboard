from dash.dependencies import Input, Output
import pandas as pd

from webapp.app.config import BASEDIR
from utils.load_data import stock_data

# Import prediction assets
from dashboard.prediction import data_sequencer, time_sequencer
from webapp.app.ml_models.LSTM_model import uni_lstm # Univariant LSTM class 

def register_callbacks(dashapp):
    @dashapp.callback(
        Output(component_id="graph", component_property="figure"),
        [
            Input(component_id="input", component_property='value'), 
            Input(component_id="start_date", component_property="value"),
            Input(component_id="end_date", component_property='value'),
            Input(component_id="predict_amount", component_property='value'),
            Input(component_id="predict", component_property='value')
        ]
    )
    # dynamicly updating the graph from the name, start-date and end-date
    def new_df(name, start, end, prediction_amount, is_predict):
        try:
            df = stock_data(start, end, name, BASEDIR)
            df["Mid-Values"] = (df["High"]+df["Low"])/2
            df.reset_index(inplace=True)
            
            graph_x, predictions = [], []
            
            # Show predictions, # Add the prediction function here
            if len(is_predict) and prediction_amount and len(prediction_amount):
                num_prediction = int(prediction_amount)
                
                # lstm prediction
                lstm = uni_lstm(df['Mid-Values'].values[::-1], True,
                        TRAIN_SPLIT=210, EPOCHS=100,
                        BATCH_SIZE=10, BUFFER_SIZE=20,
                        EVALUATION_INTERVAL=5, seed=13, 
                        univariate_past_history=20,
                        univariate_future_target=0,)
                
                data = df['Mid-Values'].values[::-1]
                
                norm_data = lstm.normalize(data)
                
                 # Get prediction x-values
                prediction_x = data_sequencer(norm_data, num_prediction)
                
                
                # Get the graph x-values(timestamp) for those predictions
                graph_x = time_sequencer(df['Date'].values[::-1], num_prediction)
                # print('prediction_x')
                
                
                # predict using prediction x_values
                predictions = lstm.predict(prediction_x)
                
                # Unnormalize the predictions
                predictions = lstm.unnormalize(data, predictions.reshape(1, -1)[0])
                
        except Exception as e:
            print("Error", e)
            df =  pd.DataFrame(columns=["Date", "Mid-Values"])
        
        return {
                'data':[{'x': df["Date"], 'y': df["Mid-Values"], 'type':'line','name':name},
                        {'x': graph_x, 'y': predictions, 'type':'line', 'name':'predictions'}],
                'layout':{
                    'title': f'Stock Prices for {name}'
                    }
                }