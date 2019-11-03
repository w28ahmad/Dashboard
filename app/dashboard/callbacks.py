from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go

import pandas as pd

from webapp.app.config import BASEDIR
from utils.load_data import stock_data

# Import prediction assets
from dashboard.prediction import data_sequencer, time_sequencer
from webapp.app.ml_models.LSTM_model import uni_lstm # Univariant LSTM class 

def register_callbacks(dashapp):
    @dashapp.callback(
        Output(component_id="graph-prediction", component_property="figure"),
        [
            Input(component_id="input", component_property='value'), 
            Input(component_id="start_date", component_property="value"),
            Input(component_id="end_date", component_property='value'),
            Input(component_id="predict_amount", component_property='value'),
            Input(component_id="predict", component_property='value')
        ]
    )
    # dynamicly updating the graph from the name, start-date and end-date
    def Update_prediction(name, start, end, prediction_amount, is_predict):
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
                
            return {
                    'data':[{'x': df["Date"], 'y': df["Mid-Values"], 'type':'line','name':name},
                            {'x': graph_x, 'y': predictions, 'type':'line', 'name':'predictions'}],
                    'layout':{
                        'title': f'Stock Prices for {name}'
                        }
                    }
            
        except Exception as e:
            print("Error", e)
            df =  pd.DataFrame(columns=["Date", "Mid-Values"])
        
        
    @dashapp.callback(
        Output(component_id="graph-sentiment", component_property="figure"),
        [
            Input(component_id="input", component_property='value'), 
        ]
    )
    def update_sentiment(name):
        X = [1, 2, 3, 4, 5, 6, 7]
        Y = [2, 3, 5, 7, 9, 4, 7]
        Y2 = [3, 4, 5, -3, -2, -5, 6] 
        
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Sentiment',
                mode= 'lines',
                yaxis='y2',
                # line = dict(color = (app_colors['sentiment-plot']),
                            # width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                x=X,
                y=Y2,
                name='Volume',
                # marker=dict(color=app_colors['volume-bar']),
                )
        
        return {
                'data':[data, data2],
                'layout':go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                    yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right'),
                                    yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='sentiment'),
                                    title=f'Live sentiment for: {name}',
                                    # font={'color':app_colors['text']},
                                    # plot_bgcolor = app_colors['background'],
                                    # paper_bgcolor = app_colors['background'],
                                    showlegend=False
                                    )
                }