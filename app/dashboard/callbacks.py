from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go

import pandas as pd

from webapp.app.config import BASEDIR
from utils.load_data import stock_data

# Import prediction assets
from dashboard.prediction import data_sequencer, time_sequencer, sentiment_sequencer
from webapp.app.ml_models.LSTM_model import uni_lstm # Univariant LSTM class

# Sentiment Imports
from webapp.app.sentiment_analysis.twitterConn import twitter_sentiment

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
            Input(component_id="numberOfSentiments", component_property='value'),
        ]
    )
    def update_sentiment(name, numberOfSentiments):
        num_tweets = int(numberOfSentiments) if numberOfSentiments else 0
        print(num_tweets)

        # update mongoDB
        if num_tweets:
            filter_strings = [str(name)]
            sentiment = twitter_sentiment(filter_strings, num_tweets)
            sentiment.stream()
            
        data = sentiment_sequencer()
        
        X = data['Date']
        Y = data['polarity']
        Y2 = data['subjectivity']        
        
        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Polarity',
                mode= 'lines',
                yaxis='y2',
                # line = dict(color = (app_colors['sentiment-plot']),
                            # width = 4,)
                )

        data2 = plotly.graph_objs.Bar(
                base=0,
                x=X,
                y=Y2,
                name='Subjectivity',
                # marker=dict(color=app_colors['volume-bar']),
                )
        
        return {
                'data':[data, data2],
                'layout':go.Layout(xaxis=dict(range=[min(X),max(X)], type='category'),
                                    yaxis=dict(range=[-1,1], title='Subjectivity', side='right'),
                                    yaxis2=dict(range=[-1, 1], side='left', overlaying='y',title='Polarity'),
                                    title=f'Live sentiment for: {name}',
                                    # font={'color':app_colors['text']},
                                    # plot_bgcolor = app_colors['background'],
                                    # paper_bgcolor = app_colors['background'],
                                    showlegend=False
                                    )
                }