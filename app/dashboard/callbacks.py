from dash.dependencies import Input, Output
import pandas as pd

from webapp.app.config import BASEDIR
from utils.load_data import stock_data


def register_callbacks(dashapp):
    @dashapp.callback(
        Output(component_id="graph", component_property="figure"),
        [
            Input(component_id="input", component_property='value'), 
            Input(component_id="start_date", component_property="value"),
            Input(component_id="end_date", component_property='value'),
            Input(component_id="predict", component_property='value')
        ]
    )
    # dynamicly updating the graph from the name, start-date and end-date
    def new_df(name, start, end, is_predict):
        print(is_predict)
        try:
            df = stock_data(start, end, name, BASEDIR)
            df["Mid-Values"] = (df["High"]+df["Low"])/2
            df.reset_index(inplace=True)
            
            # Show predictions, # Add the prediction function here
            if len(is_predict):
                print('hello')
        except:
            print("Stock Name not found")
            df =  pd.DataFrame(columns=["Date", "Mid-Values"])
        
        return {
                'data':[{'x': df["Date"], 'y': df["Mid-Values"], 'type':'line','name':name},],
                'layout':{
                    'title': f'Stock Prices for {name}'
                    }
                }