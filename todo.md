# Todo:

- Research CNN's
- implement darkflow object detection
- Create a multi-variant lstm
- make sure the lstm univariate model is picked from the right place
- make the prediction n value dynamic

* Fix all the prediction dashboard bug

  - start date is wierd
  - prediction checkbox and input combo is specific to the its order

* Optimize lstm model

  - Research what Batch_size, Buffer_size and evaluation interval accomplish
  - Train a multi-variant lstm
  - Fix model.h5 location problem

- update the docker file to encorporate the new changes

- Updating the dockerfile

- Continous stock streaming into a db or a program

- Research automated investing into stocks

- following multiple stocks

# Done:

- CNN Image upload

- Create a cnn page
- predicting when to swich to different stocks
- for the sentiment analysis make sure the env variables are accessed from the right place
  - i.e app/sentiment_analysis/twitterConn.py
- Unhardcode the num_tweets variable
- Make the nlp dashboard localized - meaning that there is a button to retrain the sentiments, with a function call
- Add react to the project
  - Add a sidebar...?
- Improve the quality of the graphs And the overall dashboard
- Refine NLP data on the dashboard to remove 0.0 vals
- Twitter streamer to mongo. Turn the data into insert many, instead of insert one.
- Implement sentament analysis graphs using the twitter api
- preprocess text data in realtime
- save the data in mongodb in realtime, making sure that the data is not already there

- Find a way to get more data from qandle,

- Train and tune a better model

- show the predicted points on the graph

- fix lstm constant values

- create a sequencer - that sequences n sets of m points that are need to make the m+1'th prediction and these predictions
  are made n times for each one of the n sets,
  Note, another function for the dates of each of the n-set predictions is needed. Because we need the dates for the x values

* radio button callback for prediction points

* Find a way to abstract the dash app

  - usefull urls
  - https://medium.com/@olegkomarov_77860/how-to-embed-a-dash-app-into-an-existing-flask-app-ea05d7a2210b

* merge branch to model

* Add the new lstm-feature branch to github

* Create an lstm class:

  - find a way to save the model

* flask-web app able to search for your desired stock from a search bar.

* secure the login page and connect it to the dashboard

* Create a login Page

* find a way to import models to blueprints without having a circular import

* Get the flask app running on docker

* continous visualization of streamed data through a web graph--> dash

* Integrate dash app with flask

* login page html/css

# Commands:

- docker build -t app:latest .
- docker run -d -p 5000:5000 app:latest

- http://0.0.0.0:8050/login/

# Dash good reference examples

- https://dash-gallery.plotly.host/Portal/
