# Motivation
This machine learning dashboard showcases the applications of various machine learning models. It includes an AI trader that predicts stock outcomes and real-time Twitter sentiment analysis. Additionally, it offers object detection capabilities for real-time object detection.

- The AI trader module of the application uses a bidirectional LSTM model, which is a state-of-the-art deep learning technique for sequence prediction, making it a cutting-edge tool for stock analysis.
- The sentiment analysis for the AI trader module uses a real-time stream of tweets, processed using the Twitter API and Python's Tweepy library, demonstrating the application of NLP and social media analysis in finance.
- The object detection module of the application uses the YOLO (You Only Look Once) algorithm, which is a popular deep learning technique for real-time object detection, making it a valuable tool for security and surveillance.
- The application uses various cloud-based technologies for scalability and performance, including MongoDB for NoSQL data storage, Plotly for interactive data visualization, and AWS Lambda for serverless computing.

# Tools Used
The application was built using Python's Flask Framework and Plotly for clean visualizations of stock and ticker prices. It offers a secure login page for authentication and detailed explanations of how the models were trained and how they are applied to each application.


# Environment Variables

```text
quandle_key

SQLALCHEMY_USERNAME
SQLALCHEMY_PASSWORD
LOGIN_MANAGER_SECRET_KEY

# twitter
CONSUMER_KEY
CONSUMER_SECRET
ACCESS_TOKEN
ACCESS_TOKEN_SECRET


MONGODB_URL
```

# Run the code

```bash
python3 app/main.py
```

# Login page
The dashboard offers a secure login page for authentication.


![](/assets/Login.png)

# Stock Dashboard
The dashboard provides a real-time analysis of the AMZN stock, including the actual vs. predicted price with a one-day prediction period. It allows users to select specific stock tickers and has real-time sentiment and price analysis, along with a one-day prediction.

![](/assets/Dashboard2.png)

# Object Detection Dashboard
The dashboard uses YOLO object detection models to detect various objects in an image. Users can upload images and have certain objects in those images highlighted. The object detection dashboard offers clean visualizations, as shown in the included images

![](/assets/cnnDashboard1.png)
![](/assets/cnnDashboard2.png)
