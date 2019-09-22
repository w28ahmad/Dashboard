FROM python:3
ADD LSTM.py loadData.py SeqDataGenerator.py  /
RUN pip install requirements.txt
CMD [ "python", "./LSTM.py" ]