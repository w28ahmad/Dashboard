FROM python:3.6
COPY utils/* /utils/
ADD LSTM.py loadData.py SeqDataGenerator.py requirements.txt usedImports.py graphs.py  /
RUN pip install --default-timeout=1000 -r requirements.txt
CMD [ "python", "./LSTM.py" ]
