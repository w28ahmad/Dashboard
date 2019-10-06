FROM ubuntu:18.10

LABEL maintainer="Wahab Ahmad <wahabahmad710961@gmail.com>"

RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip
# RUN python3 -m ensurepip --upgrade
RUN pip3 install --upgrade setuptools
RUN mkdir ./app

COPY  static templates .env app.py login.py requirements.txt setting.py ./app/
ADD utils ./app/utils
WORKDIR ./app

RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python3" ]

EXPOSE 8050
CMD [ "app.py"]