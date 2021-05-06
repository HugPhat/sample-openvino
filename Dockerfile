FROM python:3.7.5-slim

RUN apt-get update && apt-get -y install g++

COPY ./requirements.txt /app/requirements.txt 

WORKDIR /app 

RUN pip3 install -r requirements.txt 

COPY . /app 

CMD  [ "sh", "product.sh"]