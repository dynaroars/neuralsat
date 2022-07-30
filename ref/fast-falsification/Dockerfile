# Dockerfile for FFN
#
# To build image:
# sudo docker build . -t ffn_image
#
# To get a shell after building the image:
# sudo docker run -i -t ffn_image bash

FROM ubuntu:20.04
COPY ./requirements.txt /FFN/requirements.txt
WORKDIR /FFN
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
COPY . /FFN
ENV PYTHONPATH=$PYTHONPATH:/FFN
