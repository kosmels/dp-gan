FROM tensorflow/tensorflow:2.3.1

# install opencv required libs
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /dp-gan

COPY ./requirements.txt /dp-gan/requirements.txt

RUN pip install -r requirements.txt
