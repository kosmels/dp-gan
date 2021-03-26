FROM tensorflow/tensorflow:2.4.1-gpu

# install opencv required libs
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-dev

WORKDIR /dp-gan

COPY ./requirements.txt /dp-gan/requirements.txt

RUN pip install -r requirements.txt
