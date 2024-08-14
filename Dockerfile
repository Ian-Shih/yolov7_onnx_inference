#docker image environment
FROM nvcr.io/nvidia/pytorch:21.08-py3

#workspace folder:app
WORKDIR /app

# apt install required packages
RUN apt-get update 
RUN apt-get install -y zip htop screen libgl1-mesa-glx

# pip install required packages
RUN pip install seaborn thop

#install requirements
COPY . /app/
RUN pip install -r requirements.txt


