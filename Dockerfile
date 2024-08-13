FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /app

RUN apt-get update 
RUN apt-get install -y zip htop screen libgl1-mesa-glx

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# 安装额外的 Python 包
RUN pip install seaborn thop

# 复制其余代码
COPY . /app/
