FROM python:3.10.5-bullseye
WORKDIR /app
COPY req.txt req.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install -r req.txt
COPY main.py main.py
CMD [ "uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5000", "--reload"]

