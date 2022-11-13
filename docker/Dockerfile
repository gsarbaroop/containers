# FROM ubuntu:latest
FROM python:3.8.1
RUN mkdir exp
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./api /exp/api
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python3", "./plot_graphs.py"]