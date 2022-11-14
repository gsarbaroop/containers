# FROM ubuntu:latest
FROM python:3.8.1
RUN mkdir exp
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./svm_gamma=0.001_C=2.joblib /exp/svm_gamma=0.001_C=2.joblib
COPY ./api /exp/api
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
ENV FLASK_APP=exp/api/api.py
# RUN python3 exp/plot_graphs.py
WORKDIR /exp
CMD ["flask","--app", "api/api", "run"]
EXPOSE 5000/tcp
# CMD ["python3","./plot_graphs.py"]