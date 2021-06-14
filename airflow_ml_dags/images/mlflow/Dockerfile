FROM python:3.6


RUN pip install mlflow && \
    pip install awscli --upgrade --user && \
    pip install boto3

ENV PORT 5000

COPY run.sh /

ENTRYPOINT ["/run.sh"]