FROM python:3.12

ENV UID=1000

RUN useradd -ms /bin/bash "python"

WORKDIR /home/python/app
COPY requirements.txt /home/python/app

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
        libpcre3 \
        libpcre3-dev \
        libgl1-mesa-dev \
        libglib2.0-0 \
        uwsgi-plugin-python3 \
    && pip3 install -r requirements.txt \
    && apt-get remove -y --purge make gcc build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY app /home/python/app/
RUN apt update -y


ARG APPVERSIONARG=NA
ENV APPVERSION=$APPVERSIONARG

EXPOSE 8000
# Run app_ws.py when the container launches
USER python
# Run app.py when the container launches
#CMD ["python3", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

