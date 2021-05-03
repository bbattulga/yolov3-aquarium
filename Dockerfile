FROM python:3.8.0

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY trainer/ .

CMD ["python", "./trainer/task.py"]