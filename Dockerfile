FROM python:3.11.5-slim-bullseye

WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./main.py /code/main.py
COPY ./musicmood.csv /code/musicmood.csv
COPY ./MoodModel.py /code/MoodModel.py

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]