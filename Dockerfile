FROM python:3.9-buster
WORKDIR /lang_class
COPY requirements.txt /lang_class 
RUN pip install -r requirements.txt
COPY . /lang_class
CMD FLASK_ENV=development FLASK_APP=app.py flask run --host=0.0.0.0 
