from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__app__)
app.config['SECRET_KEY'] = '31f214ac7307802de7160100ec7a549b'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

from server import controllers
