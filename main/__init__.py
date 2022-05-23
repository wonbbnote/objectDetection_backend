from flask import Flask
from flask_cors import CORS
from . import signup
from . import login

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

app.register_blueprint(signup.blue_signup)
app.register_blueprint(login.blue_login)