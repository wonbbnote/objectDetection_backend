import hashlib
import json
from flask import Blueprint, jsonify, request
from pymongo import MongoClient

blue_signup = Blueprint("signup", __name__, url_prefix="/signup")

client = MongoClient('localhost', 27017)
db = client.dbobject

@blue_signup.route("/", methods=["POST"])
def sign_up():
    data = json.loads(request.data)

    password = data['password']
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    user_info = bool(db.users.find_one({"user_id" : data['user_id']}))
    if not user_info:
        doc = {
            'user_id' : data['user_id'],
            'password' : password_hash
        }
        db.users.insert_one(doc)
        return jsonify({'message' : 'success'})
    else:
        print("아이디 중복!!")
        return jsonify({'message' : 'fail'})