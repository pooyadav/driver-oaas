__author__ = "Poonam Yadav"
__copyright__ = "Copyright 2007, The Databox Project"
__email__ = "p.yadav@acm.org"

import urllib3
import os
import json
from flask import Flask
import ssl



app = Flask(__name__)

@app.route("/ui")
def hello():
    return "Hello from OAAS World! OAAS APP Server is started in backgroud now!"

if __name__ == "__main__":
     print("OAAS APP Server is starting in backgroud now!")
     app.run(host='0.0.0.0', port=8080)
