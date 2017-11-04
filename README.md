# Occupancy-as-a-Service

Occupancy-as-a-Service App 

Clone this repo and build 

docker build -t  driver-oaas .

and 

docker run --name driver-oaas -i -t driver-oaas 

and

docker exec -t -i driver-oaas /bin/bash

Once you are login

cd src

python3 2A.py


## Info about setup

This docker image is installed with python 3.5 with following packages.

bleach (1.5.0)

certifi (2017.7.27.1)

click (6.7)

enum34 (1.1.6)

Flask (0.12.2)

html5lib (0.9999999)

itsdangerous (0.24)

Jinja2 (2.9.6)

Markdown (2.6.9)
MarkupSafe (1.0)
numpy (1.13.1)
olefile (0.44)
opencv-python (3.3.0.9)
pandas (0.20.3)
Pillow (4.2.1)
pip (9.0.1)
protobuf (3.4.0)
ptyprocess (0.5.2)
pycurl (7.43.0)
python-dateutil (2.6.1)
pytz (2017.3)
scikit-learn (0.19.0)
scipy (0.19.1)
setuptools (36.3.0)
simplegeneric (0.8.1)
six (1.10.0)
sklearn (0.0)
tensorflow (1.3.0rc2)
tensorflow-tensorboard (0.1.5)
urllib3 (1.21.1)
Werkzeug (0.12.2)
wheel (0.29.0)

## To - do

Currently after build, image size is 2.2 GB, so need to look what is the best way to reduce the size.

