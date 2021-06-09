online_inference
==============================
This project provides service for Heart Disease UCI classification problem (RESTApi). 

### Installation ###
docker pull muzaffarsoliyev/online_inference:1.0.0

### Run (tested on Windows) ###
docker run -p 5000:80 muzaffarsoliyev/online_inference:1.0.0

*-p 5000:80* is used to redirect port, so that you will have access to VM's 80-port via your local 5000 port.

Just run on your browser: 
http://127.0.0.1:5000/

### Usage ###
http://127.0.0.1:5000/predict 
with body, for example (json raw format): 
<pre>
{
  "data": [
    [
      63,1,3,145,233,1,0,150,0,2.3,0,0,1
    ],
    [
      37,1,2,130,250,0,1,187,0,3.5,0,0,2
    ],
    [
      60,1,0,130,253,0,1,144,1,1.4,2,1,3
    ]
  ],
  "features": [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
  ]
}</pre>

### Testing script usage (from root directory) ###
python api/make_request.py

### Image size optimization ###
- Exclusion of unnecessary file with .dockerignore
- Installation and copying only necessary files and libs
- Usage of base image (FROM python:3.6)
- Copying after running requirements installation
- Preference COPY over ADD

### Returned codes (predict get request) ###
- 200 - if request is correct
- 400 - wrong input
- 413 - payload too large (more that 20 rows of data)

----
See http://127.0.0.1:5000/docs for more information