# sample-openvino

```
usage: app.py [-h] [--age_gender] [--person_reid]

optional arguments:
  -h, --help     show this help message and exit
  --age_gender   run age gender model only
  --person_reid  run person reid only
No arguments: Run 2models
```

* get current result: **url/result [GET]** 

* Docker, to add args to app.py, go to production.sh and set. Some commands
  * build: docker build -t vino . 
  * run: docker run -dp 3000:80  vino