# NetDetect: Botnet Detection without Feature Engineering
NetDetect applies recurrent neural networks to detect devices infected by botnets in an enterprise-size network.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

### Requirements
* Docker-CE version 17.06.2-ce
* Docker Compose version 1.14.0

### Getting Started
To install, hop into Docker and install the necessary datasets.
```
docker-compose up --build
```

Now hop into Docker and download some files.
```
cd NetDetect
bash access_NetDetect.sh
```

Now you should be inside the NetDetect container.
```
service neo4j start
```

Now run unit tests to make sure everything builds ok.
```
py.test NetDetect/tests
```

### Usage:
Let's get botnet detection training up and running.
```
python3 -m NetDetect.datasets.iscx.download
python3 -m NetDetect.src.main_iscx.train
```
In a seperate window, simultaneously, run:
```
cd /NetDetect/src/main_iscx
bash run_tensorboard.sh
```

### Special thanks
* Dr. Bunn from Caltech's CACR for mentoring
* Microsoft for the $5,000 Azure Research Award
* hmishra2250 for his open source flow featurization module upon which our baseline featurization module is based.

### Contribute
I appreciate all contributions. Just make a pull request.
Contributors are listed under 'contributors.txt'.

## License
MIT License

Copyright (c) 2017 Eric Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

