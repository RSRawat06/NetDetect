# NetDetect: Botnet Detection without Feature Engineering
We apply recurrent neural networks with attention mechanisms to detect hosts infected by botnets in an enterprise network, without using any engineered features beyond basic packet properties.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

### Overview
We capture network packets (.pcap) from enterpise networks (currently supported datasets: ISOT, ISCX). We sort the packets into network flows. We create data points for each host (which is either infected/benign), where each point is a 3-dimensional tensor: a sequence of network flows involving the IP, a sequence of packets composing each network flow, and a feature vector representing each packet. To attempt to learn encoding functions that generalize across botnet species, even previously unseen species, we limit ourselves to basic out-of-the-box packet features and avoid engineered features and heuristics.

We employ an end-to-end differentiable recurrent neural network with attention mechanisms, to encode network flows and then predict based off of packet history whether a host has been infected. A simple RNN cell (LSTM/GRU) is used to encode sequences of packets, while sequences of flows are first encoded through RNN cells and then passed through a self-attention mechanism. The resulting vector is passed through a dense layer to obtain the prediction.

# Goals
* Achieve close to state-of-art scores without using a single engineered feature
* Prove or invalidate my assumption that neural network architectures designed for natural language also carry over to network logs
* Prove that our network can learn an encoding function that generalizes across botnets and potentially even across multiple network-related tasks

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
cd template
bash access_template.sh
```

Now you should be inside the template container.
```
service neo4j start
python3 -m template.datasets.generic_double_seq.download
python3 -m template.datasets.generic_flat.download
python3 -m template.datasets.generic_sequential.download
```

Go into Neo4j by visting 0.0.0.0:7474 on your local browser.
Run the following queries: 
```
CREATE INDEX ON :Resource(uri)
CALL semantics.importRDF('file:///template/datasets/apple.owl','RDF/XML', {})
```

Now run unit tests to make sure everything is awesome.
```
py.test template/tests
```

### Usage:
Let's get botnet detection training up and running.
```
python3 -m template.datasets.iscx.download
python3 -m template.src.main.train
```
In a seperate window, simultaneously, run:
```
cd /template/src/main
bash run_tensorboard.sh
```

### Special thanks
* Dr. Bunn from Caltech's CACR for mentoring
* Microsoft for the $5,000 Azure Research Award
* hmishra2250 for his open source flow featurization module upon which our baseline featurization module is based.

### Contribute
I appreciate all contributions. Just make a pull request.
Contributors are listed under `contributors.txt`.

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

