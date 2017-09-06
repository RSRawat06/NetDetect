# Botnet Detection without Feature Engineering
We apply recurrent neural networks with attention mechanisms to detect hosts infected by botnets in an enterprise network, without using any engineered features beyond basic packet properties.

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

### Overview
We capture network packets (.pcap) from enterpise networks (currently supported datasets: ISOT, ISCX). We sort the packets into network flows. We create data points for each host (which is either infected/benign), where each point is a 3-dimensional tensor: a sequence of network flows involving the IP, a sequence of packets composing each network flow, and a feature vector representing each packet. To attempt to learn encoding functions that generalize across botnet species, even previously unseen species, we limit ourselves to basic out-of-the-box packet features and avoid engineered features and heuristics.

We employ an end-to-end differentiable recurrent neural network with attention mechanisms, to encode network flows and then predict based off of packet history whether a host has been infected. A simple RNN cell (LSTM/GRU) is used to encode sequences of packets, while sequences of flows are first encoded through RNN cells and then passed through a self-attention mechanism. The resulting vector is passed through a dense layer to obtain the prediction.

# Goals
* Achieve close to state-of-art scores without using a single engineered feature
* Prove or invalidate my assumption that neural network architectures designed for natural language also carry over to network logs
* Prove that our network can learn an encoding function that generalizes across botnets and potentially even across multiple network-related tasks

### Installation
To install, you will need the newest versions of Docker CE and Docker Compose.
To initialize Docker, `cd` to the root of this project and:

```
docker-compose up --build
bash access_cluster.sh
```

To download default datasets for ISOT and ISCX, run:

```
python3 -m botnet_attention.isot.download
python3 -m botnet_attention.iscx.download
```

To initialize training on ISOT and ISCX datasets (you will need to have downloaded them), run:

```
python3 -m botnet_attention.isot.train
python3 -m botnet_attention.iscx.train
```

### What is included
* Download scripts and source for ISOT and ISCX datasets
* Network flow segmentation and basic packet feature preprocessing
* Baseline featurization module to extract engineered features to compare against our feature-engineering-free network. 
* Different recurrent neural network architectures implemented in Tensorflow

### Code breakdown
`botnet_attention` module:
* This module is for downloading the default datasets and training our models on them

`flow_featurization` module:
* This module is for generating the default ISCX/ISOT datasets that can be downloaded as detailed in `Installation`
* Unless you intend on adding a new dataset or modifying the preprocessing procedures for the default ISCX and ISOT datasets, you should not need to use this module.

### Contributions
Feel free to make a pull request if you have any suggestions.
Contributors are listed under `contributors.txt`.

### Special thanks
* Dr. Bunn from Caltech's CACR for mentoring
* Microsoft for the $5,000 Azure Research Award
* hmishra2250 for his open source flow featurization module upon which our baseline featurization module is based.

# License
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

