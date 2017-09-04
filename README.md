# Botnet Attention
Botnet detection in enterprise networks using recurrent neural networks with attention mechanisms (prototype: 6/1/2017)

[![Maintenance Intended](http://maintained.tech/badge.svg)](http://maintained.tech/)

### Overview
Scripts and a general pipeline for
* Pumping in and processing data from ISCX/ISOT
* Flow feature generation and attention-friendly preprocessing
* Custom Tensorflow model
* Variety of utilities
This codebase is outdated and the most recent public prototype as of 6/1/2017. This repository will be updated in October with a pipeline and model performing with improved accuracy on ISCX.

### Installation
`docker-compose up --build`
`bash access_cluster.sh`
To run training for ISOT:
Inside of Docker container: `python3 -m botnet_attention.isot.train`
Inside of Docker container: `python3 -m botnet_attention.iscx.train`

### Contributions
Feel free to make a pull request if you have any suggestions.
Contributors are listed under `contributors.txt`.

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

