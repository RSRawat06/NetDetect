py_v=$(which python)
virtualenv -p $py_v venv --no-site-packages
source venv/bin/activate
pip install -r requirements.txt

