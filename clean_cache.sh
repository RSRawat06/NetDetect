find . -name "__pycache__" -exec rm -rf {} \;
find . -name "*.pyc" -delete
find . -name ".cache" -exec rm -rf {} \;
echo "Success: python cache files have been deleted"
