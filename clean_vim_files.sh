find . -name "*.swo" -exec rm -rf {} \;
find . -name "*.swp" -exec rm -rf {} \;
echo "Success: vim files have been removed"
