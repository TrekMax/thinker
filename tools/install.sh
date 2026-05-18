pip uninstall pythinker -y
rm -rf dist/*
python setup.py sdist

pip install dist/pythinker*.tar.gz
