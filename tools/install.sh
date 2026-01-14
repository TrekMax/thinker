pip uninstall pythinker -y
rm -rf dist/*
python setup.py sdist bdist_wheel

python -m build
pip install dist/pythinker*.whl
