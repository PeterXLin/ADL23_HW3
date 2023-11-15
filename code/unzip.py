import zipfile

with zipfile.ZipFile("./hw3.zip", 'r') as zip_ref:
    zip_ref.extractall("./hw3")