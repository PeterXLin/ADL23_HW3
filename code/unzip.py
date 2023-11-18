import zipfile

with zipfile.ZipFile("./checkpoint/Taiwan-LLM-7B-v2.0-chat.zip", 'r') as zip_ref:
    zip_ref.extractall("./checkpoint/Taiwan-LLM-7B-v2.0-chat")