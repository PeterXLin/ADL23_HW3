import gdown

# download hw3.zip
# url = "https://drive.google.com/file/d/1n0EY5c7nyzz_fhRdmAJJVojAM9GDUrZO/view?usp=sharing"
# output = "./hw3.zip"
# gdown.download(url, output, quiet=False, fuzzy=True)

# download Taiwan-LLM-7B-v2.0-chat.zip
# url = "https://drive.google.com/file/d/1OkteUnpFGQvAU79j0OY-j2jo5J8mL_tp/view?usp=sharing"
# output = "Taiwan-LLM-7B-v2.0-chat.zip"
# gdown.download(url, output, quiet=False, fuzzy=True)

# # download adapter_checkpoint
# url = "https://drive.google.com/drive/folders/1GFir4_U3LgJZw4KDRIjBm7OGFPTh-VcOs"
# gdown.download_folder(url, quiet=True, use_cookies=False)

url = "https://drive.google.com/drive/folders/1GFir4_U3LgJZw4KDRIjBm7OGFPTh-VcO?usp=sharing"
gdown.download_folder(url, quiet=True, use_cookies=False)
