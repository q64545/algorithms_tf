#coding=utf-8

import os

catagory_features = ["cate_id", "brand_id", "province", "city"]
URL = "/user/jd_ad/ads_sz/biz/numericalDataProcess/"


UNIQUE_NAME_FILE = "./unique_name"
LAST_UNIQUE_NAME_FILE = "/home/zengpan/unique_name"
SHARED_MEM_FILE = "/home/zengpan/models"

# def download_f(feature_name):
#     remote_path = os.path.join(URL, feature_name + "_Vocab")
#     cmd = "hadoop dfs -get " + remote_path + "/*.csv ; mv *.csv " + os.path.join(vocab_file_root, feature_name)
#     result = os.popen(cmd).read()



# def download_vocab():
#     print("start downloading...")
#     for feature_name in catagory_features:
#         download_f(feature_name)
#     print("finish downloading")


def delete_dir(p):
    delete_cmd = "sudo rm -rf {}".format(p)
    os.popen(delete_cmd).read()


def clear_shared_memory():
    print("before training, clear the shared memory")
    with open(UNIQUE_NAME_FILE, "r") as f:
        unique_name = f.read().strip()

    with open(LAST_UNIQUE_NAME_FILE, "r") as f:
        last_unique_name = f.read().strip()

    model_files = map(lambda p: os.path.join(SHARED_MEM_FILE, p), os.listdir(SHARED_MEM_FILE))
    delete_files = filter(lambda p: p != os.path.join(SHARED_MEM_FILE, unique_name) and p != os.path.join(SHARED_MEM_FILE, last_unique_name), model_files)
    list(map(delete_dir, delete_files))
