#coding=utf-8
__author__ = "zeng pan"
import os

UNIQUE_NAME_FILE = "./unique_name"


def cmd_after_training():
    print("training ending, update the newest unique name ")
    upload_cmd = "mv ./unique_name /home/zengpan/"
    os.popen(upload_cmd).read()

