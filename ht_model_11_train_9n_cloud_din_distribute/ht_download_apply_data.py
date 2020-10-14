# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import os
from multiprocessing import Pool
from os.path import join, getsize
import time




URL_REGULAR_APPLY = "/user/jd_ad/ads_sz/biz/app_szad_m_dyrec_rank_train_tensorflow/type=apply/part-*"
LOCAL_DATA_DIR = "./apply_data"




def get_download_files(URL_REGULAR_APPLY, n_workers, worker_id):
    cmd = "hadoop dfs -ls " + URL_REGULAR_APPLY + "| awk '{print $8}'"
    files = os.popen(cmd).read().split("\n")
    files = list(filter(lambda f: "part-" in f, files))
    nums = len(files)
    each_nums = int(nums / n_workers)
    FILES = [files[i * each_nums:(i + 1) * each_nums] if i != n_workers - 1 else files[i * each_nums:] for i in
             range(n_workers)]
    return FILES[worker_id]


def download_files(file):
    print("downloading file {}".format(file))
    cmd = "hadoop dfs -get " + file + " ./apply_data/"
    os.popen(cmd).read()


def delete_earlier_results(result_table):
    print("deleting the earlier result of table {}".format(result_table))
    cmd = "hadoop dfs -rmr /user/jd_ad/ads_sz/biz/9n_cloud/app.db/{}/*".format(result_table)
    os.popen(cmd).read()


def start_download(apply_path, n_workers, worker_id, n_process=8):
    apply_path = os.path.join(apply_path, 'part-*')
    files = get_download_files(apply_path, n_workers, worker_id)
    pool = Pool(n_process)
    pool.map(func=download_files, iterable=files)
    pool.close()
    pool.join()


def getdirsize(dir):
   size = 0
   for root, dirs, files in os.walk(dir):
      size += sum([getsize(join(root, name)) for name in files])
   return size


def unit_test():
    t1 = time.time()
    start_download(4, 0, n_process=8)
    duration = time.time() - t1
    filesize = getdirsize(LOCAL_DATA_DIR) / 1024 / 1024 / 1024
    print('download %fsG files cost %fsec' % (filesize, duration))


if __name__ == "__main__":
    unit_test()