# -*- coding:utf-8 -*-
import os
from multiprocessing import Pool
import time


RESULT_PATH = "./results/"
# CAL_RESULT_TABLE = "app_szad_m_ht_rank_predict_res_tensor"
LZO_PATH = os.path.join(RESULT_PATH, "result_lzo")

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

if not os.path.exists(LZO_PATH):
    os.mkdir(LZO_PATH)


def upload_files(arg):
    path = arg[0]
    flag = arg[1]
    result_table = arg[2]
    cmd_upload = "hadoop dfs -put {} /user/jd_ad/ads_sz/biz/9n_cloud/app.db/{}/flag={}/".format(path, result_table, flag)
    os.popen(cmd_upload).read()



def upload_results(result_table, flag, n_process=8):
    # 将所有计算结果压缩成lzo格式
    print("compress all results files into lzo format")
    compress_cmd = 'lzop -1 {}apply_file_*'.format(RESULT_PATH)
    t1 = time.time()
    os.popen(compress_cmd).read()
    duration1 = time.time() - t1
    print('lzo compressing result files costs %fsec' % duration1)
    mv_cmd = 'mv {}*.lzo {}'.format(RESULT_PATH, LZO_PATH)
    os.popen(mv_cmd).read()
    t3 = time.time()
    rm_cmd = "hadoop dfs -rmr /user/jd_ad/ads_sz/biz/9n_cloud/app.db/{}/flag={}".format(result_table, flag)
    os.popen(rm_cmd).read()
    mkdir_cmd = "hadoop dfs -mkdir /user/jd_ad/ads_sz/biz/9n_cloud/app.db/{}/flag={}".format(result_table, flag)
    os.popen(mkdir_cmd).read()
    duration3 = time.time() - t3
    print('deleting old data on hdfs costs %fsec' % duration3)

    # # 将所有结果文件路径放入List
    t2 = time.time()
    file_result_lzo = list(map(lambda file: os.path.join(LZO_PATH, file), os.listdir(LZO_PATH)))
    pool = Pool(n_process)
    pool.map(func=upload_files, iterable=zip(file_result_lzo, [flag] * len(file_result_lzo), [result_table] * len(file_result_lzo)))
    pool.close()
    pool.join()
    duration2 = time.time() - t2
    print('upload calculating result into hdfs costs %fsec' % duration2)

    # 修复表分区
    # repair_cmd = "hive -e 'msck repair table app.app_szad_m_dyrec_rank_predict_res_tensor_new;'"
    # os.popen(repair_cmd).read()




