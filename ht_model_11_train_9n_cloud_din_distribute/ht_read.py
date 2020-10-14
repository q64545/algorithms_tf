# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import tensorflow as tf
import os
import time
import pandas as pd
import csv
import numpy as np
from functools import reduce
import random


sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
                )
sess_config.gpu_options.allow_growth = True



def _parse_txt_data(value, N):
    record_defaults = [['-1'] for _ in range(N + 1)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    # feature = col[1:]
    feature = tf.reshape(tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[1:])), 1), [N,])
    return label, feature


#
def get_input_raw_batch_distributed_datesets(data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug):
    if not train_debug:
        url_list = get_url_list_daily(data_path, data_path2)
    else:
        url_list = get_url_list(data_path, 1, 0)

    file_nums = len(url_list)
    file_nums_each_worker = int(file_nums / n_workers)
    # random.shuffle(url_list)
    # 将文件列表均分给不同worker 注: 高版本可用 tf.data.Dataset.list_files(url_list, shuffle=True).shard(n_workers, worker_id) 替代
    url_list_worker = url_list[worker_id * file_nums_each_worker:(worker_id + 1) * file_nums_each_worker] if worker_id != n_workers - 1 else url_list[worker_id * file_nums_each_worker:]
    print("worker's training data")
    print(url_list_worker)

    dataset = tf.data.TextLineDataset(url_list_worker)
    dataset = dataset.map(lambda k: _parse_txt_data(k, N))
    dataset = dataset.shuffle(buffer_size=10000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    iterator_final = dataset.make_one_shot_iterator()
    next_batch_final = iterator_final.get_next()
    label_batch, feature_batch = next_batch_final
    return feature_batch, label_batch


def get_input_raw_batch_distributed_datesets2(data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug):
    if not train_debug:
        url_list = get_url_list_daily(data_path, data_path2)
    else:
        url_list = get_url_list(data_path, 1, 0)


    filenames = tf.data.Dataset.list_files(url_list).shuffle(buffer_size=100).shard(n_workers, worker_id)
    dataset = filenames.apply(tf.contrib.data.parallel_interleave(lambda filename: tf.data.TextLineDataset(filename),
                                                                  cycle_length=10,
                                                                  # buffer_output_elements=12,
                                                                  # prefetch_input_elements=12
                                                                  ))
    dataset = dataset.map(lambda k: _parse_txt_data(k, N))
    dataset = dataset.shuffle(buffer_size=10000 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(1)
    iterator_final = dataset.make_one_shot_iterator()
    next_batch_final = iterator_final.get_next()
    label_batch, feature_batch = next_batch_final
    return feature_batch, label_batch


def get_input_raw_batch_distributed(data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug):
    if not train_debug:
        url_list = get_url_list_daily(data_path, data_path2)
    else:
        url_list = get_url_list(data_path, 1, 0)
    file_nums = len(url_list)
    file_nums_each_worker = int(file_nums / n_workers)
    # random.shuffle(url_list)
    url_list_worker = url_list[worker_id*file_nums_each_worker:(worker_id+1)*file_nums_each_worker] if worker_id != n_workers-1 else url_list[worker_id*file_nums_each_worker:]
    print("worker's training data")
    print(url_list_worker)
    filename_queue = tf.train.string_input_producer(url_list_worker, shuffle=True, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 250)
    record_defaults = [['-1'] for _ in range(N + 1)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    feature = tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[1:])), 1)

    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                                                        capacity=10000 + 3 * batch_size,
                                                        num_threads=10, min_after_dequeue=2000,
                                                        enqueue_many=True, allow_smaller_final_batch=True)
    return feature_batch, label_batch



def get_input_raw_batch_distributed_new(data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug):
    if not train_debug:
        url_list = get_url_list_daily(data_path, data_path2)
    else:
        url_list = get_url_list(data_path, 1, 0)
    file_nums = len(url_list)
    file_nums_each_worker = int(file_nums / n_workers)
    # random.shuffle(url_list)
    url_list_worker = url_list[worker_id*file_nums_each_worker:(worker_id+1)*file_nums_each_worker] if worker_id != n_workers-1 else url_list[worker_id*file_nums_each_worker:]
    print("worker's training data")
    print(url_list_worker)
    filename_queue = tf.train.string_input_producer(url_list_worker, shuffle=True, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 250)
    record_defaults = [['-1'] for _ in range(N + 1+2)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    feature = tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[2:3]+col[4:])), 1)

    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                                                        capacity=10000 + 3 * batch_size,
                                                        num_threads=10, min_after_dequeue=2000,
                                                        enqueue_many=True, allow_smaller_final_batch=True)
    return feature_batch, label_batch


def get_url_list(train_url, train_days_trom=1, train_days_to=0):
    cmd = "hadoop dfs -ls {} |awk '{{print $8}}'".format('/' + '/'.join(train_url.split('/')[3:]))
    url_list = os.popen(cmd).read().strip().split('\n')[-1*train_days_trom:-1*train_days_to] if train_days_to != 0 else os.popen(cmd).read().strip().split('\n')[-1*train_days_trom:]
    url_lists = list(map(lambda url: os.popen("hadoop dfs -ls {} |awk '{{print $8}}'".format(url)).read().strip().split('\n'), url_list))
    url_lists = reduce(lambda a, b: a+b, url_lists)
    url_lists = list(filter(lambda url: "_SUCCESS" not in url, url_lists))
    return list(map(lambda url: "hdfs://default" + url, url_lists))


def get_url_list_daily(train_url, test_url):
    return get_url_list(test_url, 1, 0) + get_url_list(train_url, 1, 0)



def init_environ():
    # 设置环境变量，让tensorflow能够访问hdfs
    cmd = os.environ['HADOOP_HDFS_HOME'] + '/bin/hadoop classpath --glob'
    CLASSPATH = os.popen(cmd).read()
    os.environ['CLASSPATH'] = CLASSPATH


def get_input_raw_batch(data_path, data_path2, batch_size, N, train_debug):
    # 优化后的版本，读取速度更快
    # url_list = tf.train.match_filenames_once(data_path)
    if not train_debug:
        url_list = get_url_list_daily(data_path, data_path2)
    else:
        url_list = get_url_list(data_path, 1, 0)
    print("training data")
    print(url_list)
    filename_queue = tf.train.string_input_producer(url_list, shuffle=True, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 250)
    record_defaults = [['-1'] for _ in range(N+1)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    feature = tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[1:])), 1)

    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                           capacity=10000 + 3 * batch_size,
                           num_threads=10, min_after_dequeue=2000,
                           enqueue_many=True, allow_smaller_final_batch=True)
    return feature_batch, label_batch


def get_input_raw_dataset():
    pass




def get_input_raw_batch_eval(data_path, batch_size, N):
    # 用于eval的版本
    url_list = get_url_list(data_path, 1, 0)
    print("eval data")
    print(url_list)
    filename_queue = tf.train.string_input_producer(url_list, shuffle=False, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 250)
    record_defaults = [['-1'] for _ in range(N + 1)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    feature = tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[1:])), 1)

    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                                                        capacity=10000 + 3 * batch_size,
                                                        num_threads=10, min_after_dequeue=2000,
                                                        enqueue_many=True, allow_smaller_final_batch=True)
    return feature_batch, label_batch

def get_input_raw_batch_eval_new(data_path, batch_size, N):
    # 用于eval的版本
    url_list = get_url_list(data_path, 1, 0)
    print("eval data")
    print(url_list)
    filename_queue = tf.train.string_input_producer(url_list, shuffle=False, num_epochs=1)
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, 250)
    record_defaults = [['-1'] for _ in range(N + 1 + 2)]
    col = tf.decode_csv(records=value,
                        record_defaults=record_defaults,
                        field_delim='\t')
    label = col[0]
    label = tf.string_to_number(label, tf.float32)
    feature = tf.concat(list(map(lambda k: tf.reshape(k, [-1, 1]), col[2:3]+col[4:])), 1)

    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], batch_size=batch_size,
                                                        capacity=10000 + 3 * batch_size,
                                                        num_threads=10, min_after_dequeue=2000,
                                                        enqueue_many=True, allow_smaller_final_batch=True)
    return feature_batch, label_batch


def get_input_apply_pandas(data_path, N):
    data = pd.read_csv(data_path, names=range(N+3), sep='\t', low_memory=False, dtype=str, quoting=csv.QUOTE_NONE, engine='c', )
    data = data.values
    return data


# def get_input_apply_raw(data_path):
#     with open(data_path, "rb") as f:

def splitApplyData(data):
    """
    Used in apply period
    :param data: data read by pandas
    :return: ids and data_input
    """
    ids = data[:, :4][:, 1:3]
    jd_action_date = data[:, 3]
    jd_similar_flag = data[:, 16]
    # 取特征数据，并删除cut_exp_rn
    data_input = data[:, 4:]
    # data_input = np.delete(data_input, [2], axis=1)
    return ids, data_input, jd_action_date, jd_similar_flag


def splitApplyDataHT(data):
    """
    Used in apply period
    :param data: data read by pandas
    :return: ids and data_input
    """
    ids = data[:, 3:5]
    action_type = data[:, 0]
    action_date = data[:, 1]
    similar_flag = data[:, 2]
    shopid = data[:,7]
    # 取特征数据，并删除cut_exp_rn
    data_input = data[:, 3:]
    # data_input = np.delete(data_input, [2], axis=1)
    return ids, action_type, action_date, similar_flag, shopid, data_input

def unit_test1():
    data_path = 'hdfs://ns1018/user/jd_ad/ads_sz/biz/app_szad_m_ht_recall_user_rank_train_tensorflow/'
    N = 33
    train_debug = True
    init_environ()
    # print(os.environ['CLASSPATH'])
    # x, y = get_input_raw_batch_distributed_datesets2(data_path, '', 4, 0, 1200, N, train_debug) # (data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug)
    # x, y = get_input_raw_batch_distributed_datesets(data_path, '', 4, 0, 1200, N,train_debug)  # (data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug)
    x, y = get_input_raw_batch_distributed(data_path, '', 4, 0, 1200, N, train_debug) # (data_path, data_path2, n_workers, worker_id, batch_size, N, train_debug)
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        nums_data = 0
        t0 = time.time()
        t = t0
        for _ in range(1000000):
            x_v, y_v = sess.run([x, y])
            nums_data += len(x_v)
            duration = time.time() - t
            t = time.time()
            sum_t = t - t0
            print("reading {} samples cost {}s, each batch cost {}s".format(nums_data, sum_t, duration))
        coord.request_stop()
        coord.join(threads)



if __name__ == "__main__":
    unit_test1()


