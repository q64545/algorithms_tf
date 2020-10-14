# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import tensorflow as tf
import os
import time
import datetime
from ht_read import get_input_raw_batch_eval, init_environ, get_input_raw_batch_eval_new
from tfconf import log_dir, train_conf
from sklearn import metrics
import numpy as np
import pandas as pd
from pandas import DataFrame
from functools import reduce
# from trainconf_din import trainconf

# data_path = "hdfs://ns1018/user/jd_ad/ads_sz/biz/app_szad_m_ht_rank_test_tensorflow/"


# 设置Tensorflow按需申请GPU资源
config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            # inter_op_parallelism_threads=25,
            # intra_op_parallelism_threads=25,
            )
config.gpu_options.allow_growth = True

# N = 44+8-3+5-3-7+1

# 设置模型加载路径
# MODEL_SAVE_PATH = "./models_debug/"

def setSuperParam():
    """
    在这里统一设置模型需要的超参，方便调试
    :return:
    """
    import importlib
    print("using {} training config".format(train_conf))
    trainconf_module = importlib.import_module(train_conf)
    return trainconf_module.trainconf



def download_eval(batch_size, data_nums, DATA_PATH, N):
    x, y_ = get_input_raw_batch_eval(DATA_PATH, batch_size, N)
    X = []
    Y_ = []
    sum = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        while sum < data_nums:
            x_v, y_v = sess.run([x, y_])
            # debug
            print(x_v.shape)
            #
            X += x_v.tolist()
            Y_ += y_v.tolist()
            sum = len(X)
            print(sum)
            if len(x_v) < batch_size:
                break
        coord.request_stop()
        coord.join(threads)
    print(len(X))
    return X, Y_



def eval_flow():
    # 根据训练脚本选定模型
    param_dict = setSuperParam()
    N = param_dict["N"]
    test_path = param_dict["test_path"]
    model_type = param_dict["MODEL_TYPE"]
    batch_size = param_dict["BATCH_SIZE"]
    batch_size_eval = param_dict["BATCH_SIZE_EVAL"]
    BATCH_SIZE = batch_size_eval
    EVAL_DATA_NUMS = param_dict["EVAL_DATA_NUMS"]
    MODEL_SAVE_PATH = log_dir
    gpu_index = 1


    # 初始化环境变量，便于tensorflow访问hdfs
    # 获取eval数据
    init_environ()
    X, Y_ = download_eval(BATCH_SIZE, EVAL_DATA_NUMS, test_path, N)
    X = np.array(X)
    with tf.device("/gpu:{}".format(gpu_index)):
        # 设置数据入口
        x = tf.placeholder(tf.string, [BATCH_SIZE, N], name='input-x')
        # 构建模型

        # param_dict = setSuperParam()
        # model_type = param_dict["MODEL_TYPE"]
        m = model_type(param_dict)
        y = m.build_inference(x, "apply")

        # variable_averages = tf.train.ExponentialMovingAverage(0.99)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(dict(map(lambda k: (k.name, k), tf.global_variables())))

    with tf.Session(config=config) as sess:
        # sess.run(tf.tables_initializer())
        with open("exp_result", "w") as f:
            while 1:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path.replace("/export/App/training_platform/PinoModel/models", MODEL_SAVE_PATH))
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    # 将数据分批次传入session
                    num_data = len(X)
                    last_num = num_data % BATCH_SIZE
                    epoch_num = int(num_data / BATCH_SIZE) if last_num == 0 else int((num_data / BATCH_SIZE) + 1)
                    Y_PRED = []
                    JDPIN = []
                    for i in range(epoch_num):
                        if i != epoch_num - 1:
                            X_batch = X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                            X_jdpin = X_batch[:, 0]
                            y_pred_i = sess.run(y, feed_dict={x: X_batch})
                            Y_PRED += y_pred_i.flatten().tolist()
                            JDPIN += X_jdpin.flatten().tolist()
                        else:
                            X_batch = X[i * BATCH_SIZE:]
                            if len(X_batch) < BATCH_SIZE:
                                BATCH_N = len(X_batch)
                                X_batch = reduce(lambda a, b: a + b, [X_batch.tolist()] * int((BATCH_SIZE / BATCH_N) + 1))
                                X_batch = X_batch[:BATCH_SIZE]
                                X_batch = np.array(X_batch)
                            else:
                                BATCH_N = BATCH_SIZE
                            y_pred_i = sess.run(y, feed_dict={x: X_batch})[:BATCH_N]
                            X_jdpin = X_batch[:BATCH_N, 0]
                            Y_PRED += y_pred_i.flatten().tolist()
                            JDPIN += X_jdpin.flatten().tolist()
                    # 计算AUC
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=Y_, y_score=Y_PRED)
                    auc_scroe = metrics.auc(fpr, tpr)
                    result = str(global_step) + "," + str(auc_scroe) + "\n"
                    # f.write(result)
                    pos_nums = sum(Y_)
                    nums = len(Y_)
                    # 计算GAUC
                    result = pd.DataFrame(zip(JDPIN, Y_PRED, Y_),columns=["pin", "pred", "label"])
                    gauc_raw = result.groupby("pin").apply(lambda k: cal_auc(k))
                    gauc_raw = list(filter(lambda k: k[1] != -1.0, gauc_raw.values.tolist()))
                    gauc_sum_weight = sum(map(lambda k: k[0], gauc_raw))
                    gauc = reduce(lambda a,b: a+b, map(lambda k: (k[0] / gauc_sum_weight)*k[1], gauc_raw))
                    print("After %s training step(s), AUC score = %g, sum samples %g, mean pctr = %g, real ctr = %g; GAUC sroce = %g, sum samples %g" % (global_step, auc_scroe, num_data, float(np.mean(Y_PRED)), pos_nums / nums, gauc, gauc_sum_weight))
                else:
                    print('No checkpoint file found')
# def test(k):
#     print(k)
#     return("test", 0.0)


def cal_auc(k):
    data = k
    if len(data) > 1 and len(set(data["label"])) == 2:
        y_pred = data["pred"].values.tolist()
        y_label = data["label"].values.tolist()
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_label, y_score=y_pred)
        auc_user = metrics.auc(fpr, tpr)
        return (len(data), auc_user)
    else:
        return (len(data), -1.0)



def unit_test():
    eval_flow()


def main():
    eval_flow()


if __name__ == "__main__":
    main()


