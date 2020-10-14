# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import numpy as np
import time
# import tensorflow as tf
# from tfconf import log_dir, train_conf, local_data_dir
from multiprocessing import Process
# from ht_download_apply_data import start_download, delete_earlier_results
# from ht_upload_results import upload_results
# from ht_read import get_input_apply_pandas, splitApplyDataHT
import os
import logging
# from functools import reduce
# import traceback
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ht_9ncloud")



#获取cluster内容
cluster_def = eval(os.getenv('TF_CONFIG'))['cluster']
# cluster = tf.train.ClusterSpec(cluster_def)

#获取job_name、task_index
task_def = eval(os.getenv('TF_CONFIG'))['task']
job_name = task_def['type']
task_index = task_def['index']

# # Define parameters
# FLAGS = tf.app.flags.FLAGS
# # For distributed
# # tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
# # tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
# tf.app.flags.DEFINE_integer("dldproN", 8, "num of processes which download data from hdfs to local")
# tf.app.flags.DEFINE_integer("runproN", 6, "num of processes which calculate the local data")
# tf.app.flags.DEFINE_integer("ulroN", 8, "num of processes which upload the result data to the hdfs")
parser = argparse.ArgumentParser()
parser.add_argument("--dldproN", help = "num of processes which download data from hdfs to local", type=int,default=8)
parser.add_argument("--runproN", help = "num of processes which calculate the local data", type=int,default=6)
parser.add_argument("--ulroN", help = "num of processes which upload the result data to the hdfs", type=int,default=8)
args = parser.parse_args()



LOCAL_DATA_DIR = "./apply_data"
RESULT_PATH = "./results"
UNIQUE_NAME_PATH = "/home/zengpan/unique_name"
MODEL_PATH = ""

if not os.path.exists(LOCAL_DATA_DIR):
    os.mkdir(LOCAL_DATA_DIR)


if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)




def setSuperParam():
    """
    在这里统一设置模型需要的超参，方便调试
    :return:
    """
    from tfconf import train_conf
    import importlib
    print("using {} training config".format(train_conf))
    trainconf_module = importlib.import_module(train_conf)
    return trainconf_module.trainconf


def get_model_path():
    global MODEL_PATH
    with open(UNIQUE_NAME_PATH, "r", encoding="utf-8") as f:
        unique_name = f.read().strip()
    model_path = "/home/zengpan/models/" + unique_name + "/models"
    MODEL_PATH = model_path
    # print("the newest dpa recomand model is " + MODEL_PATH)
    logger.info("the newest dpa recomand model is " + MODEL_PATH)




def before_applying():
    from ht_download_apply_data import delete_earlier_results, start_download
    apply_path = setSuperParam()["apply_path"]
    result_table = setSuperParam()["result_table"]
    if job_name == 'worker':
        if task_index == 0:
            delete_earlier_results(result_table)
        logger.info("worker start downloading apply data")
        start_download(apply_path, len(cluster_def['worker']), task_index, args.dldproN)
        get_model_path()


def after_applying():
    from ht_upload_results import upload_results
    result_table = setSuperParam()["result_table"]
    if job_name == 'worker':
        logger.info("worker start uploading calculated results")
        upload_results(result_table, task_index, args.ulroN)



def build_tensorflowGraph(x, param_dict, tf):
    model_type = param_dict["MODEL_TYPE"]
    n_gpus = param_dict["N_GPUS"]
    BATCH_SIZE_APPLY = param_dict["BATCH_SIZE_APPLY"]
    # m = deep_n_cross_model(param_dict)
    # model_type=param_dict["MODEL_TYPE"]
    m = model_type(param_dict)
    # 将一个batch的数据分成4份
    n_each = int(BATCH_SIZE_APPLY / n_gpus)
    Y = []
    X = []
    for i in range(n_gpus):
        with tf.variable_scope('inference_{}'.format(i)):
            if i != n_gpus - 1:
                x_i = x[i * n_each:(i + 1) * n_each]
                with tf.device('/gpu:%d' % i):
                    y_i = m.build_inference(x_i, "apply")
                Y.append(y_i)
                X.append(x_i)
                # tf.get_variable_scope().reuse_variables()
            elif i == n_gpus - 1:
                x_i = x[i * n_each:]
                with tf.device('/gpu:%d' % i):
                    y_i = m.build_inference(x_i, "apply")
                Y.append(y_i)
                X.append(x_i)
    y = tf.concat(Y, 0)
    return y


def generate_variables_to_restore(n_gpus, tf):
    variables_to_restore = dict(map(lambda k: (k.name, k), tf.global_variables()))
    # variables_to_restore_exp = dict(filter(lambda k: "ExponentialMovingAverage" in k[0],  variables_to_restore.items()))
    # variables_to_restore = dict(filter(lambda k: "ExponentialMovingAverage" not in k[0],  variables_to_restore.items()))

    VARIABLES_DICT = [dict(map(lambda kk:(kk[0].replace("inference_{}/".format(i), ""), kk[1]), filter(lambda k: k[0].split('/', 1)[0] == 'inference_{}'.format(i), variables_to_restore.items()))) for i in range(n_gpus)]
    # VARIABLES_DICT = [dict(map(lambda kk: (kk[0].split('/', 1)[1], kk[1]), filter(lambda k: k[0].split('/', 1)[0] == 'inference_{}'.format(i), variables_to_restore.items()))) for i in range(n_gpus)]
    # variables_dict0 = dict(map(lambda kk: (kk[0].split('/', 1)[1], kk[1]),
    #                            filter(lambda k: k[0].split('/', 1)[0] == 'inference_0', variables_to_restore.items())))
    # variables_dict1 = dict(map(lambda kk: (kk[0].split('/', 1)[1], kk[1]),
    #                            filter(lambda k: k[0].split('/', 1)[0] == 'inference_1', variables_to_restore.items())))
    # variables_dict2 = dict(map(lambda kk: (kk[0].split('/', 1)[1], kk[1]),
    #                            filter(lambda k: k[0].split('/', 1)[0] == 'inference_2', variables_to_restore.items())))
    # variables_dict3 = dict(map(lambda kk: (kk[0].split('/', 1)[1], kk[1]),
    #                            filter(lambda k: k[0].split('/', 1)[0] == 'inference_3', variables_to_restore.items())))
    # return variables_dict0, variables_dict1, variables_dict2, variables_dict3
    return VARIABLES_DICT



def apply_model(pid, setSuperParam, n_process):
    # 设置tensorflow相关配置
    import tensorflow as tf
    from ht_read import get_input_apply_pandas, splitApplyDataHT
    from functools import reduce
    import traceback
    # 设置Tensorflow按需申请GPU资源
    sess_config = tf.ConfigProto(
        # device_filters=['/job:ps', '/job:worker/task:%d' % task_index],
        allow_soft_placement=True,
        log_device_placement=False,
        # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
    )
    sess_config.gpu_options.allow_growth = True
    logger.info("finishing import tensorflow")
    # 获取应用参数
    # print("enter sub process {}".format(pid))
    logger.info("enter sub process {}".format(pid))
    param_dict = setSuperParam()
    BATCH_SIZE_APPLY = param_dict["BATCH_SIZE_APPLY"]
    n_gpus = param_dict["N_GPUS"]
    N = param_dict["N"]
    tt = time.time()
    file_list = os.listdir(LOCAL_DATA_DIR)
    file_list = filter(lambda x: x[:4] == 'part', file_list)
    file_list = map(lambda x: os.path.join(LOCAL_DATA_DIR, x), file_list)

    x = tf.placeholder(tf.string, [BATCH_SIZE_APPLY, N], name='input-x')
    y = build_tensorflowGraph(x, param_dict, tf)

    # variables_dict0, variables_dict1, variables_dict2, variables_dict3 = generate_variables_to_restore(variable_averages)
    VARIABLES_DICT = generate_variables_to_restore(n_gpus, tf)

    SAVER = [tf.train.Saver(variables_dicti) for variables_dicti in VARIABLES_DICT]


    # saver0 = tf.train.Saver(variables_dict0)
    # saver1 = tf.train.Saver(variables_dict1)
    # saver2 = tf.train.Saver(variables_dict2)
    # saver3 = tf.train.Saver(variables_dict3)

    with tf.Session(config=sess_config) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        model_path = ckpt.model_checkpoint_path.replace("/export/App/training_platform/PinoModel/models", MODEL_PATH)
        [saveri.restore(sess, model_path) for saveri in SAVER]
        # saver0.restore(sess, model_path)
        # saver1.restore(sess, model_path)
        # saver2.restore(sess, model_path)
        # saver3.restore(sess, model_path)
        global_index = 0
        for id, path in enumerate(file_list):
            if id % n_process != pid:
                continue
            # print('Process: %d, loading data from %s' % (pid, path))
            logger.info('Process: %d, loading data from %s' % (pid, path))
            t1 = time.time()
            data = get_input_apply_pandas(path, N)
            ids, action_type, action_date, similar_flag, shopid, data_input = splitApplyDataHT(data)
            num_data = data_input.shape[0]
            last_num = num_data % BATCH_SIZE_APPLY
            epoch_num = int(num_data / BATCH_SIZE_APPLY) if last_num == 0 else int(num_data / BATCH_SIZE_APPLY) + 1
            duration = time.time() - t1
            # print('Process: %d, loading %d apply data cost %fsec' % (pid, num_data, duration))
            logger.info('Process: %d, loading %d apply data cost %fsec' % (pid, num_data, duration))
            Y_PRED = []
            IDS = []
            TYPE = []
            DATE = []
            SIMILAR_FLAGE = []
            SHOPID=[]
            t4 = time.time()
            for i in range(epoch_num):
                try:
                    if i < epoch_num - 1:
                        data_input_batch = data_input[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        ids_i = ids[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        action_type_i = action_type[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        action_date_i = action_date[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        similar_flag_i = similar_flag[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        shopid_i = shopid[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        y_pred_i = sess.run(y, feed_dict={x: data_input_batch},
                                            # timeline开启
                                            # options=run_options,
                                            # run_metadata=run_metadata,
                                            #
                                            )
                        Y_PRED += y_pred_i.flatten().tolist()
                        IDS += ids_i.tolist()
                        TYPE += action_type_i.tolist()
                        DATE += action_date_i.tolist()
                        SIMILAR_FLAGE += similar_flag_i.tolist()
                        SHOPID += shopid_i.tolist()
                    else:
                        data_input_batch = data_input[i * BATCH_SIZE_APPLY:]
                        ids_i = ids[i * BATCH_SIZE_APPLY:]
                        action_type_i = action_type[i * BATCH_SIZE_APPLY:]
                        action_date_i = action_date[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        similar_flag_i = similar_flag[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        shopid_i = shopid[i * BATCH_SIZE_APPLY:(i + 1) * BATCH_SIZE_APPLY]
                        if data_input_batch.shape[0] < BATCH_SIZE_APPLY:
                            BATCH_N = data_input_batch.shape[0]
                            data_input_batch = reduce(lambda a, b: np.r_[a, b],
                                                        [data_input_batch] * (int(BATCH_SIZE_APPLY / BATCH_N) + 1))
                            data_input_batch = data_input_batch[:BATCH_SIZE_APPLY]
                        else:
                            BATCH_N = BATCH_SIZE_APPLY
                        # print data_input_batch.shape, n_
                        y_pred_i = sess.run(y, feed_dict={x: data_input_batch},
                                            # timeline开启
                                            # options=run_options,
                                            # run_metadata=run_metadata,
                                            #
                                            )[:BATCH_N]
                        Y_PRED += y_pred_i.flatten().tolist()
                        IDS += ids_i.tolist()
                        TYPE += action_type_i.tolist()
                        DATE += action_date_i.tolist()
                        SIMILAR_FLAGE += similar_flag_i.tolist()
                        SHOPID += shopid_i.tolist()
                except Exception as e:
                    # print("calculating error, abandon this batch, file is " + path)
                    # print('traceback.print_exc(): %s' % traceback.print_exc())
                    # print('traceback.format_exc():\n%s' % traceback.format_exc())
                    logger.info("calculating error, abandon this batch, file is " + path)
                    logger.info("traceback.print_exc(): %s" % traceback.print_exc())
                    logger.info("=================================================== the error batch is : start ===================================================")
                    list(map(logger.info, data_input_batch))
                    logger.info("=================================================== the error batch is : end   ===================================================")
                    # logger.info('traceback.format_exc():\n%s' % traceback.format_exc())
            duration4 = time.time() - t4
            # print('Process: %d, calculating %d examples cost %fsec' % (pid, num_data, duration4))
            logger.info('Process: %d, calculating %d examples cost %fsec' % (pid, num_data, duration4))
            file_name = 'apply_file_' + str(pid) + "_" + str(global_index) + '_worker_{}'.format(task_index)
            global_index += 1
            t3 = time.time()
            # print('Process: %d, writing data into %s' % (pid, file_name))
            logger.info('Process: %d, writing data into %s' % (pid, file_name))
            with open(os.path.join(RESULT_PATH, file_name), 'w', encoding='utf-8') as f:
                for i in range(len(Y_PRED)):
                    try:
                        # debug
                        # print(IDS[i][0] + '\t' + str(IDS[i][1]) + '\t' + '%.6f' % Y_PRED[i] + '\t' + DATA[i] + '\t' + str(SIMILAR_FLAGE[i]) + '\n')
                        #
                        f.write(IDS[i][0] + '\t' + str(IDS[i][1]) + '\t' + '%.6f' % Y_PRED[i] + '\t' + TYPE[i] + '\t' + DATE[i] + '\t' + str(SIMILAR_FLAGE[i]) + '\t' + str(SHOPID[i]) +'\n')
                    except Exception as e:
                        # print("writing error, abandon this line")
                        # print('traceback.print_exc(): %s' % traceback.print_exc())
                        # print('traceback.format_exc():\n%s' % traceback.format_exc())
                        logger.info("writing error, abandon this line")
                        logger.info('traceback.print_exc(): %s' % traceback.print_exc())
                        logger.info('traceback.format_exc():\n%s' % traceback.format_exc())
            duration3 = time.time() - t3
            # print('Process: %d, writing file costs %fsec' % (pid, duration3))
            logger.info('Process: %d, writing file costs %fsec' % (pid, duration3))
        duration5 = time.time() - tt
        # print('Process: %d, whole writing cost %fsec' % (pid, duration5))
        logger.info('Process: %d, whole writing cost %fsec' % (pid, duration5))



def apply_flow():
    # print("job_name: " + job_name)
    # print("task_index: {}".format(task_index))
    logger.info("job_name: " + job_name)
    logger.info("task_index: {}".format(task_index))
    # 判断是否为ps服务器
    if job_name == 'ps':
        return

    # 将之前的apply结果删除
    pp = range(args.runproN)
    PS = []
    # 逐个启动进程
    for i in pp:
        p = Process(target=apply_model, args=(i, setSuperParam, args.runproN))
        PS.append(p)
        p.start()
        time.sleep(10)

    # 主进程等待所有子进程执行完毕
    for p in PS:
        p.join()


def main(argv=None):
    # print("enter the 9n cloud")
    logger.info("enter the 9n cloud")
    before_applying()
    apply_flow()
    after_applying()


if __name__ == "__main__":
    main()