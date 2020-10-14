# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import numpy as np
import time
import tensorflow as tf
from tfconf import log_dir, train_conf, train_debug
from ht_read import get_input_raw_batch, init_environ

from cmd_after_training import cmd_after_training
from cmd_before_training import clear_shared_memory
import os


# URL_REGULAR_TRAIN = "hdfs://default/user/jd_ad/ads_sz/biz/app_szad_m_ht_rank_train_tensorflow"


# 设置Tensorflow按需申请GPU资源
sess_config = tf.ConfigProto(
                # device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index],
                allow_soft_placement=True,
                log_device_placement=False,
				inter_op_parallelism_threads=25,
            	intra_op_parallelism_threads=25,
                # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
                )
sess_config.gpu_options.allow_growth = True


# Define parameters
# FLAGS = tf.app.flags.FLAGS
#
# # For distributed
# tf.app.flags.DEFINE_string("train_dir", "/export/App/training_platform/PinoModel/models", "Directory where to write event logs and checkpoint.")
# tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


def setSuperParam():
    """
    在这里统一设置模型需要的超参，方便调试
    :return:
    """
    import importlib
    print("using {} training config".format(train_conf))
    trainconf_module = importlib.import_module(train_conf)
    return trainconf_module.trainconf

def average_gradients(tower_grads):
    with tf.name_scope('gradients_average'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_vars = (grad, v)
            average_grads.append(grad_and_vars)
        return average_grads



def before_training():
    # download_vocab()
    print("train debug flag is {}".format(train_debug))
    if not train_debug:
        # clear_shared_memory()
        pass
    else:
        pass


def after_train_flow():
    print("train debug flag is {}".format(train_debug))
    if not train_debug:
        cmd_after_training()
    else:
        pass


def debug_add_grad_hist(name, grads):
    grads_ = map(lambda k: k[0], filter(lambda g_v: name in g_v[1].name, grads))
    list(map(lambda g: tf.summary.histogram(g.name, g), grads_))


def debug_add_var_hist(name, vars):
    vars_ = filter(lambda v: name in v.name, vars)
    list(map(lambda g: tf.summary.histogram(g.name, g), vars_))



def start_train_flow():
    # 初始化环境变量，便于tensorflow访问hdfs
    init_environ()
    # 获取超参
    param_dict = setSuperParam()
    N = param_dict["N"]
    train_path = param_dict["train_path"]
    test_path0 = param_dict["test_path0"]
    model_type = param_dict["MODEL_TYPE"]
    batch_size = param_dict["BATCH_SIZE"]
    batch_size_eval = param_dict["BATCH_SIZE_EVAL"]
    n_gpus = param_dict["N_GPUS"]
    # moving_average_decay = param_dict["moving_average_decay"]
    training_steps = param_dict["training_steps"]
    MODEL_SAVE_PATH = log_dir
    MODEL_NAME = param_dict["MODEL_NAME"]
    # LOG_SAVE_PATH = param_dict["LOG_SAVE_PATH"]
    OPT = param_dict["optimal_algorithm"]
    momentum = param_dict["momentum"]
    # 全局变量
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate_base = param_dict["learning_rate_base"]
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100, 0.99)

    # 数据入口
    # url_regular_train_list = get_url_list(URL_REGULAR_TRAIN, 7)
    x, y_ = get_input_raw_batch(train_path, test_path0, batch_size, N, train_debug)
    # x_t, y_t_ = get_input_raw_batch(URL_REGULAR_TEST, batch_size_eval)

    # 将输入数据切分，每份分别给每个GPU计算
    batch_each_gpu = int(batch_size / n_gpus)
    INPUT_TENSORS = []
    Y_ = []
    for i in range(n_gpus):
        with tf.device('/gpu:%d' % i):
            if i != n_gpus - 1:
                x_i = x[i * batch_each_gpu:(i + 1) * batch_each_gpu]
                y_i = y_[i * batch_each_gpu:(i + 1) * batch_each_gpu]
            elif i == n_gpus - 1:
                x_i = x[i * batch_each_gpu:]
                y_i = y_[i * batch_each_gpu:]
            INPUT_TENSORS.append(x_i)
            Y_.append(y_i)

    model = model_type(param_dict)
    # model = wide_n_deep_model(param_dict)
    # opt = model.get_train_op()

    opt = OPT(learning_rate, momentum) if momentum else OPT(learning_rate)
    tf.summary.scalar('learning-rate', learning_rate)

    # 记录每个GPU的损失函数值
    tower_grads = []
    loss_gpu_dir = {}
    # 将神经网络的优化过程跑在不同的GPU上
    for i in range(n_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('GPU_%d' % i) as scope:
                # 在每个GPU的命名空间下获取模型，模型参数定义在第一个GPU上，其余GPU中使用模型参数的副本
                x_i = INPUT_TENSORS[i]
                y_i_ = Y_[i]
                y_i = model.build_inference(x_i, "train")
                cur_loss, cur_cross_entropy, cur_regularization_loss = model.get_loss(y_i, y_i_, scope)
                tf.summary.scalar('total_loss', cur_loss)
                tf.summary.scalar('cross_entropy_loss', cur_cross_entropy)
                tf.summary.scalar('regularization_loss', cur_regularization_loss)
                loss_gpu_dir['GPU_%d' % i] = cur_loss
                tf.get_variable_scope().reuse_variables()
                grads = opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

    # 计算变量的平均梯度
    grads = average_gradients(tower_grads)

    # 对embedding梯度监控
    if train_debug:
        debug_add_grad_hist("embedding_1", grads)
        debug_add_var_hist("embedding_1", tf.global_variables())

    # 使用平均梯度更新参数
    tf.get_variable_scope()._reuse = False
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 计算变量的滑动平均值
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(
    #     tf.trainable_variables())
    tf.get_variable_scope()._reuse = True

    # 构建好训练中需要使用的OP
    train_op = apply_gradient_op
    # train_op = tf.group(apply_gradient_op, variables_averages_op)

    saver = tf.train.Saver(dict(map(lambda k: (k.name, k), tf.global_variables())))
    init = tf.global_variables_initializer()
    summary_op = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.local_variables_initializer())
        # sess.run(tf.tables_initializer())
        init.run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # 先将之前的tensorboard日志文件转移
        # transform_tensorboardLogs(LOG_SAVE_PATH)
        summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

        # sess.run(grads)
        for step in range(training_steps):
            try:
                if step != 0 and step % 100 == 0:
                    start_time = time.time()
                    loss_value_list = sess.run([train_op] + list(loss_gpu_dir.values()))
                    loss_value_list = loss_value_list[1:]
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration
                    loss_str = ', '.join([str(i) for i in loss_value_list])
                    format_str = ('step %d, loss = ' + loss_str + ' (%.1f examples sec; %.3f sec/batch)')
                    print(format_str % (
                    step, examples_per_sec, sec_per_batch))
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)
                else:
                    _ = sess.run(train_op)
                if step % 500 == 0 or (step + 1) == training_steps:
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print("train ended!")
                break
            except Exception as e:
                print("Exception type:%s" % type(e))
                print("Unexpected Error: {}".format(e))
                sys.exit(1)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    print("enter the 9n cloud")
    before_training()
    start_train_flow()
    after_train_flow()


if __name__ == "__main__":
    tf.app.run()

