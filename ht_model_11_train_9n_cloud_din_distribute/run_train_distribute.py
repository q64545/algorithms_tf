# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import numpy as np
import time
import tensorflow as tf
from tfconf import log_dir, train_conf, train_debug
from ht_read import get_input_raw_batch, init_environ, get_input_raw_batch_distributed

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

# 获取cluster内容
cluster_def = eval(os.getenv('TF_CONFIG'))['cluster']

# 获取job_name、task_index
task_def = eval(os.getenv('TF_CONFIG'))['task']
job_name = task_def['type']
task_index = task_def['index']

# Define parameters
FLAGS = tf.app.flags.FLAGS

# For distributed
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


class DistributeTrainError(RuntimeError):
	"""用来捕获分布式训练，个别worker启动失败"""
	def __init__(self, arg):
		self.args = arg


def start_distribute_tf_server():
	print("job_name: " + job_name)
	print("task_index: {}".format(task_index))
	print("cluster_def: ")
	print(cluster_def)
	cluster = tf.train.ClusterSpec(cluster_def)
	print(cluster)
	server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
	return cluster, server


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



def start_train_flow(cluster, server):
	# 开启分布式训练流程
	issync = FLAGS.issync
	if issync == 1:
		print("synchronization distribute mode")
	elif issync == 0:
		print("asynchronous distribute mode")
	# 判断是否为ps服务器
	if job_name == 'ps':
		server.join()

	# 以下是Worker服务器做的事情
	time.sleep(10)
	is_chief = (task_index == 0)
	# # debug
	print('is_chief: {}'.format(is_chief))
	worker_device = "/job:worker/task:{}/gpu:0".format(task_index)
	print('worker_device: {}'.format(worker_device))

	with tf.device(tf.train.replica_device_setter(
			worker_device=worker_device,
			ps_device="/job:ps/gpu:0",
			cluster=cluster)):
		# 初始化环境变量，便于tensorflow访问hdfs
		init_environ()
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
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
		                              dtype=tf.int32)
		learning_rate_base = param_dict["learning_rate_base"]
		learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100, 0.99)
		x, y_ = get_input_raw_batch_distributed(train_path, test_path0, len(cluster_def['worker']), task_index,
		                                        batch_size, N, train_debug)

		# # 将输入数据切分，每份分别给每个GPU计算
		# batch_each_gpu = int(batch_size / n_gpus)
		# INPUT_TENSORS = []
		# Y_ = []
		# for i in range(n_gpus):
		# 	worker_device_gpu = "/job:worker/task:%d/gpu:%d" % (task_index, i)
		# 	with tf.device(tf.train.replica_device_setter(
		# 		worker_device=worker_device_gpu,
		# 		ps_device="/job:ps/gpu:0",
		# 		cluster=cluster)):
		# 		if i != n_gpus - 1:
		# 			x_i = x[i * batch_each_gpu:(i + 1) * batch_each_gpu]
		# 			y_i = y_[i * batch_each_gpu:(i + 1) * batch_each_gpu]
		# 		elif i == n_gpus - 1:
		# 			x_i = x[i * batch_each_gpu:]
		# 			y_i = y_[i * batch_each_gpu:]
		# 		INPUT_TENSORS.append(x_i)
		# 		Y_.append(y_i)

		model = model_type(param_dict)
		grad_opt = OPT(learning_rate, momentum) if momentum else OPT(learning_rate)
		# tf.summary.scalar('learning-rate', learning_rate)
		# 记录每个GPU的损失函数值
		# tower_grads = []
		# loss_gpu_dir = {}
		# # 将神经网络的优化过程跑在不同的GPU上
		# for i in range(n_gpus):
		# 	worker_device_gpu = "/job:worker/task:%d/gpu:%d" % (task_index, i)
		# 	with tf.device(tf.train.replica_device_setter(
		# 		worker_device=worker_device_gpu,
		# 		ps_device="/job:ps/gpu:0",
		# 		cluster=cluster)):
		# 		with tf.name_scope('GPU_%d' % i) as scope:
		# 			# 在每个GPU的命名空间下获取模型，模型参数定义在第一个GPU上，其余GPU中使用模型参数的副本
		# 			x_i = INPUT_TENSORS[i]
		# 			y_i_ = Y_[i]
		# 			y_i = model.build_inference(x_i, "train")
		# 			cur_loss, cur_cross_entropy, cur_regularization_loss = model.get_loss(y_i, y_i_, scope)
		# 			# tf.summary.scalar('total_loss', cur_loss)
		# 			# tf.summary.scalar('cross_entropy_loss', cur_cross_entropy)
		# 			# tf.summary.scalar('regularization_loss', cur_regularization_loss)
		# 			loss_gpu_dir['GPU_%d' % i] = cur_loss
		# 			tf.get_variable_scope().reuse_variables()
		# 			grads = grad_opt.compute_gradients(cur_loss)
		# 			tower_grads.append(grads)
		# # 计算变量的平均梯度
		# grads = average_gradients(tower_grads)

		# # 对embedding梯度监控
		# if train_debug:
		# 	debug_add_grad_hist("embedding_1", grads)
		# 	debug_add_var_hist("embedding_1", tf.global_variables())

		# 使用平均梯度更新参数
		# tf.get_variable_scope()._reuse = False
		# worker_device_gpu = "/job:worker/task:%d/gpu:%d" % (task_index, 0)
		# with tf.device(tf.train.replica_device_setter(
		# 		worker_device=worker_device_gpu,
		# 		ps_device="/job:ps/gpu:0",
		# 		cluster=cluster)):
		with tf.name_scope('GPU_%d' % 0) as scope:
			y = model.build_inference(x, "train")
			cur_loss, cur_cross_entropy, cur_regularization_loss = model.get_loss(y, y_, scope)
			tf.summary.scalar('total_loss', cur_loss)
			tf.summary.scalar('cross_entropy_loss', cur_cross_entropy)
			tf.summary.scalar('regularization_loss', cur_regularization_loss)
		# grads = grad_opt.compute_gradients(cur_loss)

		if issync == 1:
			# 同步模式
			opt = tf.train.SyncReplicasOptimizer(
				grad_opt,
				replicas_to_aggregate=len(cluster_def['worker']),
				total_num_replicas=len(cluster_def['worker'])
				# use_locking=True
			)
			apply_gradient_op = opt.minimize(cur_loss, global_step=global_step)
			sync_replicas_hook = opt.make_session_run_hook(is_chief=is_chief)

		else:
			# 异步模式
			apply_gradient_op = grad_opt.minimize(cur_loss, global_step=global_step)

		# tf.get_variable_scope()._reuse = True
		# 构建好训练中需要使用的OP
		train_op = apply_gradient_op
		saver = tf.train.Saver(dict(map(lambda k: (k.name, k), tf.global_variables())))
		# global_init = tf.global_variables_initializer()
		local_init = tf.local_variables_initializer()
		summary_op = tf.summary.merge_all()

		# define training hooks
		end_train_hook = tf.train.StopAtStepHook(num_steps=training_steps)
		checkpointSaverHook = tf.train.CheckpointSaverHook(checkpoint_dir=MODEL_SAVE_PATH,
		                                                   save_steps=500,
		                                                   saver=saver,
		                                                   checkpoint_basename=MODEL_NAME
		                                                   )
		summarySaverHook = tf.train.SummarySaverHook(save_steps=500,
		                                             output_dir=MODEL_SAVE_PATH,
		                                             summary_op=summary_op)
		#
		hooks = [sync_replicas_hook, end_train_hook] if issync == 1 else [end_train_hook]
		chief_only_hooks = [summarySaverHook, checkpointSaverHook]
		print("global_variables: ")
		print(tf.global_variables())

		print("local variables: ")
		print(tf.local_variables())

		print("server target:")
		print(server.target)
		print("before entering distribute monitor session")
		if not is_chief:
			time.sleep(30)
		with tf.train.MonitoredTrainingSession(master=server.target,
		                                       is_chief=is_chief,
		                                       # checkpoint_dir=MODEL_SAVE_PATH,
		                                       hooks=hooks,
		                                       chief_only_hooks=chief_only_hooks,
		                                       # save_checkpoint_secs=60,
		                                       # save_summaries_steps=500,
		                                       config=sess_config) as mon_sess:
			print("enter distribute monitor session")
			# mon_sess.run(global_init)
			# print("finish global init")
			mon_sess.run(local_init)
			print("finish local init")
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord, sess=mon_sess)
			# debug: start
			print("before get into train loop")
			# debug: end
			step = 0
			start_time = time.time()
			while not mon_sess.should_stop():
				# print("step: {}".format(step))
				try:
					if step != 0 and step % 100 == 0:
						# x_v = mon_sess.run(INPUT_TENSORS[0])
						# print(x_v.shape)
						_, global_step_value, loss_value = mon_sess.run([train_op, global_step, cur_loss])
						duration = time.time() - start_time
						sec_per_batch = duration / global_step_value
						sec_per_batch2 = duration / step
						print("After {} global steps ({} global steps), loss on training batch is {}. (whole workers: {} examples per sec, {} sec/batch; single worker: {} examples per sec, {} sec/batch)".format(
								step, global_step_value, loss_value, batch_size / sec_per_batch, sec_per_batch, batch_size / sec_per_batch2, sec_per_batch2))
					else:
						# pass
						_ = mon_sess.run(train_op)
					step += 1
				except tf.errors.OutOfRangeError as e:
					print("Exception type:%s" % type(e))
					print("Unexpected Error: {}".format(e))
					if step <= 100:
						print("the training step is {}, not finishing training but error occurs!!!".format(step))
						# coord.request_stop()
						# coord.join(threads)
						# mon_sess.close()
						raise DistributeTrainError("Worker start failed!!!!!!!!!!!!  Need to restart worker training process immediately!!!!!!!!!!!!!!")
					else:
						print("finished training!!!!")
						break
				except Exception as e:
					print("Exception type:%s" % type(e))
					print("Unexpected Error: {}".format(e))
					sys.exit(1)
			coord.request_stop()
			coord.join(threads)


def main(argv=None):
	print("start distribute tensorflow server!")
	cluster, server = start_distribute_tf_server()
	# set retry times
	retry_times = 3
	print("enter the 9n cloud")
	if task_index == 0:
		before_training()
	try:

		start_train_flow(cluster, server)
	except DistributeTrainError:
		while retry_times > 0:
			try:
				print("Restart this worker training, {} times left!!!!!!!!!!!!!!!!!!!!!!!!!".format(retry_times))
				tf.reset_default_graph()
				print("finishing reset the default graph!!!!!!!!!!!")
				print("beginning to rebuild distribute training process!!!!!!!!!!!!!")
				start_train_flow(cluster, server)
				retry_times = 0
			except DistributeTrainError:
				retry_times -= 1

	if task_index == 0:
		after_train_flow()


if __name__ == "__main__":
	main()

