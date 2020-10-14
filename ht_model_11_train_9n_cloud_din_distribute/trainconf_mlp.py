# —*- coding:utf-8 -*-
__author__ = "zeng pan"
import tensorflow as tf
from ht_models import *
import os

"""
    在这里统一设置模型需要的超参，方便调试
"""


# 全局变量，定义某些外部数据的路径
vocab_file_root = "./vocab"

trainconf = dict(
                # 训练/测试数据源
                train_path="hdfs://ns1018/user/jd_ad/ads_sz/biz/app_szad_m_ht_recall_user_rank_train_tensorflow/",
                test_path="hdfs://ns1018/user/jd_ad/ads_sz/biz/app_szad_m_ht_recall_user_rank_test_tensorflow/",
                apply_path="hdfs://ns1018/user/jd_ad/ads_sz/biz/app_szad_m_ht_recall_user_rank_apply_tensorflow/",
	            result_table="app_szad_m_ht_rank_predict_res_tensor",
                # 模型类型
                MODEL_TYPE=embed_n_mlp,

                # 数据参数
                BATCH_SIZE=1200,

                BATCH_SIZE_EVAL=200,

                BATCH_SIZE_APPLY=800,

                EVAL_DATA_NUMS=3000000,
                # 环境参数
                N_GPUS=2,

                # MODEL_SAVE_PATH="./models_debug_dcn_ad",

                MODEL_NAME="model.ckpt",

                # LOG_SAVE_PATH="./logs",

                # 行为数据pooling策略
                # action_feature_indice=43,

                # combiner="sum",

                # 特征数量
                N=33,
                # 稀疏数据配置
                hash_size=[
                    5000000,  # |-- pin: string (nullable = true)                                             0
                    20000000,  # |-- skuid: string (nullable = true)                                           1
                    200000,  # |-- cateid: string (nullable = true)                                          2
                    200000,  # |-- brandid: string (nullable = true)                                         3
                    200000,  # |-- shopid: string (nullable = true)                                          4
                    5,  # |-- sex: string (nullable = true)                                             5
                    5,  # |-- buy_sex: string (nullable = true)                                         6
                    50,  # |-- age: string (nullable = true)                                             7
                    10,  # |-- carrer: string (nullable = true)                                          8
                    2,  # |-- haschild: string (nullable = true)                                        9
                    10,  # |-- jd_lev: string (nullable = true)                                          10
                    3,  # |-- marriage: string (nullable = true)                                        11
                    50,  # |-- province: string (nullable = true)                                        12
                    500,  # |-- city: string (nullable = true)                                            13
                    2,  # |-- c_browse_sku_flag: integer (nullable = false)                             14
                    101,  # |-- c_browse_cate_nums: integer (nullable = false)                            15
                    368,  # |-- c_browse_cate_gap: integer (nullable = false)                             16
                    101,  # |-- c_browse_brand_nums: integer (nullable = false)                           17
                    368,  # |-- c_browse_brand_gap: integer (nullable = false)                            18
                    2,  # |-- c_collect_sku_flag: integer (nullable = false)                            19
                    101,  # |-- c_collect_cate_nums: integer (nullable = false)                           20
                    101,  # |-- c_collect_brand_nums: integer (nullable = false)                          21
                    2,  # |-- c_car_sku_flag: integer (nullable = false)                                22
                    101,  # |-- c_car_cate_nums: integer (nullable = false)                               23
                    101,  # |-- c_car_brand_nums: integer (nullable = false)                              24
                    2,  # |-- c_gen_sku_flag: integer (nullable = false)                                25
                    101,  # |-- c_gen_cate_nums: integer (nullable = false)                               26
                    368,  # |-- c_gen_cate_gap: integer (nullable = false)                                27
                    101,  # |-- c_gen_brand_nums: integer (nullable = false)                              28
                    368,  # |-- c_gen_brand_gap: integer (nullable = false)                               29
                    # |-- c_browse_sku_list: string (nullable = true)
                    # |-- c_browse_cate_list: string (nullable = true)
                    # |-- c_browse_brand_list: string (nullable = true)
                ],

                sku_cate_brand_index=[1,2,3],    # index of skuid, cateid and brandid in hash_size

                # 不做hash转化的特征
                no_hash = list(range(14, 30)),

                # 稀疏数据embedding配置
                embed_size=[
                    16,  # 0
                    32,  # 1
                    8,  # 2
                    8,  # 3
                    8,  # 4
                    -1,  # 5
                    -1,  # 6
                    -1,  # 7
                    -1,  # 8
                    -1,  # 9
                    -1,  # 10
                    -1,  # 11
                    4,  # 12
                    4,  # 13
                    -1,  # 14
                    -1,  # 15
                    -1,  # 16
                    -1,  # 17
                    -1,  # 18
                    -1,  # 19
                    -1,  # 20
                    -1,  # 21
                    -1,  # 22
                    -1,  # 23
                    -1,  # 24
                    -1,  # 25
                    -1,  # 26
                    -1,  # 27
                    -1,  # 28
                    -1,  # 29
                ],

                is_usingg_user_act_feature=True,

				pooling_method="sum",

				activation_func_type="relu", # "relu, prelu or dice"


                # 网络结构和参数初始化配置
                initializer_embedding_w=lambda shape: tf.truncated_normal_initializer(stddev=2.0 / tf.sqrt(float(shape[0] + shape[1]))),

                initializer_dnn_w=lambda shape: tf.truncated_normal_initializer(stddev=1.0 / tf.sqrt(float(shape[0]))),

                # initializer_crossnet_w=tf.truncated_normal_initializer(stddev=1.0),
                # initializer_act_unit_w=lambda shape: tf.truncated_normal_initializer(stddev=1.0 / tf.sqrt(float(shape[0]))),

                initializer_fc_w=lambda shape: tf.truncated_normal_initializer(stddev=1.0 / tf.sqrt(float(shape[0]))),

                deep_layers=[200,150,80],

                fc_layers=[1],

                # mini batch aware reg 配置
                is_mba_reg=False,

                is_action_mba_reg=False,

                lambda_reg_mba=0.00003,

                # 正则化配置
                regulerizer1=None,  # embedding字典正则化

                regulerizer2=tf.contrib.layers.l2_regularizer(0.00003), # activation unit参数正则化

                regulerizer3=None,

                regulerizer4=tf.contrib.layers.l1_regularizer(0.00001),

                # 训练参数
                learning_rate_base=0.00005,

                momentum=None,

                # moving_average_decay=0.99,

                training_steps=800000,

                # steps_to_validate = 1000,

                # 优化算法
                optimal_algorithm=tf.train.AdamOptimizer,
)