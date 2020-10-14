# /bin/python
# encoding=utf-8

# 9nctl resources --cluster=langfang
framework="tensorflow-1.6.0-nm-cuda-new"    # 必选

cloud_tags={'cluster':'langfang','proxy':'http://172.18.178.107','node':'dpa'}
# cloud_tags={'cluster':'langfang','proxy':'http://172.18.178.107','node':'gpu-test'}

#可选，训练数据本地路径，默认为./data, 动态修改, 运行于Cloud上时会动态修改为Cloud端路径
local_data_dir="./data"

train_conf = 'trainconf_din'

hdfs_data=True

#可选，model及checkpoint保存目录，默认为/export/App/training_platform/PinoModel/models，建议保持默认，否则会影响dump以及tensorBoard功能
log_dir="/export/App/training_platform/PinoModel/models"
#可选，使用hdfs数据时使用，预留字段，该功能尚未开发
#remote_data_path="hdfs_path"

mode = "train_distribute" # train or apply or train_distribute

if mode == "train":
    # 可选，可自定义入口文件，默认为main.py
    entrance_filename = "run_train.py"
    train_debug=False
    #################################
    global_parameters = dict()

    #################################
    # 必选，job配置
    roles=['worker']

    # 必选，对应roles中各个job的配置
    worker = dict(
        cpu=10.0, # 单位为核
        mem=100.0, # 单位为G
        disk=60.0,   # 单位为G
        gpu=2,
    )
elif mode == "train_distribute":
    # 异步分布式训练
    entrance_filename = "run_train_distribute.py"
    train_debug = False
    #################################
    global_parameters = dict(issync=0)

    #################################
    # 必选，job配置
    roles = ['ps', 'worker']

    # 必选，对应roles中各个job的配置
    ps = dict(
	    count=2,  # role数量
	    gpu=1,  # gpu数量，单位为卡
	    cpu=4.0,  # 单位为核
	    mem=15.0,  # 单位为G
	    disk=20.0,  # 单位为G
	    job_name='ps'
    )

    # 必选，对应roles中各个job的配置
    worker = dict(
	    count=5,
	    gpu=1,  # gpu数量，单位为卡
	    cpu=4.0,  # 单位为核
	    mem=40.0,  # 单位为G
	    disk=20.0,  # 单位为G
	    job_name='worker',
    )
elif mode == "apply":
    # 可选，可自定义入口文件，默认为main.py
    entrance_filename = "run_apply.py"

    #################################
    global_parameters = dict(
	    dldproN=10,
	    runproN=4,
	    ulroN=10
    )

    #################################
    # 必选，job配置
    roles = ['worker']

    # 必选，对应roles中各个job的配置
    worker = dict(
	    count=4,
	    cpu=20,
	    gpu=4,  # gpu数量，单位为卡
	    mem=250.0,
	    disk=150.0
    )

elif mode == 'eval':
    # 可选，可自定义入口文件，默认为main.py
    entrance_filename = "run_eval.py"
    log_dir = "/home/zengpan/models/fb5f09f8f85a409b96ee2b6d1004bdd9/models"
    global_parameters = dict()
    roles = ['worker']
    worker = dict(
        cpu=10.0,  # 单位为核
        mem=100.0,  # 单位为G
        disk=60.0,  # 单位为G
        gpu=1,
    )

#可选，若相同roles需不同参数，在此添加


#create after
