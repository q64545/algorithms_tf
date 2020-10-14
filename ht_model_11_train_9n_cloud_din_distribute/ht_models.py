# -*- coding:utf-8 -*-
__author__ = "zeng pan"
import tensorflow as tf
# import cPickle as pickle
from abc import ABCMeta, abstractmethod



class model(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def get_weight_variable(self, shape, regularizer, initializer, name="", partitioner=None):
        weights = tf.get_variable("weights"+name, shape, initializer=initializer, dtype=tf.float32
                                                , partitioner=partitioner)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    @abstractmethod
    def build_inference(self, x): pass

    @abstractmethod
    def get_loss(self, logit, y, scope): pass

    @abstractmethod
    def get_train_op(self): pass

    def activation_func_prelu(self, _x, name=''):
        with tf.variable_scope("prelu"):
            alphas = tf.get_variable('prelu_alpha_{}'.format(name), _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            return tf.nn.relu(_x) + alphas*(_x - tf.nn.relu(_x))

    # def activation_func_dice(self, _x, name='', is_train=True, moving_decay=0.99, eps=1e-8):
    #     # mode canbe train or apply
    #     with tf.name_scope("dice"):
    #         alphas = tf.get_variable('alpha_{}'.format(name), _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
    #         x_e, x_v = tf.nn.moments(_x, 0)
    #         ema = tf.train.ExponentialMovingAverage(moving_decay)
	#
    #         def mean_var_with_update():
    #             ema_apply_op = ema.apply([x_e, x_v])
    #             with tf.control_dependencies([ema_apply_op]):
    #                 return tf.identity(x_e), tf.identity(x_v)
	#
    #         e, v = tf.cond(tf.equal(is_train, True), mean_var_with_update, lambda : (ema.average(x_e), ema.average(x_v)))
    #         # 构建dice计算公式
    #         p = tf.nn.sigmoid((_x - e) / tf.sqrt(v+eps))
    #         return p * _x + (1 - p) * alphas * _x


    def activation_func_dice(self, _x, name='', is_train=True, moving_decay=0.999, eps=1e-8):
        # mode canbe train or apply
        with tf.variable_scope("dice_{}".format(name)):
            alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
            x_e, x_v = tf.nn.moments(_x, 0)

            batch_expect = tf.get_variable('batch_expect', x_e.get_shape(), dtype=tf.float32, trainable=False)
            batch_varriance = tf.get_variable('batch_varriance', x_v.get_shape(), dtype=tf.float32, trainable=False)
            # tf.add_to_collection("dice_e_v", [batch_expect, batch_varriance])

            batch_expect_op = tf.assign(batch_expect, x_e)
            batch_varriance_op = tf.assign(batch_varriance, x_v)

            ema = tf.train.ExponentialMovingAverage(moving_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_expect, batch_varriance])
                with tf.control_dependencies([batch_expect_op, batch_varriance_op]):
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(x_e), tf.identity(x_v)

            e, v = tf.cond(tf.equal(is_train, True), mean_var_with_update, lambda : (ema.average(batch_expect), ema.average(batch_varriance)))
            # 构建dice计算公式
            p = tf.nn.sigmoid((_x - e) / tf.sqrt(v+eps))
            return p * _x + (1 - p) * alphas * _x

    def calculate_mini_batch_aware_reg(self, w, indice, lamb):
        # batch_size = tf.shape(indice)[0]
        yy, idx, count = tf.unique_with_counts(indice)
        yy_w = tf.nn.embedding_lookup(w, yy)
        mba_reg = 0.5 * lamb * tf.reduce_sum((1./ tf.cast(count, dtype=tf.float32)) * tf.reduce_sum(tf.square(yy_w),1), 0)
        tf.add_to_collection('losses', mba_reg)





class deep_interest_network(model):
    """
	该模型采用DIN的形式
	"""

    def __init__(self, param_dict):
        super(deep_interest_network, self).__init__()
        self.param_dict = param_dict


    def get_activation_func(self, is_train=True):
        assert self.param_dict["activation_func_type"] in ('sigmoid', 'relu', 'prelu', 'dice'), "activation func type must be in sigmoid, relu, prelu and dice"
        if self.param_dict["activation_func_type"] == 'sigmoid':
            return lambda _x, name: tf.nn.sigmoid(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'relu':
            return lambda _x, name: tf.nn.relu(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'prelu':
            return lambda _x, name: self.activation_func_prelu(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'dice':
            return lambda _x, name: self.activation_func_dice(_x=_x, name=name, is_train=is_train, moving_decay=0.99, eps=1e-8)


    def build_inference(self, x, flag="train"):
        # 设置regularizer，本别对应网络的四个部分
        regularizer1 = self.param_dict["regulerizer1"] if flag == "train" else None
        regularizer2 = self.param_dict["regulerizer2"] if flag == "train" else None
        regularizer3 = self.param_dict["regulerizer3"] if flag == "train" else None
        regularizer4 = self.param_dict["regulerizer4"] if flag == "train" else None
        is_train = True if flag == "train" else False
        # 先获取需要的参数
        hash_size = self.param_dict['hash_size']
        no_hash = self.param_dict["no_hash"]
        embed_size = self.param_dict["embed_size"]
        # 根据配置获取激活函数
        act_fn = self.get_activation_func(is_train)
        # 是否启用mini-batch aware regularization
        is_mba_reg = self.param_dict["is_mba_reg"]
        lambda_reg_mba = self.param_dict["lambda_reg_mba"]
        is_action_mba_reg = self.param_dict["is_action_mba_reg"]


        # 将输入划分
        x_feature = x[:, :-3]
        x_action_lists = x[:, -3:]

        # 先将稀疏特征转换成indice
        x_sparse = []
        for i in range(len(hash_size)):
            if i in no_hash:
                # 这部分特征本身可以直接作为indice，不需要转化
                x_i = tf.string_to_number(x_feature[:, i], tf.int32)
                x_sparse.append(x_i)
            else:
                # 这部分特征可以通过哈希函数来转化成index
                x_i = tf.string_to_hash_bucket_strong(input=x_feature[:, i], num_buckets=hash_size[i], key=[679362, 964545], name="sparse_feature_{}".format(i))
                x_sparse.append(x_i)
        # 将稀疏数据转换成embedding向量
        x_embed = []
        w_action_embed = []
        x_action = []
        indice_sku_cate_brand = []
        sku_cate_brand_index = self.param_dict["sku_cate_brand_index"]
        for i in range(len(embed_size)):
            if embed_size[i] != -1:
                with tf.variable_scope("embedding_{}".format(i)):
                    if hash_size[i] <= 500000:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]))
                    elif hash_size[i] > 500000 and hash_size[i] <= 5000000:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]),
                                                           partitioner=tf.fixed_size_partitioner(5, 0))
                    elif hash_size[i] > 5000000 and hash_size[i] <= 10000000:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]),
                                                           partitioner=tf.fixed_size_partitioner(10, 0))
                    elif hash_size[i] > 10000000 and hash_size[i] <= 15000000:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]),
                                                           partitioner=tf.fixed_size_partitioner(15, 0))
                    elif hash_size[i] > 15000000 and hash_size[i] <= 20000000:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]),
                                                           partitioner=tf.fixed_size_partitioner(20, 0))
                    else:
                        weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                           self.param_dict["initializer_embedding_w"](
                                                               [hash_size[i], embed_size[i]]),
                                                           partitioner=tf.fixed_size_partitioner(30, 0))
                x_i = tf.nn.embedding_lookup(weights, x_sparse[i])

                if i in sku_cate_brand_index:  # skuid, cateid, brandid对应的embedding向量
                    w_action_embed.append(weights)
                    x_action.append(x_i)
                    indice_sku_cate_brand.append(x_sparse[i])
                    if is_train and is_mba_reg and not is_action_mba_reg:
                        # 计算mba
                        self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)
                else:
                    if is_train and is_mba_reg:
                        # 计算mba
                        self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)



            else:
                x_i = tf.one_hot(x_sparse[i], depth=hash_size[i])

            x_embed.append(x_i)


            # if i in sku_cate_brand_index: # skuid, cateid, brandid对应的embedding向量
            #     with tf.variable_scope("embedding_{}".format(i)):
            #         weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
            #                                             self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]),
            #                                             partitioner=tf.fixed_size_partitioner(20, 0))
            #         w_action_embed.append(weights)
            #         x_i = tf.nn.embedding_lookup(weights, x_sparse[i])
            #         if is_train and is_mba_reg and not is_action_mba_reg:
            #             # 计算mba
            #             self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)
            #
            #         indice_sku_cate_brand.append(x_sparse[i])
            #         x_embed.append(x_i)
            #         x_action.append(x_i)
            # else:
            #     if embed_size[i] != -1:
            #         with tf.variable_scope("embedding_{}".format(i)):
            #             if i == 0:
            #                 weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
            #                                                    self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]),
            #                                                    partitioner=tf.fixed_size_partitioner(20, 0))
            #             else:
            #                 weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
            #                                                    self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]))
            #             x_i = tf.nn.embedding_lookup(weights, x_sparse[i])
            #             if is_train and is_mba_reg:
            #                 # 计算mba
            #                 self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)
            #
            #             x_embed.append(x_i)
            #     else:
            #         x_i = tf.one_hot(x_sparse[i], depth=hash_size[i])
            #         x_embed.append(x_i)
        x_embed = tf.concat(x_embed, 1)

        # 对浏览行为建模，构建DIN
        with tf.name_scope("user_behaviours"):
            x_browse_skus_list = tf.reshape(x_action_lists[:, 0], [-1, ])
            x_browse_cates_list = tf.reshape(x_action_lists[:, 1], [-1,])
            x_browse_brand_list = tf.reshape(x_action_lists[:, 2], [-1,])
            browse_lists = [x_browse_skus_list, x_browse_cates_list, x_browse_brand_list]
            browse_names = ['skus', 'cates', 'brands']
            browse_nums = self.param_dict["browse_nums"]
            x_action_list_embeds = []
            sum_poolings = []
            x_action_list_masks = []
            for i in range(len(browse_names)):
            # for i in [0]:
                with tf.name_scope("user_browse_{}_embedding".format(browse_names[i])):
                    browse_w_embed = w_action_embed[i]
                    # x_ad_embedded = x_action[i]
                    x_browse_action = browse_lists[i] # shape of x_browse_action is [?,]
                    x_browse_action_list = tf.string_split(x_browse_action, "#")
                    x_browse_action_list_indices = tf.sparse_to_dense(x_browse_action_list.indices,
                                                                      # x_browse_action_list.dense_shape,
                                                                      [x_browse_action_list.dense_shape[0], browse_nums[i]],
                                                                      tf.string_to_hash_bucket_strong(x_browse_action_list.values,
                                                                                                      num_buckets=browse_w_embed.get_shape()[0].value,
                                                                                                      key=[679362, 964545],
                                                                                                      name="sparse_user_browse_{}".format(browse_names[i])), -1)
                    indice_mask = tf.reshape(tf.not_equal(x_browse_action_list_indices, -1), [-1, browse_nums[i]])
                    x_action_list_masks.append(indice_mask)
                    x_action_list_embed = tf.reshape(tf.nn.embedding_lookup(browse_w_embed, x_browse_action_list_indices), [-1, browse_nums[i], browse_w_embed.get_shape()[1].value])
                    if is_train and is_action_mba_reg:
                        # 计算mba
                        indice_action = tf.concat([tf.string_to_hash_bucket_strong(x_browse_action_list.values,
                                                                                   num_buckets=browse_w_embed.get_shape()[0].value,
                                                                                   key=[679362, 964545]), indice_sku_cate_brand[i]], 0)
                        self.calculate_mini_batch_aware_reg(browse_w_embed, indice_action, lambda_reg_mba)
                    x_action_list_embeds.append(x_action_list_embed)

            with tf.name_scope("activation_unit"):
                act_unit_hidden_layers = self.param_dict["act_unit_hidden_layers"]
                action_indexs = self.param_dict["action_indexs"]
                # for i in range(len(x_action_list_embeds)):
                for i in action_indexs:
                    x_action_list_embed = x_action_list_embeds[i]
                    x_ad_embedded = x_action[i]
                    indice_mask = x_action_list_masks[i]
                    # 外积：笛卡尔积矩阵拉平向量
                    # out_product_list = tf.map_fn(lambda action_emb: tf.reshape(tf.matmul(tf.expand_dims(action_emb, 2), tf.expand_dims(x_ad_embedded, 1)), [-1, x_ad_embedded.shape[1].value ** 2]),
                    #                              tf.transpose(x_action_list_embed, [1, 0, 2]))

                    # 近似外积：向量相减再concat向量点积

                    x_action_list_embed_new = tf.transpose(x_action_list_embed, [1, 0, 2])

                    concat_list = [tf.concat([x_action_list_embed_new[ii], x_action_list_embed_new[ii] - x_ad_embedded, x_action_list_embed_new[ii]*x_ad_embedded, x_ad_embedded], 1)
                                        for ii in range(x_action_list_embed_new.shape[0].value)]


                    act_unit_in = concat_list[0].shape[1].value
                    act_in = concat_list
                    with tf.variable_scope("activation_unit_{}_list".format(browse_names[i])):
                        for ii in range(len(act_unit_hidden_layers)):
                            weights_act_unit = self.get_weight_variable([act_unit_in, act_unit_hidden_layers[ii]],
                                                                        regularizer3,
                                                                        self.param_dict["initializer_act_unit_w"](
                                                                            [act_unit_in, act_unit_hidden_layers[ii]]),
                                                                        name='_act_unit_w_{}'.format(ii))
                            biases_act_unit = tf.get_variable("biases_{}_act_unit".format(ii), [act_unit_hidden_layers[ii]],
                                                              initializer=tf.constant_initializer(0.0),
                                                              dtype=tf.float32)

                            act_out = list(map(lambda act_in_i: act_fn(tf.matmul(act_in_i[0], weights_act_unit) + biases_act_unit, name="act_func_{}_{}".format(ii, act_in_i[1])), zip(act_in, range(len(act_in)))))

                            # act_out = [tf.expand_dims(act_fn(tf.matmul(act_in[ii], weights_act_unit) + biases_act_unit, name="act_func_{}_{}".format(i, ii)), 0)
                            #                 for ii in range(act_in.shape[0].value)]
                            act_in =act_out
                            act_unit_in = act_in[0].shape[1].value
                        act_output_in = act_in
                        act_output_unit = act_unit_in
                        weights_act_unit_output = self.get_weight_variable([act_output_unit, 1], regularizer3, self.param_dict["initializer_act_unit_w"]([act_output_unit, 1]), name='_act_unit_output_w')
                        biases_act_unit_output = tf.get_variable("biases_act_unit_output", [1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

                        act_output_out = tf.concat(list(map(lambda act_output_i: tf.expand_dims(tf.matmul(act_output_i, weights_act_unit_output) + biases_act_unit_output, 0), act_output_in)), 0)
                        # act_output_out = tf.concat([tf.expand_dims(tf.matmul(act_output_in[iii], weights_act_unit_output) + biases_act_unit_output, 0) for iii in range(act_output_in.shape[0].value)], 0)
                    active_weight_score = tf.transpose(act_output_out, [1, 0, 2])
                    # 将空缺行为的权重设置为0.0
                    padding = tf.zeros_like(active_weight_score)
                    active_weight_score_t = tf.where(tf.expand_dims(indice_mask, 2), active_weight_score, padding)
                    with tf.name_scope("weight_sum_pooling"):
                        sum_pooling = tf.reduce_sum(x_action_list_embed * active_weight_score_t, 1)
                    sum_poolings.append(sum_pooling)
            x_deep_in = tf.concat([x_embed, tf.concat(sum_poolings, 1)], 1)

        # 构建deep模块
        with tf.name_scope("deep_network"):
            deep_layers = self.param_dict["deep_layers"]
            for i in range(len(deep_layers)):
                with tf.variable_scope("dnn_layer_{}".format(i)):
                    weights = self.get_weight_variable([x_deep_in.shape[1].value, deep_layers[i]], regularizer2,
                                                        self.param_dict["initializer_dnn_w"](
                                                            [x_deep_in.shape[1].value, deep_layers[i]]))
                    biases = tf.get_variable("biases", [deep_layers[i]], initializer=tf.constant_initializer(0.0),
                                                dtype=tf.float32)
                    layer_i = act_fn(tf.matmul(x_deep_in, weights) + biases, name="deep_mlp_{}".format(i))
                    x_deep_in = layer_i


        # 构建输出模块full connect
        x_fc_in = x_deep_in
        with tf.name_scope("fc_layers"):
            fc_layers = self.param_dict['fc_layers']
            for i in range(len(fc_layers)):
                with tf.variable_scope("fc_layers_{}".format(i)):
                    weights = self.get_weight_variable([x_fc_in.shape[1].value, fc_layers[i]], regularizer4,
                                                       self.param_dict["initializer_fc_w"](
                                                           [x_fc_in.shape[1].value, fc_layers[i]]))
                    biases = tf.get_variable("biases", [fc_layers[i]], initializer=tf.constant_initializer(0.0),
                                             dtype=tf.float32)
                    layer_i = tf.nn.sigmoid(tf.matmul(x_fc_in, weights) + biases)
                    x_fc_in = layer_i
        logit = x_fc_in
        return logit


    def get_loss(self, logit, y_, scope):
        logit = tf.reshape(logit, (-1,))
        with tf.name_scope("loss-regularization"):
            cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(logit, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - logit, 1e-10, 1.0)))
            regularization_loss = tf.add_n(tf.get_collection("losses", scope))
            loss = cross_entropy + regularization_loss
        return loss, cross_entropy, regularization_loss



class embed_n_mlp(model):
    """
	该模型采用embeddingg&mlp的形式
	"""

    def __init__(self, param_dict):
        super(embed_n_mlp, self).__init__()
        self.param_dict = param_dict


    def get_activation_func(self, is_train=True):
        assert self.param_dict["activation_func_type"] in ('sigmoid', 'relu', 'prelu', 'dice'), "activation func type must be in sigmoid, relu, prelu and dice"
        if self.param_dict["activation_func_type"] == 'sigmoid':
            return lambda _x, name: tf.nn.sigmoid(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'relu':
            return lambda _x, name: tf.nn.relu(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'prelu':
            return lambda _x, name: self.activation_func_prelu(_x, name=name)
        elif self.param_dict["activation_func_type"] == 'dice':
            return lambda _x, name: self.activation_func_dice(_x=_x, name=name, is_train=is_train, moving_decay=0.99, eps=1e-8)


    def build_inference(self, x, flag="train"):
        # 设置regularizer，本别对应网络的四个部分
        regularizer1 = self.param_dict["regulerizer1"] if flag == "train" else None
        regularizer2 = self.param_dict["regulerizer2"] if flag == "train" else None
        regularizer3 = self.param_dict["regulerizer3"] if flag == "train" else None
        regularizer4 = self.param_dict["regulerizer4"] if flag == "train" else None
        is_train = True if flag == "train" else False
        # 先获取需要的参数
        hash_size = self.param_dict['hash_size']
        no_hash = self.param_dict["no_hash"]
        embed_size = self.param_dict["embed_size"]
        # browse_nums = self.param_dict["browse_nums"] # browse_nums = [20, 10, 10]
        # 根据配置获取激活函数
        act_fn = self.get_activation_func(is_train)
        # 是否启用mini-batch aware regularization
        is_mba_reg = self.param_dict["is_mba_reg"]
        lambda_reg_mba = self.param_dict["lambda_reg_mba"]
        is_action_mba_reg = self.param_dict["is_action_mba_reg"]


        # 将输入划分
        x_feature = x[:, :-3]
        x_action_lists = x[:, -3:]

        # 先将稀疏特征转换成indice
        x_sparse = []
        for i in range(len(hash_size)):
            if i in no_hash:
                # 这部分特征本身可以直接作为indice，不需要转化
                x_i = tf.string_to_number(x_feature[:, i], tf.int32)
                x_sparse.append(x_i)
            else:
                # 这部分特征可以通过哈希函数来转化成index
                x_i = tf.string_to_hash_bucket_strong(input=x_feature[:, i], num_buckets=hash_size[i], key=[679362, 964545], name="sparse_feature_{}".format(i))
                x_sparse.append(x_i)
        # 将稀疏数据转换成embedding向量
        x_embed = []
        w_action_embed = []
        x_action = []
        indice_sku_cate_brand = []
        sku_cate_brand_index = self.param_dict["sku_cate_brand_index"]
        for i in range(len(embed_size)):
            if i in sku_cate_brand_index: # skuid, cateid, brandid对应的embedding向量
                with tf.variable_scope("embedding_{}".format(i)):
                    weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                        self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]),
                                                        partitioner=tf.fixed_size_partitioner(10, 0))
                    w_action_embed.append(weights)
                    x_i = tf.nn.embedding_lookup(weights, x_sparse[i])
                    if is_train and is_mba_reg and not is_action_mba_reg:
                        # 计算mba
                        self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)

                    indice_sku_cate_brand.append(x_sparse[i])
                    x_embed.append(x_i)
                    x_action.append(x_i)
            else:
                if embed_size[i] != -1:
                    with tf.variable_scope("embedding_{}".format(i)):
                        if i == 0:
                            weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                               self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]),
                                                               partitioner=tf.fixed_size_partitioner(10, 0))
                        else:
                            weights = self.get_weight_variable([hash_size[i], embed_size[i]], regularizer1,
                                                               self.param_dict["initializer_embedding_w"]([hash_size[i], embed_size[i]]))
                        x_i = tf.nn.embedding_lookup(weights, x_sparse[i])
                        if is_train and is_mba_reg:
                            # 计算mba
                            self.calculate_mini_batch_aware_reg(weights, x_sparse[i], lambda_reg_mba)
                        x_embed.append(x_i)
                else:
                    x_i = tf.one_hot(x_sparse[i], depth=hash_size[i])
                    x_embed.append(x_i)
        x_embed = tf.concat(x_embed, 1)
        x_deep_in = x_embed
        is_usingg_user_act_feature = self.param_dict["is_usingg_user_act_feature"]
        if is_usingg_user_act_feature:
            pooling_method = self.param_dict["pooling_method"]
            # 对浏览行为建模，构建行为embedding向量
            with tf.name_scope("user_behaviours"):
                x_browse_skus_list = tf.reshape(x_action_lists[:, 0], [-1,])
                x_browse_cates_list = tf.reshape(x_action_lists[:, 1], [-1,])
                x_browse_brand_list = tf.reshape(x_action_lists[:, 2], [-1,])
                browse_lists = [x_browse_skus_list, x_browse_cates_list, x_browse_brand_list]
                browse_names = ['skus', 'cates', 'brands']
                x_action_list_embeds = []
                for i in range(len(browse_names)):
                    with tf.name_scope("user_browse_{}_embedding".format(browse_names[i])):
                        browse_w_embed = w_action_embed[i]
                        # x_ad_embedded = x_action[i]
                        x_browse_action = browse_lists[i] # shape of x_browse_action is [?,]
                        x_browse_action_list = tf.string_split(x_browse_action, "#")
                        x_browse_action_list_indices = tf.SparseTensor(x_browse_action_list.indices,
                                                                       tf.string_to_hash_bucket_strong(x_browse_action_list.values,
                                                                                                          num_buckets=browse_w_embed.get_shape()[0].value,
                                                                                                          key=[679362, 964545],
                                                                                                          name="sparse_user_browse_{}".format(browse_names[i])),
                                                                       x_browse_action_list.dense_shape,)
                        x_action_list_embed = tf.nn.embedding_lookup_sparse(browse_w_embed, sp_ids=x_browse_action_list_indices,
                                                                                 sp_weights=None, combiner=pooling_method)
                        if is_train and is_action_mba_reg:
                            # 计算mba
                            indice_action = tf.concat([tf.string_to_hash_bucket_strong(x_browse_action_list.values,
                                                                                       num_buckets=browse_w_embed.get_shape()[0].value,
                                                                                       key=[679362, 964545]), indice_sku_cate_brand[i]], 0)
                            self.calculate_mini_batch_aware_reg(browse_w_embed, indice_action, lambda_reg_mba)
                        x_action_list_embeds.append(x_action_list_embed)
                x_deep_in = tf.concat([x_deep_in, tf.concat(x_action_list_embeds, 1)], 1)


        # 构建deep模块
        with tf.name_scope("deep_network"):
            deep_layers = self.param_dict["deep_layers"]
            for i in range(len(deep_layers)):
                with tf.variable_scope("dnn_layer_{}".format(i)):
                    weights = self.get_weight_variable([x_deep_in.shape[1].value, deep_layers[i]], regularizer2,
                                                        self.param_dict["initializer_dnn_w"](
                                                            [x_deep_in.shape[1].value, deep_layers[i]]))
                    biases = tf.get_variable("biases", [deep_layers[i]], initializer=tf.constant_initializer(0.0),
                                                dtype=tf.float32)
                    layer_i = act_fn(tf.matmul(x_deep_in, weights) + biases, name="deep_mlp_{}".format(i))
                    x_deep_in = layer_i


        # 构建输出模块full connect
        x_fc_in = x_deep_in
        with tf.name_scope("fc_layers"):
            fc_layers = self.param_dict['fc_layers']
            for i in range(len(fc_layers)):
                with tf.variable_scope("fc_layers_{}".format(i)):
                    weights = self.get_weight_variable([x_fc_in.shape[1].value, fc_layers[i]], regularizer4,
                                                       self.param_dict["initializer_fc_w"](
                                                           [x_fc_in.shape[1].value, fc_layers[i]]))
                    biases = tf.get_variable("biases", [fc_layers[i]], initializer=tf.constant_initializer(0.0),
                                             dtype=tf.float32)
                    layer_i = tf.nn.sigmoid(tf.matmul(x_fc_in, weights) + biases)
                    x_fc_in = layer_i
        logit = x_fc_in
        return logit


    def get_loss(self, logit, y_, scope):
        logit = tf.reshape(logit, (-1,))
        with tf.name_scope("loss-regularization"):
            cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(logit, 1e-10, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - logit, 1e-10, 1.0)))
            regularization_loss = tf.add_n(tf.get_collection("losses", scope))
            loss = cross_entropy + regularization_loss
        return loss, cross_entropy, regularization_loss