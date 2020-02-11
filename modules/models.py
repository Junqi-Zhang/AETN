import tensorflow as tf


def get_activation(activation):
    switch = {
        'relu': tf.nn.relu,
        'crelu': tf.nn.crelu,
        'elu': tf.nn.elu,
        'leakyrelu': tf.nn.leaky_relu,
        'prelu': tf.keras.layers.PReLU,
        'tanh': tf.tanh,
        'sigmoid': tf.sigmoid
    }

    try:
        act = switch[activation.lower()]
    except KeyError:
        raise NotImplementedError
    else:
        return act


class AETN(object):

    def __init__(self, params, inputs, mode):
        self.training = mode == tf.estimator.ModeKeys.TRAIN
        self.parse_params(params)
        self.parse_inputs(inputs)
        self.show_constructs()
        self.construct_weights()

    def parse_params(self, params):
        self.train_mlm = params['train_mlm'] # Bool for train the BERT-like task of masked app prediction

        if params['q_dims'] is None:
            params['q_dims'] = params['p_dims'][::-1]
        else:
            assert params['q_dims'][0] == params['p_dims'][-1], "Input and output dimension must equal each other for autoencoders."
            assert params['q_dims'][-1] == params['p_dims'][0], "Latent dimension for -p and q- mismatches."
        self.dae_dims = params['q_dims'] + params['p_dims'][1:]

        assert params['u_dims'] != None, "u_dims cannot be None." # dim for user embeddings
        self.u_dims = params['u_dims']
        self.bottleneck_dims = [self.dae_dims[1], self.u_dims, self.dae_dims[1]]

        assert params['l_dims'] != None, "l_dims cannot be None."
        self.classifier_dims = [self.u_dims] + params['l_dims'] # prepare for the future fine-tune tasks

        # settings for the transformers
        self.ffn_dim = params['ffn_dim']
        self.num_translayer = params['num_translayer']
        self.num_header = params['num_header']

        self.num_date = params['num_date']
        self.length_his = params['length_his']
        self.softmap = params['softmap']
        self.softmap_vocab = params['softmap_vocab']

        self.image_dropout_rate = params.get('image_dropout_rate', 0)
        self.classifier_dropout_rate = params.get('classifier_dropout_rate', 0)
        self.attention_dropout_rate = params.get('attention_dropout_rate', 0)
        self.ffn_dropout_rate = params.get('ffn_dropout_rate', 0)

        self.lambda_L2 = params.get('lambda_L2', 1.5e-7)
        self.reg = tf.contrib.layers.l2_regularizer(self.lambda_L2)

    def show_constructs(self):
        print("Train with MLM:", self.train_mlm, end='\t')
        print("DAE structure is:", self.dae_dims, end='\t')
        print("BOTTLENECK structure is:", self.bottleneck_dims, end='\t')
        print("CLASSIFIER structure is:", self.classifier_dims, end='\t')
        print("TRANSFORMER structure is:", [self.num_translayer, self.num_header, self.ffn_dim])

    def parse_inputs(self, inputs):
        self.image = inputs[0] # retention
        # installation
        self.newdate = tf.to_int32(inputs[1]) # (batch_size, length_his)
        self.newapp = tf.to_int32(inputs[2]) # (batch_size, length_his)
        self.newmask = tf.to_int32(inputs[3]) # (batch_size, length_his)
        self.newbertindex = tf.to_int32(inputs[4]) # (batch_size, 3)
        self.newbertmask = tf.to_int32(inputs[5]) # (batch_size, length_his)
        # uninstallation
        self.lossdate = tf.to_int32(inputs[6]) # (batch_size, length_his)
        self.lossapp = tf.to_int32(inputs[7]) # (batch_size, length_his)
        self.lossmask = tf.to_int32(inputs[8]) # (batch_size, length_his)
        self.lossbertindex = tf.to_int32(inputs[9]) # (batch_size, 3)
        self.lossbertmask = tf.to_int32(inputs[10]) # (batch_size, length_his)

    def construct_weights(self):
        self.dae_weights = []
        self.dae_biases = []

        self.classifier_weights = []
        self.classifier_biases = []

        self.bottleneck_weights = []
        self.bottleneck_biases = []

        self.transformer_weights = []

        # set apps embedding matrix weith category ID
        self.label_weight = tf.gather(tf.get_variable(
            name="label_weight", shape=[self.softmap_vocab, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer()), self.softmap)

        for i, (d_in, d_out) in enumerate(zip(self.dae_dims[:-1], self.dae_dims[1:])):
            if i == 0:
                weight_key = "dae_weight_{}to{}".format(i,i+1)
                print('label_weight', weight_key)
                self.dae_weights.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()) + self.label_weight)
            elif i != len(self.dae_dims) - 2:
                weight_key = "dae_weight_{}to{}".format(i,i+1)
                print('normal_weight', weight_key)
                self.dae_weights.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))
            else:
                print('tie_weight', i) # weight tying
                self.dae_weights.append(tf.transpose(self.dae_weights[0]))

            bias_key = "dae_bias_{}".format(i+1)
            self.dae_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        for i, (d_in, d_out) in enumerate(zip(self.classifier_dims[:-1], self.classifier_dims[1:])):
            weight_key = "classifier_weight_{}to{}".format(i,i+1)
            bias_key = "classifier_bias_{}".format(i+1)
            print(weight_key)
            self.classifier_weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer()))
            self.classifier_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        for i, (d_in, d_out) in enumerate(zip(self.bottleneck_dims[:-1], self.bottleneck_dims[1:])):
            weight_key = "bottleneck_weight_{}to{}".format(i,i+1)
            bias_key = "bottleneck_bias_{}".format(i+1)
            print(weight_key)
            self.bottleneck_weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer()))
            self.bottleneck_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        self.app_embeddings = tf.concat([self.dae_weights[0], tf.zeros(shape=[1, self.dae_dims[1]])], 0)
        # date embeddings
        self.position_embeddings = tf.concat(
            [tf.get_variable(name="position_embeddings", shape=[self.num_date, self.dae_dims[1]],
                            initializer=tf.contrib.layers.xavier_initializer()),
            tf.zeros(shape=[1, self.dae_dims[1]])], 0)
        # behavior type embeddings
        self.new_embeddings = tf.concat([tf.zeros(shape=[1, self.dae_dims[1]]),
            tf.get_variable(name="new_embeddings", shape=[1, self.dae_dims[1]],
                initializer=tf.contrib.layers.xavier_initializer())], 0)
        self.loss_embeddings = tf.concat([tf.zeros(shape=[1, self.dae_dims[1]]),
            tf.get_variable(name="loss_embeddings", shape=[1, self.dae_dims[1]],
                initializer=tf.contrib.layers.xavier_initializer())], 0)
        self.img_embeddings = tf.get_variable(name="img_embeddings", shape=[1, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer())
        # embedding for bert mask
        self.bert_mask_embeddings = tf.get_variable(name="bert_mask_embeddings", shape=[1, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer())

    def multihead_attention(self, queries, keys, values, query_masks, key_masks):
        '''
        queries: A 3d tensor with shape of [N, T_q, d_q].
        keys: A 3d tensor with shape of [N, T_k, d_k].
        values: A 3d tensor with shape of [N, T_v, d_k].
        query_masks:A 2d tensor with shape of [N, T_q]
        key_masks:A 2d tensor with shape of [N, T_k]

        return: A 3d tensor with shape of [N, T_q, d_model]
        '''

        d_model = queries.get_shape().as_list()[-1]

        Q_layer = tf.layers.Dense(d_model, use_bias=False)
        Q = Q_layer(queries)
        self.transformer_weights += Q_layer.weights
        K_layer = tf.layers.Dense(d_model, use_bias=False)
        K = K_layer(keys)
        self.transformer_weights += K_layer.weights
        V_layer = tf.layers.Dense(d_model, use_bias=False)
        V = V_layer(values)
        self.transformer_weights += V_layer.weights

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_header, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_header, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_header, axis=2), axis=0) # (h*N, T_v, d_model/h)
        query_masks = tf.tile(query_masks, [self.num_header, 1]) # (h*N, T_q)
        key_masks = tf.tile(key_masks, [self.num_header, 1]) # (h*N, T_k)

        # Attention
        outputs = self.scaled_dot_product_attention(Q_, K_, V_, query_masks, key_masks) # (h*N, T_q, d_model/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_header, axis=0), axis=2) # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        return tf.contrib.layers.layer_norm(outputs)

    def scaled_dot_product_attention(self, Q, K, V, Q_masks, K_masks):
        '''
        Q: Packed queries. 3d tensor. [h*N, T_q, d_model/h].
        K: Packed keys. 3d tensor. [h*N, T_k, d_model/h].
        V: Packed values. 3d tensor. [h*N, T_v, d_model/h].
        Q_masks: Packed masks. 2d tensor. [h*N, T_q].
        K_masks: Packed masks. 2d tensor. [h*N, T_k].

        return: A 3d tensor with shape of [h*N, T_q, d_model/h]
        '''

        d_k = K.get_shape().as_list()[-1] # d_model/h

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        padding_num = -1e+7
        K_masks = tf.expand_dims(K_masks, 1) # (h*N, 1, T_k)
        K_masks = tf.tile(K_masks, [1, tf.shape(Q)[1], 1]) # (h*N, T_q, T_k)
        paddings = tf.ones_like(outputs) * padding_num
        outputs = tf.where(tf.equal(K_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # softmax
        outputs = tf.nn.softmax(outputs)

        # query masking
        Q_masks = tf.expand_dims(Q_masks, -1) # (h*N, T_q, 1)
        Q_masks = tf.tile(Q_masks, [1, 1, tf.shape(K)[1]]) # (h*N, T_q, T_k)
        outputs = outputs * tf.to_float(Q_masks)

        # dropout
        outputs = tf.layers.dropout(outputs, self.attention_dropout_rate, training=self.training)

        return tf.matmul(outputs, V) # [h*N, T_q, d_model/h]

    def ffn(self, inputs):
        # Inner layer
        ffn_layer_1 = tf.layers.Dense(self.ffn_dim, activation=get_activation('relu'))
        outputs = ffn_layer_1(inputs)
        self.transformer_weights += ffn_layer_1.weights

        outputs = tf.layers.dropout(outputs, self.ffn_dropout_rate, training=self.training)

        # Outer layer
        d_model = inputs.get_shape().as_list()[-1]
        ffn_layer_2 = tf.layers.Dense(d_model)
        outputs = ffn_layer_2(outputs)
        self.transformer_weights += ffn_layer_2.weights

        # Residual connection
        outputs += inputs

        return tf.contrib.layers.layer_norm(outputs)

    def forward_pass(self):
        model_outputs = []

        # embedding layer for transformer encoder
        new_app_emb = tf.nn.embedding_lookup(self.app_embeddings, self.newapp) # (N, length, d_model)
        loss_app_emb = tf.nn.embedding_lookup(self.app_embeddings, self.lossapp) # (N, length, d_model)

        new_pos_emb = tf.nn.embedding_lookup(self.position_embeddings, self.newdate) + \
                        tf.nn.embedding_lookup(self.new_embeddings, self.newmask)
        loss_pos_emb = tf.nn.embedding_lookup(self.position_embeddings, self.lossdate) + \
                        tf.nn.embedding_lookup(self.loss_embeddings, self.lossmask)

        mask_emb = tf.nn.embedding_lookup(self.bert_mask_embeddings,
                                            tf.zeros_like(self.newbertmask, tf.int32)) # (N, length, d_model)

        new_bert_masks = tf.tile(tf.expand_dims(self.newbertmask * self.training * self.train_mlm, 2), [1, 1, self.dae_dims[1]]) # (N, length, d_model)
        loss_bert_masks = tf.tile(tf.expand_dims(self.lossbertmask * self.training * self.train_mlm, 2), [1, 1, self.dae_dims[1]]) # (N, length, d_model)

        # representation for masked installation and uninstallation
        new_emb = tf.where(tf.equal(new_bert_masks, 0), new_app_emb, mask_emb) + new_pos_emb
        loss_emb = tf.where(tf.equal(loss_bert_masks, 0), loss_app_emb, mask_emb) + loss_pos_emb

        img = tf.math.l2_normalize(self.image, 1) # original retention
        img = tf.layers.dropout(img, self.image_dropout_rate, training=self.training)

        # dae part
        for i, (w, b) in enumerate(zip(self.dae_weights, self.dae_biases)):
            if i == 0:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
                img_emb = tf.expand_dims(img, 1) # (N, 1, d_model)
            if i == 1:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
                # user_embeddings = img # (N, 128)
            if i == 2:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
            if i == 3:
                logits_dae = tf.matmul(img,w)+b
                model_outputs.append(logits_dae)

        # retention for transformer encoder
        img_emb = img_emb + tf.nn.embedding_lookup(self.img_embeddings,
                                tf.zeros_like(tf.reduce_sum(img_emb, 2), tf.int32)) # (N ,1, d_model)
        imgmask = tf.ones_like(tf.reduce_sum(img_emb, 2), tf.int32) # (N, 1)

        # The transformer Encoder
        for trans_layer in range(self.num_translayer):
            # The modification in multi-head self attention
            img_new_emb = tf.concat([tf.tile(img_emb, [1, tf.shape(new_emb)[1], 1]), new_emb], 2) # (N, length_his, 2*d_model)
            img_loss_emb = tf.concat([tf.tile(img_emb, [1, tf.shape(loss_emb)[1], 1]), loss_emb], 2) # (N, length_his, 2*d_model)
            img_his_emb = self.ffn(self.multihead_attention(queries=tf.concat([img_emb, new_emb, loss_emb], 1),
                                                            keys=tf.concat([img_new_emb, img_loss_emb], 1),
                                                            values=tf.concat([img_new_emb, img_loss_emb], 1),
                                                            query_masks=tf.concat([imgmask, self.newmask, self.lossmask], 1),
                                                            key_masks=tf.concat([self.newmask, self.lossmask], 1))) # (N, 2*length_his+1, 512)
            img_emb = tf.expand_dims(img_his_emb[:, 0, :], 1) # (N, 1, d_model)
            new_emb = img_his_emb[:, 1:self.length_his+1, :] # (N, length_his, d_model)
            loss_emb = img_his_emb[:, -self.length_his:, :] # (N, length_his, d_model)

        user_embeddings = tf.matmul(tf.squeeze(img_emb, [1]),self.bottleneck_weights[0]) + self.bottleneck_biases[0] # (N, 128)
        user_embeddings = get_activation('tanh')(user_embeddings) # (N, 128)
        model_outputs.append(user_embeddings)

        # outputs for the BERT-like masked apps prediction, also tied weights with the app embedding matrix
        logits_bert_new = tf.einsum('ijk,kl->ijl', tf.batch_gather(new_emb,self.newbertindex), self.dae_weights[-1]) + self.dae_biases[-1]
        model_outputs.append(logits_bert_new)
        logits_bert_loss = tf.einsum('ijk,kl->ijl', tf.batch_gather(loss_emb,self.lossbertindex), self.dae_weights[-1]) + self.dae_biases[-1]
        model_outputs.append(logits_bert_loss)

        # main reconstruction of retention
        img_emb = tf.matmul(user_embeddings,self.bottleneck_weights[1]) + self.bottleneck_biases[1] # (N, d_model)
        img_emb = get_activation('leakyrelu')(img_emb) # (N, d_model)
        logits_image = tf.matmul(img_emb, self.dae_weights[-1]) + self.dae_biases[-1] # (N, 10000)
        model_outputs.append(logits_image)

        use_emb = tf.expand_dims(user_embeddings, 1) # (N, 1, 128)

        reduce_dim_layer = tf.layers.Dense(self.dae_dims[1])

        # The modification in self-attention of transformer decoder
        img_new_emb = tf.concat([tf.tile(use_emb, [1, tf.shape(new_emb)[1], 1]), new_pos_emb], 2) # (N ,length_his, 128+d_model)
        img_new_emb = reduce_dim_layer(img_new_emb) # (N ,length_his, d_model)

        img_loss_emb = tf.concat([tf.tile(use_emb, [1, tf.shape(loss_emb)[1], 1]), loss_pos_emb], 2) # (N, length_his, 128+d_model)
        img_loss_emb = reduce_dim_layer(img_loss_emb) # (N ,length_his, d_model)

        self.bottleneck_weights += reduce_dim_layer.weights
        # The transformer decoder
        his_emb = self.ffn(self.multihead_attention(queries=tf.concat([img_new_emb, img_loss_emb], 1),
                                                        keys=tf.concat([img_new_emb, img_loss_emb], 1),
                                                        values=tf.concat([img_new_emb, img_loss_emb], 1),
                                                        query_masks=tf.concat([self.newmask, self.lossmask], 1),
                                                        key_masks=tf.concat([self.newmask, self.lossmask], 1))) # (N, 2*length_his, 512)
        # Main reconstruction of the installation and uninstallation
        logits_new = tf.einsum('ijk,kl->ijl', his_emb[:, :self.length_his, :], self.dae_weights[-1]) + self.dae_biases[-1] # (N, length_his, 10000)
        model_outputs.append(logits_new)
        logits_loss = tf.einsum('ijk,kl->ijl', his_emb[:, -self.length_his:, :], self.dae_weights[-1]) + self.dae_biases[-1] # (N, length_his, 10000)
        model_outputs.append(logits_loss)

        # Prepare for the future fine-tine tasks
        classifier_emb = tf.layers.dropout(user_embeddings, self.classifier_dropout_rate,
                                            training=self.training)

        for i, (w, b) in enumerate(zip(self.classifier_weights, self.classifier_biases)):
            if i != len(self.classifier_weights) - 1:
                classifier_emb = tf.matmul(classifier_emb,w)+b
                classifier_emb = get_activation('leakyrelu')(classifier_emb)
            else:
                logits_classifier = tf.matmul(classifier_emb,w)+b
                model_outputs.append(logits_classifier)

        return model_outputs

    def compute_dae_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.dae_weights[:2])
        return tf.contrib.layers.apply_regularization(self.reg, self.dae_weights)

    def compute_bottleneck_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights[0])
        return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights)

    def compute_transformer_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights[:self.num_translayer*7])
        return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights)

    def compute_classifier_L2(self):
        return tf.contrib.layers.apply_regularization(self.reg, self.classifier_weights)


class VAETN(object):

    def __init__(self, params, inputs, mode):
        self.training = mode == tf.estimator.ModeKeys.TRAIN
        self.parse_params(params)
        self.parse_inputs(inputs)
        self.show_constructs()
        self.construct_weights()

    def parse_params(self, params): # The same with the AETN
        self.train_mlm = params['train_mlm']

        if params['q_dims'] is None:
            params['q_dims'] = params['p_dims'][::-1]
        else:
            assert params['q_dims'][0] == params['p_dims'][-1], "Input and output dimension must equal each other for autoencoders."
            assert params['q_dims'][-1] == params['p_dims'][0], "Latent dimension for -p and q- mismatches."
        self.dae_dims = params['q_dims'] + params['p_dims'][1:]

        assert params['u_dims'] != None, "u_dims cannot be None."
        self.u_dims = params['u_dims']
        self.bottleneck_dims = [self.dae_dims[1], self.u_dims, self.dae_dims[1]]

        assert params['l_dims'] != None, "l_dims cannot be None."
        self.classifier_dims = [self.u_dims] + params['l_dims']

        self.ffn_dim = params['ffn_dim']
        self.num_translayer = params['num_translayer']
        self.num_header = params['num_header']
        self.num_date = params['num_date']
        self.length_his = params['length_his']
        self.softmap = params['softmap']
        self.softmap_vocab = params['softmap_vocab']

        self.image_dropout_rate = params.get('image_dropout_rate', 0)
        self.classifier_dropout_rate = params.get('classifier_dropout_rate', 0)
        self.attention_dropout_rate = params.get('attention_dropout_rate', 0)
        self.ffn_dropout_rate = params.get('ffn_dropout_rate', 0)

        self.lambda_L2 = params.get('lambda_L2', 1.5e-7)
        self.reg = tf.contrib.layers.l2_regularizer(self.lambda_L2)

    def show_constructs(self):
        print("Train with MLM:", self.train_mlm, end='\t')
        print("DAE structure is:", self.dae_dims, end='\t')
        print("BOTTLENECK structure is:", self.bottleneck_dims, end='\t')
        print("CLASSIFIER structure is:", self.classifier_dims, end='\t')
        print("TRANSFORMER structure is:", [self.num_translayer, self.num_header, self.ffn_dim])

    def parse_inputs(self, inputs):
        self.image = inputs[0]
        self.newdate = tf.to_int32(inputs[1])
        self.newapp = tf.to_int32(inputs[2])
        self.newmask = tf.to_int32(inputs[3])
        self.newbertindex = tf.to_int32(inputs[4])
        self.newbertmask = tf.to_int32(inputs[5])
        self.lossdate = tf.to_int32(inputs[6])
        self.lossapp = tf.to_int32(inputs[7])
        self.lossmask = tf.to_int32(inputs[8])
        self.lossbertindex = tf.to_int32(inputs[9])
        self.lossbertmask = tf.to_int32(inputs[10])

    def construct_weights(self):
        self.dae_weights = []
        self.dae_biases = []

        self.classifier_weights = []
        self.classifier_biases = []

        self.bottleneck_weights = []
        self.bottleneck_biases = []

        self.transformer_weights = []

        self.label_weight = tf.gather(tf.get_variable(
            name="label_weight", shape=[self.softmap_vocab, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer()), self.softmap)

        for i, (d_in, d_out) in enumerate(zip(self.dae_dims[:-1], self.dae_dims[1:])):
            if i == 0:
                weight_key = "dae_weight_{}to{}".format(i,i+1)
                print('label_weight', weight_key)
                self.dae_weights.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()) + self.label_weight)
            elif i != len(self.dae_dims) - 2:
                weight_key = "dae_weight_{}to{}".format(i,i+1)
                print('normal_weight', weight_key)
                self.dae_weights.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))
            else:
                print('tie_weight', i)
                self.dae_weights.append(tf.transpose(self.dae_weights[0]))

            bias_key = "dae_bias_{}".format(i+1)
            self.dae_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        for i, (d_in, d_out) in enumerate(zip(self.classifier_dims[:-1], self.classifier_dims[1:])):
            weight_key = "classifier_weight_{}to{}".format(i,i+1)
            bias_key = "classifier_bias_{}".format(i+1)
            print(weight_key)
            self.classifier_weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer()))
            self.classifier_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        for i, (d_in, d_out) in enumerate(zip(self.bottleneck_dims[:-1], self.bottleneck_dims[1:])):
            weight_key = "bottleneck_weight_{}to{}".format(i,i+1)
            bias_key = "bottleneck_bias_{}".format(i+1)
            print(weight_key)
            self.bottleneck_weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer()))
            self.bottleneck_biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(stddev=0.001)))

        self.app_embeddings = tf.concat([self.dae_weights[0], tf.zeros(shape=[1, self.dae_dims[1]])], 0)
        self.position_embeddings = tf.concat(
            [tf.get_variable(name="position_embeddings", shape=[self.num_date, self.dae_dims[1]],
                            initializer=tf.contrib.layers.xavier_initializer()),
            tf.zeros(shape=[1, self.dae_dims[1]])], 0)
        self.new_embeddings = tf.concat([tf.zeros(shape=[1, self.dae_dims[1]]),
            tf.get_variable(name="new_embeddings", shape=[1, self.dae_dims[1]],
                initializer=tf.contrib.layers.xavier_initializer())], 0)
        self.loss_embeddings = tf.concat([tf.zeros(shape=[1, self.dae_dims[1]]),
            tf.get_variable(name="loss_embeddings", shape=[1, self.dae_dims[1]],
                initializer=tf.contrib.layers.xavier_initializer())], 0)
        self.img_embeddings = tf.get_variable(name="img_embeddings", shape=[1, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer())
        self.bert_mask_embeddings = tf.get_variable(name="bert_mask_embeddings", shape=[1, self.dae_dims[1]],
            initializer=tf.contrib.layers.xavier_initializer())

    def multihead_attention(self, queries, keys, values, query_masks, key_masks):
        '''
        queries: A 3d tensor with shape of [N, T_q, d_q].
        keys: A 3d tensor with shape of [N, T_k, d_k].
        values: A 3d tensor with shape of [N, T_v, d_v]. T_v = T_k
        query_masks:A 2d tensor with shape of [N, T_q]
        key_masks:A 2d tensor with shape of [N, T_k]

        return: A 3d tensor with shape of [N, T_q, d_model]
        '''

        d_model = queries.get_shape().as_list()[-1]

        Q_layer = tf.layers.Dense(d_model, use_bias=False)
        Q = Q_layer(queries)
        self.transformer_weights += Q_layer.weights
        K_layer = tf.layers.Dense(d_model, use_bias=False)
        K = K_layer(keys)
        self.transformer_weights += K_layer.weights
        V_layer = tf.layers.Dense(d_model, use_bias=False)
        V = V_layer(values)
        self.transformer_weights += V_layer.weights

        # Split and concat
        Q_ = tf.concat(tf.split(Q, self.num_header, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_header, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_header, axis=2), axis=0) # (h*N, T_v, d_model/h)
        query_masks = tf.tile(query_masks, [self.num_header, 1]) # (h*N, T_q)
        key_masks = tf.tile(key_masks, [self.num_header, 1]) # (h*N, T_k)

        # Attention
        outputs = self.scaled_dot_product_attention(Q_, K_, V_, query_masks, key_masks) # (h*N, T_q, d_model/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_header, axis=0), axis=2) # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        return tf.contrib.layers.layer_norm(outputs)

    def scaled_dot_product_attention(self, Q, K, V, Q_masks, K_masks):
        '''
        Q: Packed queries. 3d tensor. [h*N, T_q, d_model/h].
        K: Packed keys. 3d tensor. [h*N, T_k, d_model/h].
        V: Packed values. 3d tensor. [h*N, T_v, d_model/h].
        Q_masks: Packed masks. 2d tensor. [h*N, T_q].
        K_masks: Packed masks. 2d tensor. [h*N, T_k].

        return: A 3d tensor with shape of [h*N, T_q, d_model/h]
        '''

        d_k = K.get_shape().as_list()[-1] # d_model/h

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (h*N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        padding_num = -1e+7
        K_masks = tf.expand_dims(K_masks, 1) # (h*N, 1, T_k)
        K_masks = tf.tile(K_masks, [1, tf.shape(Q)[1], 1]) # (h*N, T_q, T_k)
        paddings = tf.ones_like(outputs) * padding_num
        outputs = tf.where(tf.equal(K_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # softmax
        outputs = tf.nn.softmax(outputs)

        # query masking
        Q_masks = tf.expand_dims(Q_masks, -1) # (h*N, T_q, 1)
        Q_masks = tf.tile(Q_masks, [1, 1, tf.shape(K)[1]]) # (h*N, T_q, T_k)
        outputs = outputs * tf.to_float(Q_masks)

        # dropout
        outputs = tf.layers.dropout(outputs, self.attention_dropout_rate, training=self.training)

        return tf.matmul(outputs, V) # [h*N, T_q, d_model/h]

    def ffn(self, inputs):
        # Inner layer
        ffn_layer_1 = tf.layers.Dense(self.ffn_dim, activation=get_activation('relu'))
        outputs = ffn_layer_1(inputs)
        self.transformer_weights += ffn_layer_1.weights

        outputs = tf.layers.dropout(outputs, self.ffn_dropout_rate, training=self.training)

        # Outer layer
        d_model = inputs.get_shape().as_list()[-1]
        ffn_layer_2 = tf.layers.Dense(d_model)
        outputs = ffn_layer_2(outputs)
        self.transformer_weights += ffn_layer_2.weights

        # Residual connection
        outputs += inputs

        return tf.contrib.layers.layer_norm(outputs)

    def forward_pass(self):
        model_outputs = []

        new_app_emb = tf.nn.embedding_lookup(self.app_embeddings, self.newapp) # (N, length, d_model)
        loss_app_emb = tf.nn.embedding_lookup(self.app_embeddings, self.lossapp) # (N, length, d_model)

        new_pos_emb = tf.nn.embedding_lookup(self.position_embeddings, self.newdate) + \
                        tf.nn.embedding_lookup(self.new_embeddings, self.newmask)
        loss_pos_emb = tf.nn.embedding_lookup(self.position_embeddings, self.lossdate) + \
                        tf.nn.embedding_lookup(self.loss_embeddings, self.lossmask)

        mask_emb = tf.nn.embedding_lookup(self.bert_mask_embeddings,
                                            tf.zeros_like(self.newbertmask, tf.int32)) # (N, length, d_model)

        new_bert_masks = tf.tile(tf.expand_dims(self.newbertmask * self.training * self.train_mlm, 2), [1, 1, self.dae_dims[1]]) # (N, length, d_model)
        loss_bert_masks = tf.tile(tf.expand_dims(self.lossbertmask * self.training * self.train_mlm, 2), [1, 1, self.dae_dims[1]]) # (N, length, d_model)

        new_emb = tf.where(tf.equal(new_bert_masks, 0), new_app_emb, mask_emb) + new_pos_emb
        loss_emb = tf.where(tf.equal(loss_bert_masks, 0), loss_app_emb, mask_emb) + loss_pos_emb

        img = tf.math.l2_normalize(self.image, 1)
        img = tf.layers.dropout(img, self.image_dropout_rate, training=self.training)

        for i, (w, b) in enumerate(zip(self.dae_weights, self.dae_biases)):
            if i == 0:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
                img_emb = tf.expand_dims(img, 1) # (N, 1, d_model)
            if i == 1:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
                # user_embeddings = img # (N, 128)
            if i == 2:
                img = tf.matmul(img,w)+b
                img = get_activation('leakyrelu')(img)
            if i == 3:
                logits_dae = tf.matmul(img,w)+b # (N, 10000)
                model_outputs.append(logits_dae)

        img_emb = img_emb + tf.nn.embedding_lookup(self.img_embeddings,
                                tf.zeros_like(tf.reduce_sum(img_emb, 2), tf.int32)) # (N ,1, d_model)
        imgmask = tf.ones_like(tf.reduce_sum(img_emb, 2), tf.int32) # (N, 1)

        for trans_layer in range(self.num_translayer):
            # The vanilla multi-head self-attention based transformer encoder
            img_his_emb = self.ffn(self.multihead_attention(queries=tf.concat([img_emb, new_emb, loss_emb], 1),
                                                            keys=tf.concat([img_emb, new_emb, loss_emb], 1),
                                                            values=tf.concat([img_emb, new_emb, loss_emb], 1),
                                                            query_masks=tf.concat([imgmask, self.newmask, self.lossmask], 1),
                                                            key_masks=tf.concat([imgmask, self.newmask, self.lossmask], 1))) # (N, 2*length_his+1, d_model)
            img_emb = tf.expand_dims(img_his_emb[:, 0, :], 1) # (N, 1, d_model)
            new_emb = img_his_emb[:, 1:self.length_his+1, :] # (N, length_his, d_model)
            loss_emb = img_his_emb[:, -self.length_his:, :] # (N, length_his, d_model)

        user_embeddings = tf.matmul(tf.squeeze(img_emb, [1]),self.bottleneck_weights[0]) + self.bottleneck_biases[0] # (N, 128)
        user_embeddings = get_activation('tanh')(user_embeddings) # (N, 128)
        model_outputs.append(user_embeddings)

        logits_bert_new = tf.einsum('ijk,kl->ijl', tf.batch_gather(new_emb,self.newbertindex), self.dae_weights[-1]) + self.dae_biases[-1] # (N, 3, 10000)
        model_outputs.append(logits_bert_new)
        logits_bert_loss = tf.einsum('ijk,kl->ijl', tf.batch_gather(loss_emb,self.lossbertindex), self.dae_weights[-1]) + self.dae_biases[-1] # (N, 3, 10000)
        model_outputs.append(logits_bert_loss)

        img_emb = tf.matmul(user_embeddings,self.bottleneck_weights[1]) + self.bottleneck_biases[1] # (N, d_model)
        img_emb = get_activation('leakyrelu')(img_emb) # (N, d_model)
        logits_image = tf.matmul(img_emb, self.dae_weights[-1]) + self.dae_biases[-1] # (N, 10000)
        model_outputs.append(logits_image)

        use_emb = tf.expand_dims(user_embeddings, 1) # (N, 1, 128)

        increase_dim_layer = tf.layers.Dense(self.dae_dims[1])

        use_emb = increase_dim_layer(use_emb) # (N, 1, d_model)

        self.bottleneck_weights += increase_dim_layer.weights

        # The vanilla multi-head self-attention based transformer encoder
        his_emb = self.ffn(self.multihead_attention(queries=tf.concat([new_pos_emb, loss_pos_emb], 1),
                                                        keys=tf.concat([use_emb, new_pos_emb, loss_pos_emb], 1),
                                                        values=tf.concat([use_emb, new_pos_emb, loss_pos_emb], 1),
                                                        query_masks=tf.concat([self.newmask, self.lossmask], 1),
                                                        key_masks=tf.concat([imgmask, self.newmask, self.lossmask], 1))) # (N, 2*length_his, d_model)
        logits_new = tf.einsum('ijk,kl->ijl', his_emb[:, :self.length_his, :], self.dae_weights[-1]) + self.dae_biases[-1] # (N, length_his, 10000)
        model_outputs.append(logits_new)
        logits_loss = tf.einsum('ijk,kl->ijl', his_emb[:, -self.length_his:, :], self.dae_weights[-1]) + self.dae_biases[-1] # (N, length_his, 10000)
        model_outputs.append(logits_loss)


        classifier_emb = tf.layers.dropout(user_embeddings, self.classifier_dropout_rate,
                                            training=self.training)

        for i, (w, b) in enumerate(zip(self.classifier_weights, self.classifier_biases)):
            if i != len(self.classifier_weights) - 1:
                classifier_emb = tf.matmul(classifier_emb,w)+b
                classifier_emb = get_activation('leakyrelu')(classifier_emb)
            else:
                logits_classifier = tf.matmul(classifier_emb,w)+b
                model_outputs.append(logits_classifier)

        return model_outputs

    def compute_dae_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.dae_weights[:2])
        return tf.contrib.layers.apply_regularization(self.reg, self.dae_weights)

    def compute_bottleneck_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights[0])
        return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights)

    def compute_transformer_L2(self, only_part=False):
        if only_part:
            return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights[:self.num_translayer*7])
        return tf.contrib.layers.apply_regularization(self.reg, self.bottleneck_weights)

    def compute_classifier_L2(self):
        return tf.contrib.layers.apply_regularization(self.reg, self.classifier_weights)
