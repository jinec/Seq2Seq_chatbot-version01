#encoding=utf8
import pdb
import random
import copy
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data_03_utils

class S2SModel(object):
    def __init__(self,
                source_vocab_size,
                target_vocab_size,#翻译系统中有2套VC表
                buckets,
                size,
                dropout,
                num_layers, #LSTM的层数
                max_gradient_norm,
                batch_size,
                learning_rate,
                num_samples, #分批softmax的样本量，默认为512
                forward_only=False,#只有前向的话，说明是 测试过程。这里是双向
                dtype=tf.float32):
        # init member variales
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # LSTM cells
        cell = tf.contrib.rnn.BasicLSTMCell(size) #size=输出单元的维度，即字的维度
        #输入：内有一个维度映射_linear，
        #用函数_call_中的[w,bias]将[inputs, h]转换为second dimension of weight variable=4*self._num_units=size的输出
        #输出：BasicLSTMCell.call()输出的是 new_h, new_state;话说这个函数实例化后怎么调用呢?直接cell（），不用带.call,这个风格跟init（）很相似
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)#加上 dropout，这里是 保留参数！
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)#一共是2层

        output_projection = None #求出具体的字的one_hot

        print('开启投影：{}'.format(num_samples))
        w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)#字的维度[6865,512]，这部分功能和_linear的不重合吗？_linear的功能这里没用啊。
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [self.target_vocab_size],dtype=dtype)
        output_projection = (w, b) #定义了一个输出投影层。

        def loss_function(labels, logits):#从512升到6865，然后副采样，求loss
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32) #（6865，512）
                local_b = tf.cast(b, tf.float32)#（6865）
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(tf.nn.sampled_softmax_loss(weights=local_w_t, biases=local_b,labels=labels,inputs=local_inputs,num_sampled=num_samples, num_classes=self.target_vocab_size),dtype)

        # seq2seq_f
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            # Encoder.先将cell进行deepcopy，因为seq2seq模型是两个相同的模型，但是模型参数不共享，所以encoder和decoder要使用两个不同的RnnCell
            tmp_cell = copy.deepcopy(cell)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,decoder_inputs,tmp_cell, num_encoder_symbols=source_vocab_size, num_decoder_symbols=target_vocab_size,
                embedding_size=size, output_projection=output_projection, feed_previous=do_decode,# 是否用前面的预测值作为decoder的输入
                dtype=dtype)

        # inputs
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_weights = []
        #decoder_weights 是一个于decoder_outputs大小一样的0-1矩阵，该矩阵将目标序列长度以外的其他位置填0。比如[1(我),1(是),1(中),1(国),1(人),0,0,0,0,0]
        # buckets中的最后一个是最大的（即第“-1”个），0指的是编码器的20。这里是分配资源，财政预算以最大的准！
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='encoder_input_{}'.format(i)))
        for i in range(buckets[-1][1] + 1):  #位置：0：30
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='decoder_input_{}'.format(i)))
            self.decoder_weights.append(tf.placeholder(dtype,shape=[None],name='decoder_weight_{}'.format(i)))
        targets = [self.decoder_inputs[i + 1] for i in range(buckets[-1][1])] 


        if forward_only:# 测试阶段  seq2seq_f --> loss_function   --> model_with_buckets
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs, self.decoder_inputs, targets,self.decoder_weights, buckets,lambda x, y: seq2seq_f(x, y, True),softmax_loss_function=loss_function)#测试阶段，seq2seq_f、softmax_loss_function不执行
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [ tf.matmul(output, output_projection[0] ) + output_projection[1] for output in self.outputs[b]  ]
        else:#训练阶段
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(self.encoder_inputs,self.decoder_inputs,targets,self.decoder_weights,buckets,lambda x, y: seq2seq_f(x, y, False),softmax_loss_function=loss_function)
        params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if not forward_only:# 只有训练阶段才需要计算梯度和参数更新.不同点只有这一处，此外还有dropout和输出结果
            self.gradient_norms = []
            self.updates = []
            for output, loss in tqdm(zip(self.outputs, self.losses)):# 用梯度下降法优化
                gradients = tf.gradients(loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params)))
        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2)

    def step(self,session,encoder_inputs,decoder_inputs,decoder_weights,bucket_id,forward_only):
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket," " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket," " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(decoder_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket," " %d != %d." % (len(decoder_weights), decoder_size))

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
            #Tensor类型的name，多GPU时可以方便参数共享，单GPU不必..
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.decoder_weights[i].name] = decoder_weights[i]

        # 最后这一位是没东西的，所以要补齐最后一位，填充0.。0是EOS的位置，0的真正位置是2（pad）
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_feed = [self.updates[bucket_id],self.gradient_norms[bucket_id], self.losses[bucket_id]]
        else:#第一位置的loss函数，意义：形式上的统一，实际意义不大。
            output_feed = [self.losses[bucket_id]]#self.losses也是按桶区分的吗？
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])#[1个损失，20个输出]

        outputs = session.run(output_feed, input_feed)
        #len(outputs)比output_feed多一位的意义？第【0】位，是self.losses
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:]
        else:
            return None, outputs[0], outputs[1:]#20*6865

    def get_batch_data(self, bucket_dbs, bucket_id):
        data = []
        data_in = []
        bucket_db = bucket_dbs[bucket_id] #字典【标号】
        for _ in range(self.batch_size):#前面自定义的，一批量的数目；默认是 1句话
            ask, answer = bucket_db.random()#这个bucket——dbs在测试时，为TestBucket
            #训练时，从标号桶里随机选了一对问答，一共batch_size对。
            data.append((ask, answer))
            data_in.append((answer, ask))
        return data, data_in

    def get_batch(self, bucket_dbs, bucket_id, data,word_index):
        encoder_size, decoder_size = self.buckets[bucket_id] #(15, 25)
        # bucket_db = bucket_dbs[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for encoder_input, decoder_input in data:#这句话可以直接拆开列表中的问答tuple，厉害
            # encoder_input, decoder_input = random.choice(data[bucket_id])
            # encoder_input, decoder_input = bucket_db.random()
            #把输入句子转化为id
            encoder_input = data_03_utils.sentence_indice(encoder_input,word_index)#得到每个字在word_index中的位置列表
            decoder_input = data_03_utils.sentence_indice(decoder_input,word_index)
            # Encoder
            encoder_pad = [word_index['<pad>']] * ( encoder_size - len(encoder_input))#补2（pad的位置）
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))#把位置翻了过来！
            # Decoder：padding:桶长-实际解码句长（GO与EOS占了2个位置），这种说法与 ‘target == data_utils.PAD_ID’冲突！
            decoder_pad_size = decoder_size - len(decoder_input) #可看出最多29个字；最后一个字没用，后面删除了！
            decoder_inputs.append(
                [word_index['<go>']] + decoder_input +
                [word_index['<pad>']] * decoder_pad_size  #
            )
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # batch encoder
        for i in range(encoder_size): #在标准的句长里面
            batch_encoder_inputs.append(np.array(
                [encoder_inputs[j][i] for j in range(self.batch_size)],#得到每句话的字典位置列表
                dtype=np.int32
            ))#格式[[第一句的第1个字位，第一句的第1个字位]，[第一句的第2个字位，第一句的第2个字位]....]
        # batch decoder
        for i in range(decoder_size):#次数限制了decoder的最后一个数字上的文字必须被删除！
            batch_decoder_inputs.append(np.array(
                [decoder_inputs[j][i] for j in range(self.batch_size)],
                dtype=np.int32
            ))
            batch_weight = np.ones(self.batch_size, dtype=np.float32)#生成一个self.batch_size维全为1的列表
            for j in range(self.batch_size):#每一批都要进行的意思
                if i < decoder_size - 1:
                    target = decoder_inputs[j][i + 1]#提前了1位显示，探警器；只分0/1的前后就行！
                if i == decoder_size - 1 or target == word_index['<pad>']:
                    batch_weight[j] = 0.0## batch_weights最后一位为0，即最后一个字没用
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
