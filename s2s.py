#encoding:utf8

import os
import sys
import math
import time
import numpy as np
import tensorflow as tf
import data_03_utils
import s2s_model

#FLAGS.learning_rate
tf.app.flags.DEFINE_float('learning_rate',0.0003,'学习率')
tf.app.flags.DEFINE_float('max_gradient_norm',5.0,'梯度最大阈值')
tf.app.flags.DEFINE_float('dropout',1.0,'每层输出DROPOUT的大小')#怎么给了1？现在是 测试阶段，不用，所以给了1.
tf.app.flags.DEFINE_integer('batch_size', 32, '小批量梯度下降的批量大小')
tf.app.flags.DEFINE_integer('size',512,'LSTM每层神经元数量')#定义看一下，推荐使用 100-300，把它降下来，以免跑不起来出现 OOM。
tf.app.flags.DEFINE_integer('num_layers',2,'LSTM的层数')
tf.app.flags.DEFINE_integer('num_epoch', 41,'训练几轮')#怎么感觉有问题？嗯，初部估计，以前那个有per
tf.app.flags.DEFINE_integer('num_samples',512,'分批softmax的样本量')#什么意思？和 word2vec意思一样，和前面的512没有任何关系。
tf.app.flags.DEFINE_integer('num_per_epoch',10000,'每轮训练多少随机样本')#这是多少个batch
tf.app.flags.DEFINE_string('buckets_dir','D:\\oo_自然语言处理\\aa_seq2seqModel\\tf_Seq2Seq_Chatbot\\Seq2Seq_chatbot\\bucket_dbs','sqlite3数据库所在文件夹')#“桶”的意思：【问题长度，答案长度】；实例只能都小于，才能放进去；从小到大桶的选择；弊端：破坏上下文情景的连贯性，应用限制在QA类型.
tf.app.flags.DEFINE_string('model_dir','D:\\oo_自然语言处理\\aa_seq2seqModel\\tf_Seq2Seq_Chatbot\\Seq2Seq_chatbot\\model','模型保存的目录')
tf.app.flags.DEFINE_string('model_name','model3','模型保存的名称')
tf.app.flags.DEFINE_boolean('use_fp16',False, '是否使用16位浮点数（默认32位）')
tf.app.flags.DEFINE_integer('bleu', -1,'是否测试bleu')# 机械翻译的标准，Google的准则
tf.app.flags.DEFINE_boolean('test',True,'是否在测试')
tf.app.flags.DEFINE_string('DICTIONARY_PATH',"db\dictionary.json",'加载字典')
tf.app.flags.DEFINE_string('train_device', "/gpu:0",'运行设备')
tf.app.flags.DEFINE_string('test_device', "/cpu:0",'运行设备')
FLAGS = tf.app.flags.FLAGS

# 加载桶信息
buckets = [(5, 15), (10, 20), (15, 25), (20, 30)]
# 加载字典，完成“字-索引”的转换
EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'
GO = '<go>'
dictDim, dictionary, index_word, word_index = data_03_utils.load_dictionary(FLAGS.DICTIONARY_PATH, EOS, UNK, PAD, GO)

def create_model(session, forward_only):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32 #存储空间一定减少，运算速度不一定减少，因为有些数据线硬件（映射量）本身就是32位的
    model = s2s_model.S2SModel(dictDim,dictDim,buckets, FLAGS.size, FLAGS.dropout, FLAGS.num_layers,FLAGS.max_gradient_norm,FLAGS.batch_size, FLAGS.learning_rate,
        FLAGS.num_samples, forward_only, dtype )
    return model

def train():
    """训练模型"""
    print('准备数据')
    bucket_dbs = data_03_utils.read_bucket_dbs(FLAGS.buckets_dir,buckets)#
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)#列表求和
    print('共有数据 {} 条'.format(total_size))
    # 开始建模与训练
    with tf.Session() as sess:
        #　构建模型
        model = create_model(sess, False)
        print("modle is readey!")
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]#一个元素和为1的列表
        # 开始训练
        metrics = '  '.join(['\r[{}]','{:.1f}%','{}/{}','loss={:.3f}','{}/{}'])
        bars_max = 20
        with tf.device('/gpu:0'):
            for epoch_index in range(1, FLAGS.num_epoch):
                print('Epoch {}:'.format(epoch_index))
                time_start = time.time()
                epoch_trained = 0 #第一个指标
                batch_loss = []#第2个指标
                while True:
                    # 随机选择一个要训练的bucket
                    random_number = np.random.random_sample()
                    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
                    data, _ = model.get_batch_data(bucket_dbs, bucket_id) #随机抽到足量的批数据
                    encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(bucket_dbs,bucket_id,data,word_index)# 数据转换
                    _, step_loss, output = model.step(sess,encoder_inputs,decoder_inputs,decoder_weights,bucket_id,False)#预测
                    batch_loss.append(step_loss)
                    epoch_trained += FLAGS.batch_size
                    # 写显示的进度条
                    time_now = time.time()
                    time_spend = time_now - time_start
                    time_estimate = time_spend / (epoch_trained / FLAGS.num_per_epoch)
                    percent = min(100, epoch_trained / FLAGS.num_per_epoch) * 100
                    bars = math.floor(percent / 100 * bars_max)
                    sys.stdout.write(metrics.format( #类似于print
                        '=' * bars + '-' * (bars_max - bars),
                        percent,
                        epoch_trained, FLAGS.num_per_epoch,#话说，‘/’怎么出现的？
                        np.mean(batch_loss),
                        data_03_utils.time(time_spend), data_03_utils.time(time_estimate)
                    ))#打印进度条
                    sys.stdout.flush()
                    if epoch_trained >= FLAGS.num_per_epoch: #一个epoch就跑 10000个样本，就结束！
                        break
                print('\n')
                if not os.path.exists(FLAGS.model_dir):
                    os.makedirs(FLAGS.model_dir)
                if epoch_index%40==0:
                    model.saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

def test_bleu(count):
    """测试bleu"""
    from nltk.translate.bleu_score import sentence_bleu
    from tqdm import tqdm
    # 准备数据
    print('准备数据')
    bucket_dbs = data_03_utils.read_bucket_dbs(FLAGS.buckets_dir,buckets)
    bucket_sizes = []
    for i in range(len(buckets)):
        bucket_size = bucket_dbs[i].size
        bucket_sizes.append(bucket_size)
        print('bucket {} 中有数据 {} 条'.format(i, bucket_size))
    total_size = sum(bucket_sizes)
    print('共有数据 {} 条'.format(total_size))
    # bleu设置0的话，默认对所有样本采样
    if count <= 0:
        count = total_size
    buckets_scale = [
        sum(bucket_sizes[:i + 1]) / total_size
        for i in range(len(bucket_sizes))
    ]
    with tf.Session() as sess:
        #　构建模型
        model = create_model(sess, True)
        model.batch_size = 1
        # 初始化变量
        sess.run(tf.initialize_all_variables())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))

        total_score = 0.0
        for i in tqdm(range(count)):
            # 选择一个要训练的bucket
            random_number = np.random.random_sample()
            bucket_id = min([
                i for i in range(len(buckets_scale))
                if buckets_scale[i] > random_number
            ])
            data, _ = model.get_batch_data(bucket_dbs,bucket_id)
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch(bucket_dbs,bucket_id,data,word_index)
            _, _, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                decoder_weights,
                bucket_id,
                True
            )
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ask, _ = data[0]
            all_answers = bucket_dbs[bucket_id].all_answers(ask)
            ret = data_03_utils.indice_sentence(outputs,index_word)
            if not ret:
                continue
            references = [list(x) for x in all_answers]
            score = sentence_bleu(
                references,
                list(ret),
                weights=(1.0,)
            )
            total_score += score
        print('BLUE: {:.2f} in {} samples'.format(total_score / count * 10, count))

def test():
    class TestBucket(object):
        def __init__(self, sentence):
            self.sentence = sentence
        def random(self):  # 这个地方其实还可以添加功能：把多句话分开为一个列表。
            return sentence, ''
    with tf.Session() as sess: #主要就三步代码！
        #　构建模型，即先放一个空架子。
        model = create_model(sess, True)#ture表示 测试过程，仅forward_only
        model.batch_size = 1
        # 初始化变量
        import os
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name))
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()#我干你老妈啊
        while sentence:
            #获取最小的分桶id
            bucket_id = min([ b for b in range(len(buckets))  if buckets[b][0] > len(sentence) ])#长度包括最后的'n',得到1
            #输入句子处理.，得到问答对data，__接收的是答问对
            data, _ = model.get_batch_data( {bucket_id: TestBucket(sentence)}, bucket_id)
            #下一句：文字的ids化，输入句子还被反转了一下。<class 'list'>: [array([2，，]), array([2]), array([2]), array([1])..]
            encoder_inputs, decoder_inputs, decoder_weights = model.get_batch( {bucket_id: TestBucket(sentence)},  bucket_id, data,word_index )
            #下一句：
            _, _, output_logits = model.step(sess,encoder_inputs,decoder_inputs,decoder_weights, bucket_id,True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            ret = data_03_utils.indice_sentence(outputs,index_word)
            print(ret)#输出答案。
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(self):
    if FLAGS.bleu > -1:
        test_bleu(FLAGS.bleu)
    elif FLAGS.test:
        test()
    else:
        train()

if __name__ == '__main__':
    np.random.seed(0)
    # tf.set_random_seed(0)
    tf.app.run()