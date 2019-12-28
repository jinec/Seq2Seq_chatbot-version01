版本说明：该版本是我debug的已有的seq2seq_chatbot模型。经过我debug后，已经可以跑通。

（1）修改一：跟新了预料，最新采用网络上流行的小黄鸡预料库来进行训练，为此还修改了数据处理过程。尽管如此，对话效果还不是很好。

（2）该模型中最重要的是2点，每个都很有难度。

  （2.1）seq2seq模型：这里选取的是 embedding_attention_seq2seq函数，主要有3个难点。
  
      （1）seq2seq模型本身的强大魅力.
      （2）decoder的attention()函数。      
      （3）在decoder的输入中，ci代表什么？
            目前理解，在不加入attention机制前，ci代表句意向量；加入attention机制后，代表attention（state）后的输出。
  （2.2）损失函数模型：这里选取的是，主要由两部分构成。

    （2.2.1）_compute_sampled_logits函数：Helper function for nce_loss and sampled_softmax_loss functions.
            Returns:
                out_logits: `Tensor` object with shape [batch_size, num_true + num_sampled]`, for passing to either
                            `nn.sigmoid_cross_entropy_with_logits` (NCE) or nn.softmax_cross_entropy_with_logits_v2` (sampled softmax).
                out_labels: A Tensor object with the same shape as `out_logits`.
            主要函数包括：
            （1）log_uniform_candidate_sampler            
                   The base distribution for this operation is an approximately log-uniform or Zipfian distribution:
                   `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`
            （2）  采样依靠矩阵相乘来完成
            sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)
            # inputs has shape [batch_size, dim]；sampled_w has shape [num_sampled, dim]； Apply X*W', which yields [batch_size,   
            num_sampled]
    （2.2.2）softmax_cross_entropy_with_logits_v2函数：
           sampled_losses = nn_ops.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
           注释：  Measures the probability error in discrete classification tasks in which the
                  classes are mutually exclusive (each entry is in exactly one class).  For
                  example, each CIFAR-10 image is labeled with one and only one label: an image
                  can be a dog or a truck, but not both.

                  **NOTE:**  While the classes are mutually exclusive, their probabilities
                  need not be.  All that is required is that each row of `labels` is
                  a valid probability distribution.  If they are not, the computation of the
                  gradient will be incorrect.

                  If using exclusive `labels` (wherein one and only
                  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.
           中文翻译：
                 注意：尽管所有的类别是互斥的，它们的概率可能不是互斥的。因此，就要求 labels 中的每一行都是一个有效的概率分布，如果不是，将会导致计                  算的梯度不正确如果使用互斥 labels （即每次有且仅有一个类别），参见： sparse_softmax_cross_entropy_with_logits。
                 1.相同点
                  两者都是先经过softmax处理，然后来计算交叉熵，并且最终的结果是一样的，再强调一遍，最终结果都一样。那既然有了  
                  softmax_cross_entropy_with_logits 这个方法，那么sparse_softmax_cross_entropy_with_logit 有何用？
                  按照《TensorFlow实战Google深度学习框架》中的说法：在只有一个正确答案的分类问题中，TensorFlow提供了
                  sparse_softmax_cross_entropy_with_logit 函数来进一步加速计算过程。例如手写体识别中，每个图片都只代表唯一数字。
                  2.不同点
                  不同点在于两者在传递参数时的形式上。
                  对于softmax_cross_entropy_with_logits 来说，其logits= 的维度是[batch_size,num_classes]，即正向传播最后的输出层结果；labels=
                  的维度也是[batch_size,num_classes]，即正确标签的one_hot形式。
                  对于sparse_softmax_cross_entropy_with_logit来说，其logits= 的维度是[batch_size,num_classes]，即正向传播最后的输出层结果；但
                  labels=的维度有所不同，此时就不再是one_hot形式的标签，而是每一个标签所代表的真实答案，其维度为[batch_size]的一个一维张量。

           
                  
           
          
    
 
