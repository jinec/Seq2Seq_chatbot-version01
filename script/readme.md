1.
   先取出数据并利用该数据文字数字化建桶
注意：详解线上部署 离线训练与线上部署
2. 运行Seq2seq_Chatbot/Seq2Seq_Chatbot中的 s2s.py softmax解释 对总流程写笔记
   对数字化过程详细讲解调试
3. seq2seq_note.py是注释版，如果想放入源码中，找出源码路径，将其改为seq2seq.py将源码替换掉即可(注：源码最好做个备份)

(1)dgk_shooter_min.conv是原始的数据，通过decode_conv.py转化成conversation.db；
(2)data_utils.py创建好db分桶

(1)读文本，QA；
（2）分桶
（3）组batch——size
（4）ids化

