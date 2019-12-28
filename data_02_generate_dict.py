# encoding:utf-8

import re
import json
#
with open('db/chinese.txt', 'r',encoding="utf-8") as fp:
    content = fp.read()
    words = list(set(list(re.findall('[\u4e00-\u9fa5]', content))))
words = sorted(words)
print(len(words))


#
with open('db/gb2312_level1.txt', 'r',encoding="utf-8") as fp:
    content = fp.read()
    gb2312_level1 = list(set(list(re.findall('[\u4e00-\u9fa5]', content))))
gb2312_level1 = sorted(gb2312_level1)
print(len(gb2312_level1))


#
with open('db/gb2312_level2.txt', 'r',encoding="utf-8") as fp:
    content = fp.read()
    gb2312_level2 = list(set(list(re.findall('[\u4e00-\u9fa5]', content))))
gb2312_level2 = sorted(gb2312_level2)
print(len(gb2312_level2))


#
chinese_punctuation = '。？！，、；：“”‘’（）《》—　'
english_punctuation = '.,;:!\'"-[](){}…<>/ '
number = '0123456789'
alphabet = 'abcdefghijklmnopqrstuvwxyz'
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
all_dictionary =      list(number) + \
    list(alphabet) + \
    list(ALPHABET) + \
    words+gb2312_level1 + gb2312_level2+ \
    list(chinese_punctuation) + \
    list(english_punctuation)

# all_dictionary = list(set(all_dictionary))
print(len(all_dictionary))
print("字典构建完成")


#
with open('db/dictionary.json', 'w',encoding="utf-8") as fp:
    json.dump(all_dictionary, fp, indent=4, ensure_ascii=False)