#!/usr/bin/env python3
#encoding=utf-8

import os
import re
import sys
import sqlite3
from collections import Counter

from tqdm import tqdm

def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False

def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if len(a) <= max_len:
            return True
    return False

def insert(a, b, cur):
    cur.execute("""
    INSERT INTO conversation (ask, answer) VALUES
    ('{}', '{}')
    """.format(a.replace("'", "''"), b.replace("'", "''")))

def insert_if(question, answer, cur, input_len=500, output_len=500):
    if valid(question, input_len) and valid(answer, output_len):
        insert(question, answer, cur)
        return 1
    return 0

def main(file_path):
    with open(file_path, 'r',encoding="utf-8") as fp:
        lines = fp.readlines()
    print('一共读取 %d 行数据' % len(lines))

    db = 'db/conversation.db'
    if os.path.exists(db):
        os.remove(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS conversation(ask text, answer text);""")
    conn.commit()

    words = Counter()
    ask = ''
    answer = ''
    inserted = 0

    for index, line in tqdm(enumerate(lines), total=len(lines)):
        if line.strip() != "E":
            ask = answer
            answer = line.strip()[2:]
            words.update(Counter(answer))#注意，频率词典作为副产品，也产生了
        else:
            inserted += insert_if(ask, answer, cur)
        # 批量提交
        if inserted != 0 and inserted % 5000 == 0:
            conn.commit()
    conn.commit()

if __name__ == '__main__':
    file_path = 'db/xiaohuangji50w_nofenci.conv'
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print('文件 {} 不存在'.format(file_path))
    else:
        main(file_path)

