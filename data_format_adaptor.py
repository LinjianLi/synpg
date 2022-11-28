"""
用于各种数据格式转换。
不同模型、项目代码用到的数据格式不一样。
"""

import argparse
from tqdm import tqdm
import h5py

def split_file():
    """
    分割成小文件。因为单个大文件，Stanford CoreNLP 的 Java 包用单线程处理太慢，还引发了 GC 异常。分成多个小文件之后，用多线程分别处理就能成功。
    """

    with open("paranmt500k-tab2newline.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    split_size = len(lines) // 500 + 1
    print("Total number of sentences: {}.\nNumber of sentences in each split (500 splits in total): {}.".format(len(lines), split_size))
    for i in tqdm(range(500)):
        with open("/home/llj/synpg/data/paranmt500k/paranmt500k-tab2newline.{part}_of_500.txt".format(part=i+1), "w", encoding="utf-8") as f:
            f.writelines(lines[split_size * i : split_size * (i + 1)])

    with open("paranmt500k-tab2newline.filelist500.txt", "w", encoding="utf-8") as f:
        for i in tqdm(range(500)):
            f.write("/home/llj/synpg/data/paranmt500k/paranmt500k-tab2newline.{part}_of_500.txt\n".format(part=i+1))

def merge_output_file():
    with open("/home/llj/synpg/data/paranmt500k-tab2newline.txt.out", "w", encoding="utf-8") as fout:
        for i in tqdm(range(500)):
            with open("/home/llj/synpg/data/paranmt500k/out/paranmt500k-tab2newline.{part}_of_500.txt.out".format(part=i+1), "r", encoding="utf-8") as f:
                lines = f.readlines()
            fout.writelines(lines)
    

def csv_to_h5():
    with open("data/parsed-paranmt500k-tab2newline.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    lines = [s.strip().split("\t") for s in lines]
    sent, synt = zip(*lines)

    # 必须要 encode 成 bytes，直接用 string 有点问题，我不知道原因。似乎和 h5py 的版本有关，v2 和 v3 对字符串的处理不同。
    # Have to encode to bytes. There are some problems when using string directly. I do not know the reason. It may have something to do with the version of h5py (v2 and v3).
    sent = [s.encode() for s in sent]
    synt = [s.encode() for s in synt]

    hf = h5py.File('data/paranmt500k.train.h5', 'w')
    hf.create_dataset('sents', data=sent)
    hf.create_dataset('synts', data=synt)
    hf.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, required=True,
                        help="operation: split, merge, csv2h5")

    args = parser.parse_args()
    print(vars(args))
    if args.op == "split":
        split_file()
    elif args.op == "merge":
        merge_output_file()
    elif args.op == "csv2h5":
        csv_to_h5()
    print("done")
