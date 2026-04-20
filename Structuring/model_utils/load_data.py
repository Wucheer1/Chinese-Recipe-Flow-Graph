from tqdm import trange
import utils_class
from utils_class import Trie


def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    # 1.获取词汇表
    print(vocab_files)
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    return lexicon_tree