# read ner
import json
import os
import seaborn as sns
from matplotlib import pyplot as plt

import random
from difflib import SequenceMatcher

import numpy as np
import synonyms
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class InputFeature(object):

    def __init__(self, input_ids, token_type_ids, attention_mask, label_ids):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids


class InputFeature_RE(object):

    def __init__(self, input_ids, token_type_ids, attention_mask, label_ids, e1_mask, e2_mask,e1_pos,e2_pos, re_len):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.re_len = re_len


def add_params(args, module, prefix, lr, no_decay,optimizer_grouped_parameters):
    params_decay = [
        p for n, p in module.named_parameters()
        if not any(nd in n for nd in no_decay) and p.requires_grad
    ]
    params_no_decay = [
        p for n, p in module.named_parameters()
        if any(nd in n for nd in no_decay) and p.requires_grad
    ]

    if params_decay:
        optimizer_grouped_parameters.append({
            "params": params_decay,
            "weight_decay": args.weight_decay,
            "lr": lr
        })
    if params_no_decay:
        optimizer_grouped_parameters.append({
            "params": params_no_decay,
            "weight_decay": 0.0,
            "lr": lr
        })


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def parse_recipe_ner(file):
    """
    读取食谱数据，生成BIO标签，并按句号和换行符切分句子。
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data['text']
    entities = data['entities']

    # 1. 初始化全为 'O' 的标签列表，长度与文本一致
    # tags 列表对应 text 中的每一个字符
    tags = ['O'] * len(text)

    # 2. 根据实体信息填充 BIO 标签
    for entity in entities:
        start = entity['start_offset']
        end = entity['end_offset']
        label = entity['label']

        # 确保索引在有效范围内
        if start < len(text) and end <= len(text):
            # 标记实体的开始 (Begin)
            tags[start] = f"B-{label}"
            # 标记实体的内部 (Inside)
            for i in range(start + 1, end):
                tags[i] = f"I-{label}"

    # 3. 按照 '。' 和 '\n' 进行句子切分
    sentences = []  # 存储切分后的字符列表
    sentence_labels = []  # 存储切分后的标签列表

    current_chars = []
    current_tags = []

    for char, tag in zip(text, tags):
        if char == '\n':
            # 遇到换行符：如果是有效句子（非空），则保存并重置
            # 注意：这里我们选择不把 '\n' 放入句子中
            if len(current_chars) > 0:
                sentences.append(current_chars)
                sentence_labels.append(current_tags)
            current_chars = []
            current_tags = []

        elif char == '。':
            # 遇到句号：将句号加入当前句子，然后保存并重置
            current_chars.append(char)
            current_tags.append(tag)
            sentences.append(current_chars)
            sentence_labels.append(current_tags)
            current_chars = []
            current_tags = []

        else:
            # 普通字符：加入当前缓冲区
            current_chars.append(char)
            current_tags.append(tag)

    # 处理文末可能剩余的字符（如果没有以符号结尾）
    if len(current_chars) > 0:
        sentences.append(current_chars)
        sentence_labels.append(current_tags)

    return sentences, sentence_labels


def read_recipe_ner_data(filepath, replace_entities=True):
    sentences_word = []
    sentences_label = []
    current_words = []
    current_labels = []
    current_segment_id = None

    # 定义需要替换的实体类型
    target_entity_types = ['Af', 'St', 'D', 'T', 'Q', 'Sf']

    def find_similar_phrases(target, top_n=2):
        """使用 Synonyms 库获取语义相近词"""
        try:
            # 获取语义相近的词语
            nearby_words, scores = synonyms.nearby(target)

            # 收集所有符合条件的候选词及其相似度
            candidates = []
            for word, score in zip(nearby_words, scores):
                if word != target and score > 0.6:  # 只排除自身和低相似度词
                    candidates.append((word, score))

            # 按相似度从高到低排序
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 选择前top_n个相似度最高的词
            similar_phrases = [word for word, score in candidates[:top_n]]

            return similar_phrases
        except Exception as e:
            print(f"在查找 '{target}' 的相似词时出错: {e}")
            return []

    def extract_entities(words, labels):
        """提取完整的实体，处理B-和I-序列"""
        entities = []
        i = 0
        while i < len(labels):
            # 检查当前标签是否是目标实体类型的开始
            is_target_entity = False
            entity_type = None

            for entity_type_prefix in target_entity_types:
                if labels[i] == f'B-{entity_type_prefix}':
                    is_target_entity = True
                    entity_type = entity_type_prefix
                    break

            if is_target_entity:
                # 找到B-开头的实体
                entity_words = [words[i]]
                entity_indices = [i]

                # 查找连续的I-标签
                j = i + 1
                while j < len(labels) and labels[j] == f'I-{entity_type}':
                    entity_words.append(words[j])
                    entity_indices.append(j)
                    j += 1

                # 将实体词合并为一个完整的实体
                entity_text = ''.join(entity_words)
                entities.append({
                    'text': entity_text,
                    'type': entity_type,
                    'indices': entity_indices,
                    'start': i,
                    'end': j
                })

                i = j  # 跳过已处理的实体
            else:
                i += 1

        return entities

    def generate_variations(words, labels):
        """为包含目标实体的句子生成多个版本，正确处理标签对齐"""
        variations = []

        # 版本1: 原句
        variations.append((words.copy(), labels.copy(), "original"))

        # 提取完整的实体
        entities = extract_entities(words, labels)

        # 如果没有目标实体，直接返回原句
        if not entities:
            return variations

        # 为每个实体找到最相似的替换词
        replacement_options = {}
        for entity in entities:
            similar_phrases = find_similar_phrases(entity['text'], top_n=2)
            replacement_options[entity['text']] = {
                'similar_phrases': similar_phrases,
                'indices': entity['indices'],
                'type': entity['type']
            }

        # 生成多个随机替换版本
        num_variations = random.randint(1, 3)

        for i in range(num_variations):
            words_variant = words.copy()
            labels_variant = labels.copy()
            replaced_entities = []

            # 随机决定要替换哪些实体
            entities_to_replace = []
            for entity_text, entity_info in replacement_options.items():
                if entity_info['similar_phrases'] and random.random() < 0.7:
                    entities_to_replace.append((entity_text, entity_info))

            # 如果没有实体被选中替换，随机选择一个实体强制替换
            if not entities_to_replace and replacement_options:
                entity_text = random.choice(list(replacement_options.keys()))
                if replacement_options[entity_text]['similar_phrases']:
                    entities_to_replace.append((entity_text, replacement_options[entity_text]))

            # 按实体在原句中的位置从后往前替换，避免索引变化影响
            entities_to_replace.sort(key=lambda x: x[1]['indices'][0], reverse=True)

            for entity_text, entity_info in entities_to_replace:
                # 随机选择一个同义词替换
                replacement = random.choice(entity_info['similar_phrases'])
                entity_type = entity_info['type']
                indices = entity_info['indices']

                # 将替换词拆分为字符
                replacement_chars = list(replacement)
                original_length = len(indices)
                replacement_length = len(replacement_chars)

                # 计算长度差异
                length_diff = replacement_length - original_length

                # 替换实体并调整标签
                if replacement_length == original_length:
                    # 长度相同，直接替换
                    for j, idx in enumerate(indices):
                        words_variant[idx] = replacement_chars[j]
                        # 保持原有的B-/I-标签模式
                        if j == 0:
                            labels_variant[idx] = f"B-{entity_type}"
                        else:
                            labels_variant[idx] = f"I-{entity_type}"

                elif replacement_length < original_length:
                    # 替换词比原实体短
                    for j in range(replacement_length):
                        words_variant[indices[j]] = replacement_chars[j]
                        if j == 0:
                            labels_variant[indices[j]] = f"B-{entity_type}"
                        else:
                            labels_variant[indices[j]] = f"I-{entity_type}"

                    # 将多余位置标记为删除
                    for j in range(replacement_length, original_length):
                        words_variant[indices[j]] = ""
                        labels_variant[indices[j]] = ""

                else:
                    # 替换词比原实体长，需要扩展位置
                    # 先处理原有位置
                    for j in range(original_length):
                        words_variant[indices[j]] = replacement_chars[j]
                        if j == 0:
                            labels_variant[indices[j]] = f"B-{entity_type}"
                        else:
                            labels_variant[indices[j]] = f"I-{entity_type}"

                    # 插入额外的字符和标签
                    extra_chars = replacement_chars[original_length:]
                    extra_labels = [f"I-{entity_type}"] * len(extra_chars)

                    # 在实体结束位置后插入
                    insert_pos = indices[-1] + 1
                    words_variant[insert_pos:insert_pos] = extra_chars
                    labels_variant[insert_pos:insert_pos] = extra_labels

                replaced_entities.append((entity_text, replacement, entity_type))

            # 过滤掉空字符串，同时保持词语和标签对齐
            filtered_data = [(w, l) for w, l in zip(words_variant, labels_variant) if w != ""]
            if filtered_data:
                filtered_words, filtered_labels = zip(*filtered_data)
                filtered_words = list(filtered_words)
                filtered_labels = list(filtered_labels)

                # 只有当确实发生了替换时才添加这个版本
                if replaced_entities:
                    variations.append((filtered_words, filtered_labels, f"variation_{i + 1}"))

        return variations, replacement_options

    with open(filepath, "r", encoding='UTF-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # 跳过空行

            # 跳过文档起始标记
            if "-DOCSTART-" in stripped_line:
                continue

            # 分割字段
            parts = stripped_line.split()

            try:
                # 解析关键字段
                first_col = parts[0]  # 第一列
                second_col = parts[1]  # 第二列
                word = parts[3]  # 第四列为单词
                label = parts[4]  # 第5列为标签
                segment_id = (first_col, second_col)  # 组合段落ID
            except IndexError:
                continue

            # 检测段落边界变化（第一列或第二列变化）
            if current_segment_id is None:
                current_segment_id = segment_id
            elif segment_id != current_segment_id:
                # 检查当前句子是否包含目标实体标签
                has_target_entities = any(
                    any(entity_type in label for entity_type in target_entity_types)
                    for label in current_labels
                )

                if has_target_entities and replace_entities:
                    # 为包含目标实体的句子生成多个版本
                    variations, replacement_options = generate_variations(current_words, current_labels)
                    for words_variation, labels_variation, version_type in variations:
                        sentences_word.append(words_variation)
                        sentences_label.append(labels_variation)
                else:
                    # 不包含目标实体或不需要替换，只添加原句
                    sentences_word.append(current_words)
                    sentences_label.append(current_labels)

                # 重置缓存
                current_words = []
                current_labels = []
                current_segment_id = segment_id

            # 添加当前词和标签
            current_words.append(word)
            current_labels.append(label)

        # 处理最后一个句子
        if current_words:
            has_target_entities = any(
                any(entity_type in label for entity_type in target_entity_types)
                for label in current_labels
            )

            if has_target_entities and replace_entities:
                # 为包含目标实体的句子生成多个版本
                variations, replacement_options = generate_variations(current_words, current_labels)
                for words_variation, labels_variation, version_type in variations:
                    sentences_word.append(words_variation)
                    sentences_label.append(labels_variation)
            else:
                # 不包含目标实体或不需要替换，只添加原句
                sentences_word.append(current_words)
                sentences_label.append(current_labels)

    return sentences_word, sentences_label


def calc_sentence_entity_accuracy(all_true_label, all_pred_label, all_sentence_test, initial_label):
    """
    返回 F1 最低的 50 条句子（句子长度 > 10 字），包含：
      - true_entity
      - pred_entity
      - sentence
      - F1
    """

    # BIO 映射
    id2label = {v: k for k, v in initial_label.items()}

    # ----------- BIO → spans（带起止位置）-------------
    def bio_to_spans(label_ids, tokens):
        spans = []
        n = len(label_ids)
        i = 0
        while i < n:
            tag = id2label[label_ids[i]]
            if tag.startswith("B-"):
                ent_type = tag[2:]
                start = i
                ent_tokens = [tokens[i]]
                i += 1
                while i < n and id2label[label_ids[i]] == f"I-{ent_type}":
                    ent_tokens.append(tokens[i])
                    i += 1
                end = i - 1
                spans.append((ent_type, "".join(ent_tokens), start, end))
            else:
                i += 1
        return spans

    # ----------- span overlap 判断 -----------
    def spans_overlap(s1, e1, s2, e2):
        return not (e1 < s2 or e2 < s1)

    sentence_scores = []

    for true_seq, pred_seq, tokens in zip(all_true_label, all_pred_label, all_sentence_test):

        true_spans = bio_to_spans(list(true_seq), tokens)
        pred_spans = bio_to_spans(list(pred_seq), tokens)

        sentence = "".join(tokens)

        # ---------- 统计 TP FP FN ----------
        TP = 0
        FP = 0
        FN = 0

        matched_pred = set()

        # True spans → check 匹配的 pred spans
        for t_type, t_text, t_start, t_end in true_spans:
            matched = False
            for idx, (p_type, p_text, p_start, p_end) in enumerate(pred_spans):
                if idx in matched_pred:
                    continue
                if spans_overlap(t_start, t_end, p_start, p_end) and p_type == t_type:
                    TP += 1
                    matched_pred.add(idx)
                    matched = True
                    break
            if not matched:
                FN += 1  # 真实存在但预测没识别

        # 多余预测（匹配不到 true）
        FP = len(pred_spans) - len(matched_pred)

        # F1
        if TP == 0:
            f1 = 0.0
        else:
            f1 = (2 * TP) / (2 * TP + FP + FN)

        # 收集实体
        true_entity_info = [{"name": t_text, "type": t_type} for (t_type, t_text, _, _) in true_spans]
        pred_entity_info = [{"name": p_text, "type": p_type} for (p_type, p_text, _, _) in pred_spans]

        # -------- 加入句子长度限制 --------
        if len(sentence) > 10:
            sentence_scores.append({
                "sentence": sentence,
                "pred_entity": pred_entity_info,
                "true_entity": true_entity_info,
                "f1": f1
            })

    # 排序 F1 从小到大
    sentence_scores = sorted(sentence_scores, key=lambda x: x["f1"])

    # 返回最差 50 条
    return sentence_scores[:50]


def read_labels_from_file(filepath, args):
    with open(filepath) as f:
        labels_data = f.read()
        json_labels = json.loads(labels_data)
        if args.task == 'NER':
            if args.IO_mode == 'BIO':
                id2label_train = json_labels['train-BIO']
            elif args.IO_mode == 'BIOES':
                id2label_train = json_labels['train-BIOES']
            else:
                id2label_train = json_labels["train"]

            id2label_dev = json_labels["dev"]
            id2label_test = id2label_train

            id2proxy_label_train = json_labels["proxy_train"]
            id2proxy_label_dev = json_labels["proxy_dev"]
            id2proxy_label_test = json_labels["proxy_test"]

            id2label = []
            id2label.extend(id2label_train)
            # id2label.extend(id2label_dev)
            # id2label.extend(id2label_test)
            id2label.insert(0, "O")

            label2id = {}
            for i, label in enumerate(id2label):
                label2id[label] = i

            id2proxy_label = []
            id2proxy_label.extend(id2proxy_label_train)
            # id2proxy_label.extend(id2proxy_label_dev)
            # id2proxy_label.extend(id2proxy_label_test)
            id2proxy_label.insert(0, "other")

            proxy_label2id = {}
            for i, label in enumerate(id2proxy_label):
                proxy_label2id[label] = i
        else:
            id2label_train = json_labels["train"]
            id2label_dev = json_labels["dev"]
            id2label_test = id2label_train

            id2proxy_label_train = json_labels["proxy_train"]
            id2proxy_label_dev = json_labels["proxy_dev"]
            id2proxy_label_test = json_labels["proxy_test"]

            id2label = []
            id2label.extend(id2label_train)
            label2id = {}
            for i, label in enumerate(id2label):
                label2id[label] = i

            id2proxy_label = []
            id2proxy_label.extend(id2proxy_label_train)
            proxy_label2id = {}
            for i, label in enumerate(id2proxy_label):
                proxy_label2id[label] = i

        initial_label = json_labels["train"]
        return (
            id2label, id2label_train, id2label_dev, id2label_test, label2id,
            id2proxy_label, id2proxy_label_train, id2proxy_label_dev,
            id2proxy_label_test, proxy_label2id, initial_label
        )


def convert_label_to_id_ner(labels, args):
    labels_ids = []
    if args.IO_mode == "BIO":
        new_labels = labels

    elif args.IO_mode == "BIOES":
        new_labels = convert_BIOES(labels)
    else:
        new_labels = convert_IO(labels)

    map2id = args.label2id
    for item in new_labels:
        tag = []
        for label in item:
            tag.append(map2id.get(label))  # Default to 'O' if label not found
        labels_ids.append(tag)

    return labels_ids


def convert_label_to_id_ner_cn(labels, args):
    labels_io = []
    for item in labels:
        tag = []
        for label in item:
            if 'B-' in label:
                tag.append(label.split('B-')[1])
            elif 'I-' in label:
                tag.append(label.split('I-')[1])
            elif 'M-' in label:
                tag.append(label.split('M-')[1])
            elif 'E-' in label:
                tag.append(label.split('E-')[1])
            elif 'S-' in label:
                tag.append(label.split('S-')[1])
            elif '-I' in label:
                tag.append(label.split('-I')[0])
            elif '-B' in label:
                tag.append(label.split('-B')[0])
            else:
                tag.append(label)
        labels_io.append(tag)
    labels_ids = []
    if args.IO_mode == "BIO":
        new_labels = convert_BIO(labels_io)

    elif args.IO_mode == "BIOES":
        new_labels = convert_BIOES(labels_io)
    else:
        new_labels = labels_io

    map2id = args.label2id

    for item in new_labels:
        tag = []
        for label in item:
            tag.append(map2id.get(label))  # Default to 'O' if label not found
        labels_ids.append(tag)

    return labels_ids


def convert_BIO(labels):
    new_labels = []
    for item in labels:
        tag = []
        prev_label = "O"
        for label in item:
            if label == "O":
                tag.append("O")
            elif label != prev_label:
                tag.append(f"B-{label}")
            else:
                tag.append(f"I-{label}")
            prev_label = label
        new_labels.append(tag)
    return new_labels


def convert_IO(labels):
    io_tags = []
    for tags in labels:
        for tag in tags:
            if tag == 'O':
                io_tags.append('O')
            else:
                # 去掉B-/I-前缀，只保留实体类型
                io_tags.append(tag.split('-')[1])
    return io_tags


def convert_BIOES(labels):
    bioes_tags = []
    for item in labels:
        current_tags = []
        length = len(item)

        for i in range(length):
            current = item[i]
            next_tag = item[i + 1] if i + 1 < length else "O"

            if current.startswith("B-"):
                entity_type = current.split("-")[1]
                if next_tag.startswith("I-") and next_tag.split("-")[1] == entity_type:
                    current_tags.append(f"B-{entity_type}")  # 实体开头
                else:
                    current_tags.append(f"S-{entity_type}")  # 单独实体

            elif current.startswith("I-"):
                entity_type = current.split("-")[1]
                if next_tag.startswith("I-") and next_tag.split("-")[1] == entity_type:
                    current_tags.append(f"I-{entity_type}")  # 实体中间
                else:
                    current_tags.append(f"E-{entity_type}")  # 实体结尾

            else:  # 'O'
                current_tags.append("O")

        bioes_tags.append(current_tags)
    return bioes_tags


def GetDataLoader_NER(args, sentences, labels_ids, batch_size, ignore_o_sentence=True):
    sentences_filtered = []
    labels_ids_filtered = []
    if ignore_o_sentence:
        for sentence, label_ids in zip(sentences, labels_ids):
            if sum(label_ids) > 0:
                sentences_filtered.append(sentence)
                labels_ids_filtered.append(label_ids)
    else:
        sentences_filtered = sentences
        labels_ids_filtered = labels_ids

    features = []
    for sentence, label_ids in zip(sentences_filtered, labels_ids_filtered):
        features.append(convert_to_feature_ner(sentence, label_ids, args))

    dataset = convert_features_to_dataset_ner(features)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader


def merge_ner_to_json(original_filepath, sentences, pred_ids_list, label_map):
    """
    Args:
        original_filepath: 原始JSON文件路径，用于读取 text
        sentences: 分词后的句子列表 [['准', '备', ...], ...]
        pred_ids_list: 预测的ID列表 [array([2,2,0...]), ...]
        label_map: ID到Label的映射列表 ['O', 'F', ...]
    Returns:
        更新后的数据字典
    """
    # 1. 读取原始数据
    with open(original_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    full_text = data['text']
    new_entities = []

    # 2. 对齐逻辑
    # text_cursor 指向 full_text 当前匹配到的位置
    text_cursor = 0
    entity_id_counter = 1

    # 用于缓存当前正在构建的实体
    current_entity = None
    # { "start": int, "label": str }

    # 遍历每一句话
    for sent_chars, sent_preds in zip(sentences, pred_ids_list):
        # 遍历句子中的每一个字和对应的预测结果
        for char, label_id in zip(sent_chars, sent_preds):
            label_str = label_map[label_id]

            # --- 关键步骤：在 full_text 中找到当前 char ---
            # 只要 full_text 当前字符不等于 sentence 中的 char，就跳过（通常是空格或换行）
            while text_cursor < len(full_text) and full_text[text_cursor] != char:
                # 如果跳过的过程中存在正在构建的实体，且当前是非实体字符，这里简化处理不中断，
                # 但通常实体内部不包含换行。如果需要严格断开，可以在这里处理。
                # 现行逻辑：只匹配有效字符的Offset。
                text_cursor += 1

            if text_cursor >= len(full_text):
                break  # 防止越界

            # 此时 full_text[text_cursor] == char，这是有效字符

            # --- 实体合并逻辑 ---
            if label_str == 'O':
                # 如果当前是 O，且之前有正在构建的实体，说明实体结束了
                if current_entity:
                    new_entities.append({
                        "id": entity_id_counter,
                        "label": current_entity['label'],
                        "start_offset": current_entity['start'],
                        "end_offset": text_cursor  # 上一个字符结束的位置
                    })
                    entity_id_counter += 1
                    current_entity = None
            else:
                # 如果当前是实体标签 (如 'Ac', 'F' 等)
                if current_entity:
                    if current_entity['label'] == label_str:
                        # 标签相同，延续当前实体，不做操作，继续循环
                        pass
                    else:
                        # 标签不同（如从 'Ac' 变成了 'F'），先结束上一个，再开始下一个
                        new_entities.append({
                            "id": entity_id_counter,
                            "label": current_entity['label'],
                            "start_offset": current_entity['start'],
                            "end_offset": text_cursor
                        })
                        entity_id_counter += 1
                        # 开启新实体
                        current_entity = {
                            "start": text_cursor,
                            "label": label_str
                        }
                else:
                    # 之前没有实体，开启新实体
                    current_entity = {
                        "start": text_cursor,
                        "label": label_str
                    }

            # 移动原始文本指针
            text_cursor += 1

        # 句子结束时，通常如果实体还在，可以选择闭合，
        # 或者如果该数据集允许跨行实体，则保留 current_entity。
        # 针对菜谱数据，通常句子结束意味着实体结束（防止跨行噪音）。
        if current_entity:
            new_entities.append({
                "id": entity_id_counter,
                "label": current_entity['label'],
                "start_offset": current_entity['start'],
                "end_offset": text_cursor
            })
            entity_id_counter += 1
            current_entity = None

    # 3. 更新 data
    data['entities'] = new_entities

    # 移除可能存在的旧字段（视情况而定）
    # if 'relations' in data: del data['relations']

    return data


def convert_to_feature_ner(sentence, label_ids, args):
    max_seq_length = args.max_seq_length
    sentence_tokens = []
    label_ids_tokens = []
    for word, label_id in zip(sentence, label_ids):
        word_tokens = args.tokenizer.tokenize(word)
        label_id_tokens = [label_id] + [-1] * (len(word_tokens) - 1)
        if len(word_tokens) == 0:  # Meet special space character
            word_tokens = args.tokenizer.tokenize('[UNK]')
            label_id_tokens = [label_id]
        sentence_tokens.extend(word_tokens)
        label_ids_tokens.extend(label_id_tokens)

    sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]
    label_ids_tokens = [-1] + label_ids_tokens + [-1]

    input_ids = args.tokenizer.convert_tokens_to_ids(sentence_tokens)

    padding_length = max_seq_length - len(input_ids)
    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length
        label_ids_tokens += [-1] * padding_length
    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]
        label_ids_tokens = label_ids_tokens[:max_seq_length]

    token_type_ids = [0] * max_seq_length

    assert len(input_ids_padded) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(label_ids_tokens) == max_seq_length
    return InputFeature(input_ids_padded, token_type_ids, attention_mask, label_ids_tokens)


def convert_features_to_dataset_ner(features):
    # convert to Tensors
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_attention_mask = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_ids for feature in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
    return dataset


def convert_ids_to_labels(true_labels, pred_labels, id2label, label2id):
    merged_true = []
    merged_pred = []
    true_flat = true_labels.reshape(-1)
    pred_flat = pred_labels.reshape(-1)
    new_label = ['O'] + id2label.copy()
    new_id = ['O'] + label2id.copy()
    for true_id, pred_id in zip(true_flat, pred_flat):
        # 转换标签ID到名称
        true_name = new_label[true_id]
        pred_name = new_label[pred_id]

        # 合并B/I前缀
        merged_true.append(true_name.split("-")[-1] if "-" in true_name else true_name)
        merged_pred.append(pred_name.split("-")[-1] if "-" in pred_name else pred_name)

    true_ids = np.array([new_id.index(label) for label in merged_true])  # 未知标签默认映射到'O'
    pred_ids = np.array([new_id.index(label) for label in merged_pred])
    return true_ids, pred_ids


def generate_ner_confusion_matrix(true_ids, pred_ids):
    """
    输入:
        true_ids: 真实标签的整数列表, 如 [2, 1, 0, ...]
        pred_ids: 预测标签的整数列表, 如 [2, 1, 0, ...]
    输出:
        confusion_data: 嵌套字典 {True_Label: {Pred_Label: Count}}
    """

    # 1. 定义映射关系 (下标 -> 标签名)
    # 这里的顺序必须严格对应你提供的 0-8 的含义
    label_map = ['O', 'T', 'N', 'I', 'F-eq', 'T-eq', 'T-comp', 'F-comp', 'F-part-of', 'T-part-of', 'V-tm', "Null"]

    # 2. 初始化嵌套字典
    # 结构: {真实类别: {预测类别: 0}}
    # 预先将所有值设为0，保证输出包含所有类别，即使某个类别从未出现
    confusion_data = {
        row_label: {col_label: 0 for col_label in label_map}
        for row_label in label_map
    }

    # 3. 遍历列表进行统计
    for t_idx, p_idx in zip(true_ids, pred_ids):
        # 确保索引是整数 (防止传入的是 tensor 或 numpy scalar)
        t_idx = int(t_idx)
        p_idx = int(p_idx)

        # 根据索引获取标签名称
        # 例如: 0 -> 'O', 1 -> 'F'
        true_name = label_map[t_idx]
        pred_name = label_map[p_idx]

        # 计数 +1
        # 含义: 真实是 true_name, 被预测成了 pred_name
        confusion_data[true_name][pred_name] += 1

    return confusion_data

# label_ids(batch,seq_len) --> (batch, num_class, seq_len, seq_len)
def get_entity_type(label_id):
    if label_id == 0:
        return None
    elif label_id % 2 == 1:  # B标签
        return (label_id - 1) // 2
    else:                     # I标签
        return (label_id - 2) // 2


def convert_bio_to_entity_matrix(label_ids, num_entity_types, attention_mask=None):
    batch_size, seq_len = label_ids.shape
    device = label_ids.device
    entity_matrix = torch.zeros(
        (batch_size, num_entity_types, seq_len, seq_len),
        dtype=torch.long,
        device=device
    )

    for batch_idx in range(batch_size):
        current_labels = label_ids[batch_idx]
        valid_length = seq_len
        if attention_mask is not None:
            valid_length = attention_mask[batch_idx].sum().item()
            current_labels = current_labels[:valid_length]

        current_entity = None
        for pos in range(valid_length):
            label_id = current_labels[pos].item()
            if label_id < 0:  # 过滤负值标签
                continue

            if label_id == 0:
                if current_entity is not None:
                    # 结束当前实体
                    start, end, entity_type = current_entity
                    if entity_type < num_entity_types:
                        entity_matrix[batch_idx, entity_type, start, end] = 1
                    current_entity = None
                continue

            entity_type = get_entity_type(label_id)
            if entity_type is None or entity_type >= num_entity_types:
                continue

            if label_id % 2 == 1:  # B标签
                if current_entity is not None:
                    # 结束前一个实体
                    start, end, prev_type = current_entity
                    if prev_type < num_entity_types:
                        entity_matrix[batch_idx, prev_type, start, end] = 1
                # 开始新实体
                current_entity = (pos, pos, entity_type)
            else:  # I标签
                if current_entity is not None and current_entity[2] == entity_type:
                    # 更新结束位置
                    current_entity = (current_entity[0], pos, entity_type)
                else:
                    # 结束前一个实体（如果有）
                    if current_entity is not None:
                        start, end, prev_type = current_entity
                        if prev_type < num_entity_types:
                            entity_matrix[batch_idx, prev_type, start, end] = 1
                    current_entity = None

        # 处理最后一个实体
        if current_entity is not None:
            start, end, entity_type = current_entity
            if entity_type < num_entity_types:
                entity_matrix[batch_idx, entity_type, start, end] = 1

    return entity_matrix
# -------------------------- RE -----------------------------------


def collect_error_cases_by_label(all_pred_labels, all_true_labels, all_dict, label_map):
    # id → label 映射
    id2label = {v: k for k, v in label_map.items()}

    # 初始化错误案例字典（每个类别一个 list）
    error_dict = {label: [] for label in id2label.values()}
    print(len(all_dict))
    print(len(all_pred_labels))
    # 遍历所有样本（索引一一对应）
    for true_id, pred_id, info in zip(all_true_labels, all_pred_labels, all_dict):

        # id 不在映射中，跳过
        if true_id not in id2label:
            continue

        true_label = id2label[true_id]
        pred_label = id2label.get(pred_id, None)

        # 仅收集预测错误的样本
        if true_id != pred_id:
            # 每类最多 5 个案例
            if len(error_dict[true_label]) < 5:
                error_dict[true_label].append({
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "info": info
                })

    # 保存到 results.json
    with open(r"C:\Users\Admin\Desktop\Last\temp\data.json", "w", encoding="utf-8") as f:
        json.dump(error_dict, f, ensure_ascii=False, indent=4)

    return error_dict


# 结果为{entity1: { start:i, end: j }, entity2: , sentence1: , sentence2: , label: }
def build_re_dict(file1_path, file2_path):
    # 读取第一个文件，提取编码和标签
    with open(file1_path, 'r', encoding='utf-8') as file1:
        lines1 = file1.readlines()

    # 读取第二个文件，提取单词和句子信息
    with open(file2_path, 'r', encoding='utf-8') as file2:
        lines2 = file2.readlines()
    # 将第二个文件的内容解析为字典，方便快速查找
    sentence_info = {}
    for line in lines2:
        parts = line.strip().split()
        if len(parts) >= 5:
            key = (parts[0], parts[1])  # 段号和句号作为键
            word_key = (parts[0], parts[1], parts[2])  # 编码作为键（段号，句号，单词号）
            word = parts[3]  # 单词
            label = parts[4]

            if key not in sentence_info:
                sentence_info[key] = {'words': [], 'word_map': {}, 'labels': []}
            sentence_info[key]['words'].append(word)
            sentence_info[key]['word_map'][word_key] = word
            sentence_info[key]['labels'].append(label)

    dataset = []
    for line in lines1:
        parts = line.strip().split()
        if len(parts) >= 7:
            # 提取编码和标签
            key1 = (parts[0], parts[1], parts[2])  # 第一个实体的编码
            key2 = (parts[4], parts[5], parts[6])  # 第二个实体的编码
            label = parts[3]  # 标签

            # 查找实体和句子

            entity1 = get_entity_pos_from_ner(sentence_info, key1)
            entity2 = get_entity_pos_from_ner(sentence_info, key2)
            sentence1 = ' '.join(sentence_info.get((key1[0], key1[1]), {}).get('words', []))
            sentence2 = ' '.join(sentence_info.get((key2[0], key2[1]), {}).get('words', []))

            if entity1 and entity2:
                dataset.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'sentence1': sentence1,
                    'sentence2': sentence2,
                    'label': label
                })
    return dataset


# 直接提取实体文本
def get_entity_from_ner(sentence_info, entity_key):
    segment, sentence, word_pos = entity_key
    key = (segment, sentence)

    # 获取句子中的所有单词和标签
    words = sentence_info.get(key, {}).get('words', [])
    labels = sentence_info.get(key, {}).get('labels', [])
    word_map = sentence_info.get(key, {}).get('word_map', {})

    # 找到目标单词的位置，确保单词和索引值都匹配
    start_pos = -1
    i = 0
    for word_key, word in word_map.items():
        if word_key == entity_key:
            start_pos = i
            break
        i = i + 1

    if start_pos == -1:
        print(f"{word_pos} not found in sentence.")
        return ''

    if not labels[start_pos].endswith('-B'):
        print(words)
        print(entity_key)
        print(f"Start position label is not B: {labels[start_pos]}")  # 调试输出
        return ''

    # 提取完整的实体
    label_type = labels[start_pos].split('-')[0]

    # 提取完整的实体
    entity_words = []
    for i in range(start_pos, len(words)):
        current_label = labels[i]
        if current_label.startswith(label_type) and (current_label.endswith('-B') or current_label.endswith('-I')):
            entity_words.append(words[i])
        else:
            break

    return ' '.join(entity_words)


# 提取实体的位置
def get_entity_pos_from_ner(sentence_info, entity_key):
    segment, sentence, word_pos = entity_key
    key = (segment, sentence)

    # 获取句子中的所有单词和标签
    words = sentence_info.get(key, {}).get('words', [])
    labels = sentence_info.get(key, {}).get('labels', [])
    word_map = sentence_info.get(key, {}).get('word_map', {})

    # 找到目标单词的位置，确保单词和索引值都匹配
    start_pos = -1
    i = 0
    for word_key, word in word_map.items():
        if word_key == entity_key:
            start_pos = i
            break
        i = i + 1

    if start_pos == -1:
        print(words)
        print(f"{word_pos} not found in sentence.")  # 调试输出
        return ''

    if not labels[start_pos].startswith('B-'):
        print(words)
        print(entity_key)
        print(f"Start position label is not B: {labels[start_pos]}")  # 调试输出
        return ''

    # 提取完整的实体
    label_type = labels[start_pos].split('-')[0]

    # 获取字符级别的起始位置
    char_start_pos = sum(len(word) + 1 for word in words[:start_pos])  # 计算字符级别的起始位置
    char_end_pos = char_start_pos + len(words[start_pos])  # 计算字符级别的结束位置

    # 查找结束位置
    for i in range(start_pos + 1, len(words)):
        current_label = labels[i]
        if current_label.startswith(label_type) and (current_label.startswith('B-') or current_label.startswith('I-')):
            char_end_pos = sum(len(word) + 1 for word in words[:i]) + len(words[i])
        else:
            break

    return {'start': char_start_pos, 'end': char_end_pos}


def extract_relations_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 获取文本和段落分割
    text = json_data['text']
    paragraphs = text.split('\n\n')

    # 预计算每个段落的起始和结束位置
    paragraph_positions = []
    current_position = 0
    for i, para in enumerate(paragraphs):
        para_length = len(para)
        # 记录段落起始位置和长度（不包含\n\n）
        paragraph_positions.append({
            'start': current_position,
            'end': current_position + para_length,  # 包含最后一个字符的位置
            'text': para,
            'index': i  # 添加段落索引，从0开始
        })
        # 更新当前位置（加上本段长度和分隔符）
        current_position += para_length + (2 if i < len(paragraphs) - 1 else 0)

    # 创建实体ID到实体信息的映射
    entity_map = {entity['id']: entity for entity in json_data['entities']}

    # 处理每个关系
    relations_info = []

    for relation in json_data['relations']:
        from_entity = entity_map.get(relation['from_id'])
        to_entity = entity_map.get(relation['to_id'])

        if not from_entity or not to_entity:
            continue

        # 查找实体所属段落及其索引
        from_para = find_paragraph(from_entity['start_offset'], paragraph_positions)
        to_para = find_paragraph(to_entity['start_offset'], paragraph_positions)

        # 计算相对位置
        entity1_start = from_entity['start_offset'] - from_para['start']
        entity1_end = from_entity['end_offset'] - from_para['start']
        entity2_start = to_entity['start_offset'] - to_para['start']
        entity2_end = to_entity['end_offset'] - to_para['start']

        # 构建结果字典，添加段落位置标签
        relation_info = {
            'entity1': {
                'start': entity1_start,
                'end': entity1_end,
                'text': from_para['text'][entity1_start:entity1_end],
                'label': from_entity['label'],
                'paragraph_index': from_para['index']  # 添加位置标签
            },
            'entity2': {
                'start': entity2_start,
                'end': entity2_end,
                'text': to_para['text'][entity2_start:entity2_end],
                'label': to_entity['label'],
                'paragraph_index': to_para['index']  # 添加位置标签
            },
            'sentence1': from_para['text'],
            'sentence2': to_para['text'],
            'relation_label': relation['type'],
            # 也可以添加整体关系的位置信息
            'relation_position_info': {
                'entity1_paragraph': from_para['index'],
                'entity2_paragraph': to_para['index'],
                'same_paragraph': from_para['index'] == to_para['index'],
                'total_paragraph': len(paragraphs)
            }
        }

        relations_info.append(relation_info)

    return relations_info


def calculate_re_metrics(all_res, all_label, ignore_labels=['Null']):

    # 1. 数据预处理：将列表转换为集合 (Set) 以便进行 O(1) 复杂度的快速匹配
    # 使用元组 (entity1, entity2, label) 作为唯一标识
    # 注意：这里假设同一对实体在同一个列表中只出现一种关系

    pred_set = set()
    for item in all_res:
        if item['label'] not in ignore_labels:
            pred_set.add((item['entity1_text'], item['entity2_text'], item['label']))

    gt_set = set()
    for item in all_label:
        if item['label'] not in ignore_labels:
            gt_set.add((item['entity1_text'], item['entity2_text'], item['label']))

    # 2. 计算匹配数量 (TP - True Positives)
    # 取两个集合的交集，即：实体1相同 AND 实体2相同 AND 关系Label相同
    correct_matches = pred_set.intersection(gt_set)
    correct_num = len(correct_matches)

    # 3. 获取分母
    # 预测总数 (用于计算 Precision)
    pred_num = len(pred_set)
    # 真实总数 (用于计算 Recall，即你提到的“总长度按照 all_labels 计算”)
    gt_num = len(gt_set)

    # 4. 计算指标 (防止除以 0 报错)
    precision = correct_num / pred_num if pred_num > 0 else 0.0
    recall = correct_num / gt_num if gt_num > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "correct_count": correct_num,
        "pred_total": pred_num,
        "label_total": gt_num
    }

def parse_re_info(json_file_path):
    """
        读取JSON文件中的实体，进行两两全匹配（笛卡尔积，排除自指），
        并提取上下文和位置信息。
        """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # --- 1. 获取文本和段落分割 (保持不变) ---
    text = json_data['text']
    # 注意：某些操作系统或编辑器可能会将换行处理不同，这里沿用您原始的 \n\n 分割
    paragraphs = text.split('\n\n')

    # 预计算每个段落的起始和结束位置
    paragraph_positions = []
    current_position = 0
    for i, para in enumerate(paragraphs):
        para_length = len(para)
        # 记录段落起始位置和长度
        paragraph_positions.append({
            'start': current_position,
            'end': current_position + para_length,
            'text': para,
            'index': i
        })
        # 更新当前位置（加上本段长度和分隔符长度，通常是2个字符）
        current_position += para_length + (2 if i < len(paragraphs) - 1 else 0)

    # --- 2. 获取所有实体 ---
    entities = json_data['entities']

    # --- 3. 生成两两全匹配实体对 ---
    pairs_info = []

    # 使用双重循环遍历所有实体组合
    for i, entity1_data in enumerate(entities):
        for j, entity2_data in enumerate(entities):

            # 排除实体自己对自己匹配
            if i == j:
                continue

            # 如果需要去重（即 A-B 和 B-A 视为同一个），可以启用下行代码：
            # if i >= j: continue
            # 但通常关系抽取任务中 A->B 和 B->A 是不同的，这里默认为全排列。

            # 查找实体所属段落
            para1 = find_paragraph(entity1_data['start_offset'], paragraph_positions)
            para2 = find_paragraph(entity2_data['start_offset'], paragraph_positions)

            # 计算相对段落的偏移量
            e1_start_rel = entity1_data['start_offset'] - para1['start']
            e1_end_rel = entity1_data['end_offset'] - para1['start']

            e2_start_rel = entity2_data['start_offset'] - para2['start']
            e2_end_rel = entity2_data['end_offset'] - para2['start']

            # 构建结果字典
            # 注意：已移除 'relation_label' 字段
            pair_info = {
                'entity1': {
                    'id': entity1_data['id'],  # 建议保留ID以便溯源
                    'start': e1_start_rel,
                    'end': e1_end_rel,
                    # 防止切片越界（虽然正常数据不会），加个保护
                    'text': para1['text'][e1_start_rel:e1_end_rel],
                    'label': entity1_data['label'],
                    'paragraph_index': para1['index']
                },
                'entity2': {
                    'id': entity2_data['id'],
                    'start': e2_start_rel,
                    'end': e2_end_rel,
                    'text': para2['text'][e2_start_rel:e2_end_rel],
                    'label': entity2_data['label'],
                    'paragraph_index': para2['index']
                },
                'sentence1': para1['text'],
                'sentence2': para2['text'],
                'relation_position_info': {
                    'entity1_paragraph': para1['index'],
                    'entity2_paragraph': para2['index'],
                    'same_paragraph': para1['index'] == para2['index'],
                    'total_paragraph': abs(para1['index'] - para2['index'])  # 增加段落距离信息
                }
            }

            pairs_info.append(pair_info)

    return pairs_info


# 查找实体所属段落的函数
def find_paragraph(offset, paragraph_positions):
    """
    根据偏移量查找实体所在的段落

    Args:
        offset: 实体的起始偏移量
        paragraph_positions: 段落位置信息列表

    Returns:
        包含段落信息的字典，包括index字段
    """
    for para_info in paragraph_positions:
        if para_info['start'] <= offset < para_info['end']:
            return para_info

    # 如果没有找到，返回最后一个段落（处理边缘情况）
    return paragraph_positions[-1]


def find_paragraph(char_offset, paragraph_positions):
    """根据字符偏移量找到对应的段落信息"""
    for para in paragraph_positions:
        if para['start'] <= char_offset < para['end']:
            return para
    # 如果找不到返回最后一个段落（处理可能的边界问题）
    return paragraph_positions[-1]


def find_paragraph_index(char_offset, paragraphs):
    """
    根据字符偏移量找到对应的段落索引

    参数:
        char_offset: 字符偏移量
        paragraphs: 分割后的段落列表

    返回:
        段落索引(从0开始)，如果找不到返回None
    """
    current_pos = 0
    for i, para in enumerate(paragraphs):
        para_length = len(para)
        # 每个段落后面有\n\n，除了最后一个
        if i < len(paragraphs) - 1:
            para_length += 2  # 加上\n\n的长度

        if current_pos <= char_offset < current_pos + para_length:
            return i

        current_pos += para_length

    return None


def convert_label_to_id_re(re_dict, args):
    map2id = args.label2id

    all_label_ids = []
    for item in re_dict:
        all_label_ids.append(map2id[item['relation_label']])

    return all_label_ids


def GetDataLoader_RE(args, dicts, labels_ids, batch_size):
    features = []
    for re_dict, label_id in zip(dicts, labels_ids):
        features.append(convert_to_feature_re(re_dict, label_id, args))

    dataset = convert_features_to_dataset_re(features)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader


def convert_to_feature_re(re_dict, label_id, args):
    with open('./data/Chinese/Recipe/labels_ner.jsonl', 'r', encoding= 'utf-8') as f:
        map_entity = json.load(f)
    max_seq_length_re = args.max_seq_length_re
    sentence_tokens = []
    label_ids_tokens = []
    entity1_start = re_dict['entity1']['start']
    entity1_end = re_dict['entity1']['end']
    entity1_type = re_dict['entity1']['label']
    proxy_name_1 = map_entity['proxy_train'][map_entity['train'].index(entity1_type)]
    entity2_start = re_dict['entity2']['start']
    entity2_end = re_dict['entity2']['end']
    entity2_type = re_dict['entity1']['label']
    proxy_name_2 = map_entity['proxy_train'][map_entity['train'].index(entity2_type)]
    sentence_1 = re_dict['sentence1']
    sentence_2 = re_dict['sentence2']
    e1_pos = re_dict['relation_position_info']['entity1_paragraph']
    e2_pos = re_dict['relation_position_info']['entity2_paragraph']
    re_len = re_dict['relation_position_info']['total_paragraph']
    # 句子一样
    if sentence_1 == sentence_2:
        if entity1_start < entity2_start:
            sentence = (
                    "[CLS]" + sentence_1[:entity1_start] + "<e1>" + sentence_1[entity1_start:entity1_end] + "</e1>"
                    + sentence_1[entity1_end:entity2_start] + "<e2>" + sentence_1[entity2_start:entity2_end] +
                    "</e2>" + sentence_1[entity2_end:] + "[SEP]" + proxy_name_1 + "[SEP]" + proxy_name_2
            )
        else:
            # 实体2在前
            sentence = ("[CLS]" + sentence_1[:entity2_start] + "<e2>" + sentence_1[entity2_start:entity2_end] +
                        "</e2>" + sentence_1[entity2_end:entity1_start] + "<e1>" +
                        sentence_1[entity1_start:entity1_end] + "</e1>" + sentence_1[entity1_end:] + "[SEP]" + proxy_name_1 + "[SEP]" + proxy_name_2
                        )

    # 句子不一样，输入格式为[CLS] [E1] e1 [/E1]| [E2] e2 [/E2] [SEP] sentence1 [SEP] sentence2 [SEP]
    else:
        sentence = ("[CLS]" + sentence_1[:entity1_start] + "<e1>" + sentence_1[entity1_start:entity1_end] + "</e1>"
                    + sentence_1[entity1_end:] + "[SEP]" + sentence_2[:entity2_start] + "<e2>" +
                    sentence_2[entity2_start: entity2_end] + "</e2>" + sentence_2[entity2_end:]
                    + "[SEP]" + proxy_name_1 + "[SEP]" + proxy_name_2)
    sentence_tokens = args.tokenizer.tokenize(sentence)
    e11_p = sentence_tokens.index("<e1>")  # the start position of entity1
    e12_p = sentence_tokens.index("</e1>")  # the end position of entity1
    e21_p = sentence_tokens.index("<e2>")  # the start position of entity2
    e22_p = sentence_tokens.index("</e2>")  # the end position of entity2
    sentence_tokens[e11_p] = "$"
    sentence_tokens[e12_p] = "$"
    sentence_tokens[e21_p] = "#"
    sentence_tokens[e22_p] = "#"

    input_ids = args.tokenizer.convert_tokens_to_ids(sentence_tokens)
    padding_length = max_seq_length_re - len(input_ids)
    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length
    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length_re]
        input_ids_padded = input_ids[:max_seq_length_re]

    # e1 mask, e2 mask
    e1_mask = [0] * len(attention_mask)
    e2_mask = [0] * len(attention_mask)
    for i in range(e11_p, e12_p+1):
        e1_mask[i] = 1
    for i in range(e21_p, e22_p+1):
        e2_mask[i] = 1
    # 处理边界的0和1状态
    if 1 in e1_mask:
        first_e1 = e1_mask.index(1)
        last_e1 = len(e1_mask) - 1 - e1_mask[::-1].index(1)
        e1_mask[first_e1] = 0
        e1_mask[last_e1] = 0

    if 1 in e2_mask:
        first_e2 = e2_mask.index(1)
        last_e2 = len(e2_mask) - 1 - e2_mask[::-1].index(1)
        e2_mask[first_e2] = 0
        e2_mask[last_e2] = 0
    token_type_ids = [0] * max_seq_length_re
    assert len(input_ids_padded) == max_seq_length_re
    assert len(token_type_ids) == max_seq_length_re
    assert len(attention_mask) == max_seq_length_re

    return InputFeature_RE(input_ids_padded, token_type_ids, attention_mask, label_id, e1_mask, e2_mask, e1_pos, e2_pos,
                           re_len)


def plot_batch_shap_heatmaps(t_lists, c_lists, relations, save_name="compact_shap.png", args=None):
    """
    绘制纵向更紧凑的批量SHAP热力图
    """
    num_samples = len(t_lists)
    if num_samples == 0: return

    # 关键修改：每行高度从 3 缩减到 1.5 或 2
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(20, 1.8 * num_samples))

    if num_samples == 1:
        axes = [axes]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(num_samples):
        # 准备数据
        data = np.array([c_lists[i]])
        tokens = t_lists[i]
        rel = relations[i]

        # 绘制热力图
        sns.heatmap(
            data,
            xticklabels=tokens,
            yticklabels=False,
            cmap='RdBu_r',
            annot=True,
            fmt=".2f",
            center=0,
            ax=axes[i],
            # 缩小颜色条，shrink=0.5 让它不那么高
            cbar_kws={"shrink": 0.6} if i == 0 else None,
            cbar=(i == 0)
        )

        # 紧凑型标题：减小字体和间距
        result_str = "✓" if rel['is_correct'] else "✗"
        title_str = f"S{i} | True: {rel['true_label']} | Pred: {rel['pred_label']} [{result_str}]"
        axes[i].set_title(title_str, fontsize=11, pad=5)

        # 优化横坐标：减小字体，旋转角度
        axes[i].tick_params(axis='x', labelsize=9)
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 减少子图之间的垂直间距
    plt.subplots_adjust(hspace=0.6)

    # 保存
    save_dir = getattr(args, 'results_dir', './results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    print(f"紧凑型热力图已保存: {save_path}")


def convert_features_to_dataset_re(features):
    # convert to Tensors
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_attention_mask = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_e1_mask = torch.tensor([feature.e1_mask for feature in features], dtype=torch.long)
    all_e2_mask = torch.tensor([feature.e2_mask for feature in features], dtype=torch.long)
    all_e1_pos = torch.tensor([feature.e1_pos for feature in features], dtype=torch.long)
    all_e2_pos = torch.tensor([feature.e2_pos for feature in features], dtype=torch.long)
    all_re_len = torch.tensor([feature.re_len for feature in features], dtype=torch.long)
    all_label_ids = torch.tensor([feature.label_ids for feature in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids, all_e1_mask,
                            all_e2_mask, all_e1_pos, all_e2_pos, all_re_len)
    return dataset
