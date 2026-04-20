def read_recipe_ner_data_cn(filepath):
    """
    读取NER数据文件，空行作为句子分隔符
    返回:
        sentences_word: List[List[str]] 形状为 (batch_size, seq_len)
        sentences_label: List[List[str]] 形状为 (batch_size, seq_len)
    """
    batch_words = []
    batch_labels = []
    current_sentence = []
    current_labels = []

    with open(filepath, "r", encoding='UTF-8') as f:
        for line in f:
            stripped_line = line.strip()

            # 空行表示句子结束
            if not stripped_line:
                if current_sentence:  # 确保不是连续空行
                    batch_words.append(current_sentence)
                    batch_labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue

            # 跳过文档起始标记
            if "-DOCSTART-" in stripped_line:
                continue

            # 分割字段
            parts = stripped_line.split()

            try:
                word = parts[0]  # 第一列是词
                label = parts[1]  # 第二列是标签
            except IndexError:
                print(f"Error in file: {filepath}")
                print(f"Ignore invalid line: {stripped_line}")
                continue

            current_sentence.append(word)
            current_labels.append(label)

        # 处理文件末尾没有空行的情况
        if current_sentence:
            batch_words.append(current_sentence)
            batch_labels.append(current_labels)

    # 检查所有句子长度是否一致(如果不一致需要padding)
    seq_lengths = [len(sent) for sent in batch_words]
    if len(set(seq_lengths)) > 1:
        print(f"Warning: Sentences have varying lengths: {seq_lengths}")
        # 这里可以添加padding逻辑如果需要

    return batch_words, batch_labels
