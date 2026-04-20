import csv
import json
import os
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import seaborn as sns
from model import BertModelNER, BertModelRE
from model_utils.egp import MetricsCalculator
from utils import read_recipe_ner_data, convert_label_to_id_ner, convert_to_feature_ner, build_re_dict, \
    convert_label_to_id_re, convert_to_feature_re, convert_ids_to_labels, extract_relations_from_json, \
    calc_sentence_entity_accuracy, collect_error_cases_by_label, generate_ner_confusion_matrix, parse_recipe_ner, \
    merge_ner_to_json, parse_re_info, calculate_re_metrics
from utils_cn import read_recipe_ner_data_cn
import shap


def evaluate_ner(args):
    # --- 1. 数据读取部分 (保持不变) ---
    if "Chinese-test" in args.dataset:
        all_sentence_test, all_label_test = read_recipe_ner_data_cn(args.filepath_test)
    else:
        all_files = [
            os.path.join(args.filepath_test, f) for f in os.listdir(args.filepath_test)
        ]
        all_sentence_test = []
        all_label_test = []
        for filepath in all_files:
            sentence, label = read_recipe_ner_data(filepath)
            all_sentence_test.extend(sentence)
            all_label_test.extend(label)

    label_ids = convert_label_to_id_ner(all_label_test, args)
    test_features = []
    for sentence, label_ids in zip(all_sentence_test, label_ids):
        test_features.append(convert_to_feature_ner(sentence, label_ids, args))

    input_ids = torch.stack([torch.tensor(feature.input_ids) for feature in test_features])
    attention_mask = torch.stack([torch.tensor(feature.attention_mask) for feature in test_features])
    label_ids = torch.stack([torch.tensor(feature.label_ids) for feature in test_features])

    # --- 2. 模型加载部分 (保持不变) ---
    ckpt_dir = './checkpoint/' \
               + args.task + '/' + args.pretrained_model + '-' + str(args.seed) + '/' \
               + args.IO_mode + '-' + args.dataset + '/' + args.ner_model_structure + '/' \
               + 'model-ner' + '-' + str(args.k_folds) + '.ckpt'

    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint not found: {ckpt_dir}")
        return {}, {}

    checkpoint_bert_model = torch.load(ckpt_dir, map_location=args.device)
    ner_model = BertModelNER(args).to(args.device)
    ner_model.load_state_dict(checkpoint_bert_model['model_state_dict'])

    # --- 3. 预测与结果保存设置 ---
    output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
    os.makedirs(output_dir, exist_ok=True)

    # 定义保存 Bad Case 的 CSV 文件路径
    bad_case_file = os.path.join(output_dir, f"bad_cases_{args.dataset}.csv")
    print(f"Bad cases will be saved to: {bad_case_file}")

    results_file = open(output_dir + args.IO_mode + '-' + args.dataset + '.txt', 'a')
    batch_size = 1
    num_batches = len(input_ids) // batch_size + 1

    all_prediction = 0
    all_f1 = 0
    all_recall = 0
    all_true_labels = []
    all_pred_labels = []

    ner_model.eval()

    # 设定阈值：F1 低于此值的句子将被记录到 CSV
    LOW_F1_THRESHOLD = 0.8

    # 打开 CSV 文件准备写入
    # encoding='utf-8-sig' 是为了让 Excel 正确显示中文，防止乱码
    with open(bad_case_file, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头：句子ID, F1分数, 词, 真实标签, 预测标签, 完整原句(方便上下文理解)
        writer.writerow(["Sentence_ID", "Sentence_F1", "Word", "True_Label", "Pred_Label", "Full_Sentence_Text"])

        with torch.no_grad():
            for i in tqdm(range(num_batches - 1), desc="Evaluating", dynamic_ncols=True):
                batch_input_ids = input_ids[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_attention_mask = attention_mask[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_label_ids = label_ids[i * batch_size:(i + 1) * batch_size].to(args.device)

                # 获取原始 Token
                current_tokens = all_sentence_test[i]

                _, logits, labels = ner_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    label_ids=batch_label_ids
                )

                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = labels.cpu().numpy()

                batch_true_labels, batch_pred_labels = convert_ids_to_labels(
                    labels,
                    pred_ids,
                    id2label=args.id2label_test,
                    label2id=args.initial_label
                )

                all_true_labels.extend(batch_true_labels)
                all_pred_labels.extend(batch_pred_labels)

                # 计算单句 F1
                precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_ids,
                                                                           average='macro', zero_division=0)

                if precision is None or recall is None or f1 is None:
                    num_batches = num_batches - 1
                else:
                    all_prediction += precision
                    all_recall += recall
                    all_f1 += f1

                    # === 保存 Bad Case 到 CSV ===
                    if f1 < LOW_F1_THRESHOLD:
                        # 处理数据格式
                        curr_true_seq = batch_true_labels[0] if isinstance(batch_true_labels[0],
                                                                           list) else batch_true_labels
                        curr_pred_seq = batch_pred_labels[0] if isinstance(batch_pred_labels[0],
                                                                           list) else batch_pred_labels

                        full_sentence_str = " ".join(current_tokens)

                        # 确保长度一致，避免 zip 丢失数据或报错
                        length = min(len(current_tokens), len(curr_true_seq), len(curr_pred_seq))

                        for idx in range(length):
                            word = current_tokens[idx]
                            true_lbl = curr_true_seq[idx]
                            pred_lbl = curr_pred_seq[idx]

                            # 仅写入行
                            writer.writerow([
                                i,  # 句子ID (Batch Index)
                                f"{f1:.4f}",  # 当前句子的 F1 分数
                                word,  # 词
                                true_lbl,  # 真实标签
                                pred_lbl,  # 预测标签
                                full_sentence_str if idx == 0 else ""  # 完整句子仅在每个句子的第一行显示，保持表格整洁
                            ])

                        # 如果需要分隔不同句子，可以插入一个空行（可选）
                        # writer.writerow([])

    # --- 4. 汇总指标计算 (保持不变) ---
    metric = {"f1": all_f1 / (num_batches - 1),
              "precision": all_prediction / (num_batches - 1),
              "recall": all_recall / (num_batches - 1),
              }

    precision_per_label, recall_per_label, f1_per_label, support = precision_recall_fscore_support(
        all_true_labels,
        all_pred_labels,
        average=None,
        labels=np.arange(len(args.initial_label)),
        zero_division=0
    )

    per_metric = {"f1": f1_per_label,
                  "precision": precision_per_label,
                  "recall": recall_per_label,
                  "num": support}

    print(metric, file=results_file)
    results_file.close()

    return metric, per_metric


def predict_recipe_ner(args, json_dir):
    # 加载模型
    ckpt_dir = './checkpoint/' \
               + args.task + '/' + args.pretrained_model + '-' + str(args.seed) + '/' \
               + args.IO_mode + '-' + args.dataset + '/' + args.ner_model_structure + '/' \
               + 'model-ner' + '-' + str(args.k_folds) + '.ckpt'

    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint not found: {ckpt_dir}")
        return {}, {}

    checkpoint_bert_model = torch.load(ckpt_dir, map_location=args.device)
    ner_model = BertModelNER(args).to(args.device)
    ner_model.load_state_dict(checkpoint_bert_model['model_state_dict'])
    ner_model.eval()
    all_files = [
        os.path.join(json_dir, f) for f in os.listdir(json_dir)
    ]
    for filepath in all_files:
        test_features = []
        sentences, label_lists = parse_recipe_ner(filepath)
        label_ids = [[args.label2id[label] for label in sent_labels] for sent_labels in label_lists]
        for sentence, label_id in zip(sentences, label_ids):
            test_features.append(convert_to_feature_ner(sentence, label_id, args))
        input_ids = torch.stack([torch.tensor(feature.input_ids) for feature in test_features])
        attention_mask = torch.stack([torch.tensor(feature.attention_mask) for feature in test_features])
        label_ids = torch.stack([torch.tensor(feature.label_ids) for feature in test_features])
        batch_size = 1
        num_batches = len(input_ids) // batch_size + 1
        pred_list = []
        with torch.no_grad():
            for i in tqdm(range(num_batches - 1), desc="Evaluating", dynamic_ncols=True):
                batch_input_ids = input_ids[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_attention_mask = attention_mask[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_label_ids = label_ids[i * batch_size:(i + 1) * batch_size].to(args.device)

                _, logits, labels = ner_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    label_ids=batch_label_ids
                )
                pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = labels.cpu().numpy()

                batch_true_labels, batch_pred_labels = convert_ids_to_labels(
                    labels,
                    pred_ids,
                    id2label=args.id2label_test,
                    label2id=args.initial_label
                )
                pred_list.append(batch_pred_labels)
        map_labels = ['O'] + args.initial_label
        updated_data = merge_ner_to_json(filepath, sentences, pred_list, map_labels)
        file_name = os.path.basename(filepath)
        output_dir = r"D:\Data_Store\Dataset\Chinese_recipe_flow_graph\Data\Folds\error\RE\error"
        save_path = os.path.join(output_dir, file_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)


def evaluate_re(args):
    # read test data
    all_files = [
        os.path.join(args.filepath_test, f) for f in os.listdir(args.filepath_test)
    ]
    all_dict = []
    for file in all_files:
        re_dict = extract_relations_from_json(file)
        all_dict.append(re_dict)
    flattened_dict = [item for sublist in all_dict for item in sublist]
    all_label_ids = convert_label_to_id_re(flattened_dict, args)

    test_features = []
    for re_dict, label_id in zip(flattened_dict, all_label_ids):
        test_features.append(convert_to_feature_re(re_dict, label_id, args))

    input_ids = torch.stack(
        [torch.tensor(feature.input_ids) for feature in test_features])
    attention_mask = torch.stack(
        [torch.tensor(feature.attention_mask) for feature in test_features])
    label_ids = torch.stack(
        [torch.tensor(feature.label_ids) for feature in test_features])
    e1_masks = torch.stack(
        [torch.tensor(feature.e1_mask) for feature in test_features])
    e2_masks = torch.stack(
        [torch.tensor(feature.e2_mask) for feature in test_features])
    e1_pos = torch.stack(
        [torch.tensor(feature.e1_pos) for feature in test_features])
    e2_pos = torch.stack(
        [torch.tensor(feature.e2_pos) for feature in test_features])
    re_len = torch.stack(
        [torch.tensor(feature.re_len) for feature in test_features])
    # load model
    ckpt_dir = './checkpoint/' \
               + args.task \
               + '/' \
               + args.pretrained_model \
               + '-' + str(args.seed) + '/' \
               + args.dataset + '/' + args.re_model_structure + '/' + 'model-re' + '-' + str(args.k_folds) + '.ckpt'
    checkpoint_bert_model = torch.load(ckpt_dir, map_location=args.device)
    re_model = BertModelRE(args).to(args.device)
    re_model.load_state_dict(checkpoint_bert_model['model_state_dict'])

    # predict
    output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
    os.makedirs(output_dir, exist_ok=True)

    results_file = open(output_dir + args.dataset + '.txt', 'a')
    batch_size = 10
    num_batches = len(input_ids) // batch_size + 1

    all_true_labels = []
    all_pred_labels = []

    all_prediction = 0
    all_f1 = 0
    all_recall = 0
    re_model.eval()
    print(
        f"###########{args.dataset}-{args.task}-{args.pretrained_model}-{args.re_model_structure}-{args.seed}-{args.k_folds}############",
        file=results_file)
    with torch.no_grad():
        for i in tqdm(range(num_batches - 1), desc="Evaluating", dynamic_ncols=True):
            batch_input_ids = input_ids[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_attention_mask = attention_mask[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_label_ids = label_ids[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_e1_masks = e1_masks[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_e2_masks = e2_masks[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_e1_pos = e1_pos[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_e2_pos = e2_pos[i * batch_size:(i + 1) * batch_size].to(args.device)
            batch_re_len = re_len[i * batch_size:(i + 1) * batch_size].to(args.device)
            _, logits, label = re_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                label_ids=batch_label_ids,
                e1_mask=batch_e1_masks,
                e2_mask=batch_e2_masks,
                e1_pos= batch_e1_pos,
                e2_pos=batch_e2_pos,
                re_len=batch_re_len
            )
            pred_ids = torch.argmax(logits, dim=-1)
            pred_ids = pred_ids.cpu().numpy()
            label = label.cpu().numpy()
            # print(f"pred_ids:{pred_ids} labels:{label}")
            all_true_labels.extend(label.flatten().tolist())  # 展平后转换为列表
            all_pred_labels.extend(pred_ids.flatten().tolist())

            precision, recall, f1, _ = precision_recall_fscore_support(label, pred_ids,
                                                                       average='macro', zero_division=0)
            if precision is None or recall is None or f1 is None:
                num_batches = num_batches - 1
            else:
                print(f"Batch {i + 1}/{num_batches} - "
                      f"Precision: {precision:.4f}, "
                      f"Recall: {recall:.4f}, "
                      f"F1 Score: {f1:.4f}")
                all_prediction = all_prediction + precision
                all_recall = all_recall + recall
                all_f1 = all_f1 + f1

    precision_per_label, recall_per_label, f1_per_label, support = precision_recall_fscore_support(
        all_true_labels,
        all_pred_labels,
        average=None,
        labels=np.arange(len(args.initial_label)),  # 假设args.initial_label包含所有标签
        zero_division=0
    )
    metric = {"f1": all_f1 / (num_batches - 1),
              "precision": all_prediction / (num_batches - 1),
              "recall": all_recall / (num_batches - 1),
              }
    per_metric = {"f1": f1_per_label,
                  "precision": precision_per_label,
                  "recall": recall_per_label,
                  "num": support}
    print(all_true_labels)
    print(all_pred_labels)
    error = generate_ner_confusion_matrix(all_true_labels, all_pred_labels)
    print(error)

    print(metric, file=results_file)

    return metric, per_metric


def explain_single_sample(args, sample_idx=0):
    print(f"--- 正在执行样本索引 {sample_idx} 的深度解释 ---")

    # 1. 数据准备 (复用你之前的逻辑)
    all_files = [os.path.join(args.filepath_test, f) for f in os.listdir(args.filepath_test)]
    all_dict = []
    for file in all_files:
        all_dict.append(extract_relations_from_json(file))
    flattened_dict = [item for sublist in all_dict for item in sublist]

    # 获取所有标签 ID
    all_label_ids = convert_label_to_id_re(flattened_dict, args)

    # 转化为特征列表
    test_features = []
    for re_dict, label_id in zip(flattened_dict, all_label_ids):
        test_features.append(convert_to_feature_re(re_dict, label_id, args))

    if sample_idx >= len(test_features):
        print(f"错误: 索引 {sample_idx} 越界")
        return None, None, None

    # 2. 加载模型
    ckpt_dir = os.path.join('./checkpoint/', args.task, f"{args.pretrained_model}-{args.seed}",
                            args.dataset, args.re_model_structure, f"model-re-{args.k_folds}.ckpt")
    checkpoint = torch.load(ckpt_dir, map_location=args.device)
    re_model = BertModelRE(args).to(args.device)
    re_model.load_state_dict(checkpoint['model_state_dict'])
    re_model.eval()

    # 3. 调用 SHAP 计算核心逻辑 (analyze_re_with_shap)
    try:
        # 这里的 analyze_re_with_shap 需返回 tokens, contributions, pred_idx
        tokens_list, contrib_list, pred_idx = analyze_re_with_shap(
            re_model=re_model,
            sample_idx=sample_idx,
            test_features=test_features,
            tokenizer=args.tokenizer,
            args=args
        )

        # 4. 提取标签名称
        # 假设你的 args 中存储了 id2label_train 映射表
        true_label_id = test_features[sample_idx].label_ids
        true_label_name = args.id2label_train[true_label_id]
        pred_label_name = args.id2label_train[pred_idx]

        relation_info = {
            "true_label": true_label_name,
            "pred_label": pred_label_name,
            "is_correct": bool(true_label_id == pred_idx)
        }

        return tokens_list, contrib_list, relation_info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None


def analyze_re_with_shap(re_model, sample_idx, test_features, tokenizer, args):
    re_model.eval()
    feature = test_features[sample_idx]
    input_ids_tensor = torch.tensor([feature.input_ids]).to(args.device)

    with torch.no_grad():
        orig_p1_mask, orig_p2_mask = re_model.get_proxy_masks(input_ids_tensor)

    def custom_predict(input_ids_np):
        batch_size = input_ids_np.shape[0]
        ids_tensor = torch.tensor(input_ids_np, dtype=torch.long).to(args.device)

        def expand_to_batch(arr):
            t = torch.tensor(arr).to(args.device)
            return t.unsqueeze(0).repeat(batch_size, *([1] * t.dim()))

        # 准备模型输入特征
        params = {
            "input_ids": ids_tensor,
            "attention_mask": expand_to_batch(feature.attention_mask),
            "e1_mask": expand_to_batch(feature.e1_mask),
            "e2_mask": expand_to_batch(feature.e2_mask),
            "e1_pos": expand_to_batch(feature.e1_pos),
            "e2_pos": expand_to_batch(feature.e2_pos),
            "re_len": expand_to_batch(feature.re_len),
            "p1_mask": orig_p1_mask.repeat(batch_size, 1),
            "p2_mask": orig_p2_mask.repeat(batch_size, 1)
        }

        with torch.no_grad():
            _, logits, _ = re_model(**params)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    # SHAP 计算
    background = np.full((1, len(feature.input_ids)), tokenizer.pad_token_id or 0)
    explainer = shap.KernelExplainer(custom_predict, background)
    shap_values = explainer.shap_values(np.array([feature.input_ids]), nsamples=500)

    # 提取预测类别
    initial_probs = custom_predict(np.array([feature.input_ids]))
    pred_idx = np.argmax(initial_probs[0])

    # 鲁棒性提取 SHAP 值
    if isinstance(shap_values, list):
        current_shap_values = shap_values[pred_idx][0]
    else:
        current_shap_values = shap_values[0, :, pred_idx] if shap_values.ndim == 3 else shap_values[0]

    # --- 数据对齐与清洗 ---
    all_tokens = tokenizer.convert_ids_to_tokens(feature.input_ids)
    shap_len = len(current_shap_values)

    # 过滤特殊 Token (如 [PAD]) 并保留对应的贡献度
    clean_tokens = []
    clean_contributions = []

    for i, token in enumerate(all_tokens):
        if i >= shap_len: break
        if token not in ['[PAD]']:  # 如果需要保留 [CLS]/[SEP]，从列表中移除即可
            clean_tokens.append(token)
            # 转换为标准 Python float 方便后续处理
            clean_contributions.append(float(current_shap_values[i]))

    # 可选：如果你仍然想生成图片，可以在这里保留之前的 plt 代码

    return clean_tokens, clean_contributions, pred_idx


def predict_re(args, json_dir):
    ckpt_dir = './checkpoint/' \
               + args.task \
               + '/' \
               + args.pretrained_model \
               + '-' + str(args.seed) + '/' \
               + args.dataset + '/' + args.re_model_structure + '/' + 'model-re' + '-' + str(args.k_folds) + '.ckpt'
    checkpoint_bert_model = torch.load(ckpt_dir, map_location=args.device)
    re_model = BertModelRE(args).to(args.device)
    re_model.load_state_dict(checkpoint_bert_model['model_state_dict'])
    re_model.eval()
    all_files = [
        os.path.join(json_dir, f) for f in os.listdir(json_dir)
    ]
    file_names = [os.path.basename(f) for f in all_files]
    val_dir = r'D:\Data_Store\Dataset\Chinese_recipe_flow_graph\Data\Folds\error\RE\test'
    for file in file_names:
        pre_file = os.path.join(json_dir, file)
        label_file = os.path.join(val_dir, file)
        test_features = []
        pre_data = parse_re_info(pre_file)
        labels = extract_relations_from_json(label_file)

        for pre_dict in pre_data:
            test_features.append(convert_to_feature_re(pre_dict, [], args))
        input_ids = torch.stack(
            [torch.tensor(feature.input_ids) for feature in test_features])
        attention_mask = torch.stack(
            [torch.tensor(feature.attention_mask) for feature in test_features])
        e1_masks = torch.stack(
            [torch.tensor(feature.e1_mask) for feature in test_features])
        e2_masks = torch.stack(
            [torch.tensor(feature.e2_mask) for feature in test_features])
        e1_pos = torch.stack(
            [torch.tensor(feature.e1_pos) for feature in test_features])
        e2_pos = torch.stack(
            [torch.tensor(feature.e2_pos) for feature in test_features])
        re_len = torch.stack(
            [torch.tensor(feature.re_len) for feature in test_features])
        output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
        os.makedirs(output_dir, exist_ok=True)

        results_file = open(output_dir + args.dataset + '.txt', 'a')
        batch_size = 10
        num_batches = len(input_ids) // batch_size + 1
        print(
            f"###########{args.dataset}-{args.task}-{args.pretrained_model}-{args.re_model_structure}-{args.seed}-{args.k_folds}############",
            file=results_file)
        all_pred = []
        with torch.no_grad():
            for i in tqdm(range(num_batches - 1), desc="Evaluating", dynamic_ncols=True):
                batch_input_ids = input_ids[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_attention_mask = attention_mask[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_e1_masks = e1_masks[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_e2_masks = e2_masks[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_e1_pos = e1_pos[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_e2_pos = e2_pos[i * batch_size:(i + 1) * batch_size].to(args.device)
                batch_re_len = re_len[i * batch_size:(i + 1) * batch_size].to(args.device)

                _, logits, _ = re_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    e1_mask=batch_e1_masks,
                    e2_mask=batch_e2_masks,
                    e1_pos=batch_e1_pos,
                    e2_pos=batch_e2_pos,
                    re_len=batch_re_len
                )
                pred_ids = torch.argmax(logits, dim=-1)
                pred_ids = pred_ids.cpu().numpy()
                all_pred.extend(pred_ids)
                # label = label.cpu().numpy()
        # --- 核心步骤：将预测标签与原始实体对匹配 ---
        all_res = []
        all_label = []
        for idx, pred_label_id in enumerate(all_pred):
            # 找到对应的原始实体信息
            origin_pair = pre_data[idx]
            # 将结果整合
            res = {
                'entity1_text': origin_pair['entity1']['text'],
                'entity2_text': origin_pair['entity2']['text'],
                'label': args.initial_label[int(pred_label_id)]
            }
            all_res.append(res)
        for true_data in labels:
            res_true = {
                'entity1_text': true_data['entity1']['text'],
                'entity2_text': true_data['entity2']['text'],
                'label': true_data['relation_label']
            }
            all_label.append(res_true)
        answer = calculate_re_metrics(all_res, all_label)
        print(answer, file=results_file)





