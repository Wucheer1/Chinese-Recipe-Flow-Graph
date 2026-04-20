# This is a sample Python script.
import argparse

import numpy as np
import torch
from transformers import BertTokenizer

from evaluate import evaluate_ner, evaluate_re, predict_recipe_ner, predict_re, explain_single_sample
from train import train_ner, train_re
from utils import read_labels_from_file, set_seeds, plot_batch_shap_heatmaps

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_train", default=None, type=str, help="train file path of the data")
    parser.add_argument("--filepath_test", default=None, type=str, help="test file path of the data")
    parser.add_argument("--dataset", type=str, default='English-Recipe', help="dataset kind")
    parser.add_argument("--filepath_labels", default=None, type=str, help="path of the label")
    parser.add_argument("--task", default='NER', type=str, help="NER or RE")
    parser.add_argument("--results_dir", default='./results', type=str, help="path of results")
    # ------------------------------------ NER ----------------------------------------
    parser.add_argument("--batch_size_ner", default=24, type=int, help="train batch_size of NER")
    parser.add_argument("--train_LR_ner", default=3e-5, type=float, help="train_source_LR")
    parser.add_argument("--train_epoches_ner", default=10, type=int, help="train_epochs_ner")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--lex_model", action='store_true', default=False,
                       help="if use lexicon deep learning model")
    parser.add_argument("--crf", action='store_true', default=False,
                       help="if use crf")
    parser.add_argument("--crf_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--egp", action='store_true', default=False,
                       help="if use efficient-global-pointer")
    parser.add_argument("--egp_learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for egp")
    parser.add_argument("--lstm", action='store_true', default=False,
                       help="if use lstm")
    parser.add_argument("--lstm_learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for lstm")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                             "learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--IO_mode", type=str, default='BIO', help=" BIOES or BIO or IO")
    parser.add_argument("--ner_model_structure", type=str, default='BERT+CRF', help="structure of model")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length")
    parser.add_argument("--label_num_ner", default=11, type=int, help="label of ner num")
    # —---------------------------------------RE-------------------------------------------------
    parser.add_argument("--batch_size_re", default=12, type=int, help="train batch_size of RE")
    parser.add_argument("--train_epoches_re", default=5, type=int, help="train_epoches_re")
    parser.add_argument("--weight_decay_re", default=0.01, type=float, help="Weight decay if we apply some for re")
    parser.add_argument("--learning_rate_re", default=2e-5, type=float,
                        help="The initial learning rate for Adam for re")
    parser.add_argument("--adam_epsilon_re", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer for re")
    parser.add_argument("--warmup_steps_re", default=0, type=float,
                        help="RE warmup")
    parser.add_argument("--dropout_rate_re", default=0.1, type=int, help="dropout_rate of re")
    parser.add_argument("--max_seq_length_re", default=512, type=int, help="The maximum total input sequence length re")
    parser.add_argument("--re_model_structure", type=str, default='BERT', help="structure of model")
    # --------------------------------------all--------------------------------------------------
    parser.add_argument("--hidden_size", default=768, type=int, help="model size")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=int, help="model size")
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str, help="pretrained_model")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--gpu_id", type=int, default=0, help="select on which gpu to train.")

    args = parser.parse_args()
    args.task = 'RE'
    args.IO_mode = 'BIO'
    args.dataset = "Chinese-Recipe-RE-Ablation"
    # labels
    args.filepath_labels = r"D:\Project\Python\recipe-flow\data\Chinese\Recipe\labels_re.jsonl"
    labels_from_file = read_labels_from_file(args.filepath_labels, args)
    label_mapping = {
        0: 'id2label',
        1: 'id2label_train',
        2: 'id2label_dev',
        3: 'id2label_test',
        4: 'label2id',
        5: 'id2proxy_label',
        6: 'id2proxy_label_train',
        7: 'id2proxy_label_dev',
        8: 'id2proxy_label_test',
        9: 'proxy_label2id',
        10: 'initial_label'
    }
    for i, label_index in enumerate(range(11)):
        label_name = label_mapping.get(label_index)
        setattr(args, label_name, labels_from_file[i])
    # print(args.label2id)
    print(args.initial_label)
    ############################################################################
    args.gpu_id = 0
    print('***************** working on gpu id: ', args.gpu_id, ' *****************')
    args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.n_gpu = 0 if not torch.cuda.is_available else torch.cuda.device_count()
    args.n_gpu = min(1, args.n_gpu)
    set_seeds(args)
    args.pretrained_model = "bert-base-chinese"
    if args.task == 'NER':
        args.ner_model_structure = "BERT-CRF"
        args.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        all_metrics = {"f1": 0, "precision": 0, "recall": 0}
        all_per_metrics = {
            "f1": np.zeros(len(args.initial_label)),
            "precision": np.zeros(len(args.initial_label)),
            "recall": np.zeros(len(args.initial_label)),
            "num": np.zeros(len(args.initial_label))
        }
        num_folds = 10
        # args.filepath_train = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/NER/fold_0/train"
        # args.filepath_test = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/NER/fold_0/test"
        for i in range(1):
            print(f"###############################K-FOLDS:{i}###################################")
            args.filepath_train = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/NER/fold_{i}/train"
            args.filepath_test = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/NER/fold_{i}/test"
            args.k_folds = i
            args.lstm = False
            args.egp = False
            args.crf = True
            args.lex_model = False
            train_ner(args)
            metric, per_metric = evaluate_ner(args)
            predict_recipe_ner(args, r"D:\Data_Store\Dataset\Chinese_recipe_flow_graph\Data\Folds\error\RE\test")

            # 累加总体指标
            for key in ['f1', 'precision', 'recall']:
                all_metrics[key] += metric[key]

            # 累加标签级指标
            for key in ['f1', 'precision', 'recall', 'num']:
                all_per_metrics[key] += per_metric[key]

            # 在循环内输出当前fold的标签级评估结果
            output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
            results_file = open(output_dir + args.IO_mode + '-' + args.dataset + '.txt', 'a')

            print(f"\nFold {i} - Per-Class Metrics:", file=results_file)
            for idx, label in enumerate(args.initial_label):
                f1 = per_metric['f1'][idx]
                precision = per_metric['precision'][idx]
                recall = per_metric['recall'][idx]
                support = per_metric['num'][idx]
                print(
                    f"Label '{label}': F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={int(support)}",
                    file=results_file)

            # 也可以输出当前fold的总体指标
            print(f"Fold {i} - Overall Metrics: {metric}", file=results_file)
            results_file.close()

            # 计算并输出平均指标（保持不变）
        average_metrics = {key: value / num_folds for key, value in all_metrics.items()}
        average_per_metrics = {
            key: value / num_folds
            for key, value in all_per_metrics.items()
        }

        output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
        results_file = open(output_dir + args.IO_mode + '-' + args.dataset + '.txt', 'a')
        print(f"\nAverage Metrics: {average_metrics}", file=results_file)
        print("\nPer-Class Average Metrics:", file=results_file)
        for idx, label in enumerate(args.initial_label):
            f1 = average_per_metrics['f1'][idx]
            precision = average_per_metrics['precision'][idx]
            recall = average_per_metrics['recall'][idx]
            support = average_per_metrics['num'][idx]
            print(
                f"Label '{label}': F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={int(support)}",
                file=results_file)
        results_file.close()

    elif args.task == 'RE':
        args.re_model_structure = "all"
        args.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        special_tokens = {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']}
        args.tokenizer.add_special_tokens(special_tokens)
        all_metrics = {"f1": 0, "precision": 0, "recall": 0}
        all_per_metrics = {
            "f1": np.zeros(len(args.initial_label)),
            "precision": np.zeros(len(args.initial_label)),
            "recall": np.zeros(len(args.initial_label)),
            "num": np.zeros(len(args.initial_label))
        }
        num_folds = 10

        output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
        results_file = open(output_dir + args.dataset + '.txt', 'a')
        for i in range(1):
            print(f"###############################K-FOLDS:{i}###################################")
            args.filepath_train = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/RE_a/fold_{i}/train"
            args.filepath_test = f"D:/Data_Store/Dataset/Chinese_recipe_flow_graph/Data/Folds/RE_a/fold_{i}/test"
            args.k_folds = i
            t_lists = []
            c_lists = []
            relations = []
            for j in range(30):
                t_list, c_list, relation = explain_single_sample(args, sample_idx=j)
                t_lists.append(t_list)
                c_lists.append(c_list)
                relations.append(relation)
            plot_batch_shap_heatmaps(t_lists, c_lists, relations, args=args)
            print(t_lists)
            print(c_lists)
            print(relations)
            train_re(args)
            json_path = r'D:\Data_Store\Dataset\Chinese_recipe_flow_graph\Data\Folds\error\RE\error'
            metric, per_metric = evaluate_re(args)
            # predict_re(args, json_path)
            for key in ['f1', 'precision', 'recall']:
                all_metrics[key] += metric[key]

            # 累加标签级指标
            for key in ['f1', 'precision', 'recall', 'num']:
                all_per_metrics[key] += per_metric[key]

            print(f"\nFold {i} Results:", file=results_file)
            print(
                f"Overall - F1: {metric['f1']:.4f}, Precision: {metric['precision']:.4f}, Recall: {metric['recall']:.4f}",
                file=results_file)

            print("Per-Class Metrics:", file=results_file)
            for idx, label in enumerate(args.initial_label):
                f1 = per_metric['f1'][idx]
                precision = per_metric['precision'][idx]
                recall = per_metric['recall'][idx]
                support = per_metric['num'][idx]
                print(f"  {label}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={int(support)}",
                      file=results_file)

            print("-" * 80, file=results_file)
            results_file.flush()  # 确保立即写入文件
        average_metrics = {key: value / num_folds for key, value in all_metrics.items()}
        average_per_metrics = {
            key: value / num_folds
            for key, value in all_per_metrics.items()
        }
        output_dir = args.results_dir + '/' + args.task + '/' + args.pretrained_model + '/'
        results_file = open(output_dir + args.dataset + '.txt', 'a')
        print(f"Average Metrics: {average_metrics}", file=results_file)
        print("\nPer-Class Average Metrics:", file=results_file)
        for idx, label in enumerate(args.initial_label):
            f1 = average_per_metrics['f1'][idx]
            precision = average_per_metrics['precision'][idx]
            recall = average_per_metrics['recall'][idx]
            support = average_per_metrics['num'][idx]
            print(
                f"Label '{label}': F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Support={int(support)}",
                file=results_file)
