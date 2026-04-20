import json
import os

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import read_recipe_ner_data, convert_label_to_id_ner, GetDataLoader_NER, build_re_dict, \
    convert_label_to_id_re, GetDataLoader_RE, extract_relations_from_json, add_params

from model import BertModelNER, BertModelRE
from utils_cn import read_recipe_ner_data_cn


def train_ner(args):
    if "Chinese-test" in args.dataset:
        all_sentence_train, all_label_train = read_recipe_ner_data_cn(args.filepath_train)
    else:
        all_files = [
            os.path.join(args.filepath_train, f) for f in os.listdir(args.filepath_train)
        ]
        all_sentence_train = []
        all_label_train = []
        for filepath in all_files:
            sentence_train, label_train = read_recipe_ner_data(filepath)
            all_sentence_train.extend(sentence_train)
            all_label_train.extend(label_train)
    label_train_ids = convert_label_to_id_ner(all_label_train, args)

    dataloader_train = GetDataLoader_NER(args=args,
                                         sentences=all_sentence_train,
                                         labels_ids=label_train_ids,
                                         batch_size=args.batch_size_ner,
                                         ignore_o_sentence=True)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_model_ner = BertModelNER(args).to(args.device)
    optimizer_grouped_parameters = []
    if args.crf:
        add_params(args, module=bert_model_ner.encoder, prefix="encoder", lr=args.learning_rate, no_decay=no_decay,
                   optimizer_grouped_parameters=optimizer_grouped_parameters)
        add_params(args, bert_model_ner.crf, "crf", args.crf_learning_rate, no_decay, optimizer_grouped_parameters)
        add_params(args, bert_model_ner.classifier, "classifier", args.crf_learning_rate, no_decay,
                   optimizer_grouped_parameters)

    if args.lstm:
        add_params(args, bert_model_ner.bilstm, "bilstm", args.lstm_learning_rate, no_decay,
                   optimizer_grouped_parameters)

    if args.egp:
        add_params(args, bert_model_ner.egp, "egp", args.egp_learning_rate, no_decay, optimizer_grouped_parameters)
    seen_params = set()
    for group in optimizer_grouped_parameters:
        for p in group["params"]:
            param_id = id(p)
            if param_id in seen_params:
                raise ValueError(f"参数重复 {param_id}，请检查模块命名")
            seen_params.add(param_id)
    optimizer_ner = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer_ner = torch.optim.Adam(bert_model_ner.parameters(), lr=args.train_LR_ner)
    num_train_epochs = args.train_epoches_ner
    num_update_steps_per_epoch = len(dataloader_train)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer_ner,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # train
    for epoch in range(num_train_epochs):  # 训练多个 epoch
        bert_model_ner.train()
        batch_iterator_ner = tqdm(dataloader_train, desc="batch_iterator_ner", disable=False,
                                  dynamic_ncols=True, ncols=100)
        loss = 0
        for step, batch in enumerate(batch_iterator_ner):
            optimizer_ner.zero_grad()
            # print(batch_stage1)
            loss_ner, _1, _2 = \
                bert_model_ner(
                    input_ids=batch[0].to(args.device),
                    token_type_ids=batch[1].to(args.device),
                    attention_mask=batch[2].to(args.device),
                    label_ids=batch[3].to(args.device),
                )
            # compute gradient and do step
            loss_ner.backward()
            optimizer_ner.step()
            lr_scheduler.step()
            loss = loss_ner + loss
        print(loss / num_update_steps_per_epoch)
    # save model
    ckpt_dir = './checkpoint/' \
               + args.task \
               + '/' \
               + args.pretrained_model \
               + '-' + str(args.seed) + '/' + args.IO_mode + '-' + args.dataset + '/' + args.ner_model_structure + '/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir, 'model-ner' + '-' + str(args.k_folds) + '.ckpt')
    torch.save({'model_state_dict': bert_model_ner.state_dict()},
               os.path.join(ckpt_dir, 'model-ner' + '-' + str(args.k_folds) + '.ckpt'))
    pass




def train_re(args):
    all_files = [
        os.path.join(args.filepath_train, f) for f in os.listdir(args.filepath_train)
    ]
    all_dict = []
    for file in all_files:
        re_dict = extract_relations_from_json(file)
        all_dict.append(re_dict)
    flattened_dict = [item for sublist in all_dict for item in sublist]
    with open(r'D:\Project\Python\recipe-flow\data\Chinese\Recipe\Aug_relation_with_positions.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    flattened_dict.extend(json_data)

    all_label_ids = convert_label_to_id_re(flattened_dict, args)

    # # all_entity_pos = get_entity_pos(flattened_dict, args)
    dataloader_train = GetDataLoader_RE(args=args,
                                        dicts=flattened_dict,
                                        labels_ids=all_label_ids,
                                        batch_size=args.batch_size_re,
                                        )
    num_train_epochs = args.train_epoches_re
    num_update_steps_per_epoch = len(dataloader_train)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    bert_model_re = BertModelRE(args).to(args.device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert_model_re.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay_re,
        },
        {
            "params": [p for n, p in bert_model_re.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_re = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate_re,
        eps=args.adam_epsilon_re,
    )
    scheduler_re = get_linear_schedule_with_warmup(
        optimizer_re,
        num_warmup_steps=args.warmup_steps_re,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_train_epochs):  # 训练多个 epoch
        bert_model_re.train()
        batch_iterator_re = tqdm(dataloader_train, desc="batch_iterator_re", disable=False,
                                 dynamic_ncols=True, ncols=100)
        loss = 0
        for step, batch in enumerate(batch_iterator_re):
            optimizer_re.zero_grad()
            # print(batch_stage1)

            loss_re, _1, _2 = \
                bert_model_re(
                    input_ids=batch[0].to(args.device),
                    token_type_ids=batch[1].to(args.device),
                    attention_mask=batch[2].to(args.device),
                    label_ids=batch[3].to(args.device),
                    e1_mask=batch[4].to(args.device),
                    e2_mask=batch[5].to(args.device),
                    e1_pos=batch[6].to(args.device),
                    e2_pos=batch[7].to(args.device),
                    re_len=batch[8].to(args.device),
                )
            # compute gradient and do step
            loss_re.backward()
            torch.nn.utils.clip_grad_norm_(
                bert_model_re.parameters(),
                max_norm=1.0  # 建议初始值
            )
            optimizer_re.step()
            scheduler_re.step()
            loss = loss_re + loss
            # visualize_problem_sample(batch, args)
            if torch.isnan(loss_re).any():
                print(f"第{step}步检测到NaN损失")
                visualize_problem_sample(batch, args)  # 可视化问题样本
                continue

        print(loss / num_update_steps_per_epoch)
    ckpt_dir = './checkpoint/' \
               + args.task \
               + '/' \
               + args.pretrained_model \
               + '-' + str(args.seed) + '/' + args.dataset + '/' + args.re_model_structure + '/'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir, 'model-re' + '-' + str(args.k_folds) + '.ckpt')
    torch.save({'model_state_dict': bert_model_re.state_dict()},
               os.path.join(ckpt_dir, 'model-re' + '-' + str(args.k_folds) + '.ckpt'))
    pass


def visualize_problem_sample(batch, args):
    """可视化问题样本"""
    sample_idx = 0  # 查看批次中第一个样本
    input_ids = batch[0][sample_idx].cpu().tolist()
    tokens = args.tokenizer.convert_ids_to_tokens(input_ids)
    label_id = batch[3][sample_idx].item()
    print("\n问题样本分析：")
    print(f"标签ID: {label_id}")
    print("原始文本:", args.tokenizer.decode(input_ids, skip_special_tokens=False))
    print("Token列表:", tokens)

    # 可视化实体位置
    e1_mask = batch[4][sample_idx].cpu().tolist()
    e2_mask = batch[5][sample_idx].cpu().tolist()

    print("\n实体掩码分布：")
    print("E1掩码:", ''.join(['X' if m else '_' for m in e1_mask]))
    print("E2掩码:", ''.join(['X' if m else '_' for m in e2_mask]))
    # 保存问题样本
    problem_sample = {
        'input_ids': input_ids,
        'attention_mask': batch[2][sample_idx].cpu().tolist(),
        'e1_mask': e1_mask,
        'e2_mask': e2_mask,
        'label': batch[3][sample_idx].item()
    }
    torch.save(problem_sample, "problem_sample.pt")
