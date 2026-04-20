from abc import ABC

import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from model_utils.crf import CRF
from model_utils.egp import EffiGlobalPointer
from model_utils.lexicon import LXBertModel
from utils import convert_bio_to_entity_matrix


class BertModelNER(nn.Module):
    def __init__(self, args):
        super(BertModelNER, self).__init__()
        self.args = args

        # BERT 编码器
        self.encoder = BertModel.from_pretrained(args.pretrained_model)

        # 可选 BiLSTM 层
        if args.lstm:
            self.bilstm = nn.LSTM(
                input_size=args.hidden_size,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            classifier_input_size = 256 * 2
        else:
            classifier_input_size = args.hidden_size

        # 序列标注分类器
        self.classifier = nn.Linear(classifier_input_size, len(args.id2label_train) + 1)

        # 可选实体抽取模块 EGP（Efficient Global Pointer）
        if args.egp:
            self.egp = EffiGlobalPointer(
                hidden_size=args.hidden_size,
                ent_type_size=len(args.initial_label),
                inner_dim=64,
                RoPE=True
            )
            # 多任务损失融合的权重（自动可训练）
            self.loss_weights = nn.Parameter(torch.ones(2))

        # CRF 层用于序列标注
        self.crf = CRF(num_tags=len(args.id2label_train) + 1, batch_first=True)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, label_ids=None):
        # 将 BIO 标签转换为实体对矩阵（用于 EGP）
        label_matrix = convert_bio_to_entity_matrix(
            label_ids,
            num_entity_types=len(self.args.initial_label),
            attention_mask=attention_mask
        )

        # 编码器输出
        bert_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )[0]

        # 可选 LSTM
        if self.args.lstm:
            lstm_output, _ = self.bilstm(bert_output)
            sequence_output = self.dropout(lstm_output)
        else:
            sequence_output = self.dropout(bert_output)

        # --- 任务输出 ---
        total_loss = 0
        losses = {}

        seq_logits = self.classifier(sequence_output)

        # --- CRF 损失 ---
        if label_ids is not None:
            crf_loss = -self.crf(seq_logits, label_ids, mask=attention_mask.bool())
            losses['crf'] = crf_loss.item()

        # --- 如果使用 EGP，多任务加权 ---
        if self.args.egp:
            entity_logits = self.egp(bert_output, attention_mask)
            if label_matrix is not None:
                egp_loss = multilabel_categorical_crossentropy(entity_logits, label_matrix)
                losses['egp'] = egp_loss.item()

                weights = torch.softmax(self.loss_weights, dim=0)
                total_loss = egp_loss * weights[0] + crf_loss * weights[1]
            else:
                total_loss = crf_loss  # fallback

        else:
            # 如果不使用 egp，total_loss 就等于 crf_loss
            total_loss = crf_loss

        # 展平特征用于评估
        logits_flat = seq_logits.view(-1, seq_logits.size(-1))
        labels_flat = label_ids.view(-1)

        valid_indices = torch.where(labels_flat >= 0)[0]
        filtered_logits = logits_flat[valid_indices]
        filtered_labels = labels_flat[valid_indices]

        return total_loss, filtered_logits, filtered_labels


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    EGP专用损失函数 (支持四维张量)
    :param y_pred: logits (batch_size, num_entity_types, seq_len, seq_len)
    :param y_true: label_matrix (batch_size, num_entity_types, seq_len, seq_len)
    :return: loss scalar
    """
    # 分离正负样本
    y_pred = (1 - 2 * y_true) * y_pred  # 正样本保持原值，负样本取反

    # 负样本损失计算（掩码正样本）
    y_pred_neg = y_pred - y_true * 1e12  # 将正样本的logits置为极小值
    y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim=-1)

    # 正样本损失计算（掩码负样本）
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # 将负样本的logits置为极小值
    y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim=-1)

    # 计算logsumexp
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    # 聚合损失
    loss = (neg_loss + pos_loss).mean()
    return loss


class BertLexiconModelNER(BertPreTrainedModel):
    def __init__(self, config, pretrained_embeddings, num_labels):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = LXBertModel(config)
        self.dropout = nn.Dropout(config.HP_dropout)
        self.num_labels = num_labels
        self.hidden2tag = nn.Linear(config.hidden_size, num_labels + 2)
        self.crf = CRF(num_labels, torch.cuda.is_available())

        self.init_weights()

        ## init the embedding
        self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            matched_word_ids=None,
            matched_word_mask=None,
            boundary_ids=None,
            labels=None,
            flag="Train"
    ):
        matched_word_embeddings = self.word_embeddings(matched_word_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=matched_word_mask,
            boundary_ids=boundary_ids
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.hidden2tag(sequence_output)

        if flag == 'Train':
            assert labels is not None
            loss = self.crf.neg_log_likelihood_loss(logits, attention_mask, labels)
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (loss, preds)
        elif flag == 'Predict':
            _, preds = self.crf._viterbi_decode(logits, attention_mask)
            return (preds,)


# -------------------------------------RE------------------------------------
class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class SentencePositionEncoder(nn.Module):
    def __init__(self, hidden_size, max_len=512, k=32):
        """
        :param hidden_size: BERT的隐藏层维度 (e.g., 768)
        :param max_len: 支持的最大句子/食谱长度
        :param k: 相对位置截断范围 [-K, K] 的 K 值
        """
        super(SentencePositionEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.k = k

        # --- 1. 绝对位置编码 (Fixed Sinusoidal) ---
        # 预计算 PE 矩阵，registered buffer 不会被视为模型参数更新，但会随模型保存
        self.register_buffer('pe', self._get_sinusoid_encoding_table(max_len, hidden_size))

        # --- 2. 相对/可学习位置编码 (Learnable) ---
        # 对应文中的 W_rel。虽然通常 Relative 是两点间距离，
        # 但在此上下文中，它作为单句特征，我们将其实现为可学习的位置Embedding
        # 这里的 clip 逻辑通常用于索引查找
        self.rel_embedding = nn.Embedding(2 * k + 1, hidden_size)
        self.attention_proj = nn.Linear(hidden_size, 1)
        # --- 3. 可学习权重 Lambda ---
        # 对应公式 (1) 中的 λ，初始化为 0.5 或其他值
        self.lambda_weight = nn.Parameter(torch.tensor(0.5))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """生成正余弦位置编码表"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table)

    def forward(self, pos_ids, re_len):
        """
                :param pos_ids: [batch] 当前实体的句子位置 (e.g., [5])
                :param re_len: [batch] 当前食谱的总句数 (e.g., [20])
                :return: [batch, hidden_size]
                """
        batch_size = pos_ids.size(0)

        # --- A. 绝对位置特征 (P_abs) ---
        safe_pos_ids = pos_ids.clamp(0, self.max_len - 1)
        p_abs = self.pe[safe_pos_ids]  # [batch, dim]

        # --- B. 相对位置特征 (P_rel) ---

        # 1. 生成 [0, 1, ..., max_len-1] 的序列 [1, max_len]
        seq_range = torch.arange(self.max_len, device=pos_ids.device).unsqueeze(0)

        # 2. 生成相对距离矩阵: j - i
        # [1, max_len] - [batch, 1] -> [batch, max_len]
        # 结果就是你想要的: [-5, -4, ..., 14, ...]
        relative_dist = seq_range - pos_ids.unsqueeze(1)

        # 3. 截断 (Clip) 到 [-k, k] 范围，并平移索引以适应 Embedding
        # Embedding索引不能为负，所以我们将范围映射到 [0, 2k]
        # -k -> 0, 0 -> k, k -> 2k
        dist_clipped = relative_dist.clamp(-self.k, self.k) + self.k

        # 4. 获取相对位置嵌入
        # [batch, max_len] -> [batch, max_len, dim]
        rel_embeds = self.rel_embedding(dist_clipped)

        # --- C. 聚合 (利用 re_len 进行 Mask) ---

        # 1. 生成 Mask: 标记哪些位置是有效的 (index < re_len)
        # [1, max_len] < [batch, 1] -> [batch, max_len]
        mask = seq_range < re_len.unsqueeze(1)

        # 2. 计算注意力权重 (Formula 2 提到的 attention)
        # 这里简化为一个自适应的聚合，计算每个相对位置的重要性
        attn_scores = self.attention_proj(rel_embeds).squeeze(-1)  # [batch, max_len]

        # 3. Mask fill (将无效位置的权重设为极小值)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)

        # 4. Softmax 归一化
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [batch, max_len, 1]

        # 5. 加权求和得到最终的 P_rel
        # sum([batch, max_len, dim] * [batch, max_len, 1]) -> [batch, dim]
        p_rel_aggregated = torch.sum(rel_embeds * attn_weights, dim=1)

        # --- D. 融合 ---
        pos_feature = self.lambda_weight * p_abs + (1 - self.lambda_weight) * p_rel_aggregated

        return pos_feature


class BertModelRE(nn.Module):
    def __init__(self, args):
        super(BertModelRE, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(args.pretrained_model)
        self.sent_pos_encoder = SentencePositionEncoder(args.hidden_size, max_len=100)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.cls_fc_layer = FCLayer(args.hidden_size, args.hidden_size, args.dropout_rate_re)
        self.entity_fc_layer = FCLayer(args.hidden_size, args.hidden_size, args.dropout_rate_re)

        self.classifier = FCLayer(
            args.hidden_size * 7,
            len(self.args.id2label_train),
            args.dropout_rate_re,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        # e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        # length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
        #
        # # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        # sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        # avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        # return avg_vector
        e_mask_unsqueeze = e_mask.unsqueeze(1).float()
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1).float()  # [batch_size, 1]

        # 强制有效实体长度
        length_tensor = length_tensor.clamp(min=1e-7)  # 防止除零

        # 安全矩阵乘法
        sum_vector = torch.bmm(e_mask_unsqueeze, hidden_output).squeeze(1)
        avg_vector = sum_vector / length_tensor

        # 异常值过滤
        avg_vector = torch.nan_to_num(avg_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        return avg_vector

    def get_proxy_masks(self, input_ids, sep_token_id=102):
        """
        根据 [SEP] 动态生成 p1_mask 和 p2_mask
        input_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        p1_mask = torch.zeros_like(input_ids).float()
        p2_mask = torch.zeros_like(input_ids).float()

        for i in range(batch_size):
            # 找到当前 batch 中所有 [SEP] 的位置
            sep_indices = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]

            # 根据你的拼接逻辑：
            # 最后一个 [SEP] 之后是 Proxy 2 的一部分或结尾
            # 倒数第二个 [SEP] 和 最后一个 [SEP] 之间是 Proxy 2
            # 倒数第三个 [SEP] 和 倒数第二个 [SEP] 之间是 Proxy 1

            if len(sep_indices) >= 2:
                idx_last = sep_indices[-1]
                idx_prev = sep_indices[-2]

                # Proxy 2: 在倒数第二个 SEP 和最后一个 SEP 之间
                # 注意：根据你的拼接，proxy2 在最后一个 SEP 之后
                # 格式：... [SEP] proxy1 [SEP] proxy2
                p2_start = idx_last + 1
                # 找到 padding 前的有效边界
                p2_end = (input_ids[i] != 0).nonzero(as_tuple=True)[0][-1] + 1
                p2_mask[i, p2_start:p2_end] = 1

                # Proxy 1: 在倒数第二个 SEP 和最后一个 SEP 之间
                p1_start = idx_prev + 1
                p1_end = idx_last
                p1_mask[i, p1_start:p1_end] = 1

        return p1_mask, p2_mask

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, label_ids=None,
                e1_mask=None, e2_mask=None, e1_pos=None, e2_pos=None, re_len=None,p1_mask=None, p2_mask=None):
        if p1_mask is None or p2_mask is None:
            p1_mask, p2_mask = self.get_proxy_masks(input_ids)
        p1_mask, p2_mask = self.get_proxy_masks(input_ids)
        bert_output_raw = \
            self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = bert_output_raw[0]
        pooled_output = bert_output_raw[1]

        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_t = self.entity_average(sequence_output, p1_mask)  # 自动处理多token平均
        e2_t = self.entity_average(sequence_output, p2_mask)
        if e1_pos is not None and e2_pos is not None and re_len is not None:
            pos_feat_1 = self.sent_pos_encoder(e1_pos, re_len)
            pos_feat_2 = self.sent_pos_encoder(e2_pos, re_len)
            e1_p = self.layer_norm(e1_h + pos_feat_1)
            e2_p = self.layer_norm(e2_h + pos_feat_2)
        else:
            e1_p, e2_p = e1_h, e2_h

        concat_h = torch.cat([pooled_output, e1_h, e2_h, e1_p, e2_p, e1_t, e2_t], dim=-1)
        # (batch_size, num_label)
        logits = self.classifier(concat_h)
        assert not torch.isnan(sequence_output).any(), "BERT输出含NaN"
        assert not torch.isinf(e1_h).any(), "实体1特征含Inf"
        assert not torch.isnan(logits).any(), "Logits含NaN"
        loss = None
        if label_ids is not None:
            if len(self.args.id2label_train) == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, len(self.args.id2label_train)), label_ids.view(-1))

        return loss, logits, label_ids


