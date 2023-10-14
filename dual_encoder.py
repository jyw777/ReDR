import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel




class RankRaceModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type

        if self.encoder_type == 'bert':
            self.encoder = BertModel.from_pretrained(args.plm_path, cache_dir='')
        if self.encoder_type == 'bertu':
            self.encoder = BertModel.from_pretrained(args.plm_path, cache_dir='')

        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', label_smoothing=args.label_smoothing)
        self.softmax = nn.Softmax(dim=1)
        self.args = args
        self.cnt = 0


    def pooler(self, last_hidden_state, attention_mask):
        if self.pooler_type == 'mean':
            pooler_output = torch.sum(last_hidden_state * attention_mask.unsqueeze(2), dim=1)
            pooler_output /= torch.sum(attention_mask, dim=1, keepdim=True)
        if self.pooler_type == 'max':
            pooler_output = torch.max(last_hidden_state * attention_mask.unsqueeze(2), dim=1).values
        return pooler_output

    def sentence_encoding(self, encoder, input_ids, attention_mask, token_type_ids):
        encoder_outputs = encoder(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  # token_type_ids=token_type_ids,
                                  return_dict=True)
        last_hidden_state = encoder_outputs['last_hidden_state']
        pooler_output = self.pooler(last_hidden_state, attention_mask)
        if self.args.sim_type == "cos":
            pooler_output = F.normalize(pooler_output, dim=-1)
        return pooler_output

    def forward(self, src_input_ids, src_attention_mask, src_token_type_ids, tgt_input_ids, tgt_attention_mask,
                tgt_token_type_ids, src_ids=None, inference=False):
        # [bs, d_model]
        src_pooler_output = self.sentence_encoding(self.encoder, src_input_ids, src_attention_mask, src_token_type_ids)
        tgt_pooler_output = self.sentence_encoding(self.encoder, tgt_input_ids, tgt_attention_mask, tgt_token_type_ids)


        if not inference:
            # [bs, bs]
            predict_logits = src_pooler_output.mm(tgt_pooler_output.t())
            predict_logits /= self.temperature
            # loss
            batch_size = src_pooler_output.shape[0]
            logit_mask = (src_ids.unsqueeze(1).repeat(1, batch_size) == src_ids.unsqueeze(0).repeat(batch_size,1)).float() - torch.eye(
                batch_size).to(src_ids.device)
            predict_logits -= logit_mask * 100000000
            label = torch.arange(0, predict_logits.shape[0]).to(src_input_ids.device)
            predict_loss = self.args.ce_w * self.ce_loss(predict_logits, label)
            soft_label = self.args.sl_s * src_pooler_output.mm(src_pooler_output.t()) + self.args.sl_t * tgt_pooler_output.mm(tgt_pooler_output.t())

            if self.args.kd:
                soft_label_kd = soft_label / self.args.temperature2
                soft_label_kd = self.softmax(soft_label_kd)
                soft_label_kd = soft_label_kd.detach()
                predict_loss += self.args.kd_w * self.ce_loss(predict_logits, soft_label_kd)

            if self.args.hn:
                eye_label = torch.eye(batch_size, batch_size).to(src_ids.device)
                predict_logits_ss = src_pooler_output.mm(src_pooler_output.t())
                predict_logits_tt = tgt_pooler_output.mm(tgt_pooler_output.t())
                logit_mask = (src_ids.unsqueeze(1).repeat(1, batch_size) == src_ids.unsqueeze(0).repeat(batch_size,
                                                                                                        1)).float() - torch.eye(
                    batch_size).to(src_ids.device)
                predict_logits_ss -= logit_mask * 100000000
                predict_logits_tt -= logit_mask * 100000000
                predict_logits_hn = self.args.hn_ss * predict_logits_ss + self.args.hn_tt * predict_logits_tt + \
                    self.args.hn_st * predict_logits
                binary_predict_logits_hn = torch.zeros(batch_size, 2).to(src_input_ids.device)
                label_hn = torch.zeros(batch_size, 2).to(src_input_ids.device)
                for i in range(0, batch_size):
                    idxs_hn = [i, int(torch.topk(predict_logits_hn[i] - eye_label[i] * 100000000, 1).indices[0])]
                    binary_predict_logits_hn[i] = predict_logits[i][idxs_hn]
                    label_hn[i] = soft_label[i][idxs_hn]
                label_hn = label_hn / self.args.temperature3
                label_hn = self.softmax(label_hn)
                label_hn = label_hn.detach()
                predict_loss += self.args.hn_w * self.ce_loss(binary_predict_logits_hn, label_hn)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            return predict_loss, acc
        else:
            return src_pooler_output, tgt_pooler_output













