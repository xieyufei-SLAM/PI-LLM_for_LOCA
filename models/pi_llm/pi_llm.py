
import torch
import torch.nn as nn

from models.pi_llm.sace import SaCE
from models.pi_llm.msw_cnn import MSWCNN
from models.pi_llm.s2t import S2T


class TinyBackbone(nn.Module):

    def __init__(self, vocab_size=32000, hidden_size=512, nlayers=4, nhead=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, nhead, dim_feedforward=hidden_size * 2,
            dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward(self, inputs_embeds, attention_mask=None):
        return self.encoder(inputs_embeds)


def load_backbone(llm_path=None, lora_r=16, lora_alpha=32, lora_dropout=0.1,
                  dtype=torch.bfloat16):

    if llm_path is None:
        return TinyBackbone(), None, 512, False # 验证用

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(llm_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        llm_path, torch_dtype=dtype, output_hidden_states=True)
    base = model.model               # LlamaModel (无 lm_head)
    for p in base.parameters():      # 冻结主干
        p.requires_grad_(False)
    cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="FEATURE_EXTRACTION")
    base = get_peft_model(base, cfg)
    return base, tok, model.config.hidden_size, True


class PILLM(nn.Module):

    def __init__(self, llm_path=None, d_model=256, n_var=10,
                 lora_r=16, lora_alpha=32, max_prompt_len=96, dtype=torch.float32):
        super().__init__()
        self.backbone, self.tokenizer, self.H, self.is_llm = load_backbone(
            llm_path, lora_r, lora_alpha, dtype=dtype)
        self.d_model = d_model
        self.max_prompt_len = max_prompt_len

        self.sace = SaCE(d_model=d_model)
        self.msw = MSWCNN(in_ch=14, d_out=d_model)
        self.s2t = S2T(in_dim=4, d_model=d_model)

        self.prefix_proj = nn.Linear(self.H, d_model)
        # MSW / SaCE 特征 -> 主干 hidden 维度
        self.msw_to_h = nn.Linear(d_model, self.H)
        self.sace_to_h = nn.Linear(d_model, self.H)
        # 回归头：6 + 2 + 2 = 10
        self.head6 = nn.Linear(self.H, 6)
        self.head2a = nn.Linear(self.H, 2)
        self.head2b = nn.Linear(self.H, 2)

    def embed_prompt(self, prompts, device):
        """prompts(list[str]) -> token embeddings, attention_mask, prefix 向量。"""
        if self.is_llm:
            enc = self.tokenizer(
                prompts, return_tensors='pt', padding=True, truncation=True,
                max_length=self.max_prompt_len).to(device)
            emb = self.backbone.get_input_embeddings()(enc['input_ids'])
            return emb, enc['attention_mask']
        # tiny 回退：用哈希把 prompt 映射为伪 token(仅占位，prompt 段内相同)
        ids = torch.zeros(len(prompts), 8, dtype=torch.long, device=device)
        emb = self.backbone.embed_tokens(ids)
        mask = torch.ones(len(prompts), 8, device=device)
        return emb, mask

    def _run_backbone(self, inputs_embeds, attention_mask):
        if self.is_llm:
            bb_dtype = self.backbone.get_input_embeddings().weight.dtype
            inputs_embeds = inputs_embeds.to(bb_dtype)
        out = self.backbone(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask)
        if self.is_llm:
            return out.last_hidden_state
        return out

    def forward(self, history, suffix, prompts):

        device = history.device
        text_emb, text_mask = self.embed_prompt(prompts, device)
        text_emb = text_emb.to(history.dtype)

        prefix = self.prefix_proj(text_emb.mean(dim=1))         # [B,d]

        # SaCE / MSW / S2T
        loss_sace, z_suffix = self.sace(suffix, prefix)
        msw_feat = self.msw(history)                            # [B,d]
        loss_s2t, _ = self.s2t(suffix)

        msw_tok = self.msw_to_h(msw_feat).unsqueeze(1)          # [B,1,H]
        sace_tok = self.sace_to_h(z_suffix).unsqueeze(1)        # [B,1,H]
        seq = torch.cat([text_emb, msw_tok.to(text_emb.dtype),
                         sace_tok.to(text_emb.dtype)], dim=1)
        ext_mask = torch.cat(
            [text_mask, torch.ones(text_mask.size(0), 2, device=device)], dim=1)

        hidden = self._run_backbone(seq, ext_mask)              # [B,L+2,H]
        last = hidden[:, -1, :].float()                         # 末位 soft token
        pred = torch.cat([self.head6(last), self.head2a(last),
                          self.head2b(last)], dim=1)            # [B,10]
        return {'pred': pred, 'loss_sace': loss_sace,
                'loss_s2t': loss_s2t, 'prefix': prefix}
