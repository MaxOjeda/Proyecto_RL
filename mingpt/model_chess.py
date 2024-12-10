import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout) #attn_pdrop
        self.resid_drop = nn.Dropout(config.dropout) # resid_pdrop
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """GPT for Chess"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Cada estado es 8x8x12 = 768
        self.state_emb = nn.Linear(768, config.n_embd)
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())

        # Embedding para la goals
        self.goal_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # Global embedding para timesteps
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))

        self.drop = nn.Dropout(config.dropout) # embd_pdrop

        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.config.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def configure_optimizers(self, train_config):
        """
        Separa los parámetros en los que llevan weight decay y los que no.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if p.requires_grad:
                    fpn = '%s.%s' % (mn, pn) if mn else pn
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights de Linear decaen
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                    else:
                        no_decay.add(fpn)

        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # Validar intersección
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "Algunos parámetros caen en ambas categorías: %s" % str(inter_params)
        assert len(param_dict.keys() - union_params) == 0, "Algunos parámetros no fueron clasificados"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        print(f"Weight decay: {train_config.weight_decay}")
        return optimizer


    def forward(self, states, actions, targets=None, goals=None, timesteps=None):
        B, T, _, _, _ = states.size()
        state_embeddings = self.state_emb(states.reshape(B,T,-1))
        action_embeddings = self.action_embeddings(actions.squeeze(-1))
        goal_embeddings = self.goal_emb(goals.float())

        # (goal, state, action)
        # Si targets no es None, tenemos acciones conocidas, sino, estamos en inferencia
        if targets is not None:
            seq_len = 3*T
        else:
            seq_len = 2*T

        token_embeddings = torch.zeros((B, seq_len, self.config.n_embd), dtype=torch.float32, device=states.device)

        # goal en 0,3,6,...
        token_embeddings[:, 0::3, :] = goal_embeddings
        # state en 1,4,7,...
        token_embeddings[:, 1::3, :] = state_embeddings
        # action en 2,5,8,...
        if targets is not None:
            token_embeddings[:, 2::3, :] = action_embeddings

        # Expandir timesteps
        # timestep hay 3 tokens (goal, state, action)
        if targets is not None:
            expanded_timesteps = torch.zeros(B, 3*T, dtype=torch.int64, device=timesteps.device)
            for i in range(T):
                expanded_timesteps[:, i*3] = timesteps[:, i]
                expanded_timesteps[:, i*3+1] = timesteps[:, i]
                expanded_timesteps[:, i*3+2] = timesteps[:, i]
        else:
            # Sin acciones (inferencia)
            expanded_timesteps = torch.zeros(B, 2*T, dtype=torch.int64, device=timesteps.device)
            for i in range(T):
                expanded_timesteps[:, i*2] = timesteps[:, i]
                expanded_timesteps[:, i*2+1] = timesteps[:, i]

        all_global_pos_emb = self.global_pos_emb.repeat(B,1,1)
        timesteps_expanded = expanded_timesteps.unsqueeze(-1).repeat(1,1,self.config.n_embd)
        global_position_embeddings = torch.gather(all_global_pos_emb, 1, timesteps_expanded)

        x = self.drop(token_embeddings + global_position_embeddings[:,:seq_len,:] + self.pos_emb[:, :seq_len, :])
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # Sólo las posiciones de acción (2::3) deben predecir la acción
        if targets is not None:
            logits = logits[:,2::3,:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            loss = None

        return logits, loss
