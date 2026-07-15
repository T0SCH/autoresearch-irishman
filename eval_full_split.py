"""
Full-val-split top-1/top-5 evaluation for a saved checkpoint.

Assignment-grade metric: top-1/top-5 over ALL next-char positions in the
validation split, each tune evaluated exactly once. This is NOT quick_eval's
single-batch sample -- it's the "over all sequences in the test set" number
AUFGABENSTELLUNG.md asks for.

Model class is copied from train.py (which has no __main__ guard and would
start training on import). Keep in sync with train.py's CharRNN if it changes.

Usage: uv run eval_full_split.py checkpoint.pt
"""
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, load_tunes


@dataclass
class RNNConfig:
    vocab_size: int = 100
    embed_size: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    pad_token_id: int = 0


class CharRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_size)
        assert config.embed_size == config.hidden_size, "weight tying requires embed_size == hidden_size"
        assert config.num_layers == 3, "residual connection below is hardcoded for exactly 3 layers"
        self.rnn1 = nn.LSTM(input_size=config.embed_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.rnn3 = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                             num_layers=1, batch_first=True)
        self.drop = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.head.weight = self.embed.weight
        self.h0 = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(config.num_layers, 1, config.hidden_size))
        self.last_hidden = None

    def forward(self, idx, targets=None, reduction='mean', init_state=None):
        x = self.embed(idx)
        if init_state is None:
            h0 = self.h0.expand(-1, x.size(0), -1).contiguous()
            c0 = self.c0.expand(-1, x.size(0), -1).contiguous()
        else:
            h0, c0 = init_state
        x1, (h1, c1) = self.rnn1(x, (h0[0:1], c0[0:1]))
        x2, (h2, c2) = self.rnn2(x1, (h0[1:2], c0[1:2]))
        y2 = x1 + x2
        x3, (h3, c3) = self.rnn3(y2, (h0[2:3], c0[2:3]))
        x = y2 + x3
        self.last_hidden = (torch.cat([h1, h2, h3], dim=0).detach(), torch.cat([c1, c2, c3], dim=0).detach())
        x = self.drop(x)
        logits = self.head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=self.config.pad_token_id, reduction=reduction)
            return loss
        return logits


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run eval_full_split.py checkpoint.pt", file=sys.stderr)
        sys.exit(1)
    ckpt_path = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    tokenizer = Tokenizer.from_directory()
    bos = tokenizer.get_bos_token_id()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = RNNConfig(**ckpt["config"])
    model = CharRNN(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {ckpt_path}")
    print(f"Config: {config}")

    val_tunes = load_tunes("val")
    print(f"Val tunes: {len(val_tunes)}")

    top1_correct = 0
    top5_correct = 0
    total_positions = 0

    with torch.no_grad():
        for tune in val_tunes:
            ids = tokenizer.encode(tune, prepend=bos)
            if len(ids) > MAX_SEQ_LEN:
                ids = ids[:MAX_SEQ_LEN]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(x)                       # [1, T, vocab]
            pred_logits = logits[0, :-1]            # [T-1, vocab] -- predicts next char
            targets = x[0, 1:]                      # [T-1]
            top5 = pred_logits.topk(5, dim=-1).indices
            top1 = top5[:, 0]
            top1_correct += (top1 == targets).sum().item()
            top5_correct += ((top5 == targets.unsqueeze(-1)).any(dim=-1)).sum().item()
            total_positions += targets.size(0)

    top1_acc = top1_correct / total_positions
    top5_acc = top5_correct / total_positions
    print(f"Total next-char positions: {total_positions}")
    print(f"top1_acc: {top1_acc:.6f}")
    print(f"top5_acc: {top5_acc:.6f}")


if __name__ == "__main__":
    main()