import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

INPUT_MODEL   = "model/gemma300-vi-trimmed-138"
PRUNED_MODEL  = "model/gemma300-vi-trimmed-pruned"
FP16_MODEL    = "model/gemma300-vi-trimmed-fp16"


def param_summary(model, label):
    total  = sum(p.numel() for p in model.parameters())
    embed  = model.get_input_embeddings().weight.numel()
    layers = len(model.layers)
    dtype  = next(model.parameters()).dtype
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\n[{label}]")
    print(f"  dtype              : {dtype}")
    print(f"  layers             : {layers}")
    print(f"  intermediate_size  : {model.config.intermediate_size}")
    print(f"  total params       : {total/1e6:.1f}M")
    print(f"  embed params       : {embed/1e6:.1f}M")
    print(f"  other params       : {(total-embed)/1e6:.1f}M")
    print(f"  est. size          : {size_mb:.1f} MB")


def load_model(path):
    cfg   = AutoConfig.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, config=cfg, local_files_only=True)
    tok   = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tok


# depth pruning
def prune_layers(model, keep_every_n=2):
    original_layers = list(model.layers)
    kept_indices    = list(range(0, len(original_layers), keep_every_n))
    kept_layers     = [original_layers[i] for i in kept_indices]

    model.layers = torch.nn.ModuleList(kept_layers)
    model.config.num_hidden_layers = len(kept_layers)

    if hasattr(model.config, 'layer_types') and model.config.layer_types:
        model.config.layer_types = [model.config.layer_types[i] for i in kept_indices]

    print(f"\nLayer pruning: {len(original_layers)} -> {len(kept_layers)} layers")
    return model


# width pruning
def reduce_ffn_width(model, keep_ratio=0.75):
    """Shrink FFN intermediate_size by selecting neurons with highest L2 norm.

    For each MLP block:
      - gate_proj / up_proj : (intermediate_size, hidden_size) → keep top-K rows by L2 norm
      - down_proj           : (hidden_size, intermediate_size) → keep corresponding columns

    Keep_ratio=0.75 → 25% reduction (1152 → 864).
    Keep_ratio=0.50 → 50% reduction (1152 → 576).
    """
    original_size = model.config.intermediate_size
    new_size      = int(original_size * keep_ratio)
    # align to multiple of 64 for GPU efficiency
    new_size      = (new_size // 64) * 64
    print(f"\nFFN width reduction: {original_size} -> {new_size} (keep_ratio={keep_ratio})")

    for layer_idx, layer in enumerate(model.layers):
        mlp = layer.mlp

        gate_w = mlp.gate_proj.weight.data  # (intermediate_size, hidden_size)
        up_w   = mlp.up_proj.weight.data
        down_w = mlp.down_proj.weight.data  # (hidden_size, intermediate_size)

        # select top-K neurons by L2 norm of gate_proj rows
        norms        = gate_w.norm(dim=1)          # (intermediate_size,)
        top_indices  = norms.topk(new_size).indices # top K indices
        top_indices  = top_indices.sort().values    # keep original order

        # create new smaller linear layers
        new_gate = torch.nn.Linear(gate_w.shape[1], new_size, bias=mlp.gate_proj.bias is not None)
        new_up   = torch.nn.Linear(up_w.shape[1],   new_size, bias=mlp.up_proj.bias is not None)
        new_down = torch.nn.Linear(new_size, down_w.shape[0], bias=mlp.down_proj.bias is not None)

        with torch.no_grad():
            new_gate.weight.copy_(gate_w[top_indices])
            new_up.weight.copy_(up_w[top_indices])
            new_down.weight.copy_(down_w[:, top_indices])
            if mlp.gate_proj.bias is not None:
                new_gate.bias.copy_(mlp.gate_proj.bias[top_indices])
                new_up.bias.copy_(mlp.up_proj.bias[top_indices])
            if mlp.down_proj.bias is not None:
                new_down.bias.copy_(mlp.down_proj.bias)

        mlp.gate_proj = new_gate
        mlp.up_proj   = new_up
        mlp.down_proj = new_down

    model.config.intermediate_size = new_size
    print(f"FFN width reduction done across {len(model.layers)} layers.")
    return model


# ── Step 3: FP16 dtype cast ───────────────────────────────────────────────────

def cast_fp16(model):
    model = model.half()
    model.config.torch_dtype = "float16"
    print("\nCast to fp16.")
    return model

def save_model(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print(f"Saved to: {path}")

if __name__ == "__main__":
    print("Loading trimmed model...")
    model, tok = load_model(INPUT_MODEL)
    model.eval()
    param_summary(model, "trimmed (input)")

    # Step 1 — layer pruning
    model = prune_layers(model, keep_every_n=2)
    param_summary(model, "after layer pruning")

    # Step 2 — FFN width reduction (25%)
    model = reduce_ffn_width(model, keep_ratio=0.75)
    param_summary(model, "after FFN width reduction")
    save_model(model, tok, PRUNED_MODEL)

    # Step 3 — FP16 cast
    model = cast_fp16(model)
    param_summary(model, "after fp16 cast")
    save_model(model, tok, FP16_MODEL)
