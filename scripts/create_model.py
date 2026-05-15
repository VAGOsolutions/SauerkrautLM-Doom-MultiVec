"""
DOOM MultiVec Model Creator
============================

Creates a ModernBERT model with Hash Embeddings configured for DOOM ASCII
game-state understanding. Saves model, config, tokenizer, and the custom
modeling file (for trust_remote_code loading) to disk.

Adapted from sauerkrautlm_reasonir/train/create_modernbert_hash.py.

Usage:
    python scripts/create_model.py
    python scripts/create_model.py --num_layers 5 --output models/doom-multivec-5L
"""

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn
from transformers import ModernBertConfig

# Ensure package is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from doom_multivec.model.modeling_doom_hash import HashEmbedding, ModernBertHashModel
from doom_multivec.model.tokenizer import create_ascii_tokenizer, save_tokenizer


def count_params(model):
    """Count parameters separately for embedding and transformer layers."""
    total = 0
    embedding = 0
    transformer = 0

    for name, p in model.named_parameters():
        total += p.numel()
        if "embedding" in name.lower() or "hash" in name.lower() or "tok_embed" in name.lower():
            embedding += p.numel()
        else:
            transformer += p.numel()

    return total, embedding, transformer


def main():
    parser = argparse.ArgumentParser(description="Create DOOM MultiVec model with Hash Embeddings")
    parser.add_argument("--vocab_size", type=int, default=128,
                        help="Vocabulary size (~100 used ASCII tokens + padding, default: 128)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size (default: 128)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers (default: 3, later: 5)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Attention heads (head_dim = hidden/heads, default: 4 -> 32)")
    parser.add_argument("--intermediate_size", type=int, default=512,
                        help="FFN intermediate size (default: 512)")
    parser.add_argument("--hash_projections", type=int, default=16,
                        help="Hash embedding projections (default: 16)")
    parser.add_argument("--depth_bins", type=int, default=16,
                        help="Number of depth quantization bins. 0 disables the "
                             "depth embedding entirely. Must match the bin count "
                             "used to build the training dataset (default: 16).")
    parser.add_argument("--max_position_embeddings", type=int, default=1536,
                        help="Maximum sequence length (default: 1536)")
    parser.add_argument("--local_attention", type=int, default=128,
                        help="Local attention window size (default: 128)")
    parser.add_argument("--global_attn_every_n_layers", type=int, default=3,
                        help="Global attention every N layers (default: 3)")
    parser.add_argument("--output", type=str, default="models/doom-multivec-5L",
                        help="Output directory (default: models/doom-multivec-5L)")
    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec - Model Creator")
    print("=" * 60)

    # ---------------------------------------------------------------
    # 1. Create tokenizer
    # ---------------------------------------------------------------
    print("\n[1/4] Creating ASCII tokenizer...")
    tokenizer = create_ascii_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Model max length: {tokenizer.model_max_length}")

    # ---------------------------------------------------------------
    # 2. Create config
    # ---------------------------------------------------------------
    print("\n[2/4] Creating model config...")

    config = ModernBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        model_type="modernbert",
        architectures=["ModernBertHashModel"],

        # Attention config
        attention_bias=False,
        attention_dropout=0.0,
        local_attention=args.local_attention,
        global_attn_every_n_layers=args.global_attn_every_n_layers,
        local_rope_theta=10000.0,
        global_rope_theta=160000.0,

        # Activation / normalization
        hidden_activation="gelu",
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        layer_norm_eps=1e-5,
        norm_eps=1e-5,
        mlp_bias=False,
        norm_bias=False,

        # Initialization
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,

        # No weight tying for hash embeddings
        tie_word_embeddings=False,
        use_cache=False,

        # Token IDs matching our tokenizer
        pad_token_id=tokenizer.pad_token_id,     # 0 = [PAD]
        bos_token_id=tokenizer.cls_token_id,     # 2 = [CLS]
        eos_token_id=tokenizer.sep_token_id,     # 3 = [SEP]
        cls_token_id=tokenizer.cls_token_id,     # 2 = [CLS]
        sep_token_id=tokenizer.sep_token_id,     # 3 = [SEP]
    )

    # Store hash projection count in config for trust_remote_code loading
    config.hash_projections = args.hash_projections

    # Without this, ModernBertHashModel defaults depth_bins=0 and silently drops depth_ids at forward.
    config.depth_bins = args.depth_bins

    # auto_map so AutoModel.from_pretrained uses our custom class
    config.auto_map = {
        "AutoModel": "modeling_doom_hash.ModernBertHashModel",
    }

    print(f"  Architecture:")
    print(f"    Hidden Size:      {args.hidden_size}")
    print(f"    Layers:           {args.num_layers}")
    print(f"    Heads:            {args.num_heads} (head_dim={args.hidden_size // args.num_heads})")
    print(f"    Intermediate:     {args.intermediate_size}")
    print(f"    Hash Projections: {args.hash_projections}")
    print(f"    Depth Bins:       {args.depth_bins}{' (disabled)' if args.depth_bins == 0 else ''}")
    print(f"    Max Positions:    {args.max_position_embeddings}")
    print(f"    Local Attention:  {args.local_attention}")
    print(f"    Global every N:   {args.global_attn_every_n_layers}")

    # Embedding parameter comparison
    standard_emb = args.vocab_size * args.hidden_size
    hash_emb_params = args.vocab_size * args.hash_projections + args.hash_projections * args.hidden_size
    savings = (1 - hash_emb_params / standard_emb) * 100

    print(f"\n  Embedding Parameters:")
    print(f"    Standard: {standard_emb:,} ({standard_emb / 1e6:.4f}M)")
    print(f"    Hash:     {hash_emb_params:,} ({hash_emb_params / 1e6:.4f}M)")
    print(f"    Savings:  {savings:.1f}%")

    # ---------------------------------------------------------------
    # 3. Create model
    # ---------------------------------------------------------------
    print("\n[3/4] Creating model...")

    model = ModernBertHashModel(config)

    total, emb_params, trans_params = count_params(model)
    print(f"\n  Parameter Breakdown:")
    print(f"    Embeddings:   {emb_params:>10,} ({emb_params / 1e6:.4f}M)")
    print(f"    Transformer:  {trans_params:>10,} ({trans_params / 1e6:.4f}M)")
    print(f"    Total:        {total:>10,} ({total / 1e6:.4f}M)")

    # ---------------------------------------------------------------
    # 4. Save everything
    # ---------------------------------------------------------------
    print(f"\n[4/4] Saving to {args.output}...")

    os.makedirs(args.output, exist_ok=True)

    # Save model weights + config
    model.save_pretrained(args.output)
    config.save_pretrained(args.output)

    # Save tokenizer
    save_tokenizer(tokenizer, args.output)

    # Copy modeling_doom_hash.py alongside for trust_remote_code
    src_modeling = os.path.join(
        os.path.dirname(__file__), "..", "src", "doom_multivec", "model", "modeling_doom_hash.py"
    )
    dst_modeling = os.path.join(args.output, "modeling_doom_hash.py")
    shutil.copy2(src_modeling, dst_modeling)
    print(f"  Copied modeling_doom_hash.py for trust_remote_code")

    # Save hash config
    hash_config = {
        "projections": args.hash_projections,
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
    }
    with open(os.path.join(args.output, "hash_config.json"), "w") as f:
        json.dump(hash_config, f, indent=2)

    # Save model info
    info = {
        "model_type": "DOOM-MultiVec-Hash",
        "description": "ModernBERT + Hash Embeddings for DOOM ASCII game-state understanding",
        "parameters": {
            "total": f"{total / 1e6:.4f}M",
            "total_raw": total,
            "embeddings": f"{emb_params / 1e6:.4f}M",
            "embeddings_raw": emb_params,
            "transformer": f"{trans_params / 1e6:.4f}M",
            "transformer_raw": trans_params,
        },
        "architecture": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_attention_heads": args.num_heads,
            "intermediate_size": args.intermediate_size,
            "hash_projections": args.hash_projections,
            "depth_bins": args.depth_bins,
            "max_position_embeddings": args.max_position_embeddings,
            "local_attention": args.local_attention,
            "global_attn_every_n_layers": args.global_attn_every_n_layers,
        },
        "tokenizer": {
            "type": "char-level ASCII",
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
        },
        "colbert": {
            "embedding_size": 32,
            "query_length": 48,
            "document_length": 1100,
        },
        "embedding_savings": f"{savings:.1f}%",
    }
    with open(os.path.join(args.output, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Saved files:")
    for fname in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, fname)
        size = os.path.getsize(fpath)
        if size > 1024:
            print(f"    {fname:40s} {size / 1024:.1f} KB")
        else:
            print(f"    {fname:40s} {size} B")

    print(f"\n{'=' * 60}")
    print(f"Model saved to: {args.output}")
    print(f"Total parameters: {total:,} ({total / 1e6:.4f}M)")
    print(f"{'=' * 60}")

    print(f"\nNext steps:")
    print(f"  1. Verify:  python -c \"from transformers import AutoModel; m = AutoModel.from_pretrained('{args.output}', trust_remote_code=True); print(m)\"")
    print(f"  2. Data:    python scripts/collect_data.py")
    print(f"  3. Train:   python scripts/train.py --model {args.output}")


if __name__ == "__main__":
    main()
