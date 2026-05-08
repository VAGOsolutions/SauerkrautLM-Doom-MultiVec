# DOOM agent enhanced via DPO and MCTS
---

## Notes

Agent does well in defend the center. In deathmatch, it has trouble killing enemies, gets right up close to the enemies, and gets stuck in corners

---

## Quick Start

```bash
conda create -n "doom_agent" python==3.10
conda activate doom_agent
pip install -e ".[dev]"

# Watch the trained model play
python scripts/play_doom_visual.py --model models/doom-multivec-trained --scenario defend_the_center

#Watch it do poorly in deathmatch
python scripts/play_doom_visual.py --model models/doom-multivec-trained --scenario deathmatch       


# Run the benchmark
python scripts/benchmark.py --agent multivec --model models/doom-multivec-trained --episodes 10 --realtime
```

---


## Project Structure

```
doom_multivec/
  src/doom_multivec/
    ascii/          # Frame-to-ASCII conversion
    model/          # ModernBERT-Hash model, tokenizer, classifier
    doom/           # VizDoom engine wrapper
    training/       # Dataset builder, action mapping
    inference/      # Real-time inference engine
  scripts/          # CLI scripts (train, benchmark, play, record, export)
  models/           # Base models + trained checkpoint
  docs/             # MkDocs Material documentation

```

