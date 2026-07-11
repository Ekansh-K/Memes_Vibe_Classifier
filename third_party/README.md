# Vendored reference codebases

Cloned for implementation reference (Hate-CLIPper fusion, MemeCLIP adapters).

| Repo | Upstream | Used for |
|---|---|---|
| `hateclipper/` | https://github.com/gokulkarthik/hateclipper | Align / cross FIM fusion logic |
| `MemeCLIP/` | https://github.com/SiddhantBikram/MemeCLIP | Residual adapters + cosine classifier ideas |

**Runtime code lives in `src/hateclipper_mmhs/`** (clean MMHS port with soft labels). Do not train via third_party Lightning scripts directly — they target Hateful Memes / PrideMM formats.

Re-clone if missing:

```bash
git clone --depth 1 https://github.com/gokulkarthik/hateclipper.git third_party/hateclipper
git clone --depth 1 https://github.com/SiddhantBikram/MemeCLIP.git third_party/MemeCLIP
```
