"""Patch vram_feasibility_test.ipynb to use correct Qwen3-VL API."""
import json
from pathlib import Path

p = Path(__file__).parent.parent / 'notebooks' / 'vram_feasibility_test.ipynb'
text = p.read_text(encoding='utf-8')

# Fix 1: wrong model class
text = text.replace('Qwen3VLForConditionalGeneration', 'AutoModelForImageTextToText')

# Fix 2: Add min_pixels to processor calls
text = text.replace(
    "max_pixels=448*448)",
    "min_pixels=256*256, max_pixels=448*448)",
)

# Fix 3: make sure AutoModelForImageTextToText is imported everywhere
# (replaces any leftover import that would have Qwen3VLForConditionalGeneration)
text = text.replace(
    "Qwen3VLForConditionalGeneration, AutoProcessor",
    "AutoModelForImageTextToText, AutoProcessor",
)

p.write_text(text, encoding='utf-8')
print('vram_feasibility_test.ipynb patched.')
