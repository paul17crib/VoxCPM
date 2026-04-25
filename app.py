import os
import sys
import logging
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    "`Control Instruction`: *\"A young girl with a soft, sweet voice. "
    "Speaks slowly with a melancholic, slightly tsundere tone.\"*  \n"
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    "`Control Instruction`: *\"Relaxed young male voice, slightly nasal, "
    "lazy drawl, very casual and chill.\"*  \n"
    "`Target Text`: *\"Dude, did you see that set? The waves out there are totally gnarly today. "
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
    "\n\n"
    "**Example 3 — Calm & Authoritative Narrator**  \n"
    "`Control Instruction`: *\"Middle-aged male voice, deep and measured, "
    "clear articulation, steady pace, documentary-style narration.\"*  \n"
    "`Target Text`: *\"In the vast expanse of the cosmos, every star tells a story "
    "billions of years in the making.\"*"
    # Personal note: added a 4th example for expressive storytelling use-case
    "\n\n"
    "**Example 4 — Warm Bedtime Storyteller**  \n"
    "`Control Instruction`: *\"Warm, gentle adult voice, soft and unhurried, "
    "soothing tone suitable for bedtime stories.\"*  \n"
    "`Target Text`: *\"Once upon a time, in a forest where the fireflies never slept, "
    "a little fox discovered a door no one else could see.\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 "
)
