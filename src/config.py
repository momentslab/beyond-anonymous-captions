"""Qwen3-VL generation parameters — BAC."""

MIN_PIXELS = 128 * 32 * 32       # 131072
MAX_IMAGE_PIXELS = 512 * 32 * 32  # 524288

MAX_NEW_TOKENS = 64
QA_MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.0

MODEL_PRESETS = {
    "2b": {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "batch_size": 8,
    },
    "4b": {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "batch_size": 2,
    },
    "8b": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "batch_size": 2,
    },
}

DEFAULT_MODEL = MODEL_PRESETS["2b"]["model"]

# Local adapter dirs (relative to the project root); downloaded from HF if missing.
LORA_PRESETS = {
    "2b": "models/lora/2b",
    "4b": "models/lora/4b",
    "8b": "models/lora/8b",
}

# HF model repo holding one adapter subfolder per size (2b/, 4b/, 8b/).
LORA_HF_REPO = "momentslab/beyond-anonymous-captions"

# Movies held out for evaluation with ``train.py --eval`` (movie-level split:
# every shot of these movies goes to eval, all other training movies to train).
EVAL_MOVIES = [
    "0004_Charade",
    "0005_Chinatown",
    "0013_Halloween",
    "0020_Raising_Arizona",
    "0028_The_Crying_Game",
    "0029_The_Graduate",
    "0031_The_Lost_Weekend",
    "1002_Harry_Potter_and_the_Half-Blood_Prince",
    "1018_Body_Of_Lies",
    "1044_Pride_And_Prejudice_Disk_Two",
    "3014_CAPTAIN_AMERICA",
    "3043_KATY_PERRY_PART_OF_ME",
    "3046_LARRY_CROWNE",
    "3075_THE_SITTER",
    "3088_WHATS_YOUR_NUMBER",
]

# Default LoRA fine-tune (combin_2): lr=5e-5, r=16, α=32, 1 epoch.
FINETUNE_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

FINETUNE_TRAINING = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5.0e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "optim": "adamw_torch",
    "max_grad_norm": 1.0,
    "seed": 42,
}
