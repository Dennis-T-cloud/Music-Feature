# Handoff Manual — Win RTX 3060 台式机操作指南
**任务**：下载 Aria 基础模型 → Stage 2 改进重训 → 生成 MIDI  
**预计总时间**：环境配置 15 min + 模型下载 10 min + 训练 30 min + 生成 5 min  
**完成后**：把产物传回 Mac，在 Mac 上运行对比图脚本

---

## Step 0：把项目同步到 Win 机器

在 Mac 上推送最新代码：
```bash
cd ~/Desktop/CSE153/Music-Feature
git add -A && git commit -m "add analysis plots and handoff manual"
git push
```

在 Win 机器上克隆（或 pull）：
```powershell
git clone <your-repo-url> Music-Feature
cd Music-Feature
```

> 如果没有 git remote，也可以用 U盘 / 局域网共享直接复制整个 `Music-Feature/` 文件夹。  
> **必须包含**：`Task1/train/`、`Task1/result/stage1_maestro_best_adapter/`、`data/maestro-v3.0.0/`

---

## Step 1：安装 Conda 环境（只需做一次）

```powershell
# 如果没有 Anaconda/Miniconda，先去官网装 Miniconda
# https://docs.conda.io/en/latest/miniconda.html

conda create -n cse153 python=3.11 -y
conda activate cse153

# 安装 PyTorch（CUDA 12.1，适配大多数 RTX 3060 驱动）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install transformers peft pandas matplotlib mido pretty_midi huggingface_hub safetensors tqdm
```

验证 CUDA 可用：
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 应输出: True  NVIDIA GeForce RTX 3060
```

---

## Step 2：下载 Aria 基础模型（约 2.63 GB）

```powershell
conda activate cse153
cd Music-Feature

# 下载权重（主要文件，2.63 GB）
python Task1/train/aria_download_weights.py ^
  --repo-id loubb/aria-medium-base ^
  --filename model.safetensors ^
  --local-dir Task1/.vendor/aria-hf

# 下载其余配置文件（几 MB，一次性）
python -c "
from huggingface_hub import hf_hub_download
repo = 'loubb/aria-medium-base'
local = 'Task1/.vendor/aria-hf'
files = ['config.json','tokenizer_config.json','modeling_aria.py',
         'tokenization_aria.py','configuration_aria.py','__init__.py']
for f in files:
    try:
        hf_hub_download(repo_id=repo, filename=f, local_dir=local, local_dir_use_symlinks=False)
        print('Downloaded', f)
    except Exception as e:
        print('Skip', f, e)
"
```

验证下载完整：
```powershell
dir Task1\.vendor\aria-hf
# 应看到 model.safetensors、config.json、modeling_aria.py 等文件
```

---

## Step 3：Stage 2 改进重训

> **改动说明**：相比原来的 Stage 2，本次提高了 dropout（0.05→0.15）、降低了学习率（1e-5→5e-6）、减少步数（300→200）并启用 early stopping，目标是缓解过拟合。

```powershell
conda activate cse153
cd Music-Feature

python Task1/train/aria_finetune_maestro.py ^
  --model-dir Task1/.vendor/aria-hf ^
  --maestro-root data/maestro-v3.0.0 ^
  --split train ^
  --composer "Chopin" ^
  --title-contains "Etude" ^
  --max-files 64 ^
  --train-mode lora ^
  --resume-adapter Task1/result/stage1_maestro_best_adapter ^
  --lora-r 8 ^
  --lora-alpha 16 ^
  --lora-dropout 0.15 ^
  --lr 5e-6 ^
  --max-steps 200 ^
  --eval-split test ^
  --eval-max-files 0 ^
  --eval-every 25 ^
  --early-stop-patience 3 ^
  --early-stop-min-delta 0.001 ^
  --save-dir Task1/result/stage2_chopin_improved ^
  --seed 42
```

训练结束后检查结果：
```powershell
# 查看最后几行输出中的 eval loss
# 关键判断：
#   min(eval_loss) < 1.4106  → 改善了，继续 Step 4
#   min(eval_loss) >= 1.4106 → 没有改善，跳过 Step 4，直接用原始 adapter
```

训练日志保存在：`Task1/result/stage2_chopin_improved/training_log.csv`

---

## Step 4：生成 MIDI（用改进后的 adapter）

### 4a. 生成原始 MIDI
```powershell
conda activate cse153
cd Music-Feature

python Task1/train/aria_generate.py ^
  --model-id Task1/.vendor/aria-hf ^
  --adapter-dir Task1/result/stage2_chopin_improved ^
  --temperature 0.85 ^
  --max-length 4096 ^
  --output outputs/symbolic_unconditioned_improved.mid ^
  --seed 42
```

### 4b. 修复 tempo bug（每次生成都必须做）

> **说明**：Aria 的 `to_midi()` 有一个已知 bug，会把 BPM 值直接写入 MIDI 的微秒字段，导致播放时长只有 0.006 秒。下面脚本自动修复。

```powershell
python -c "
import mido
src = 'outputs/symbolic_unconditioned_improved.mid'
dst = 'outputs/symbolic_unconditioned.mid'
mid = mido.MidiFile(src)
new = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
for track in mid.tracks:
    t = mido.MidiTrack()
    for msg in track:
        if msg.type == 'set_tempo':
            t.append(mido.MetaMessage('set_tempo', tempo=500000, time=msg.time))
        else:
            t.append(msg)
    new.tracks.append(t)
new.save(dst)
import pretty_midi
pm = pretty_midi.PrettyMIDI(dst)
print(f'Fixed: {pm.get_end_time():.1f}s, {sum(len(i.notes) for i in pm.instruments)} notes')
"
```

预期输出：`Fixed: ~30.0s, ~150 notes`（比原来的 24s/96 notes 更长更丰富）

---

## Step 5：打包产物传回 Mac

需要传回的文件（其余不需要）：

```
Task1/result/stage2_chopin_improved/
    training_log.csv                ← 新的训练日志（用于对比图）
    adapter_model.safetensors       ← 新的 LoRA adapter
    adapter_config.json

outputs/
    symbolic_unconditioned.mid      ← 修复 tempo 后的新生成 MIDI
    symbolic_unconditioned_improved.mid   ← 原始（未修复）备份
```

传输方式任选：
- `git add` 后推送（注意 `.safetensors` 是大文件，需要 git-lfs 或手动排除）
- U盘 / 局域网共享直接复制

---

## Step 6：回到 Mac，更新对比图

把文件放到对应目录后，在 Mac 上运行：

```bash
cd ~/Desktop/CSE153/Music-Feature

# 重新生成所有图表（含新旧 Stage 2 对比）
/opt/anaconda3/envs/cse153-hw4/bin/python Task1/report/generate_plots.py
```

如果新的 `training_log.csv` 在 `stage2_chopin_improved/`，还需要手动在对比图脚本中加入新结果的那一行 — 告知 Mac 端操作者（或我，届时直接来问）。

---

## 快速回滚

如果 Step 3 训练结果不理想（eval_loss ≥ 1.4106），无需任何操作：

- 原始 Stage 2 adapter 仍在 `Task1/result/stage2_chopin_etude_best_adapter/`
- 原始修复后 MIDI 仍在 `outputs/symbolic_unconditioned.mid`（已是正确版本）
- 直接跳过 Step 4，仅把 `training_log.csv` 传回 Mac 用于记录"尝试过但无改善"

---

## 参考：各步骤产物一览

| Step | 产物 | 路径 |
|---|---|---|
| 2 | Aria 基础模型 | `Task1/.vendor/aria-hf/` |
| 3 | 改进 Stage 2 adapter + 训练日志 | `Task1/result/stage2_chopin_improved/` |
| 4 | 新生成 MIDI（已修复 tempo） | `outputs/symbolic_unconditioned.mid` |
| 6 | 更新后的对比图 | `Task1/result/plots/` |
