# Privacy Cloud

Qwen3-1.7B 端侧隐私 Agent、文本云推理、HE-only、HE-MPC 和 FedLoRA 的源码交付。使用 `bash run.sh help` 查看入口。面向 Linux 和 WSL。

| 目录 | 用途 |
| --- | --- |
| `privacy_infer/` | 本地 Qwen3、LangGraph 路由、工具权限、DeepSeek 出站审核、隐私模型旁路 |
| `he_only_v0/` | OpenFHE CKKS CPU 与 Perseus CUDA 的密态生成、回灌和解密 |
| `he_mpc_v0/` | BFV 与 MP-SPDZ 两方计算的端侧客户端、独立求值端和原生依赖构建 |
| `scripts/` | 公开数据获取、数据生成、非 IID 划分和构建时质量检查 |
| `fed_lora_v1/` | 联邦微调、可选 DP 与安全聚合、集中式 LoRA |

## 本地 Agent

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[agent,train]'
cp config.example.json config.local.json
bash run.sh agent --config config.local.json --route local
```

在 `config.local.json` 填入现有 Qwen3-1.7B Safetensors 检查点目录。若使用相对路径，以运行命令时的目录为基准。Agent 从终端隐藏输入读取任务和上下文，也接受仅含 `task` 和 `context` 的 UTF-8 JSON。可使用 `--allow-tool calculator` 或 `--allow-tool text_stats` 启用本地工具。

文本云路径使用 `agent --route cloud --allow-cloud`，从本地环境读取 `DEEPSEEK_API_KEY`，并要求在交互终端核对本次脱敏正文后输入 `SEND`。旧版 ProSan/ STAMP 的离线预览可用 `infer sanitize`。旧版显式发送可以保留高敏感、高相关的原文。新 LoRA 可用 `agent --shadow-adapter /path/to/adapter` 在端侧查看统计。

## 密态推理

HE-only 和 HE-MPC 由独立命令启动。整模型密态自回归是待验收的 v0 路径。

**HE-only CPU** 使用与系统及 Python 匹配的 OpenFHE 1.4.1 环境。CPU 运行时需要 NumPy 和 Transformers tokenizer。近似配置由安装了 PyTorch 和 Transformers 的环境生成。

```bash
bash run.sh calibrate --model "$MODEL" --output approximation.local.json
QWEN_HE_PYTHON=/path/to/openfhe-venv/bin/python bash run.sh he-only \
  --model "$MODEL" --approximation approximation.local.json --work runs/he1 --tokens 2
```

该入口在同一主机启动独立客户端和求值进程。Perseus CUDA 路径先在 `he_only_v0/.venv` 准备 Python 3.12+、NumPy 和 Transformers，再用 `CUDA_HOME=/path/to/cuda python he_only_v0/build_perseus_native.py` 构建 SM86 原生扩展。使用 `he-gpu` 或 `he-multi` 替换上面的 `he-only`。双卡时设置不同的物理卡 `QWEN_FHE_MODEL_GPU`、`QWEN_FHE_BOOTSTRAP_GPU`。内置 Perseus 源码适用随附的 BUSL-1.1 条款。

**HE-MPC** 在端侧与远端各准备同一版本的本项目源码和相同的 Qwen3 检查点。两个 Python 环境均需 OpenFHE 1.4.1 与 NumPy，端侧还需 Transformers。两端运行 `python he_mpc_v0/prepare_native.py` 获取并校验 MP-SPDZ v0.4.4，也可传入同版本的 `--archive`。用 `bash he_mpc_v0/build_native.sh` 在远端构建 Haswell 兼容运行时，使用 `bundle_native.py` 与 `install_native.py --bundle` 将同一运行时安装到客户端的 MP-SPDZ 目录。

```bash
HE_MPC_PYTHON=/path/to/client-venv/bin/python bash run.sh he-mpc \
  --model "$MODEL" --ssh-host whu216 \
  --remote-root /srv/privacy-cloud/he_mpc_v0 \
  --remote-python /srv/openfhe-venv/bin/python \
  --remote-model /srv/models/Qwen3-1.7B --steps 2
```

在用户自己的 OpenSSH 配置中设置目标别名、跳板和主机指纹。原文默认通过终端隐藏输入读取。v0 支持最多 4 个生成 token 和 32 个前缀位置。客户端运行目录包含本地生成结果，请按用户数据保管。

## 数据与训练

```bash
bash run.sh fetch-data --out data/raw
bash run.sh build-data --raw data/raw --tokenizer "$MODEL" --out data/v0
bash run.sh revise-data --raw data/raw --old data/v0 --tokenizer "$MODEL" --out data/v1
bash run.sh pack-train --dataset data/v1 --output runs/training-bundle.tgz
mkdir -p runs/fed
tar -xzf runs/training-bundle.tgz -C runs/fed
bash run.sh fed-train --root runs/fed --model "$MODEL"
```

`pack-train` 包含训练与验证数据及客户端清单。训练可以在准备好 CUDA 和对应 Qwen3 权重的服务器执行。启用可选隐私模块时，将 `privacy.example.json` 的设置写入运行目录 `config.json` 的 `privacy` 字段，再按实验配置修改开关。默认执行无 DP 与 MPC 的基线。集中式小试入口为 `bash run.sh pilot --dataset data/v1 --model "$MODEL" --out runs/pilot`。
