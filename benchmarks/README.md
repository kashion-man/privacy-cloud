# 推理实测数据

本目录保存 Qwen3-1.7B 的原始计时样本、运行环境、源码哈希和统计汇总。元数据中的账号路径与 GPU UUID 已做去标识化，复现时需设置本机模型与输出路径。

| 文件 | 内容 |
| --- | --- |
| [local_inference.json](results/local_inference.json) | RTX 4060 Laptop 单卡的 16 组配置及逐次计时 |
| [local_sources.json](results/local_sources.json) | 单卡测试源码哈希与仓库提交 |
| [parallel_inference.json](results/parallel_inference.json) | WHU188 单卡与双卡的 18 次结果、环境、日志及源码哈希 |
| [parallel_summary.json](results/parallel_summary.json) | 远端三次重复的中位数 |
| [prefix_inference.json](results/prefix_inference.json) | 相同前缀复用测试，支持项目首页的 TTFT 与输出速率指标 |

## 本地单卡

RTX 4060 Laptop，Qwen3-1.7B FP16，固定生成 32 token。每组预热 2 次、测量 5 次。系统测试关闭前缀缓存，CUDA Graph、prefill 编译与 ZOS 异步解码组合称为 ZOS。下表为中位数，批量场景报告聚合吞吐。

| 批量 | 输入 token | Eager 吞吐 | ZOS 吞吐 | 吞吐提升 | Eager TTFT | ZOS TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 61.65 token/s | 66.04 token/s | 7.1% | 19.74 ms | 20.51 ms |
| 1 | 512 | 50.35 token/s | 61.35 token/s | 21.9% | 60.43 ms | 56.94 ms |
| 8 | 128 | 375.57 token/s | 400.56 token/s | 6.7% | 84.64 ms | 87.09 ms |
| 8 | 512 | 220.96 token/s | 245.39 token/s | 11.1% | 335.32 ms | 331.23 ms |

四种执行模式的生成 token 哈希通过一致性检查。这组负载最高吞吐为 batch 8、128-token 输入的 CUDA Graph 模式，达到 404.86 token/s。计时排除模型加载、预热与分词。

在 dzyy-vllm 的 `vllm-v3/` 下运行原入口。

```bash
python -B -u bench/run.py e2e --model "$MODEL" \
  --batches 1 8 --prompt-lengths 128 512 --output-tokens 32 \
  --warmup 2 --repeats 5 --out "$RESULT"
```

## 远端并行

WHU188 的 5、7 号 RTX 4090，Qwen3-1.7B BF16，固定生成 128 token、关闭前缀缓存。两个 GPU 通过 PIX PCIe 拓扑连接。每个配置独立运行三次，取中位数。TTFT 和单请求 decode 使用 256-token 输入，batch 吞吐使用相同输入长度和 batch 8。decode 速率排除首 token，batch 吞吐包含 prefill。

| 配置 | TTFT | 单请求 decode | batch 8 吞吐 |
| --- | ---: | ---: | ---: |
| 单卡 eager | 21.56 ms | 58.7 token/s | 417.0 token/s |
| 单卡 CUDA Graph | 20.98 ms | 223.5 token/s | 1,422.4 token/s |
| 双卡 TP + CUDA Graph | 30.90 ms | 233.8 token/s | 1,176.6 token/s |
| 双卡 PP + CUDA Graph | 27.03 ms | 167.0 token/s | 1,083.5 token/s |

双卡 TP 相对单卡 CUDA Graph 的单请求 decode 提升 4.6%，batch 8 吞吐下降 17.3%。

在相同的 32 请求、128-token 输入和 128-token 输出负载中，单卡 CUDA Graph 为 4,924.5 token/s，双卡 DP 为 5,372.6 token/s，吞吐提升 9.1%。每 GPU 的 KV 块数为 128、序列上限为 64。DP 默认 token 预算为 8,192，单卡参考为 4,096。该负载与上表不同。

远端使用原 `vllm-v4/bench/bench_v4.py`。单卡 DP 参考由外部编排调用该文件的 `make_engine`、`run_dp_replicas` 及原测量函数，使用单卡和相同请求负载。

```bash
python -B -u bench/bench_v4.py --model "$MODEL" --device-ids 0 --retries 1
python -B -u bench/bench_v4.py --model "$MODEL" --device-ids 0 --retries 1 --cg
python -B -u bench/bench_v4.py --model "$MODEL" --device-ids 0,1 --retries 1 --tp 2 --cg
python -B -u bench/bench_v4.py --model "$MODEL" --device-ids 0,1 --retries 1 --pp 2 --cg
python -B -u bench/bench_v4.py --model "$MODEL" --device-ids 0,1 --retries 1 --dp 2 --n-reqs 32 --cg
```

服务器为共享环境，运行前检查所用 GPU 空闲。环境为 Python 3.10、PyTorch 2.8.0+cu128、FlashAttention 2.8.3、FlashInfer 0.5.3、Triton 3.5.0。独立 Python 使用专用目录中的 GLIBC 2.35。环境启动适配将 `all_gather_single` 指向同签名的 `all_gather_into_tensor`。18 次运行均完成，源码清单的哈希一致。PP 日志有 Python 资源跟踪器自动清理 semaphore 的提示。本轮跨并行配置的输出数值一致性尚待单独验证。

## 前缀复用

RTX 4060 Laptop，FP16，同一 512-token 前缀，固定输出 4 token，预热一次后测量五次。禁用前缀缓存和 Radix 缓存的 warm 结果分别为 TTFT 77.94 / 34.56 ms、单请求输出速率 22.06 / 29.74 token/s。TTFT 下降 55.7%，输出速率提升 34.8%。来源为 dzyy-vllm 的 `vllm-v1/bench/run.py prefix` 测试，加载和预热排除在计时外。

上述数据来源于 dzyy-vllm 各自的 benchmark 入口。Privacy Cloud 的请求隔离还包含 KV 清理等工作，项目内 Agent 的端到端耗时需要按其入口单独测量。
