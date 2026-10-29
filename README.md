# video2：Qwen 零样本视频摘要基线

使用 **Qwen2.5-VL-7B-Instruct**，对整个视频均匀抽帧后生成摘要，再与人工参考摘要计算 ROUGE-1 / ROUGE-2 / ROUGE-L F1。支持英文与中文，自定义 JSONL 数据集，不微调、不提供 few-shot 示例。参考摘要只进入评分阶段，绝不进入模型消息。

这是**视觉视频摘要**实验：不读取音轨、ASR、字幕文件或外部知识。视频画面中自带的字幕仍可能被模型读取。结果不能代表音视频联合理解，也不是 SumMe/TVSum 的关键帧选择任务。

## 1. 安装

Python >= 3.10；建议在带 CUDA 的机器上运行真实模型。7B 半精度权重本身约 14 GB，推理还需要视觉编码器、KV cache 和激活内存；推荐从 24 GB 显存、32 帧设置开始，实际需求取决于视频尺寸、生成长度和运行时。CPU 可用于功能测试，7B 推理很慢且需要足够内存。

```bash
cd /workspace/video2  # 其他机器使用实际仓库路径
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference]'
```

固定 `transformers==4.57.1`、`torch==2.8.0` 和 `torchvision==0.23.0`，避免 Qwen 视频接口与 PyTorch/torchvision 不匹配。CUDA/CPU wheel 请按 [PyTorch 官方安装指引](https://pytorch.org/get-started/previous-versions/)选择匹配的 2.8.0/0.23.0 组合，然后安装本项目。无需 FlashAttention 或 `qwen-vl-utils`：抽帧由 OpenCV 完成，时间信息显式传给 processor。

只对已有结果评分不需要任何第三方依赖：

```bash
python -m pip install -e .
```

首次使用模型需访问 Hugging Face 下载权重；也可通过 `--model /path/to/local/model` 指定包含模型、processor、tokenizer 的完整本地目录。下载需要允许 Hugging Face 及实际重定向到的存储域名。不要在代码或数据文件中写入 token。

## 2. 准备数据

每行一个 JSON 对象，`id` 必须唯一且为字符串，`video` 是本地视频路径。相对路径**以 JSONL 文件所在目录为基准**。`references` 是同一视频的一个或多个人工摘要；缺省或 `[]` 表示仅推理、不评分。

```json
{"id":"clip_001","video":"../data/clip_001.mp4","references":["A person chops vegetables and adds them to a pan."]}
{"id":"clip_002","video":"../data/clip_002.mp4","references":["A dog catches a ball in a field.","A dog runs over grass to catch a ball."]}
```

复制 [examples/dataset.jsonl](examples/dataset.jsonl) 并替换视频和摘要。示例仅展示格式，不包含实际视频，也不是可直接报告结果的测试集。中文实验请提供中文参考摘要，并使用 `--language zh`。所有样本在一次运行中使用相同摘要语言和提示词。请使用独立测试集；zero-shot 不代表模型预训练时一定没见过这些视频。

先实际解码检查文件与抽帧，不加载模型：

```bash
video2 validate --dataset examples/dataset.jsonl --num-frames 32
```

## 3. 零样本推理与评测

```bash
video2 run \
  --dataset examples/dataset.jsonl \
  --output outputs/qwen25vl7b-test \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda --dtype bfloat16 \
  --language en --num-frames 32 \
  --max-frame-size 448 --max-new-tokens 256 --max-words 100
```

若 GPU 不支持 BF16，使用 `--dtype float16`；`--device auto --dtype auto` 可让 Transformers/Accelerate 自动分配设备。CPU 使用 `--device cpu --dtype float32`。可替换为同架构的 `Qwen/Qwen2.5-VL-3B-Instruct`；当前实现专门使用 Qwen2.5-VL 类，不直接兼容 Qwen3-VL 或纯文本 Qwen。

中文摘要：将上面命令的 `--language en` 改为 `--language zh`。`--max-words` 此时表示提示词中的汉字预算。该预算是模型指令，**不是硬截断**；实际生成上限由 `--max-new-tokens` 控制。

采样与实验控制：

- 默认均匀取 32 帧，包含首尾帧；`--num-frames` 必须为 >=2 的偶数。短视频重复已有帧，保持所需长度。固定帧数可能遗漏短暂事件，长视频尤其明显；比较模型时保持相同采样配置。
- 抽帧先将最长边限制在 448，再由模型 processor 按 28 的倍数缩放。默认每帧像素预算为 `128*28*28` 到 `256*28*28`，可用 `--min-pixels` / `--max-pixels` 设置；processor 可能放大小帧，因此最终尺寸不一定受 448 限制。长宽比保持一致，过于极端的长宽比会由 processor 拒绝。
- 有效 FPS 来自采样时间跨度，而非硬编码为原始 FPS。OpenCV 依赖帧数和 FPS 元数据；可变帧率视频需要精确时间信息时，请先规范化为恒定帧率。重复或取整抽帧的实际时间戳会保存；传给 Qwen 的 FPS 表示平均采样间隔。
- 固定 seed、greedy decoding、`do_sample=False`、`num_beams=1`。跨设备/依赖版本不保证逐位相同。正式对比请用 `--revision <Hugging-Face-commit-SHA>` 固定模型版本，并保存依赖环境。
- 显存不足时降低帧数、像素预算或生成长度，或改用 3B；这会改变实验条件，应单独报告。

默认英文提示词定义在 [prompts.py](src/video2/prompts.py)：要求按时间顺序概括主要事件，仅使用可见证据，不编造对话、声音、身份或动机；没有示例和参考答案。完整实际提示词保存到每次实验的配置中。

## 4. 结果与离线重评分

输出目录必须尚不存在，避免覆盖旧实验。逐条写出并 flush 结果；单个视频失败会保留错误并继续其他视频，最终退出码为 1。模型加载失败会保存失败状态，不生成伪造预测。暂不支持断点续跑；重试使用新目录。

| 文件 | 内容 |
| --- | --- |
| `config.json` | 提示词、模型参数、解析到的模型 commit、seed、抽帧参数、运行状态、库版本和代码 Git 状态 |
| `dataset.jsonl` | 本次数据集快照（绝对视频路径、参考摘要、文件大小与 mtime） |
| `predictions.jsonl` | 每视频摘要/错误、耗时（抽帧+生成，不含模型加载）、时间戳与 FPS |
| `metrics.json` | ROUGE 宏平均、逐视频得分、成功/失败/缺失/实际评分数量 |

视频文件不会复制进结果目录；大小与 mtime 也不是视频内容哈希。请另外保留固定测试集。`config.json` 记录原始 JSONL 的 SHA-256；正式实验建议提交代码、固定依赖，并将 `python -m pip freeze` 的输出与实验归档。

已有预测可在没有 GPU、视频文件或模型依赖时重新评分：

```bash
video2 evaluate \
  --dataset outputs/qwen25vl7b-test/dataset.jsonl \
  --predictions outputs/qwen25vl7b-test/predictions.jsonl \
  --language en \
  --output outputs/qwen25vl7b-test/rescored.json
```

导入其他来源的预测时，每行使用 `{"id":"clip_001","status":"ok","summary":"..."}`；失败记录为 `{"id":"clip_002","status":"error","error":"..."}`。按 ID 对齐，不依赖行顺序；重复/未知 ID、空摘要会报错。离线评分若存在失败、缺失预测或没有可评分样本，退出码为 1，同时仍输出可用报告。

**评分定义与限制：**

- 得分范围 0–1；ROUGE-1/2 是剪裁计数的 n-gram F1，ROUGE-L 是整段 token 序列 LCS F1，非 ROUGE-Lsum。
- 英文小写单词/数字分词，保留单词内部的英文撇号，不做 stemming；中文按 CJK 单字并保留英文词/数字分词。不是标准数据集官方评分器，不能直接与采用不同分词或 stemming 的论文数值比较。
- 多个参考摘要先分别评分求均值，再对视频做宏平均，不取最佳参考。结果只包含**成功且有参考摘要**的视频；必须同时报告 coverage，避免失败样本被排除后虚高。`without_references` 只统计成功但没有参考摘要的样本。无可评分样本时 `macro_scores` 是 `null`，不伪报 0 或通过。
- ROUGE 仅衡量词面重合，不保证事实正确、时序正确或没有幻觉。建议人工检查主要事件覆盖、时序和幻觉；本项目不把词面得分称为完整视频理解表现。

## 5. 测试

```bash
python -m unittest discover -s tests -v
```

测试包括数据校验、中英文 ROUGE、多参考聚合、缺失/失败覆盖统计、参考摘要隔离、实验输出保护、抽帧错误处理和真实生成视频解码。安装完整推理依赖后还会运行离线的小型随机权重 Qwen 接口测试（无需下载模型）；没有可选依赖时相关集成测试会明确跳过。随机权重测试只验证处理器/生成接口，不验证摘要质量。

从未安装包的 checkout 运行也可使用 `PYTHONPATH=src python -m video2 ...` 和 `PYTHONPATH=src python -m unittest discover -s tests -v`。
