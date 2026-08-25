# asr-decoder

面向 PyTorch CTC 模型的流式解码器，支持：

- 贪心搜索和 Prefix Beam Search；
- 热词偏置与 N-best；
- token 概率和时间戳。

不绑定具体模型或语言。输入是形状为 `(frames, vocabulary_size)` 的 log probabilities；logits 请先执行
`torch.log_softmax(logits, dim=-1)`。

## 安装

```bash
pip install asr-decoder
```

仓库同时提供集成 SentencePiece 的 Rust crate 和 C++17 静态库，分别见
[`rust/`](rust/README.md) 和 [`cpp/`](cpp/)。两者都接收连续、按行存储的 log probability
矩阵，并支持流式搜索、N-best、时间戳，以及使用对应的 SentencePiece tokenizer 编码文本热词。

## 基本用法

```python
from asr_decoder import CTCDecoder, DecodingQuality

decoder = CTCDecoder(
    blank_id=0,
    decoding_quality=DecodingQuality.BALANCED,
)
result = decoder.prefix_beam_search(log_probabilities, finalize=True)
best_tokens = result["tokens"][0]
```

只需要贪心搜索时：

```python
result = decoder.greedy_search(log_probabilities, finalize=True)
```

可选的 `DecodingQuality`：`LOW_LATENCY`、`BALANCED`（默认）、`HIGH_ACCURACY`。

`DecodingQuality` 控制 Prefix Beam Search 的搜索预算和延迟；`HotwordStrength` 只控制热词偏置的强弱。两者相互独立：
没有配置热词时，`HotwordStrength` 不起作用；使用贪心搜索时，`DecodingQuality` 也不起作用。

## 热词

优先使用模型 tokenizer 生成的 token ID：

```python
from asr_decoder import CTCDecoder, HotwordStrength

decoder = CTCDecoder(
    context_token_ids=[tokenizer.encode("停止"), tokenizer.encode("Live Captions")],
    hotword_strength=HotwordStrength.BALANCED,
    blank_id=0,
)
```

`HotwordStrength` 提供 `CONSERVATIVE`、`BALANCED`、`AGGRESSIVE` 和
`LOW_FALSE_ACTIVATION` 四档。SentencePiece 可使用
`tokenize_sentencepiece_contexts(tokenizer, hotwords)`，以保留词边界。

需要完全控制热词参数时，可传入 `ContextPolicy`；它与 `hotword_strength` 不能同时设置。

## 流式与时间戳

```python
decoder = CTCDecoder(
    context_token_ids=hotword_token_ids,
    frame_shift_ms=40.0,
)
stream = decoder.create_stream()

partial = stream.prefix_beam_search(first_chunk)
final = stream.prefix_beam_search(last_chunk, finalize=True)
```

`frame_shift_ms` 是相邻 CTC 输出帧之间的间隔，不是采样率或输入帧率。未设置时仍返回帧索引，但毫秒字段为 `None`。
一次流中不要混用两种搜索；用 `finalize=True` 或 `reset()` 开始新流。

每次搜索都会返回：

- `tokens`：token ID；
- `timestamps`：token 的起止帧和可选的毫秒时间；
- `probs`、`scores`、`gating`：通过对应的 `return_*` 参数开启。

## 公开测试集参考结果

SenseVoiceSmall 的一次解码对比；每格为“不开热词 / 开热词”。

| 数据集 | 错误率 | 热词召回 | 热词精度 | F1 |
| --- | --- | --- | --- | --- |
| SeACo test | CER 10.41% / 8.40% | 51.59% / 77.28% | 99.39% / 99.18% | 67.92% / 86.87% |
| IS21 clean | WER 3.92% / 3.47% | 83.54% / 87.39% | 99.832% / 99.819% | 90.96% / 93.19% |
| IS21 other | WER 8.22% / 7.53% | 68.06% / 74.33% | 99.721% / 99.693% | 80.91% / 85.17% |
| Earnings-22 test | WER 25.10% / 24.98% | 24.87% / 26.97% | 98.567% / 98.677% | 39.72% / 42.36% |

来源：SenseVoiceSmall 的 [官方仓库](https://github.com/QwenAudio/SenseVoice) 和
[ModelScope 模型](https://www.modelscope.cn/models/iic/SenseVoiceSmall)；SeACo 的
[公开热词测试集](https://github.com/R1ckShi/SeACo-Paraformer)；IS21 的
[官方测试文件与评分脚本](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias)；
Earnings-22 的 [Contextual Earnings-22 数据集](https://huggingface.co/datasets/argmaxinc/contextual-earnings22)。
表格是本项目的复测结果，不是上游项目的官方成绩。
