# asr-decoder (Rust)

Python 实现的 Rust 对应版本，支持流式贪心搜索、Prefix Beam Search、N-best、token
概率与时间戳，以及基于 token ID 或 SentencePiece 文本的热词偏置。crate 不依赖具体推理框架；
SentencePiece 使用原生绑定并静态构建官方实现。

构建时需要 CMake 和 C++ 编译器，但不需要预先安装 SentencePiece。

输入为连续的、按行存储的 `(frames, vocabulary_size)` `f32` log probabilities：

```rust
use asr_decoder::{
    CTCDecoder, DecoderConfig, LogProbabilities, PrefixBeamSearchOptions,
};

let values = vec![
    -5.0, -0.1, -4.0,
    -0.1, -5.0, -4.0,
    -5.0, -4.0, -0.1,
];
let probabilities = LogProbabilities::new(&values, 3, 3)?;

let mut decoder = CTCDecoder::new(DecoderConfig {
    blank_id: 0,
    frame_shift_ms: Some(40.0),
    ..DecoderConfig::default()
})?;
let result = decoder.prefix_beam_search(
    probabilities,
    PrefixBeamSearchOptions {
        finalize: true,
        ..PrefixBeamSearchOptions::default()
    },
)?;
assert_eq!(result.tokens[0], vec![1, 2]);
# Ok::<(), asr_decoder::DecoderError>(())
```

流式输入时，对同一个 decoder 连续调用搜索方法。中间块保持 `finalize: false`，最后一块设置
`finalize: true`。也可以从模板调用 `create_stream()` 创建共享配置、状态独立的流。

热词直接使用模型 tokenizer 的 token ID：

```rust
use asr_decoder::{CTCDecoder, DecoderConfig, HotwordStrength};

let decoder = CTCDecoder::new(DecoderConfig {
    context_token_ids: vec![vec![23, 41], vec![81, 19, 7]],
    hotword_strength: Some(HotwordStrength::Balanced),
    ..DecoderConfig::default()
})?;
# Ok::<(), asr_decoder::DecoderError>(())
```

也可以在模型生命周期内加载一次 SentencePiece，然后直接传文本热词：

```rust
use asr_decoder::{
    CTCDecoder, DecoderConfig, SentencePieceConfig, SentencePieceTokenizer,
};

let tokenizer = SentencePieceTokenizer::open(
    "/path/to/sentencepiece.model",
    SentencePieceConfig::default(),
)?;
let decoder = CTCDecoder::new(DecoderConfig {
    contexts: vec!["停止".to_owned(), "Live Captions".to_owned()],
    sentencepiece_tokenizer: Some(tokenizer.clone()),
    ..DecoderConfig::default()
})?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`SentencePieceTokenizer` 内部使用 `Arc` 共享已经加载的 processor；clone、创建多个 decoder
或 stream 都不会重新读取模型。独立热词默认保留 SentencePiece 的 `▁` 词首语义；需要允许
词内匹配时设置 `SentencePieceConfig::add_word_boundary = false`。

如果声学模型 token ID 与 SentencePiece ID 不一致，在 `SentencePieceConfig::symbol_table`
中提供 piece 到声学 ID 的精确映射。缺失 piece 且没有 `<unk>` 时会返回错误，不会静默丢 token。
`ContextPolicy` 和 `hotword_strength` 仍然二选一。
