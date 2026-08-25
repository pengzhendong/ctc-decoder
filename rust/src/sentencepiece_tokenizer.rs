use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use prost::Message;
use sentencepiece::{PieceWithId, SentencePieceProcessor};

const WORD_BOUNDARY: &str = "▁";

#[derive(Clone, Debug)]
pub struct SentencePieceConfig {
    pub symbol_table: HashMap<String, usize>,
    pub add_word_boundary: bool,
}

impl Default for SentencePieceConfig {
    fn default() -> Self {
        Self {
            symbol_table: HashMap::new(),
            add_word_boundary: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentencePieceTokenization {
    pub context_token_ids: Vec<Vec<usize>>,
    pub word_boundary_token_ids: HashSet<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    Io(String),
    SentencePiece(String),
    InvalidModel(String),
    MissingPiece { piece: String, context: String },
}

impl Display for TokenizerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "failed to read SentencePiece model: {message}"),
            Self::SentencePiece(message) => write!(formatter, "SentencePiece error: {message}"),
            Self::InvalidModel(message) => {
                write!(formatter, "invalid SentencePiece model: {message}")
            }
            Self::MissingPiece { piece, context } => write!(
                formatter,
                "SentencePiece piece {piece:?} is missing from the symbol table for context {context:?}"
            ),
        }
    }
}

impl Error for TokenizerError {}

#[derive(Clone, Debug)]
pub struct SentencePieceTokenizer {
    inner: Arc<TokenizerInner>,
}

#[derive(Debug)]
struct TokenizerInner {
    processor: SentencePieceProcessor,
    symbol_table: HashMap<String, usize>,
    unknown_id: Option<usize>,
    add_word_boundary: bool,
    word_boundary_token_ids: HashSet<usize>,
}

impl SentencePieceTokenizer {
    pub fn open(
        model_path: impl AsRef<Path>,
        config: SentencePieceConfig,
    ) -> Result<Self, TokenizerError> {
        let model =
            fs::read(model_path.as_ref()).map_err(|error| TokenizerError::Io(error.to_string()))?;
        Self::from_serialized_proto(&model, config)
    }

    pub fn from_serialized_proto(
        model: &[u8],
        config: SentencePieceConfig,
    ) -> Result<Self, TokenizerError> {
        let processor = SentencePieceProcessor::from_serialized_proto(model)
            .map_err(|error| TokenizerError::SentencePiece(error.to_string()))?;
        let model_proto = ModelProto::decode(model)
            .map_err(|error| TokenizerError::InvalidModel(error.to_string()))?;
        let word_boundary_token_ids = model_proto
            .pieces
            .iter()
            .enumerate()
            .filter_map(|(id, piece)| {
                let piece = piece.piece.as_deref()?;
                if !piece.starts_with(WORD_BOUNDARY) {
                    return None;
                }
                if config.symbol_table.is_empty() {
                    Some(id)
                } else {
                    config.symbol_table.get(piece).copied()
                }
            })
            .collect();
        let unknown_id = config.symbol_table.get("<unk>").copied();
        Ok(Self {
            inner: Arc::new(TokenizerInner {
                processor,
                symbol_table: config.symbol_table,
                unknown_id,
                add_word_boundary: config.add_word_boundary,
                word_boundary_token_ids,
            }),
        })
    }

    pub fn tokenize(
        &self,
        contexts: &[String],
    ) -> Result<SentencePieceTokenization, TokenizerError> {
        let mut context_token_ids = Vec::with_capacity(contexts.len());
        for context in contexts {
            let text = context.trim();
            if text.is_empty() {
                continue;
            }
            let pieces = self.encode_context(text)?;
            let mut ids = Vec::with_capacity(pieces.len());
            for piece in pieces {
                let id = if self.inner.symbol_table.is_empty() {
                    piece.id as usize
                } else if let Some(id) = self.inner.symbol_table.get(&piece.piece) {
                    *id
                } else if let Some(id) = self.inner.unknown_id {
                    id
                } else {
                    return Err(TokenizerError::MissingPiece {
                        piece: piece.piece,
                        context: text.to_owned(),
                    });
                };
                ids.push(id);
            }
            if !ids.is_empty() {
                context_token_ids.push(ids);
            }
        }
        Ok(SentencePieceTokenization {
            context_token_ids,
            word_boundary_token_ids: self.inner.word_boundary_token_ids.clone(),
        })
    }

    pub fn word_boundary_token_ids(&self) -> &HashSet<usize> {
        &self.inner.word_boundary_token_ids
    }

    fn encode_context(&self, text: &str) -> Result<Vec<PieceWithId>, TokenizerError> {
        let raw = self
            .inner
            .processor
            .encode(text)
            .map_err(|error| TokenizerError::SentencePiece(error.to_string()))?;
        if !self.inner.add_word_boundary || text.starts_with(WORD_BOUNDARY) {
            return Ok(raw);
        }

        let marked = self
            .inner
            .processor
            .encode(&format!("{WORD_BOUNDARY}{text}"))
            .map_err(|error| TokenizerError::SentencePiece(error.to_string()))?;
        if marked
            .first()
            .is_some_and(|piece| piece.piece != WORD_BOUNDARY)
        {
            Ok(marked)
        } else {
            Ok(raw)
        }
    }
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<ModelPiece>,
    #[prost(message, optional, tag = "2")]
    trainer_spec: Option<TrainerSpec>,
    #[prost(message, optional, tag = "3")]
    normalizer_spec: Option<NormalizerSpec>,
}

#[derive(Clone, PartialEq, Message)]
struct ModelPiece {
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    #[prost(float, optional, tag = "2")]
    score: Option<f32>,
    #[prost(int32, optional, tag = "3")]
    piece_type: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct TrainerSpec {
    #[prost(int32, optional, tag = "3")]
    model_type: Option<i32>,
    #[prost(int32, optional, tag = "4")]
    vocab_size: Option<i32>,
    #[prost(int32, optional, tag = "40")]
    unk_id: Option<i32>,
    #[prost(int32, optional, tag = "41")]
    bos_id: Option<i32>,
    #[prost(int32, optional, tag = "42")]
    eos_id: Option<i32>,
    #[prost(int32, optional, tag = "43")]
    pad_id: Option<i32>,
}

#[derive(Clone, PartialEq, Message)]
struct NormalizerSpec {
    #[prost(string, optional, tag = "1")]
    name: Option<String>,
    #[prost(bool, optional, tag = "3")]
    add_dummy_prefix: Option<bool>,
    #[prost(bool, optional, tag = "4")]
    remove_extra_whitespaces: Option<bool>,
    #[prost(bool, optional, tag = "5")]
    escape_whitespaces: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CTCDecoder, DecoderConfig};

    fn piece(text: &str, score: f32, piece_type: i32) -> ModelPiece {
        ModelPiece {
            piece: Some(text.to_owned()),
            score: Some(score),
            piece_type: Some(piece_type),
        }
    }

    fn test_model() -> Vec<u8> {
        ModelProto {
            pieces: vec![
                piece("<unk>", 0.0, 2),
                piece(WORD_BOUNDARY, -10.0, 1),
                piece("l", -1.0, 1),
                piece("i", -1.0, 1),
                piece("v", -1.0, 1),
                piece("e", -1.0, 1),
                piece("▁live", 0.0, 1),
            ],
            trainer_spec: Some(TrainerSpec {
                model_type: Some(1),
                vocab_size: Some(7),
                unk_id: Some(0),
                bos_id: Some(-1),
                eos_id: Some(-1),
                pad_id: Some(-1),
            }),
            normalizer_spec: Some(NormalizerSpec {
                name: Some("identity".to_owned()),
                add_dummy_prefix: Some(false),
                remove_extra_whitespaces: Some(false),
                escape_whitespaces: Some(true),
            }),
        }
        .encode_to_vec()
    }

    #[test]
    fn tokenizes_with_exact_sentencepiece_ids_and_word_boundaries() {
        let model = test_model();
        let tokenizer =
            SentencePieceTokenizer::from_serialized_proto(&model, SentencePieceConfig::default())
                .unwrap();
        let contexts = vec!["live".to_owned()];
        let tokenization = tokenizer.tokenize(&contexts).unwrap();
        assert_eq!(tokenization.context_token_ids, vec![vec![6]]);
        assert_eq!(tokenization.word_boundary_token_ids, HashSet::from([1, 6]));

        let substring_tokenizer = SentencePieceTokenizer::from_serialized_proto(
            &model,
            SentencePieceConfig {
                add_word_boundary: false,
                ..SentencePieceConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            substring_tokenizer
                .tokenize(&contexts)
                .unwrap()
                .context_token_ids,
            vec![vec![2, 3, 4, 5]]
        );
    }

    #[test]
    fn remaps_ids_without_dropping_pieces_or_inventing_boundaries() {
        let model = test_model();
        let symbol_table = [
            ("<unk>", 1),
            (WORD_BOUNDARY, 2),
            ("l", 3),
            ("i", 4),
            ("v", 5),
            ("e", 6),
            ("▁live", 7),
        ]
        .into_iter()
        .map(|(piece, id)| (piece.to_owned(), id))
        .collect();
        let tokenizer = SentencePieceTokenizer::from_serialized_proto(
            &model,
            SentencePieceConfig {
                symbol_table,
                add_word_boundary: true,
            },
        )
        .unwrap();
        let contexts = vec!["live".to_owned()];
        let tokenization = tokenizer.tokenize(&contexts).unwrap();
        assert_eq!(tokenization.context_token_ids, vec![vec![7]]);
        assert_eq!(tokenization.word_boundary_token_ids, HashSet::from([2, 7]));

        let unknown_tokenizer = SentencePieceTokenizer::from_serialized_proto(
            &model,
            SentencePieceConfig {
                symbol_table: HashMap::from([("<unk>".to_owned(), 8)]),
                add_word_boundary: true,
            },
        )
        .unwrap();
        let unknown = unknown_tokenizer.tokenize(&contexts).unwrap();
        assert_eq!(unknown.context_token_ids, vec![vec![8]]);
        assert!(unknown.word_boundary_token_ids.is_empty());
    }

    #[test]
    fn reuses_loaded_model_across_tokenizer_and_decoder_copies() {
        let model = test_model();
        let model_path = std::env::temp_dir().join(format!(
            "asr-decoder-sentencepiece-{}.model",
            std::process::id()
        ));
        fs::write(&model_path, model).unwrap();
        let tokenizer =
            SentencePieceTokenizer::open(&model_path, SentencePieceConfig::default()).unwrap();
        fs::remove_file(&model_path).unwrap();

        let shared = tokenizer.clone();
        drop(tokenizer);
        assert_eq!(
            shared
                .tokenize(&["live".to_owned()])
                .unwrap()
                .context_token_ids,
            vec![vec![6]]
        );
        let decoder = CTCDecoder::new(DecoderConfig {
            contexts: vec!["live".to_owned()],
            sentencepiece_tokenizer: Some(shared),
            ..DecoderConfig::default()
        });
        assert!(decoder.is_ok());
    }

    #[test]
    fn rejects_missing_symbol_and_ambiguous_decoder_contexts() {
        let model = test_model();
        let tokenizer = SentencePieceTokenizer::from_serialized_proto(
            &model,
            SentencePieceConfig {
                symbol_table: HashMap::from([("l".to_owned(), 3)]),
                add_word_boundary: true,
            },
        )
        .unwrap();
        assert!(matches!(
            tokenizer.tokenize(&["live".to_owned()]),
            Err(TokenizerError::MissingPiece { .. })
        ));
        assert!(CTCDecoder::new(DecoderConfig {
            context_token_ids: vec![vec![1]],
            contexts: vec!["live".to_owned()],
            sentencepiece_tokenizer: Some(tokenizer),
            ..DecoderConfig::default()
        })
        .is_err());
    }
}
