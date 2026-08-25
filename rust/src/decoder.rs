use std::collections::HashSet;

use crate::context_graph::ContextGraph;
use crate::prefix_score::PrefixScore;
use crate::{
    ContextPolicy, DecoderError, DecodingQuality, HotwordStrength, LogProbabilities,
    SentencePieceTokenizer, Timestamp,
};

#[derive(Clone, Debug, Default)]
pub struct DecoderConfig {
    pub context_token_ids: Vec<Vec<usize>>,
    pub contexts: Vec<String>,
    pub sentencepiece_tokenizer: Option<SentencePieceTokenizer>,
    pub context_policy: Option<ContextPolicy>,
    pub hotword_strength: Option<HotwordStrength>,
    pub decoding_quality: DecodingQuality,
    pub blank_id: usize,
    pub frame_shift_ms: Option<f64>,
    pub word_boundary_token_ids: HashSet<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixBeamSearchOptions {
    pub beam_size: Option<usize>,
    pub token_beam_size: Option<usize>,
    pub finalize: bool,
    pub return_token_probabilities: bool,
    pub return_scores: bool,
    pub return_gating_diagnostics: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodeMode {
    Greedy,
    PrefixBeam,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SearchConfig {
    pub beam_size: usize,
    pub token_beam_size: usize,
    pub token_prune_threshold: f64,
    pub vocabulary_size: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SearchDiagnostics {
    pub gating_acoustic_scale_sum: f64,
    pub gating_factor_sum: f64,
    pub gating_frames: usize,
    pub nonblank_gating_frames: usize,
    pub evaluated_context_candidates: usize,
    pub gathered_context_candidates: usize,
    pub max_context_candidates_per_frame: usize,
    pub context_candidate_limit_hits: usize,
    pub boundary_suppressed_context_candidates: usize,
}

/// Stateful streaming decoder. Call `reset` or finalize a search before
/// switching between greedy and prefix beam search.
#[derive(Clone, Debug)]
pub struct CTCDecoder {
    pub(crate) context_policy: ContextPolicy,
    pub(crate) context_graph: Option<ContextGraph>,
    pub(crate) context_max_token_id: Option<usize>,
    pub(crate) decoding_quality: DecodingQuality,
    pub(crate) blank_id: usize,
    pub(crate) frame_shift_ms: Option<f64>,
    pub(crate) word_boundary_token_ids: HashSet<usize>,
    pub(crate) processed_frames: usize,
    pub(crate) diagnostics: SearchDiagnostics,
    pub(crate) search_config: Option<SearchConfig>,
    pub(crate) decode_mode: Option<DecodeMode>,
    pub(crate) greedy_tokens: Vec<usize>,
    pub(crate) greedy_spans: Vec<(usize, usize)>,
    pub(crate) greedy_token_log_probabilities: Vec<f64>,
    pub(crate) last_greedy_token: Option<usize>,
    pub(crate) hypotheses: Vec<(Vec<usize>, PrefixScore)>,
}

impl CTCDecoder {
    pub fn new(mut config: DecoderConfig) -> Result<Self, DecoderError> {
        if !config.contexts.is_empty() && !config.context_token_ids.is_empty() {
            return Err(DecoderError::InvalidConfig(
                "provide contexts or context_token_ids, not both",
            ));
        }
        if !config.contexts.is_empty() {
            let tokenizer =
                config
                    .sentencepiece_tokenizer
                    .as_ref()
                    .ok_or(DecoderError::InvalidConfig(
                        "sentencepiece_tokenizer is required when contexts are provided",
                    ))?;
            let tokenization = tokenizer.tokenize(&config.contexts)?;
            config.context_token_ids = tokenization.context_token_ids;
            config
                .word_boundary_token_ids
                .extend(tokenization.word_boundary_token_ids);
        }
        if config.context_policy.is_some() && config.hotword_strength.is_some() {
            return Err(DecoderError::InvalidConfig(
                "provide context_policy or hotword_strength, not both",
            ));
        }
        if config.word_boundary_token_ids.contains(&config.blank_id) {
            return Err(DecoderError::InvalidConfig(
                "word_boundary_token_ids must not include blank_id",
            ));
        }
        if let Some(frame_shift_ms) = config.frame_shift_ms {
            if !frame_shift_ms.is_finite() || frame_shift_ms <= 0.0 {
                return Err(DecoderError::InvalidConfig(
                    "frame_shift_ms must be finite and greater than zero",
                ));
            }
        }
        if config
            .context_token_ids
            .iter()
            .flatten()
            .any(|&token| token == config.blank_id)
        {
            return Err(DecoderError::InvalidConfig(
                "context token IDs must not contain blank_id",
            ));
        }
        let context_policy = config.context_policy.unwrap_or_else(|| {
            ContextPolicy::for_strength(config.hotword_strength.unwrap_or_default())
        });
        context_policy.validate()?;
        let validated_graph =
            ContextGraph::from_token_ids(&config.context_token_ids, &context_policy);
        let context_max_token_id = validated_graph.as_ref().map(|graph| graph.max_token_id);
        let context_graph = if context_policy.completion_bonus > 0.0 {
            validated_graph
        } else {
            None
        };
        let mut decoder = Self {
            context_policy,
            context_graph,
            context_max_token_id,
            decoding_quality: config.decoding_quality,
            blank_id: config.blank_id,
            frame_shift_ms: config.frame_shift_ms,
            word_boundary_token_ids: config.word_boundary_token_ids,
            processed_frames: 0,
            diagnostics: SearchDiagnostics::default(),
            search_config: None,
            decode_mode: None,
            greedy_tokens: Vec::new(),
            greedy_spans: Vec::new(),
            greedy_token_log_probabilities: Vec::new(),
            last_greedy_token: None,
            hypotheses: Vec::new(),
        };
        decoder.reset();
        Ok(decoder)
    }

    pub fn reset(&mut self) {
        self.processed_frames = 0;
        self.diagnostics = SearchDiagnostics::default();
        self.search_config = None;
        self.decode_mode = None;
        self.greedy_tokens.clear();
        self.greedy_spans.clear();
        self.greedy_token_log_probabilities.clear();
        self.last_greedy_token = None;
        self.hypotheses = vec![(Vec::new(), PrefixScore::initial())];
    }

    pub fn create_stream(&self) -> Self {
        let mut stream = self.clone();
        stream.reset();
        stream
    }

    pub fn frame_shift_ms(&self) -> Option<f64> {
        self.frame_shift_ms
    }

    pub(crate) fn validate_input(&self, input: LogProbabilities<'_>) -> Result<(), DecoderError> {
        if self.blank_id >= input.vocabulary_size() {
            return Err(DecoderError::InvalidInput(
                "blank_id must be smaller than vocabulary_size",
            ));
        }
        if self
            .context_max_token_id
            .is_some_and(|token| token >= input.vocabulary_size())
        {
            return Err(DecoderError::InvalidInput(
                "context token IDs must be smaller than vocabulary_size",
            ));
        }
        for frame in 0..input.frames() {
            let row = input.frame(frame);
            if row
                .iter()
                .any(|value| value.is_nan() || *value == f32::INFINITY)
                || !row.iter().any(|value| value.is_finite())
            {
                return Err(DecoderError::InvalidInput(
                    "every frame must contain at least one finite log probability",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn lock_decode_mode(&mut self, mode: DecodeMode) -> Result<(), DecoderError> {
        match self.decode_mode {
            None => self.decode_mode = Some(mode),
            Some(current) if current == mode => {}
            Some(_) => return Err(DecoderError::DecodeModeChanged),
        }
        Ok(())
    }

    pub(crate) fn timestamps(&self, spans: &[(usize, usize)]) -> Vec<Timestamp> {
        spans
            .iter()
            .map(|&(start, end)| Timestamp {
                start_frame: start,
                end_frame: end,
                start_ms: self.frame_shift_ms.map(|shift| start as f64 * shift),
                end_ms: self.frame_shift_ms.map(|shift| end as f64 * shift),
            })
            .collect()
    }
}
