use std::path::Path;

use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

use crate::data::jsonl::TrainingExample;

#[derive(Debug, Clone)]
pub enum TokenizerSpec {
    Whitespace,
    HfPath(String),
}

#[derive(Debug, Clone)]
pub enum ResolvedTokenizer {
    Whitespace,
    Hf(Box<HfTokenizer>),
}

#[derive(Debug, Clone)]
pub struct TokenizedExample {
    pub prompt_tokens: Vec<u32>,
    pub completion_tokens: Vec<u32>,
}

impl TokenizerSpec {
    pub fn from_cli(raw: &str) -> Self {
        let value = raw.trim();
        match value {
            "" | "whitespace" => Self::Whitespace,
            _ if value.contains(':') => {
                let path = value.split_once(':').map(|(_, p)| p).unwrap_or(value);
                if path.is_empty() {
                    Self::Whitespace
                } else {
                    Self::HfPath(path.to_owned())
                }
            }
            _ => Self::HfPath(value.to_owned()),
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(anyhow::anyhow!("missing tokenizer: {}", path.display()));
        }
        Ok(Self::HfPath(path.display().to_string()))
    }

    pub fn resolve(&self) -> Result<ResolvedTokenizer> {
        match self {
            Self::Whitespace => Ok(ResolvedTokenizer::Whitespace),
            Self::HfPath(path) => {
                let path = Path::new(path);
                if !path.exists() {
                    return Err(anyhow::anyhow!("missing tokenizer: {}", path.display()));
                }
                let tokenizer = HfTokenizer::from_file(path)
                    .map_err(|err| anyhow::anyhow!("missing tokenizer: {err}"))?;
                Ok(ResolvedTokenizer::Hf(Box::new(tokenizer)))
            }
        }
    }

    pub fn tokenize(&self, example: &TrainingExample) -> TokenizedExample {
        let tokenizer = self.resolve().unwrap_or(ResolvedTokenizer::Whitespace);
        tokenizer.tokenize(example)
    }
}

impl ResolvedTokenizer {
    pub fn tokenize(&self, example: &TrainingExample) -> TokenizedExample {
        match self {
            Self::Whitespace => TokenizedExample {
                prompt_tokens: whitespace_tokenize(&example.prompt),
                completion_tokens: whitespace_tokenize(&example.completion),
            },
            Self::Hf(tokenizer) => {
                let prompt = tokenizer
                    .encode(example.prompt.as_str(), false)
                    .map(|enc| enc.get_ids().to_vec())
                    .unwrap_or_default();
                let completion = tokenizer
                    .encode(example.completion.as_str(), false)
                    .map(|enc| enc.get_ids().to_vec())
                    .unwrap_or_default();
                TokenizedExample {
                    prompt_tokens: prompt,
                    completion_tokens: completion,
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Whitespace => whitespace_tokenize(text),
            Self::Hf(tokenizer) => tokenizer
                .encode(text, false)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default(),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::Whitespace => ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" "),
            Self::Hf(tokenizer) => tokenizer.decode(ids, true).unwrap_or_default(),
        }
    }
}

fn whitespace_tokenize(text: &str) -> Vec<u32> {
    text.split_whitespace().map(stable_token_id).collect()
}

fn stable_token_id(piece: &str) -> u32 {
    let mut hash = 2166136261u32;
    for byte in piece.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}
