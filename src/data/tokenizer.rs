use crate::data::jsonl::TrainingExample;

#[derive(Debug, Clone)]
pub enum TokenizerSpec {
    Whitespace,
    Named(String),
}

#[derive(Debug, Clone)]
pub struct TokenizedExample {
    pub prompt_tokens: Vec<u32>,
    pub completion_tokens: Vec<u32>,
}

impl TokenizerSpec {
    pub fn from_cli(raw: &str) -> Self {
        match raw.trim() {
            "" | "whitespace" => Self::Whitespace,
            value => Self::Named(value.to_owned()),
        }
    }

    pub fn tokenize(&self, example: &TrainingExample) -> TokenizedExample {
        match self {
            Self::Whitespace => TokenizedExample {
                prompt_tokens: whitespace_tokenize(&example.prompt),
                completion_tokens: whitespace_tokenize(&example.completion),
            },
            Self::Named(_name) => TokenizedExample {
                prompt_tokens: whitespace_tokenize(&example.prompt),
                completion_tokens: whitespace_tokenize(&example.completion),
            },
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
