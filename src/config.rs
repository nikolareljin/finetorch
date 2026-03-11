use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::model::lora::LoraConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub model: ModelConfig,
    pub lora: LoraConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub base_path: String,
    pub backend: String,
    #[serde(default)]
    pub tokenizer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub device: String,
    pub batch_size: usize,
    pub gradient_accumulation: usize,
    pub lr: f64,
    pub epochs: usize,
    pub max_steps: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default)]
    pub weight_decay: f64,
    #[serde(default)]
    pub seed: u64,
    pub output_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub train_path: String,
    pub val_path: String,
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(default = "default_true")]
    pub shuffle: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub metrics: Vec<String>,
}

impl RunConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }
}

fn default_max_seq_len() -> usize {
    2048
}

fn default_true() -> bool {
    true
}
