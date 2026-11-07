use anyhow::Result;

use crate::model::lora::LoraConfig;

#[derive(Debug, Clone)]
pub struct ForwardBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub labels: Vec<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    pub loss: Option<f32>,
    pub logits_shape: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    pub supports_lora: bool,
    pub supports_qlora: bool,
    pub supports_training: bool,
    pub supports_generation: bool,
    pub supported_devices: Vec<String>,
}

use crate::config::TrainingConfig;

pub trait LlmBackend {
    fn load_model(&mut self, base_path: &str) -> Result<()>;
    fn apply_lora(&mut self, config: &LoraConfig) -> Result<()>;
    fn forward(&mut self, batch: &ForwardBatch) -> Result<ForwardOutput>;
    fn save_adapters(&self, output_path: &str) -> Result<()>;

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    fn training_supported(&self) -> bool {
        self.capabilities().supports_training
    }

    fn set_training_config(&mut self, _config: &TrainingConfig) {}

    fn train_step(&mut self, _batch: &ForwardBatch, _global_step: usize) -> Result<f32> {
        Err(anyhow::anyhow!("training not supported"))
    }

    fn save_checkpoint(&self, _dir: &str, _step: usize, _config_hash: &str) -> Result<()> {
        Err(anyhow::anyhow!("training not supported"))
    }

    fn load_checkpoint(&mut self, _dir: &str, _expected_hash: &str) -> Result<usize> {
        Err(anyhow::anyhow!("training not supported"))
    }

    fn generate(&self, _prompt: &[u32], _max_new_tokens: usize) -> Result<Vec<u32>> {
        Err(anyhow::anyhow!("generation not supported"))
    }
}
