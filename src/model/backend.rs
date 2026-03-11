use anyhow::Result;

use crate::model::lora::LoraConfig;

#[derive(Debug, Clone)]
pub struct ForwardBatch {
    pub token_batches: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct ForwardOutput {
    pub loss: Option<f32>,
    pub logits_shape: Vec<usize>,
}

pub trait LlmBackend {
    fn load_model(&mut self, base_path: &str) -> Result<()>;
    fn apply_lora(&mut self, config: &LoraConfig) -> Result<()>;
    fn forward(&mut self, batch: &ForwardBatch) -> Result<ForwardOutput>;
    fn save_adapters(&self, output_path: &str) -> Result<()>;
}
