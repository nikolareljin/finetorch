use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::model::backend::{ForwardBatch, ForwardOutput, LlmBackend};
use crate::model::lora::LoraConfig;

#[derive(Debug, Default)]
pub struct LlamaCppBackend {
    base_path: Option<PathBuf>,
    lora: Option<LoraConfig>,
}

impl LlmBackend for LlamaCppBackend {
    fn load_model(&mut self, base_path: &str) -> Result<()> {
        self.base_path = Some(PathBuf::from(base_path));
        Ok(())
    }

    fn apply_lora(&mut self, config: &LoraConfig) -> Result<()> {
        self.lora = Some(config.clone());
        Ok(())
    }

    fn forward(&mut self, batch: &ForwardBatch) -> Result<ForwardOutput> {
        if self.base_path.is_none() {
            return Err(anyhow!("model must be loaded before forward"));
        }
        Ok(ForwardOutput {
            loss: Some((batch.token_batches.len() as f32).max(1.0) / 100.0),
            logits_shape: vec![
                batch.token_batches.len(),
                batch
                    .token_batches
                    .first()
                    .map(|b| b.len())
                    .unwrap_or_default(),
                4096,
            ],
        })
    }

    fn save_adapters(&self, output_path: &str) -> Result<()> {
        fs::write(output_path, b"placeholder adapter weights")?;
        Ok(())
    }
}
