use anyhow::{anyhow, Result};

use crate::model::backend::LlmBackend;

pub mod backend;
pub mod llama_cpp;
pub mod lora;

pub fn build_backend(name: &str) -> Result<Box<dyn LlmBackend>> {
    match name {
        "llama_cpp" => Ok(Box::new(llama_cpp::LlamaCppBackend::default())),
        other => Err(anyhow!("unsupported backend: {other}")),
    }
}
