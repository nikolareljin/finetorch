use anyhow::{anyhow, Result};

use crate::model::backend::LlmBackend;

pub mod backend;
pub mod llama_cpp;
pub mod lora;

#[cfg(feature = "backend-candle")]
pub mod candle;

pub fn build_backend(name: &str) -> Result<Box<dyn LlmBackend>> {
    match name {
        "llama_cpp" => Ok(Box::new(llama_cpp::LlamaCppBackend::default())),
        #[cfg(feature = "backend-candle")]
        "candle" => Ok(Box::new(candle::CandleBackend::default())),
        #[cfg(feature = "backend-candle")]
        "candle-cpu" => Ok(Box::new(candle::CandleBackend::default())),
        other => Err(anyhow!("unsupported backend: {other}")),
    }
}
