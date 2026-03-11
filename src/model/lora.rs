use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub use_qlora: bool,
    pub r: usize,
    pub alpha: usize,
    pub dropout: f32,
    #[serde(default)]
    pub target_modules: Vec<String>,
}

fn default_true() -> bool {
    true
}
