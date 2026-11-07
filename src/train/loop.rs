use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use crate::config::RunConfig;
use crate::data::loader::{build_batches, BatchingConfig};
use crate::data::sharding::{read_shards, validate_dataset_dir};
use crate::data::tokenizer::TokenizerSpec;
use crate::model::backend::{ForwardBatch, LlmBackend};

#[derive(Debug, Clone)]
pub struct TrainingArtifacts {
    pub adapter_path: PathBuf,
    pub log_path: PathBuf,
}

pub fn run_training(config: &RunConfig, backend: &mut dyn LlmBackend) -> Result<TrainingArtifacts> {
    backend.load_model(&config.model.base_path)?;
    if config.lora.enabled {
        backend.apply_lora(&config.lora)?;
    }
    if !backend.training_supported() {
        return Err(anyhow!("training not supported"));
    }
    backend.set_training_config(&config.training);

    let tokenizer = TokenizerSpec::from_cli(&config.model.tokenizer).resolve()?;

    let train_manifest_dir = if std::path::Path::new(&config.data.train_path)
        .join("manifest.json")
        .exists()
    {
        std::path::PathBuf::from(&config.data.train_path)
    } else {
        std::path::Path::new(&config.data.train_path)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(&config.data.train_path))
    };
    validate_dataset_dir(&train_manifest_dir)?;
    let examples = read_shards(&config.data.train_path)?;
    let batches = build_batches(
        &examples,
        &tokenizer,
        &BatchingConfig {
            batch_size: config.training.batch_size,
            max_seq_len: config.data.max_seq_len,
            shuffle: config.data.shuffle,
            seed: config.training.seed,
            truncation: config.data.truncation,
        },
    )?;

    if batches.is_empty() {
        return Err(anyhow!("empty dataset: no training batches"));
    }

    let output_dir = PathBuf::from(&config.training.output_dir);
    fs::create_dir_all(&output_dir)?;

    let log_path = output_dir.join("train_metrics.jsonl");
    let adapter_path = output_dir.join("adapter.safetensors");
    let mut log_file = File::create(&log_path)?;

    let mut global_step = 0usize;
    let total = config.training.max_steps;
    while global_step < total {
        let batch_idx = global_step % batches.len();
        let batch = &batches[batch_idx];
        let forward_batch = ForwardBatch {
            input_ids: batch.input_ids.clone(),
            attention_mask: batch.attention_mask.clone(),
            labels: batch.labels.clone(),
        };
        let loss = backend.train_step(&forward_batch, global_step)?;
        writeln!(
            log_file,
            "{}",
            serde_json::to_string(&serde_json::json!({
                "step": global_step,
                "loss": loss,
                "device": config.training.device,
                "gradient_accumulation": config.training.gradient_accumulation,
            }))?
        )?;
        global_step += 1;
    }

    backend.save_adapters(adapter_path.to_str().unwrap_or("adapter.safetensors"))?;

    Ok(TrainingArtifacts {
        adapter_path,
        log_path,
    })
}
