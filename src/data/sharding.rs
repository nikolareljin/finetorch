use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use crate::data::jsonl::TrainingExample;
use crate::data::tokenizer::TokenizerSpec;

#[derive(Debug, Clone)]
pub struct DatasetPrepConfig {
    pub tokenizer: TokenizerSpec,
    pub train_ratio: f32,
    pub shard_size: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetPrepSummary {
    pub train_examples: usize,
    pub val_examples: usize,
}

#[derive(Debug, Serialize)]
struct PreparedRecord<'a> {
    prompt: &'a str,
    completion: &'a str,
    prompt_tokens: usize,
    completion_tokens: usize,
}

pub fn prepare_dataset(
    examples: &[TrainingExample],
    output_dir: impl AsRef<Path>,
    config: &DatasetPrepConfig,
) -> Result<DatasetPrepSummary> {
    let output_dir = output_dir.as_ref();
    let train_dir = output_dir.join("train");
    let val_dir = output_dir.join("val");
    fs::create_dir_all(&train_dir)?;
    fs::create_dir_all(&val_dir)?;

    let mut rows = examples.to_vec();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    rows.shuffle(&mut rng);

    let split_at = ((rows.len() as f32) * config.train_ratio).round() as usize;
    let split_at = split_at.min(rows.len());
    let (train_rows, val_rows) = rows.split_at(split_at);

    write_split(train_rows, &train_dir, config)?;
    write_split(val_rows, &val_dir, config)?;

    let mut manifest = File::create(output_dir.join("manifest.json"))?;
    writeln!(
        manifest,
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "train_examples": train_rows.len(),
            "val_examples": val_rows.len(),
            "shard_size": config.shard_size,
        }))?
    )?;

    Ok(DatasetPrepSummary {
        train_examples: train_rows.len(),
        val_examples: val_rows.len(),
    })
}

fn write_split(rows: &[TrainingExample], dir: &Path, config: &DatasetPrepConfig) -> Result<()> {
    for (index, chunk) in rows.chunks(config.shard_size.max(1)).enumerate() {
        let shard_path = dir.join(format!("shard-{:05}.jsonl", index));
        let mut file = File::create(shard_path)?;
        for row in chunk {
            let tokenized = config.tokenizer.tokenize(row);
            let prepared = PreparedRecord {
                prompt: &row.prompt,
                completion: &row.completion,
                prompt_tokens: tokenized.prompt_tokens.len(),
                completion_tokens: tokenized.completion_tokens.len(),
            };
            writeln!(file, "{}", serde_json::to_string(&prepared)?)?;
        }
    }
    Ok(())
}
