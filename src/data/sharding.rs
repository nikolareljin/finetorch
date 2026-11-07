use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::data::jsonl::TrainingExample;
use crate::data::tokenizer::TokenizerSpec;

pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub tokenizer: String,
    pub train_ratio: f32,
    pub shard_size: usize,
    pub train_examples: usize,
    pub val_examples: usize,
    pub train_shards: usize,
    pub val_shards: usize,
    pub max_prompt_tokens: usize,
    pub max_completion_tokens: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct PreparedRecord {
    prompt: String,
    completion: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

pub fn prepare_dataset(
    examples: &[TrainingExample],
    output_dir: impl AsRef<Path>,
    config: &DatasetPrepConfig,
) -> Result<DatasetPrepSummary> {
    if examples.is_empty() {
        return Err(anyhow::anyhow!("empty dataset: no training examples"));
    }

    let output_dir = output_dir.as_ref();
    let train_dir = output_dir.join("train");
    let val_dir = output_dir.join("val");
    fs::create_dir_all(&train_dir)?;
    fs::create_dir_all(&val_dir)?;

    let tokenizer = config.tokenizer.resolve()?;
    let mut rows = examples.to_vec();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    rows.shuffle(&mut rng);

    let split_at = ((rows.len() as f32) * config.train_ratio).round() as usize;
    let split_at = split_at.min(rows.len());
    let split_at = if rows.is_empty() {
        0
    } else {
        split_at.max(1).min(rows.len())
    };
    let (train_rows, val_rows) = rows.split_at(split_at);

    let train_shards = write_split(train_rows, &train_dir, config, &tokenizer)?;
    let val_shards = write_split(val_rows, &val_dir, config, &tokenizer)?;

    let max_prompt_tokens = examples
        .iter()
        .map(|example| tokenizer.tokenize(example).prompt_tokens.len())
        .max()
        .unwrap_or(0);
    let max_completion_tokens = examples
        .iter()
        .map(|example| tokenizer.tokenize(example).completion_tokens.len())
        .max()
        .unwrap_or(0);

    let manifest = DatasetManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        tokenizer: format!("{:?}", config.tokenizer),
        train_ratio: config.train_ratio,
        shard_size: config.shard_size,
        train_examples: train_rows.len(),
        val_examples: val_rows.len(),
        train_shards,
        val_shards,
        max_prompt_tokens,
        max_completion_tokens,
    };

    let mut file = File::create(output_dir.join("manifest.json"))?;
    write!(file, "{}", serde_json::to_string_pretty(&manifest)?)?;

    Ok(DatasetPrepSummary {
        train_examples: train_rows.len(),
        val_examples: val_rows.len(),
    })
}

fn write_split(
    rows: &[TrainingExample],
    dir: &Path,
    config: &DatasetPrepConfig,
    tokenizer: &crate::data::tokenizer::ResolvedTokenizer,
) -> Result<usize> {
    if dir.exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                fs::remove_file(path)?;
            }
        }
    }
    let shard_size = config.shard_size.max(1);
    let chunks: Vec<&[TrainingExample]> = if rows.is_empty() {
        vec![]
    } else {
        rows.chunks(shard_size).collect()
    };
    for (index, chunk) in chunks.iter().enumerate() {
        let shard_path = dir.join(format!("shard-{:05}.jsonl", index));
        let mut file = File::create(shard_path)?;
        for row in *chunk {
            let tokenized = tokenizer.tokenize(row);
            let prepared = PreparedRecord {
                prompt: row.prompt.clone(),
                completion: row.completion.clone(),
                prompt_tokens: tokenized.prompt_tokens.len(),
                completion_tokens: tokenized.completion_tokens.len(),
            };
            writeln!(file, "{}", serde_json::to_string(&prepared)?)?;
        }
    }
    Ok(chunks.len())
}

// simpler shard count fix below by actual count not used; override
fn _shard_count_placeholder() {}

pub fn validate_dataset_dir(dir: impl AsRef<Path>) -> Result<DatasetManifest> {
    let dir = dir.as_ref();
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(anyhow::anyhow!("bad manifest: missing manifest.json"));
    }
    let raw = fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_str(&raw).map_err(|err| anyhow::anyhow!("bad manifest: {err}"))?;
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(anyhow::anyhow!(
            "bad manifest: unsupported schema version {}",
            manifest.schema_version
        ));
    }

    let train_dir = dir.join("train");
    let val_dir = dir.join("val");
    if !train_dir.is_dir() {
        return Err(anyhow::anyhow!("bad manifest: missing train directory"));
    }
    if !val_dir.is_dir() {
        return Err(anyhow::anyhow!("bad manifest: missing val directory"));
    }

    let train_shards = count_shards(&train_dir);
    let val_shards = count_shards(&val_dir);
    if train_shards != manifest.train_shards {
        return Err(anyhow::anyhow!(
            "bad manifest: train shard count does not match"
        ));
    }
    if val_shards != manifest.val_shards {
        return Err(anyhow::anyhow!(
            "bad manifest: val shard count does not match"
        ));
    }

    let train_examples = count_examples(&train_dir)?;
    let val_examples = count_examples(&val_dir)?;
    if train_examples != manifest.train_examples {
        return Err(anyhow::anyhow!(
            "bad manifest: train example count does not match"
        ));
    }
    if val_examples != manifest.val_examples {
        return Err(anyhow::anyhow!(
            "bad manifest: val example count does not match"
        ));
    }

    Ok(manifest)
}

pub fn count_shards(dir: &Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("shard-") && n.ends_with(".jsonl"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

pub fn count_examples(dir: &Path) -> Result<usize> {
    let mut total = 0;
    if !dir.exists() {
        return Ok(0);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    total += 1;
                }
            }
        }
    }
    Ok(total)
}

pub fn read_shards(dir: impl AsRef<Path>) -> Result<Vec<TrainingExample>> {
    let dir = dir.as_ref();
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("jsonl"))
        .collect();
    entries.sort();
    for path in entries {
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: PreparedRecord = serde_json::from_str(&line)
                .with_context(|| format!("parsing {}", path.display()))?;
            out.push(TrainingExample {
                prompt: record.prompt,
                completion: record.completion,
            });
        }
    }
    Ok(out)
}
