use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::data::jsonl::read_examples;
use crate::data::sharding::{prepare_dataset, DatasetPrepConfig};
use crate::data::tokenizer::TokenizerSpec;

#[derive(Debug, Args)]
pub struct PrepareDatasetArgs {
    #[arg(long)]
    pub input: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value = "whitespace")]
    pub tokenizer: String,
    #[arg(long, default_value_t = 0.95)]
    pub train_ratio: f32,
    #[arg(long, default_value_t = 2048)]
    pub shard_size: usize,
}

pub fn run(args: PrepareDatasetArgs) -> Result<()> {
    let examples = read_examples(&args.input)?;
    let config = DatasetPrepConfig {
        tokenizer: TokenizerSpec::from_cli(&args.tokenizer),
        train_ratio: args.train_ratio,
        shard_size: args.shard_size,
    };
    let summary = prepare_dataset(&examples, &args.output, &config)?;
    println!(
        "prepared dataset: train_examples={} val_examples={} output={}",
        summary.train_examples,
        summary.val_examples,
        args.output.display()
    );
    Ok(())
}
