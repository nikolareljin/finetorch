use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::config::RunConfig;
use crate::data::jsonl::read_examples;
use crate::eval::metrics::MetricSummary;
use crate::model::build_backend;

#[derive(Debug, Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: String,
    #[arg(long)]
    pub dataset: PathBuf,
}

pub fn run(args: EvalArgs) -> Result<()> {
    let config = RunConfig::from_file(&args.config)?;
    let examples = read_examples(&args.dataset)?;
    let mut backend = build_backend(&config.model.backend)?;
    backend.load_model(&config.model.base_path)?;

    let summary = MetricSummary::from_examples(&examples);
    println!(
        "eval scaffold complete: examples={} perplexity={:.3} exact_match={:.3} bleu={:.3} rouge_l={:.3}",
        examples.len(),
        summary.perplexity,
        summary.exact_match,
        summary.bleu,
        summary.rouge_l
    );
    Ok(())
}
