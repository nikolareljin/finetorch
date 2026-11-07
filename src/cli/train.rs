use anyhow::Result;
use clap::Args;

use crate::config::RunConfig;
use crate::model::build_backend;
use crate::train::r#loop::run_training;

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long)]
    pub config: String,
    #[arg(long)]
    pub resume_from: Option<String>,
}

pub fn run(args: TrainArgs) -> Result<()> {
    let config = RunConfig::from_file(&args.config)?;
    let mut backend = build_backend(&config.model.backend)?;
    let artifacts = run_training(&config, backend.as_mut(), args.resume_from.as_deref())?;
    println!(
        "training scaffold complete: adapters={} logs={}",
        artifacts.adapter_path.display(),
        artifacts.log_path.display()
    );
    Ok(())
}
