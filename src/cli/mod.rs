use anyhow::Result;
use clap::{Parser, Subcommand};

pub mod eval;
pub mod prepare;
pub mod train;

#[derive(Debug, Parser)]
#[command(
    name = "finetorch",
    version,
    about = "Rust-native LLM finetuning toolkit"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    PrepareDataset(prepare::PrepareDatasetArgs),
    Train(train::TrainArgs),
    Eval(eval::EvalArgs),
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::PrepareDataset(args) => prepare::run(args),
        Commands::Train(args) => train::run(args),
        Commands::Eval(args) => eval::run(args),
    }
}
