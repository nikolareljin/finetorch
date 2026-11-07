use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;

use crate::config::RunConfig;
use crate::data::jsonl::read_examples;
use crate::data::tokenizer::TokenizerSpec;
use crate::eval::metrics;
use crate::model::backend::ForwardBatch;
use crate::model::build_backend;

#[derive(Debug, clap::Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub config: String,
    #[arg(long)]
    pub dataset: PathBuf,
    #[arg(long)]
    pub compare_base: bool,
    #[arg(long)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalSummary {
    pub examples: usize,
    pub perplexity: f32,
    pub exact_match: f32,
    pub bleu: f32,
    pub rouge_l: f32,
    pub base: Option<Box<EvalSummary>>,
}

pub fn run(args: EvalArgs) -> Result<()> {
    let config = RunConfig::from_file(&args.config)?;
    let examples = read_examples(&args.dataset)?;
    let summary = evaluate(&config, &examples, config.lora.enabled)?;

    let base_summary = if args.compare_base {
        Some(Box::new(evaluate(&config, &examples, false)?))
    } else {
        None
    };

    let full = EvalSummary {
        examples: summary.examples,
        perplexity: summary.perplexity,
        exact_match: summary.exact_match,
        bleu: summary.bleu,
        rouge_l: summary.rouge_l,
        base: base_summary,
    };

    print_summary(&full);

    let out_dir = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(&config.training.output_dir));
    fs::create_dir_all(&out_dir)?;
    fs::write(
        out_dir.join("eval_summary.json"),
        serde_json::to_string_pretty(&full)?,
    )?;

    Ok(())
}

fn evaluate(
    config: &RunConfig,
    examples: &[crate::data::jsonl::TrainingExample],
    lora_enabled: bool,
) -> Result<EvalSummary> {
    let mut cfg = config.clone();
    cfg.lora.enabled = lora_enabled;
    let mut backend = build_backend(&cfg.model.backend)?;
    backend.load_model(&cfg.model.base_path)?;
    if cfg.lora.enabled {
        let _ = backend.apply_lora(&cfg.lora);
    }

    let tokenizer = TokenizerSpec::from_cli(&cfg.model.tokenizer).resolve()?;

    // perplexity
    let mut loss_sum = 0.0f32;
    let mut loss_count = 0usize;
    for example in examples {
        let prompt_tokens = tokenizer.tokenize(example).prompt_tokens;
        let completion_tokens = tokenizer.tokenize(example).completion_tokens;
        if completion_tokens.is_empty() {
            continue;
        }
        let mut seq = prompt_tokens.clone();
        seq.extend_from_slice(&completion_tokens);
        let mut labels = vec![-100i64; prompt_tokens.len()];
        for token in &completion_tokens {
            labels.push(*token as i64);
        }
        let max_seq = cfg.data.max_seq_len;
        if seq.len() > max_seq {
            if cfg.data.truncation {
                seq.truncate(max_seq);
                labels.truncate(max_seq);
            } else {
                continue;
            }
        }
        let batch = ForwardBatch {
            input_ids: vec![seq],
            attention_mask: vec![vec![1u8; labels.len()]],
            labels: vec![labels],
        };
        if let Some(loss) = backend.forward(&batch)?.loss {
            loss_sum += loss;
            loss_count += 1;
        }
    }
    let perplexity = if loss_count > 0 {
        (loss_sum as f64 / loss_count as f64).exp() as f32
    } else {
        1.0
    };

    // text metrics
    let mut em_sum = 0.0f32;
    let mut bleu_sum = 0.0f32;
    let mut rouge_sum = 0.0f32;
    let mut n = 0usize;
    for example in examples {
        let prompt_ids = tokenizer.encode(&example.prompt);
        let completion_ids = tokenizer.encode(&example.completion);
        let max_new = completion_ids.len().clamp(1, 64);
        let generated = backend
            .generate(&prompt_ids, max_new)
            .unwrap_or_else(|_| prompt_ids.clone());
        let new_tokens = if generated.len() > prompt_ids.len() {
            &generated[prompt_ids.len()..]
        } else {
            &[]
        };
        let pred_text = tokenizer.decode(new_tokens);
        let em = if pred_text.trim() == example.completion.trim() {
            1.0
        } else {
            0.0
        };
        let bleu = metrics::bleu_score(&pred_text, &example.completion);
        let rouge = metrics::rouge_l_score(&pred_text, &example.completion);
        em_sum += em;
        bleu_sum += bleu;
        rouge_sum += rouge;
        n += 1;
    }
    let examples_count = examples.len();
    let denom = if n == 0 { 1 } else { n } as f32;
    Ok(EvalSummary {
        examples: examples_count,
        perplexity,
        exact_match: em_sum / denom,
        bleu: bleu_sum / denom,
        rouge_l: rouge_sum / denom,
        base: None,
    })
}

fn print_summary(summary: &EvalSummary) {
    println!(
        "eval complete: examples={} perplexity={:.3} exact_match={:.3} bleu={:.3} rouge_l={:.3}",
        summary.examples, summary.perplexity, summary.exact_match, summary.bleu, summary.rouge_l
    );
    if let Some(base) = &summary.base {
        println!(
            "base comparison: perplexity={:.3} exact_match={:.3} bleu={:.3} rouge_l={:.3}",
            base.perplexity, base.exact_match, base.bleu, base.rouge_l
        );
    }
}
