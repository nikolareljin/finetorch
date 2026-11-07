use anyhow::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::data::jsonl::TrainingExample;
use crate::data::tokenizer::ResolvedTokenizer;

#[derive(Debug, Clone)]
pub struct PreparedBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub labels: Vec<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct BatchingConfig {
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub shuffle: bool,
    pub seed: u64,
    pub truncation: bool,
}

pub fn build_batches(
    examples: &[TrainingExample],
    tokenizer: &ResolvedTokenizer,
    config: &BatchingConfig,
) -> Result<Vec<PreparedBatch>> {
    if examples.is_empty() {
        return Err(anyhow::anyhow!("empty dataset: no training examples"));
    }

    let batch_size = config.batch_size.max(1);
    let mut sequences: Vec<(Vec<u32>, Vec<i64>)> = Vec::with_capacity(examples.len());

    for example in examples {
        let prompt_tokens = tokenizer.tokenize(example).prompt_tokens;
        let completion_tokens = tokenizer.tokenize(example).completion_tokens;

        let mut input_ids = prompt_tokens.clone();
        input_ids.extend_from_slice(&completion_tokens);

        let mut labels = vec![-100i64; prompt_tokens.len()];
        for token in &completion_tokens {
            labels.push(*token as i64);
        }

        if input_ids.len() > config.max_seq_len {
            if config.truncation {
                input_ids.truncate(config.max_seq_len);
                labels.truncate(config.max_seq_len);
            } else {
                return Err(anyhow::anyhow!(
                    "over-long sequence: {} tokens exceeds max_seq_len {}",
                    input_ids.len(),
                    config.max_seq_len
                ));
            }
        }

        // Ensure at least one token.
        if input_ids.is_empty() {
            return Err(anyhow::anyhow!("empty sequence after tokenization"));
        }

        sequences.push((input_ids, labels));
    }

    if config.shuffle {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        sequences.shuffle(&mut rng);
    }

    let mut batches = Vec::new();
    for chunk in sequences.chunks(batch_size) {
        let max_len = chunk.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(chunk.len());
        let mut attention_mask = Vec::with_capacity(chunk.len());
        let mut labels = Vec::with_capacity(chunk.len());

        for (ids, labs) in chunk {
            let mut padded_ids = ids.clone();
            let mut mask = vec![1u8; ids.len()];
            let mut padded_labels = labs.clone();

            while padded_ids.len() < max_len {
                padded_ids.push(0);
                mask.push(0);
                padded_labels.push(-100);
            }

            input_ids.push(padded_ids);
            attention_mask.push(mask);
            labels.push(padded_labels);
        }

        batches.push(PreparedBatch {
            input_ids,
            attention_mask,
            labels,
        });
    }

    Ok(batches)
}
