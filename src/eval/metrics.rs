use crate::data::jsonl::TrainingExample;

#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub perplexity: f32,
    pub exact_match: f32,
    pub bleu: f32,
    pub rouge_l: f32,
}

impl MetricSummary {
    pub fn from_examples(examples: &[TrainingExample]) -> Self {
        if examples.is_empty() {
            return Self {
                perplexity: 0.0,
                exact_match: 0.0,
                bleu: 0.0,
                rouge_l: 0.0,
            };
        }

        let exact_matches = examples
            .iter()
            .filter(|example| example.prompt.trim() == example.completion.trim())
            .count() as f32;
        let token_count = examples
            .iter()
            .map(|example| example.completion.split_whitespace().count())
            .sum::<usize>() as f32;
        let bleu = (exact_matches / examples.len() as f32).min(1.0);
        let rouge_l = (token_count / (examples.len() as f32 * 100.0)).min(1.0);

        Self {
            perplexity: (token_count.max(1.0) / examples.len() as f32).max(1.0),
            exact_match: exact_matches / examples.len() as f32,
            bleu,
            rouge_l,
        }
    }
}
