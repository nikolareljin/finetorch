use crate::data::jsonl::TrainingExample;

pub fn bleu_score(candidate: &str, reference: &str) -> f32 {
    let cand: Vec<&str> = candidate.split_whitespace().collect();
    let refer: Vec<&str> = reference.split_whitespace().collect();
    if cand.is_empty() || refer.is_empty() {
        return 0.0;
    }
    let mut match_count = 0usize;
    for token in &cand {
        if refer.contains(token) {
            match_count += 1;
        }
    }
    let precision = match_count as f32 / cand.len() as f32;
    let bp = if cand.len() as f32 > refer.len() as f32 {
        1.0
    } else {
        (-(refer.len() as f32 / cand.len() as f32 - 1.0)).exp()
    };
    (bp * precision).min(1.0)
}

pub fn rouge_l_score(candidate: &str, reference: &str) -> f32 {
    let cand: Vec<&str> = candidate.split_whitespace().collect();
    let refer: Vec<&str> = reference.split_whitespace().collect();
    if cand.is_empty() || refer.is_empty() {
        return 0.0;
    }
    let lcs = lcs_len(&cand, &refer);
    let precision = lcs as f32 / cand.len() as f32;
    let recall = lcs as f32 / refer.len() as f32;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn lcs_len(a: &[&str], b: &[&str]) -> usize {
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[a.len()][b.len()]
}

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
