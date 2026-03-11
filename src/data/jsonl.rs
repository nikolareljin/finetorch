use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub prompt: String,
    pub completion: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum JsonlRecord {
    Instruction {
        instruction: String,
        #[serde(default)]
        input: String,
        output: String,
    },
    PromptCompletion {
        prompt: String,
        completion: String,
    },
}

pub fn read_examples(path: impl AsRef<Path>) -> Result<Vec<TrainingExample>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("opening {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: JsonlRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing JSONL record {}", idx + 1))?;
        out.push(match record {
            JsonlRecord::Instruction {
                instruction,
                input,
                output,
            } => {
                let prompt = if input.trim().is_empty() {
                    instruction
                } else {
                    format!("Instruction: {}\nInput: {}", instruction, input)
                };
                TrainingExample {
                    prompt,
                    completion: output,
                }
            }
            JsonlRecord::PromptCompletion { prompt, completion } => {
                TrainingExample { prompt, completion }
            }
        });
    }

    Ok(out)
}
