#[cfg(feature = "backend-candle")]
use std::collections::HashMap;
#[cfg(feature = "backend-candle")]
use std::fs;
#[cfg(feature = "backend-candle")]
use std::path::PathBuf;

#[cfg(feature = "backend-candle")]
use anyhow::Result;
#[cfg(feature = "backend-candle")]
#[cfg(feature = "backend-candle")]
use finetorch::model::candle::CandleBackend;

fn main() {
    #[cfg(feature = "backend-candle")]
    {
        run().unwrap();
    }
    #[cfg(not(feature = "backend-candle"))]
    {
        eprintln!("gen_fixture requires backend-candle feature");
        std::process::exit(1);
    }
}

#[cfg(feature = "backend-candle")]
fn run() -> Result<()> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "fixture".to_string()),
    );
    fs::create_dir_all(&out)?;
    let model_dir = out.join("model");
    fs::create_dir_all(&model_dir)?;

    let mut vocab = HashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    for i in 1..32 {
        vocab.insert(format!("tok{i}"), i as u32);
    }
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        },
        "pre_tokenizer": {
            "type": "Whitespace"
        },
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": true
        }
    });
    fs::write(
        out.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer)?,
    )?;

    let backend = CandleBackend::default();
    backend.save_pretrained(&model_dir)?;

    let train = r#"{"prompt":"hello","completion":"tok1 tok2"}
{"prompt":"world","completion":"tok3 tok4"}
"#;
    fs::write(out.join("train.jsonl"), train)?;
    println!("fixture ready: {}", out.display());
    Ok(())
}
