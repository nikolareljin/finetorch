use std::path::PathBuf;

#[cfg(feature = "backend-candle")]
use crate::artifact::{write_artifact_manifest, ArtifactManifest};
#[cfg(feature = "backend-candle")]
use crate::config::RunConfig;
#[cfg(feature = "backend-candle")]
use crate::model::backend::LlmBackend;
use anyhow::Result;

#[derive(Debug, clap::Args)]
pub struct MergeArgs {
    #[arg(long)]
    pub base: PathBuf,
    #[arg(long)]
    pub adapter: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub config: Option<String>,
}

pub fn run(args: MergeArgs) -> Result<()> {
    #[cfg(feature = "backend-candle")]
    {
        use crate::model::candle::CandleBackend;
        let mut backend = CandleBackend::default();
        backend.load_model(&args.base.display().to_string())?;
        backend.reload_adapter(&args.adapter.display().to_string())?;
        backend.merge_adapters()?;
        if args.output.exists() {
            return Err(anyhow::anyhow!(
                "output already exists: {}",
                args.output.display()
            ));
        }
        backend.save_pretrained(&args.output)?;
        let mut config_hash = None;
        if let Some(path) = &args.config {
            if let Ok(cfg) = RunConfig::from_file(path) {
                config_hash = Some(cfg.config_hash());
            }
        }
        let manifest = ArtifactManifest {
            schema_version: crate::artifact::ARTIFACT_MANIFEST_SCHEMA_VERSION,
            kind: "merged_model".to_string(),
            source: serde_json::json!({
                "base": args.base,
                "adapter": args.adapter,
            }),
            outputs: vec![args.output.display().to_string()],
            backend: Some("candle".to_string()),
            config_hash,
        };
        write_artifact_manifest(args.output.join("artifact_manifest.json"), &manifest)?;
        println!("merged model: {}", args.output.display());
        Ok(())
    }
    #[cfg(not(feature = "backend-candle"))]
    {
        let _ = (args,);
        Err(anyhow::anyhow!("merge requires backend-candle feature"))
    }
}
