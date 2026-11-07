use std::fs;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const ARTIFACT_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub kind: String,
    pub source: serde_json::Value,
    pub outputs: Vec<String>,
    pub backend: Option<String>,
    pub config_hash: Option<String>,
}

pub fn write_artifact_manifest(path: impl AsRef<Path>, manifest: &ArtifactManifest) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}
