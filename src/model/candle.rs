use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::ops;
use serde::Deserialize;

use crate::config::TrainingConfig;
use crate::model::backend::{BackendCapabilities, ForwardBatch, ForwardOutput, LlmBackend};
use crate::model::lora::LoraConfig;

#[derive(Debug, Clone, Deserialize, Default)]
struct HfConfig {
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ModelMemory {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    seed: u64,
}

impl Default for ModelMemory {
    fn default() -> Self {
        Self {
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 32,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            seed: 42,
        }
    }
}

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    _in_features: usize,
    lora_a: Option<Var>,
    lora_b: Option<Var>,
    lora_r: usize,
    lora_alpha: usize,
    lora_scaling: f64,
    has_lora: bool,
}

impl Linear {
    fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let _out_features = weight.dims()[0];
        let in_features = weight.dims()[1];
        Self {
            weight,
            bias,
            _in_features: in_features,
            lora_a: None,
            lora_b: None,
            lora_r: 0,
            lora_alpha: 0,
            lora_scaling: 1.0,
            has_lora: false,
        }
    }

    fn apply_lora(&mut self, cfg: &LoraConfig) -> Result<()> {
        self.has_lora = cfg.enabled;
        self.lora_r = cfg.r;
        self.lora_alpha = cfg.alpha;
        self.lora_scaling = if cfg.r > 0 {
            cfg.alpha as f64 / cfg.r as f64
        } else {
            1.0
        };
        if !cfg.enabled {
            return Ok(());
        }
        let in_features = self.weight.dims()[1];
        let out_features = self.weight.dims()[0];
        let mut data = vec![0.0f32; in_features * cfg.r];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f32) * 0.01).sin() * 0.01;
        }
        let a = Var::from_tensor(&Tensor::from_vec(
            data,
            (in_features, cfg.r),
            self.weight.device(),
        )?)?;
        let data = vec![0.0f32; cfg.r * out_features];
        let b = Var::from_tensor(&Tensor::from_vec(
            data,
            (cfg.r, out_features),
            self.weight.device(),
        )?)?;
        self.lora_a = Some(a);
        self.lora_b = Some(b);
        Ok(())
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let in_features = self.weight.dims()[1];
        let out_features = self.weight.dims()[0];
        let orig = input.dims().to_vec();
        let flat = input.reshape((input.elem_count() / in_features, in_features))?;
        let base = flat.matmul(&self.weight.t()?)?;
        let mut out_shape = orig[..orig.len() - 1].to_vec();
        out_shape.push(out_features);
        let base = base.reshape(out_shape.as_slice())?;
        let base = match &self.bias {
            Some(b) => base.broadcast_add(b)?,
            None => base,
        };
        if self.has_lora {
            if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
                let in_features = a.as_tensor().dims()[0];
                let r = a.as_tensor().dims()[1];
                let out_features = b.as_tensor().dims()[1];
                let orig = input.dims().to_vec();
                let flat = input.reshape((input.elem_count() / in_features, in_features))?;
                let tmp = flat.matmul(a.as_tensor())?;
                let tmp = tmp.reshape((flat.dims()[0], r))?;
                let lora_flat = tmp.matmul(b.as_tensor())?;
                let mut out_shape = orig[..orig.len() - 1].to_vec();
                out_shape.push(out_features);
                let lora = lora_flat.reshape(out_shape.as_slice())?;
                let lora = (lora * self.lora_scaling)?;
                return Ok((base + lora)?);
            }
        }
        Ok(base)
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(x.rank() - 1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(normed.broadcast_mul(&self.weight)?)
    }
}

struct Block {
    norm1: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm2: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm1.forward(x)?;
        let q = self.q.forward(&h)?;
        let k = self.k.forward(&h)?;
        let v = self.v.forward(&h)?;
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scale = 1.0 / (q.dims()[2] as f64).sqrt();
        let scores = (scores * scale)?;
        let seq = q.dims()[1];
        let mut mask = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask[i * seq + j] = f32::NEG_INFINITY;
            }
        }
        let causal = Tensor::from_vec(mask, (seq, seq), self.q.weight.device())?;
        let scores = (scores + causal)?;
        let attn = ops::softmax(&scores, 2)?;
        let ctx = attn.matmul(&v)?;
        let attn_out = self.o.forward(&ctx)?;
        let x = (x + attn_out)?;
        let h = self.norm2.forward(&x)?;
        let gate = self.gate.forward(&h)?;
        let up = self.up.forward(&h)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        let hidden = (silu_gate * up)?;
        let mlp_out = self.down.forward(&hidden)?;
        Ok((x + mlp_out)?)
    }
}

pub struct CandleBackend {
    device: Device,
    config: ModelMemory,
    tok_embeddings: Tensor,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    lora_config: Option<LoraConfig>,
    train_vars: Vec<Var>,
    adapter_vars: Vec<(String, Var)>,
    opt_step: usize,
    moment1: Vec<Var>,
    moment2: Vec<Var>,
    training_lr: f64,
    weight_decay: f64,
    warmup_steps: usize,
}

impl CandleBackend {
    pub fn build_random(cfg: &ModelMemory, seed: u64, device: &Device) -> Result<Self> {
        // embedding
        let mut w = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
        for (i, v) in w.iter_mut().enumerate() {
            let x = ((i as u64).wrapping_mul(1103515245).wrapping_add(seed)) as f32;
            *v = ((x / u32::MAX as f32) - 0.5) * 0.1;
        }
        let tok_embeddings = Tensor::from_vec(w, (cfg.vocab_size, cfg.hidden_size), device)?;
        let rms_eps = cfg.rms_norm_eps;
        let mut blocks = Vec::new();
        for layer in 0..cfg.num_layers {
            let make_linear = |seed: u64, out: usize, inn: usize, layer: usize| -> Linear {
                let mut w = vec![0.0f32; out * inn];
                for (i, v) in w.iter_mut().enumerate() {
                    let x = ((i as u64)
                        .wrapping_mul(97)
                        .wrapping_add(seed)
                        .wrapping_add(layer as u64 * 1000)) as f32;
                    *v = ((x / u32::MAX as f32) - 0.5) * 0.1;
                }
                let weight = Tensor::from_vec(w, (out, inn), device).unwrap();
                Linear::new(weight, None)
            };
            let block = Block {
                norm1: RmsNorm {
                    weight: Tensor::ones((cfg.hidden_size,), DType::F32, device)?,
                    eps: rms_eps,
                },
                q: make_linear(1, cfg.hidden_size, cfg.hidden_size, layer),
                k: make_linear(2, cfg.hidden_size, cfg.hidden_size, layer),
                v: make_linear(3, cfg.hidden_size, cfg.hidden_size, layer),
                o: make_linear(4, cfg.hidden_size, cfg.hidden_size, layer),
                norm2: RmsNorm {
                    weight: Tensor::ones((cfg.hidden_size,), DType::F32, device)?,
                    eps: rms_eps,
                },
                gate: make_linear(5, cfg.intermediate_size, cfg.hidden_size, layer),
                up: make_linear(6, cfg.intermediate_size, cfg.hidden_size, layer),
                down: make_linear(7, cfg.hidden_size, cfg.intermediate_size, layer),
            };
            blocks.push(block);
        }
        let norm = RmsNorm {
            weight: Tensor::ones((cfg.hidden_size,), DType::F32, device)?,
            eps: rms_eps,
        };
        let lm_head = {
            let mut w = vec![0.0f32; cfg.vocab_size * cfg.hidden_size];
            for (i, v) in w.iter_mut().enumerate() {
                let x = ((i as u64).wrapping_mul(137).wrapping_add(seed + 9)) as f32;
                *v = ((x / u32::MAX as f32) - 0.5) * 0.1;
            }
            let weight = Tensor::from_vec(w, (cfg.vocab_size, cfg.hidden_size), device)?;
            Linear::new(weight, None)
        };
        Ok(Self {
            device: device.clone(),
            config: cfg.clone(),
            tok_embeddings,
            blocks,
            norm,
            lm_head,
            lora_config: None,
            train_vars: Vec::new(),
            adapter_vars: Vec::new(),
            opt_step: 0,
            moment1: Vec::new(),
            moment2: Vec::new(),
            training_lr: 1e-3,
            weight_decay: 0.0,
            warmup_steps: 0,
        })
    }

    pub fn load_model_dir(path: &Path) -> Result<Self> {
        let cfg = if path.join("config.json").exists() {
            let raw = fs::read_to_string(path.join("config.json"))?;
            let hf: HfConfig = serde_json::from_str(&raw).unwrap_or_default();
            let mut m = ModelMemory::default();
            if let Some(v) = hf.hidden_size {
                m.hidden_size = v;
            }
            if let Some(v) = hf.num_hidden_layers {
                m.num_layers = v;
            }
            if let Some(v) = hf.num_attention_heads {
                m.num_heads = v;
            }
            if let Some(v) = hf.intermediate_size {
                m.intermediate_size = v;
            }
            if let Some(v) = hf.vocab_size {
                m.vocab_size = v;
            }
            if let Some(v) = hf.rms_norm_eps {
                m.rms_norm_eps = v;
            }
            m
        } else {
            ModelMemory::default()
        };

        let tensors = if path.join("model.safetensors").exists() {
            candle_core::safetensors::load(path.join("model.safetensors"), &Device::Cpu)?
        } else {
            let mut map = HashMap::new();
            if let Ok(rd) = fs::read_dir(path) {
                for entry in rd {
                    let entry = entry?;
                    let p = entry.path();
                    if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                        let ms = candle_core::safetensors::load(&p, &Device::Cpu)?;
                        for (k, v) in ms {
                            map.insert(k, v);
                        }
                    }
                }
            }
            map
        };

        let mut model = Self::build_random(&cfg, cfg.seed, &Device::Cpu)?;
        // override common tensors if present
        if let Some(t) = tensors
            .get("tok_embeddings.weight")
            .or_else(|| tensors.get("model.embed_tokens.weight"))
        {
            model.tok_embeddings = t.clone();
        }
        for (i, block) in model.blocks.iter_mut().enumerate() {
            let prefix = format!("layers.{i}");
            let hf_prefix = format!("model.layers.{i}");
            // norm
            if let Some(t) = tensors
                .get(&format!("{prefix}.norm1.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.input_layernorm.weight")))
            {
                block.norm1.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.norm2.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.post_attention_layernorm.weight")))
            {
                block.norm2.weight = t.clone();
            }
            // attention weights
            if let Some(t) = tensors
                .get(&format!("{prefix}.attn.q.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.self_attn.q_proj.weight")))
            {
                block.q.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.attn.k.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.self_attn.k_proj.weight")))
            {
                block.k.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.attn.v.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.self_attn.v_proj.weight")))
            {
                block.v.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.attn.o.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.self_attn.o_proj.weight")))
            {
                block.o.weight = t.clone();
            }
            // mlp
            if let Some(t) = tensors
                .get(&format!("{prefix}.mlp.gate.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.mlp.gate_proj.weight")))
            {
                block.gate.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.mlp.up.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.mlp.up_proj.weight")))
            {
                block.up.weight = t.clone();
            }
            if let Some(t) = tensors
                .get(&format!("{prefix}.mlp.down.weight"))
                .or_else(|| tensors.get(&format!("{hf_prefix}.mlp.down_proj.weight")))
            {
                block.down.weight = t.clone();
            }
        }
        if let Some(t) = tensors
            .get("norm.weight")
            .or_else(|| tensors.get("model.norm.weight"))
        {
            model.norm.weight = t.clone();
        }
        if let Some(t) = tensors
            .get("lm_head.weight")
            .or_else(|| tensors.get("lm_head.weight"))
        {
            model.lm_head.weight = t.clone();
        }
        Ok(model)
    }

    pub fn save_pretrained(&self, path: &Path) -> Result<()> {
        fs::create_dir_all(path)?;
        let cfg = serde_json::json!({
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_layers,
            "num_attention_heads": self.config.num_heads,
            "intermediate_size": self.config.intermediate_size,
            "vocab_size": self.config.vocab_size,
            "rms_norm_eps": self.config.rms_norm_eps,
        });
        fs::write(
            path.join("config.json"),
            serde_json::to_string_pretty(&cfg)?,
        )?;
        let mut tensors: Vec<(String, Tensor)> = vec![(
            "tok_embeddings.weight".to_string(),
            self.tok_embeddings.clone(),
        )];
        for (i, block) in self.blocks.iter().enumerate() {
            tensors.push((
                format!("layers.{i}.norm1.weight"),
                block.norm1.weight.clone(),
            ));
            tensors.push((format!("layers.{i}.attn.q.weight"), block.q.weight.clone()));
            tensors.push((format!("layers.{i}.attn.k.weight"), block.k.weight.clone()));
            tensors.push((format!("layers.{i}.attn.v.weight"), block.v.weight.clone()));
            tensors.push((format!("layers.{i}.attn.o.weight"), block.o.weight.clone()));
            tensors.push((
                format!("layers.{i}.norm2.weight"),
                block.norm2.weight.clone(),
            ));
            tensors.push((
                format!("layers.{i}.mlp.gate.weight"),
                block.gate.weight.clone(),
            ));
            tensors.push((format!("layers.{i}.mlp.up.weight"), block.up.weight.clone()));
            tensors.push((
                format!("layers.{i}.mlp.down.weight"),
                block.down.weight.clone(),
            ));
        }
        tensors.push(("norm.weight".to_string(), self.norm.weight.clone()));
        tensors.push(("lm_head.weight".to_string(), self.lm_head.weight.clone()));
        let mut map = HashMap::new();
        for (name, tensor) in &tensors {
            map.insert(name.clone(), tensor.clone());
        }
        candle_core::safetensors::save(&map, path.join("model.safetensors"))
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    pub fn config_memory(&self) -> &ModelMemory {
        &self.config
    }

    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        let b = input.dims()[0];
        let s = input.dims()[1];
        let ids_raw = input.flatten_all()?.to_vec1::<u32>()?;
        let ids_flat: Vec<u32> = ids_raw
            .iter()
            .map(|x| x % self.config.vocab_size as u32)
            .collect();
        let ids_t = Tensor::from_vec(ids_flat, (b * s,), &self.device)?;
        let emb = self.tok_embeddings.index_select(&ids_t, 0)?.reshape((
            b,
            s,
            self.config.hidden_size,
        ))?;
        let mut x = emb;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.norm.forward(&x)?;
        let x_flat = x.reshape((b * s, self.config.hidden_size))?;
        let logits = self.lm_head.forward(&x_flat)?;
        Ok(logits.reshape((b, s, self.config.vocab_size))?)
    }

    pub fn train_step(&mut self, batch: &ForwardBatch, global_step: usize) -> Result<f32> {
        if self.train_vars.is_empty() {
            return Err(anyhow!("training requires apply_lora with target modules"));
        }
        if self.moment1.is_empty() {
            for var in &self.train_vars {
                self.moment1
                    .push(Var::zeros(var.shape(), var.dtype(), var.device())?);
                self.moment2
                    .push(Var::zeros(var.shape(), var.dtype(), var.device())?);
            }
        }
        let lr = if self.warmup_steps > 0 && global_step < self.warmup_steps {
            self.training_lr * (global_step as f64 + 1.0) / (self.warmup_steps as f64)
        } else {
            self.training_lr
        };
        let b = batch.input_ids.len();
        if b == 0 {
            return Err(anyhow!("empty batch"));
        }
        let s = batch.input_ids[0].len();
        let mut data = Vec::with_capacity(b * s);
        for row in &batch.input_ids {
            data.extend_from_slice(row);
        }
        let input = Tensor::from_vec(data, (b, s), &self.device)?;
        let logits = self.forward_tensor(&input)?;
        let v = self.config.vocab_size;
        let logits_flat = logits.reshape((b * s, v))?;

        let labels_for_loss: Vec<u32> = (0..(b * s))
            .map(|idx| {
                let i = idx / s;
                let j = idx % s;
                let lab = batch.labels[i][j];
                if lab == -100 {
                    0u32
                } else {
                    (lab as u32) % self.config.vocab_size as u32
                }
            })
            .collect();
        let labels_t = Tensor::from_vec(labels_for_loss, (b * s,), &self.device)?;
        let ce = candle_nn::loss::cross_entropy(&logits_flat, &labels_t)?;
        let grads = ce.backward()?;
        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let eps = 1e-8f64;
        self.opt_step += 1;
        let t = self.opt_step as f64;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        for (i, var) in self.train_vars.iter().enumerate() {
            if let Some(g) = grads.get(var) {
                let m_prev = self.moment1[i].as_tensor();
                let v_prev = self.moment2[i].as_tensor();
                let next_m = ((m_prev * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v_prev * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                self.moment1[i].set(&next_m)?;
                self.moment2[i].set(&next_v)?;
                let m_hat = (next_m * (1.0 / bc1))?;
                let v_hat = (next_v * (1.0 / bc2))?;
                let current = var.as_tensor();
                let scaled_grad = (m_hat / (v_hat.sqrt()? + eps)?)?;
                let next = ((current * (1.0 - lr * self.weight_decay))? - (scaled_grad * lr))?;
                var.set(&next)?;
            }
        }
        Ok(ce.to_vec0::<f32>()?)
    }

    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut tokens = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let b = 1usize;
            let s = tokens.len();
            let input = Tensor::from_vec(tokens.clone(), (b, s), &self.device)?;
            let logits = self.forward_tensor(&input)?;
            let last = logits.get(0)?.get(s - 1)?;
            let vals = last.to_vec1::<f32>()?;
            let mut best = 0usize;
            let mut bestv = vals[0];
            for (i, vv) in vals.iter().enumerate().skip(1) {
                if *vv > bestv {
                    bestv = *vv;
                    best = i;
                }
            }
            tokens.push(best as u32);
        }
        Ok(tokens)
    }

    pub fn merge_adapters(&mut self) {
        for block in &mut self.blocks {
            let mut linears: Vec<&mut Linear> = vec![
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.o,
                &mut block.gate,
                &mut block.up,
                &mut block.down,
            ];
            for linear in &mut linears {
                if linear.has_lora {
                    if let (Some(a), Some(b)) = (linear.lora_a.clone(), linear.lora_b.clone()) {
                        let scale = if linear.lora_r > 0 {
                            linear.lora_alpha as f32 / linear.lora_r as f32
                        } else {
                            1.0
                        };
                        // b is [r, out], a is [in, r]; delta [out, in] = b^T (out,r) @ a^T (r,in)
                        let b_t = b.as_tensor().t().unwrap();
                        let a_t = a.as_tensor().t().unwrap();
                        if let Ok(delta) = b_t.matmul(&a_t) {
                            if let Ok(delta) = delta * (scale as f64) {
                                let weight = linear.weight.clone();
                                if let Ok(new_weight) = &weight + &delta {
                                    linear.weight = new_weight;
                                }
                            }
                        }
                    }
                    linear.lora_a = None;
                    linear.lora_b = None;
                    linear.has_lora = false;
                }
            }
        }
        // clear adapter vars
        if self.lm_head.has_lora {
            if let (Some(a), Some(b)) = (self.lm_head.lora_a.clone(), self.lm_head.lora_b.clone()) {
                let scale = if self.lm_head.lora_r > 0 {
                    self.lm_head.lora_alpha as f32 / self.lm_head.lora_r as f32
                } else {
                    1.0
                };
                let a_t = a.as_tensor().t().unwrap();
                let b_t = b.as_tensor().t().unwrap();
                if let Ok(delta) = b_t.matmul(&a_t) {
                    if let Ok(delta) = delta * (scale as f64) {
                        if let Ok(new_weight) = &self.lm_head.weight + &delta {
                            self.lm_head.weight = new_weight;
                        }
                    }
                }
            }
            self.lm_head.lora_a = None;
            self.lm_head.lora_b = None;
            self.lm_head.has_lora = false;
        }
        self.adapter_vars.clear();
        self.train_vars.clear();
    }

    pub fn capabilities_report(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_lora: true,
            supports_qlora: false,
            supports_training: true,
            supports_generation: true,
            supported_devices: vec!["cpu".to_string()],
        }
    }

    pub fn training_vars(&self) -> &[Var] {
        &self.train_vars
    }

    pub fn configure_training(&mut self, cfg: &TrainingConfig) {
        self.training_lr = cfg.lr;
        self.weight_decay = cfg.weight_decay;
        self.warmup_steps = cfg.warmup_steps;
    }

    pub fn adapter_state(&self) -> &[(String, Var)] {
        &self.adapter_vars
    }

    pub fn reload_adapter(&mut self, path: &str) -> Result<()> {
        let ts = candle_core::safetensors::load(path, &Device::Cpu)?;
        for (name, var) in &self.adapter_vars {
            if let Some(t) = ts.get(name) {
                var.set(t)?;
            }
        }
        Ok(())
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        let device = Device::Cpu;
        Self::build_random(&ModelMemory::default(), 42, &device).unwrap()
    }
}

impl LlmBackend for CandleBackend {
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities_report()
    }

    fn load_model(&mut self, base_path: &str) -> Result<()> {
        if base_path.trim().is_empty() {
            return Ok(());
        }
        let p = Path::new(base_path);
        if !p.exists() {
            return Err(anyhow!("model path does not exist: {base_path}"));
        }
        let loaded = Self::load_model_dir(p)?;
        *self = loaded;
        Ok(())
    }

    fn apply_lora(&mut self, config: &LoraConfig) -> Result<()> {
        if config.use_qlora {
            return Err(anyhow!("qlora not supported"));
        }
        let allowed = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ];
        for t in &config.target_modules {
            if !allowed.contains(&t.as_str()) {
                return Err(anyhow!("unknown target module: {t}"));
            }
        }
        self.lora_config = Some(config.clone());
        let targets = config.target_modules.clone();
        self.train_vars.clear();
        self.adapter_vars.clear();

        for (i, block) in self.blocks.iter_mut().enumerate() {
            let mut pairs: Vec<(&str, &mut Linear)> = vec![
                ("q_proj", &mut block.q),
                ("k_proj", &mut block.k),
                ("v_proj", &mut block.v),
                ("o_proj", &mut block.o),
                ("gate_proj", &mut block.gate),
                ("up_proj", &mut block.up),
                ("down_proj", &mut block.down),
            ];
            for (name, linear) in pairs.iter_mut() {
                if targets.iter().any(|t| t == *name) {
                    linear.apply_lora(config)?;
                    if let Some(a) = &linear.lora_a {
                        self.train_vars.push(a.clone());
                        self.adapter_vars
                            .push((format!("blocks.{i}.{name}.lora_a"), a.clone()));
                    }
                    if let Some(b) = &linear.lora_b {
                        self.train_vars.push(b.clone());
                        self.adapter_vars
                            .push((format!("blocks.{i}.{name}.lora_b"), b.clone()));
                    }
                }
            }
        }
        if targets.iter().any(|t| t == "lm_head") {
            self.lm_head.apply_lora(config)?;
            if let Some(a) = &self.lm_head.lora_a {
                self.train_vars.push(a.clone());
                self.adapter_vars
                    .push(("lm_head.lora_a".to_string(), a.clone()));
            }
            if let Some(b) = &self.lm_head.lora_b {
                self.train_vars.push(b.clone());
                self.adapter_vars
                    .push(("lm_head.lora_b".to_string(), b.clone()));
            }
        }

        self.opt_step = 0;
        self.moment1.clear();
        self.moment2.clear();
        Ok(())
    }

    fn forward(&mut self, batch: &ForwardBatch) -> Result<ForwardOutput> {
        if batch.input_ids.is_empty() {
            return Ok(ForwardOutput {
                loss: None,
                logits_shape: vec![],
            });
        }
        // set training config if needed already
        let b = batch.input_ids.len();
        let s = batch.input_ids[0].len();
        let mut data = Vec::with_capacity(b * s);
        for row in &batch.input_ids {
            data.extend_from_slice(row);
        }
        let input = Tensor::from_vec(data, (b, s), &self.device)?;
        let logits = self.forward_tensor(&input)?;
        let logits_shape = logits.dims().to_vec();
        let loss = if batch.labels.iter().any(|r| r.iter().any(|&x| x != -100)) {
            let v = self.config.vocab_size;
            let logits_flat = logits.reshape((b * s, v))?;
            let vals = logits_flat.to_vec2::<f32>()?;
            let mut total = 0.0f64;
            let mut count = 0usize;
            for i in 0..b {
                for j in 0..s {
                    let lab = batch.labels[i][j];
                    if lab != -100 {
                        let idx = i * s + j;
                        let row = &vals[idx];
                        let label = if lab < 0 { 0usize } else { (lab as usize) % v };
                        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let sum_exp: f32 = row.iter().map(|x| (x - max).exp()).sum();
                        let log_prob = row[label] as f64 - max as f64 - (sum_exp as f64).ln();
                        total -= log_prob;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                Some((total / count as f64) as f32)
            } else {
                None
            }
        } else {
            None
        };
        Ok(ForwardOutput { loss, logits_shape })
    }

    fn set_training_config(&mut self, config: &TrainingConfig) {
        self.configure_training(config);
    }

    fn train_step(&mut self, batch: &ForwardBatch, global_step: usize) -> Result<f32> {
        self.train_step(batch, global_step)
    }

    fn save_checkpoint(&self, dir: &str, step: usize, config_hash: &str) -> Result<()> {
        let dir = Path::new(dir);
        let parent = dir.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let temp = parent.join(format!(".checkpoint-tmp-{step}"));
        if temp.exists() {
            fs::remove_dir_all(&temp)?;
        }
        fs::create_dir_all(&temp)?;
        let manifest = serde_json::json!({
            "step": step,
            "config_hash": config_hash,
        });
        fs::write(
            temp.join("checkpoint.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;

        if !self.adapter_vars.is_empty() {
            let mut map = HashMap::new();
            for (name, var) in &self.adapter_vars {
                map.insert(name.clone(), var.as_tensor().clone());
            }
            candle_core::safetensors::save(&map, temp.join("adapter.safetensors"))?;
        }
        if !self.train_vars.is_empty() {
            let mut map = HashMap::new();
            for (i, var) in self.train_vars.iter().enumerate() {
                map.insert(format!("var.{i}"), var.as_tensor().clone());
            }
            for (i, var) in self.moment1.iter().enumerate() {
                map.insert(format!("mom1.{i}"), var.as_tensor().clone());
            }
            for (i, var) in self.moment2.iter().enumerate() {
                map.insert(format!("mom2.{i}"), var.as_tensor().clone());
            }
            candle_core::safetensors::save(&map, temp.join("optimizer.safetensors"))?;
        }
        if dir.exists() {
            fs::remove_dir_all(dir)?;
        }
        fs::rename(&temp, dir)?;
        Ok(())
    }

    fn load_checkpoint(&mut self, dir: &str, expected_hash: &str) -> Result<usize> {
        let dir = Path::new(dir);
        let raw = fs::read_to_string(dir.join("checkpoint.json"))
            .map_err(|e| anyhow!("truncated checkpoint: {e}"))?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).map_err(|e| anyhow!("truncated checkpoint: {e}"))?;
        let step = value["step"]
            .as_u64()
            .ok_or_else(|| anyhow!("truncated checkpoint"))? as usize;
        let hash = value["config_hash"]
            .as_str()
            .ok_or_else(|| anyhow!("truncated checkpoint"))?;
        if hash != expected_hash {
            return Err(anyhow!("config hash mismatch"));
        }

        if dir.join("adapter.safetensors").exists() {
            let ts = candle_core::safetensors::load(dir.join("adapter.safetensors"), &self.device)?;
            for (name, var) in &self.adapter_vars {
                if let Some(t) = ts.get(name) {
                    var.set(t)?;
                }
            }
        }
        if dir.join("optimizer.safetensors").exists() {
            let ts =
                candle_core::safetensors::load(dir.join("optimizer.safetensors"), &self.device)?;
            for (i, var) in self.train_vars.iter().enumerate() {
                if let Some(t) = ts.get(&format!("var.{i}")) {
                    var.set(t)?;
                }
            }
            if self.moment1.is_empty() {
                for var in &self.train_vars {
                    self.moment1
                        .push(Var::zeros(var.shape(), var.dtype(), var.device())?);
                    self.moment2
                        .push(Var::zeros(var.shape(), var.dtype(), var.device())?);
                }
            }
            for (i, var) in self.moment1.iter().enumerate() {
                if let Some(t) = ts.get(&format!("mom1.{i}")) {
                    var.set(t)?;
                }
            }
            for (i, var) in self.moment2.iter().enumerate() {
                if let Some(t) = ts.get(&format!("mom2.{i}")) {
                    var.set(t)?;
                }
            }
            self.opt_step = step;
        }
        Ok(step)
    }

    fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.generate(prompt, max_new_tokens)
    }

    fn save_adapters(&self, output_path: &str) -> Result<()> {
        if self.adapter_vars.is_empty() {
            fs::write(output_path, "")?;
            return Ok(());
        }
        let path = Path::new(output_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut map = HashMap::new();
        for (n, v) in &self.adapter_vars {
            map.insert(n.clone(), v.as_tensor().clone());
        }
        candle_core::safetensors::save(&map, path).map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }
}
