#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub warmup_steps: usize,
    pub max_steps: usize,
}
