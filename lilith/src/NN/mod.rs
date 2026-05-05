pub mod nn;
pub mod cnn;
pub mod llm;
pub mod talk;

pub use nn::{
    NeuralCore, PipelineStage, LogLevel, LogEntry,
    InternalAi, AiModality, RequireItem, WeightLayer, degree_ai,
    encode_hdr,
};
pub use cnn::cnn::{CnnGpu, MnistDataset, WeightVisualizer, kl_divergence};
pub use llm::llm::{MiniLlm, LlmConfig, ChatBot, QaItem, load_dataset};
