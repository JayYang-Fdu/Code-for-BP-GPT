# 📁 BP-GPT Implementation

## ⚠️ File Notes
- The `GP2model` folder exceeds GitHub's size limit and cannot be uploaded directly.  
- Missing `.bin` file (`pytorch_model.bin`) can be downloaded from:  
  [Hugging Face GPT-2 Model](https://huggingface.co/openai-community/gpt2/blob/main)

## 🛠️ Requirements
- Python 3.8
- PyTorch 2.4.1 (CUDA 12.1)

## 🚀 Workflow
### 1. Data Generation
Run `main_hybrid.m` to generate:  
- Training dataset  
- Test dataset  

### 2. Model Training
Execute `LLM4BF.py` to train the BP-GPT model.

### 3. Model Testing
Run `test.py` to evaluate predicted beam performance.

## 📊 Performance Analysis
### Temporal Performance
- `TindexRate.m`: Generates Spectral Efficiency (SE) vs. time steps.

### Velocity Impact
- `Velocity.m`: Plots SE vs. different velocities.

### Noise Robustness
- `PNRRate.m`: Calculates average SE across different test SNRs.
