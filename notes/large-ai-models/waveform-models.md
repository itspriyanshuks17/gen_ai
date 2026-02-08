# Waveform Models - Developer Notes

## Introduction

Waveform models work directly with raw audio waveforms, processing audio as sequential time-domain signals.

### Key Characteristics
- Process audio as 1D sequences
- Maintain temporal resolution
- Capture fine-grained audio details

### Popular Waveform Models

#### WaveNet
- **Architecture**: Dilated convolutional network with causal convolutions
- **Key Feature**: Autoregressive generation of audio samples
- **Advantages**: High-quality audio synthesis
- **Limitations**: Slow inference due to sequential generation

#### Architecture
```mermaid
flowchart LR
    A[Audio Waveform] --> B[Dilated Conv Block 1]
    B --> C[Dilated Conv Block 2]
    C --> D[Dilated Conv Block 4]
    D --> E[Output Distribution]
    E --> F[Next Sample Prediction]
```

#### Detailed Flowchart Node Explanation

##### A: Audio Waveform
- **Purpose**: Raw audio input as time-domain signal
- **Format**: 1D sequence of amplitude values
- **Sample Rate**: Usually 16kHz, 22kHz, or 44.1kHz
- **Quantization**: Often μ-law encoded to 256 discrete values
- **Dimensions**: (batch_size, sequence_length, 1)

##### B: Dilated Conv Block 1
- **Purpose**: First layer of dilated convolution with dilation rate 1
- **Operation**: Causal convolution with receptive field expansion
- **Components**:
  - **Gated Activation**: tanh * sigmoid for gating
  - **Skip Connection**: Parallel path for gradient flow
  - **Residual Connection**: Adds input to output
- **Dilation Rate**: 1 (no dilation)

##### C: Dilated Conv Block 2
- **Purpose**: Second layer with increased receptive field
- **Dilation Rate**: 2 (skips every other sample)
- **Receptive Field**: Covers 3 samples (1 + 2) from previous layers
- **Function**: Captures patterns at 2-sample intervals
- **Same Architecture**: Gated conv + skip + residual connections

##### D: Dilated Conv Block 4
- **Purpose**: Higher layer with larger receptive field
- **Dilation Rate**: 4 (skips 3 samples between each input)
- **Receptive Field**: Exponential growth (1+2+4 = 7 samples total)
- **Advantage**: Captures long-range dependencies efficiently
- **Pattern**: Dilation doubles each layer (1, 2, 4, 8, 16...)

##### E: Output Distribution
- **Purpose**: Predicts probability distribution over next sample values
- **Computation**: Softmax over 256 possible μ-law values
- **Input**: Summed skip connections from all layers
- **Output**: Probability vector of shape (batch_size, seq_len, 256)
- **Training Target**: Next ground truth audio sample

##### F: Next Sample Prediction
- **Purpose**: Generates next audio sample for autoregressive synthesis
- **Method**: Sample from predicted probability distribution
- **Inference**: Sequential generation (one sample at a time)
- **Quality**: High fidelity due to large receptive field
- **Limitation**: Slow generation speed

#### WaveNet Data Flow Summary
1. **Audio Waveform** → Raw 1D audio sequence
2. **Dilated Conv Block 1** → Basic causal convolution (dilation=1)
3. **Dilated Conv Block 2** → Increased receptive field (dilation=2)
4. **Dilated Conv Block 4** → Large receptive field (dilation=4)
5. **Output Distribution** → Probability distribution over next sample
6. **Next Sample Prediction** → Autoregressive audio generation

#### Hinglish Explanation
WaveNet Architecture ke har component ka purpose:

**A: Audio Waveform**: Raw audio input as time-domain signal

**B: Dilated Conv Block 1**: Pehla dilated convolution layer (dilation rate 1)

**C: Dilated Conv Block 2**: Dusra layer with dilation 2, receptive field expand karta hai

**D: Dilated Conv Block 4**: Higher layer with dilation 4, long-range dependencies capture karta hai

**E: Output Distribution**: Next sample ke possible values ki probability distribution

**F: Next Sample Prediction**: Autoregressive generation ke liye next sample predict karta hai

#### Code Example: Simplified WaveNet Block

```python
import tensorflow as tf
from tensorflow import keras

def wavenet_block(x, filters, kernel_size, dilation_rate):
    # Causal convolution
    x = keras.layers.Conv1D(filters, kernel_size, 
                           padding='causal', 
                           dilation_rate=dilation_rate)(x)
    # Gated activation
    tanh_out = keras.layers.Conv1D(filters, 1, activation='tanh')(x)
    sigm_out = keras.layers.Conv1D(filters, 1, activation='sigmoid')(x)
    x = keras.layers.Multiply()([tanh_out, sigm_out])
    # Skip connection
    skip = keras.layers.Conv1D(filters, 1)(x)
    # Residual connection
    residual = keras.layers.Conv1D(filters, 1)(x)
    return keras.layers.Add()([x, residual]), skip

# Build WaveNet stack
def build_wavenet(num_blocks=4, num_layers=10):
    input_audio = keras.Input(shape=(None, 1))
    skips = []
    
    x = input_audio
    for block in range(num_blocks):
        for layer in range(num_layers):
            dilation = 2 ** layer
            x, skip = wavenet_block(x, 32, 2, dilation)
            skips.append(skip)
    
    # Sum all skip connections
    total_skip = keras.layers.Add()(skips)
    total_skip = keras.layers.ReLU()(total_skip)
    total_skip = keras.layers.Conv1D(1, 1, activation='relu')(total_skip)
    total_skip = keras.layers.Conv1D(1, 1)(total_skip)
    
    # Output logits for 256 possible values (8-bit mu-law)
    output = keras.layers.Conv1D(256, 1)(total_skip)
    output = keras.layers.Softmax()(output)
    
    return keras.Model(input_audio, output)
```

#### WaveRNN
- **Architecture**: Recurrent network with dual RNNs
- **Key Feature**: Faster inference than WaveNet
- **Advantages**: Real-time generation capability
- **Trade-off**: Slightly lower audio quality

#### WaveGlow
- **Architecture**: Flow-based generative model
- **Key Feature**: Parallel generation (non-autoregressive)
- **Advantages**: Fast inference, high quality
- **Use Case**: Voice conversion and synthesis

### Training Considerations
- **Data Preparation**: Mu-law quantization for discrete audio
- **Loss Function**: Categorical cross-entropy for next sample prediction
- **Teacher Forcing**: Use ground truth for training stability

### Applications
- Text-to-speech synthesis
- Music generation
- Audio super-resolution
- Voice conversion

### Advantages of Waveform Models
- Preserve temporal fine structure
- No lossy compression artifacts
- Direct control over waveform generation

### Challenges
- High computational requirements
- Long training times
- Memory-intensive for long sequences

### Hinglish Explanation
Waveform Models raw audio waveforms ke saath directly kaam karte hain, audio ko 1D sequential signals ke roop mein process karte hain.

**Popular Models**:
- **WaveNet**: Dilated convolutions se high-quality audio generate karta hai, par slow inference
- **WaveRNN**: Faster inference ke liye recurrent network
- **WaveGlow**: Parallel generation ke liye flow-based model

**Applications**: TTS, music generation, voice conversion.

**Advantages**: Temporal structure preserve karta hai, direct waveform control.

**Challenges**: High computation, long training.