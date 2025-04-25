import torch
import torchaudio
import librosa
from transformers import Speech2TextProcessor

# Load processor and model
def load_model_and_processor(model_name="facebook/s2t-small-librispeech-asr"):
    """
    Load a speech-to-text processor and model.

    This function initializes a Speech2TextProcessor using a pre-trained model
    specified by the `model_name` parameter. It returns the processor and the
    corresponding model.

    Args:
        model_name (str): The name of the pre-trained model to load. Defaults to
            "facebook/s2t-small-librispeech-asr".

    Returns:
        tuple: A tuple containing:
            - processor (Speech2TextProcessor): The initialized speech-to-text processor.
            - model: The corresponding model (currently undefined in the code).

    Raises:
        ValueError: If the model cannot be loaded or the processor initialization fails.
    """
    processor = Speech2TextProcessor.from_pretrained(model_name)
    return processor

# Load and preprocess audio
def load_audio(filepath, target_sr=16000):
    """
    Load an audio file, convert it to mono if necessary, and resample it to the target sample rate.
    Args:
        filepath (str): Path to the audio file to be loaded.
        target_sr (int, optional): Target sample rate in Hz. Defaults to 16000.
    Returns:
        tuple: A tuple containing:
            - waveform (torch.Tensor): The audio waveform as a 1D tensor.
            - target_sr (int): The sample rate of the returned waveform.
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    return waveform.squeeze(), target_sr

#  Normalize waveform
def normalize_audio(waveform):
    return waveform / waveform.abs().max()

#  Trim silence using librosa
def trim_silence(waveform, sr):
    trimmed, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
    return torch.tensor(trimmed)

#  Prepare inputs for the model
def prepare_inputs(waveform, processor, sampling_rate=16000):
    return processor(waveform, sampling_rate=sampling_rate, return_tensors="pt")




if __name__ == "__main__":
    # Example usage
    model_name = "facebook/s2t-small-librispeech-asr"
    processor = load_model_and_processor(model_name)

    # Path to your audio file
    audio_path = "processors/file_example_WAV_1MG.wav"
    
    # Transcribe the audio file
    waveform, sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)
    waveform = trim_silence(waveform, sr)

    inputs = prepare_inputs(waveform, processor, sr)
    input_features = inputs.input_features
    
    
    
    