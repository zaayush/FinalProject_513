import streamlit as st
import sounddevice as sd
import soundfile as sf
from pyannote.audio import Pipeline
import torch
import time

# Authenticate and download the model
# Replace "${YOUR_AUTH_TOKEN}" with your access token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Define the function to process real-time audio from the microphone
def process_realtime_audio(duration=5, filename='audio.wav', samplerate=44100, channels=2):
    st.info("Recording audio from the microphone...")
    # Start recording from the microphone
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float64')
    sd.wait()  # Wait for recording to complete

    # Save the recorded audio to a WAV file
    sf.write(filename, audio_data, samplerate)
    # Add a delay of 2 seconds

    st.audio(filename, format='audio/wav')
    # Apply speaker diarization pipeline to the recorded audio
    diarization = pipeline(filename)

    # Write the diarization output to an RTTM file
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    st.info("Diarization output saved to audio.rttm")
# Streamlit app layout and functionality
st.title("Real-time Speaker Diarization")

duration = st.slider("Select recording duration (seconds):", 1, 10, 5)
start_button = st.button("Start Recording")

if start_button:
    process_realtime_audio(duration=duration)


    # Display diarization results

   
