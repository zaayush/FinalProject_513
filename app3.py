import streamlit as st
import sounddevice as sd
import soundfile as sf
import torch
import time
import whisper  # Import the Whisper model for transcription
from pyannote.audio import Pipeline

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
    
    st.info("Diarization completed.")

    # Perform transcription on the recorded audio
    st.info("Performing transcription...")
    audio_file = "audio.wav"
    sf.write(audio_file, audio_data, samplerate)

    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio
    result = model.transcribe(audio_file)

    st.success("Transcription completed.")

    # Display diarization and transcription results
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st.write(f"Speaker {speaker}: start={turn.start:.1f}s, end={turn.end:.1f}s")
        # Attach transcript with the speaker ID based on time stamps
        for transcription in result["transcriptions"]:
            if transcription["start"] >= turn.start and transcription["end"] <= turn.end:
                st.write(f"Transcript: {transcription['text']}")

    st.audio(audio_file, format='audio/wav')

# Streamlit app layout and functionality
st.title("Real-time Speaker Diarization and Transcription")

duration = st.slider("Select recording duration (seconds):", 1, 10, 5)
start_button = st.button("Start Recording")

if start_button:
    process_realtime_audio(duration=duration)
