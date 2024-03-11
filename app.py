
import streamlit as st
import sounddevice as sd
import soundfile as sf
from pyannote.audio import Pipeline
import openai
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# Get the Pyannote token and OpenAI API key from environment variables
pyannote_token = os.getenv("PYANNOTE_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
 
# Authenticate and download the model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=pyannote_token)
 
# Initialize OpenAI client
openai.api_key = openai_api_key

transcription_buffer = ""  # Buffer to store transcriptions
 
# Define a function to delete audio segment files
def delete_audio_segment_files():
    files = [f for f in os.listdir() if f.startswith('segment_') and f.endswith('.wav')]
    for file in files:
        os.remove(file)
        
def detect_stutter_patterns(text):
    text = re.sub(r"(\b\w\b)\.\s*", r"\1 ", text)
    text = re.sub(r"(\w)-\s*", r"\1 ", text)
 
    repetition_pattern = r"(\b\w+\b)(?:\s+\1\b)+"  # Words repeated consecutively
    prolongation_pattern = r"(\w)\1{2,}"  # Characters repeated more than twice
    interjection_pattern = r"\b(uh|um|ah|uhh|ahh|umm)\b"  # Common interjections# New patterns for stutter detection

    # New patterns for stutter detection
    hesitations_pattern = r"\b(uh|um|uhm|ah|hm|hmm|eh|er)\b"  # Various hesitations
    blocks_pattern = r"\b(\w+)\s*\1\b"  # Word repetitions with a pause between
    syllable_repetition_pattern = r"(\b\w+\b)\w*\s*(\1\w*)+\b"  # Syllable repetitions

    # Detect
    repetitions = re.findall(repetition_pattern, text)
    prolongations = re.findall(prolongation_pattern, text)
    interjections = re.findall(interjection_pattern, text)
    hesitations = re.findall(hesitations_pattern, text)
    blocks = re.findall(blocks_pattern, text)
    syllable_repetitions = re.findall(syllable_repetition_pattern, text)

    # Return detected patterns
    detected = {
        "repetitions": repetitions,
        "prolongations": prolongations,
        "interjections": interjections,
        "hesitations": hesitations,
        "blocks": blocks,
        "syllable_repetitions": syllable_repetitions
    }
    return any(detected.values())

 
def get_completion(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "I'm providing a transcription of people speaker with their speaker ID. Based on the context of the conversation in the transcript please correct/complete the sentence just for last speaker ID and return only that transcript with the corrected/added words in quotes. Here is the transcript: "
                    + prompt,
                },
            ],
        )
        completion = response.choices[0].message.content.strip()
        return completion
    except Exception as e:
        return str(e)
 
# Define the function to process real-time audio from the microphone
def process_realtime_audio(duration=5, filename='audio.wav', samplerate=88200, channels=1):
    global transcription_buffer
    x = ""
    y = ""
    st.info("Recording audio from the microphone...")
    # Start recording from the microphone
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float64')
    sd.wait()  # Wait for recording to complete
 
    # Save the recorded audio to a WAV file
    sf.write(filename, audio_data, samplerate)
 
    st.audio(filename, format='audio/wav')
 
    # Apply speaker diarization pipeline to the recorded audio
    diarization = pipeline(filename)
 
    # Write the diarization output to an RTTM file
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
 
    # Extract diarization segments
    diarized_segments = []
 
    # Iterate through the speaker diarization segments
    for segment in diarization.itertracks(yield_label=True):
        start_time = int(segment[0].start * samplerate)
        end_time = int(segment[0].end * samplerate)
        speaker_id = segment[-1]  # Last element is the speaker ID
        diarized_segments.append((start_time, end_time, speaker_id))
  
    # Iterate through the voice segments for transcription
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = int(turn.start * samplerate)
        end_time = int(turn.end * samplerate)
 
        # Segment audio based on timestamps
        segment_filename = f"segment_{start_time}_{end_time}.wav"
        segment_audio = audio_data[start_time: end_time]
        sf.write(segment_filename, segment_audio, samplerate)
 
        # Transcribe the segmented audio using OpenAI's Speech-to-Text API
        with open(segment_filename, 'rb') as audio_file:
            # Check if the segment duration meets the minimum requirement
            if len(segment_audio) / samplerate >= 0.1:
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    prompt="So uhm, yeaah. ehm, uuuh. like.",
                )
                transcription_text = transcription.text  # Access the text attribute

                transcription_buffer = transcription_buffer + "\n  " + speaker + ":  " + transcription_text
 
                # Output the transcription results
                st.write(
                    f"Speaker {speaker}: {transcription_text}"
                )
            print(transcription_buffer)
            # Detect stutter
            stutter_detected = detect_stutter_patterns(transcription_buffer)
            # Get completion
            if stutter_detected:
                res = get_completion(transcription_buffer)
                st.write("Stutter detected for speaker: ", speaker)
                st.write("Transcript after stutter correction: ", res)
                
                # print(speaker, res)
                # x = x + speaker
                # y = y + res
        
    # print (x,y)
    # st.write("Stutter detected for speaker: ", x)
    # st.write("Transcript after stutter correction: ", y)
    # Visualize the waveform with diarization segments
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple']  # Add more colors as needed
    fig, ax = plt.subplots()
    ax.plot(audio_data[:, 0])
    for start_time, end_time, speaker_id in diarized_segments:
        color_index = ord(speaker_id[-1]) - ord('A')  # Get a unique color index for each speaker
        ax.axvspan(start_time, end_time, color=colors[color_index % len(colors)], alpha=0.3)  # Highlight diarized segments
    st.pyplot(fig)
 
    # Delete audio segment files
    delete_audio_segment_files()
 
# Streamlit app layout and functionality
st.title("Tale - Speaker Diarization and Context based AI completion/correction for Aphasia assistance")
 
duration = st.slider("Select recording duration (seconds):", 1, 15, 5)
start_button = st.button("Start Recording")
 
if start_button:
    # Add lines to move previous data down and display new data for each new recording
    st.write("\n" * 5)  # Add multiple empty lines to move previous data down
    process_realtime_audio(duration=duration)
