import streamlit as st
from pyannote.audio import Pipeline

# Authenticate and download the model
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# Streamlit app layout and functionality
st.title("Speaker Diarization App")

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("**Uploaded Audio**")

    # Diarization button
    if st.button("Perform Speaker Diarization"):
        # Apply pretrained pipeline
        diarization = pipeline(uploaded_file)

        # Display the result
        st.subheader("Diarization Result:")
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            st.write(f"Start={turn.start:.1f}s Stop={turn.end:.1f}s Speaker_{speaker}")
