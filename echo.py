import streamlit as st
import numpy as np
import wave
from scipy.signal import lfilter

# Echo cancellation using a basic NLMS filter
def echo_cancel(mic_input, ref_signal, filter_len=1024):
    echo_estimate = np.zeros(len(mic_input))
    h = np.zeros(filter_len)
    mu = 0.1

    for i in range(filter_len, len(mic_input)):
        x = ref_signal[i - filter_len:i]
        y = np.dot(h, x)
        e = mic_input[i] - y
        echo_estimate[i] = e
        h = h + mu * e * x / (np.dot(x, x) + 1e-6)
    return echo_estimate

# Load WAV and convert to numpy array
def load_wav(file):
    with wave.open(file, 'rb') as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        framerate = wf.getframerate()
    return audio, framerate

# Save numpy array to WAV
def save_wav(filename, data, framerate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(data.astype(np.int16).tobytes())

# --- Streamlit App ---

st.title("🎧 Echo Cancellation App")

mic_file = st.file_uploader("Upload microphone recording (with echo)", type=["wav"])
ref_file = st.file_uploader("Upload reference playback signal", type=["wav"])

if mic_file and ref_file:
    st.success("Files uploaded!")

    mic_data, rate1 = load_wav(mic_file)
    ref_data, rate2 = load_wav(ref_file)

    if rate1 != rate2:
        st.error("Sampling rates of the two files must match.")
    else:
        # Truncate to shortest length
        min_len = min(len(mic_data), len(ref_data))
        mic_data = mic_data[:min_len]
        ref_data = ref_data[:min_len]

        st.write("Processing...")
        output = echo_cancel(mic_data, ref_data)

        # Save result
        output_filename = "cleaned_output.wav"
        save_wav(output_filename, output, rate1)

        with open(output_filename, "rb") as f:
            st.audio(f.read(), format="audio/wav")
            st.download_button("Download Cleaned Audio", f, file_name="cleaned_output.wav")


streamlit run app.py
