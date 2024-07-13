import pyaudio
import numpy as np
from scipy.fftpack import fft
from playsound import playsound

# Parameters
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (mono)
RATE = 44100  # Sampling rate (samples per second)
THRESHOLD = 100000  # Threshold for detecting human voice

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for human voice...")

try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        # Convert audio data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Perform FFT on the audio data
        fft_data = fft(audio_data)
        # Compute the magnitude spectrum
        magnitude_spectrum = np.abs(fft_data)
        # Compute the average magnitude
        avg_magnitude = np.mean(magnitude_spectrum)

        # Check if the average magnitude exceeds the threshold
        if avg_magnitude > THRESHOLD:
            print("Human voice detected!")
            # Play a sound
            playsound('voice.mp3')

except KeyboardInterrupt:
    print("Stopping...")

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()