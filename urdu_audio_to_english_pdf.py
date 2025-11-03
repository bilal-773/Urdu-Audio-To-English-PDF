import streamlit as st
from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pydub import AudioSegment
import torch
import tempfile
import os

# ===== Whisper/WhisperX Setup =====
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    import whisper
    WHISPERX_AVAILABLE = False

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="🎤 Urdu Audio → English PDF", layout="centered")

# ====== Load Model ======
@st.cache_resource
def load_model():
    if WHISPERX_AVAILABLE:
        model = whisperx.load_model("large-v2", device)
    else:
        import whisper
        model = whisper.load_model("large", device=device)
    return model

model = load_model()

# ====== Helper Functions ======
def preprocess_audio(input_path):
    """Convert uploaded audio to mono 16kHz WAV."""
    temp_path = os.path.splitext(input_path)[0] + "_clean.wav"
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000).normalize()
    sound.export(temp_path, format="wav")
    return temp_path


def convert_to_pdf(text, output_path):
    """Create a simple PDF file containing the translated English text."""
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)
    y = height - 50
    c.drawString(50, y, "English Translation:")
    y -= 30

    for line in text.split("\n"):
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
        c.drawString(50, y, line)
        y -= 20

    c.save()
    return output_path


def transcribe_and_translate(audio_path):
    """Transcribe Urdu speech and translate to English."""
    clean_audio = preprocess_audio(audio_path)

    # Step 1: Urdu transcription
    progress_bar = st.progress(0)
    progress_bar.progress(20)
    if WHISPERX_AVAILABLE:
        audio = whisperx.load_audio(clean_audio)
        result = model.transcribe(audio, language="ur")
        urdu_text = result["text"]
    else:
        result = model.transcribe(clean_audio, language="ur")
        urdu_text = result["text"]

    progress_bar.progress(60)

    # Step 2: Urdu → English translation
    english_text = GoogleTranslator(source="ur", target="en").translate(urdu_text)
    progress_bar.progress(100)
    return english_text


# ====== STREAMLIT APP ======
st.title("🎤 Urdu Audio → English PDF Translator")
st.write("Upload an Urdu audio file and get a **translated English PDF transcript** automatically.")

uploaded_file = st.file_uploader("📂 Upload Urdu Audio File", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("⏳ Processing your audio, please wait...")

    try:
        english_text = transcribe_and_translate(tmp_path)
        st.success("✅ Translation complete!")

        st.subheader("📝 English Translation:")
        st.text_area("", english_text, height=250)

        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        convert_to_pdf(english_text, pdf_output)

        with open(pdf_output, "rb") as f:
            st.download_button(
                label="📄 Download English PDF",
                data=f,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_translated.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
