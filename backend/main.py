from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import requests
import aiofiles
import os
import io
from dotenv import load_dotenv

import torch

# 🧠 Load Silero VAD model & utils
print("🔄 Loading Silero VAD model...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("✅ Silero VAD loaded.")

# 🔑 Load API keys
load_dotenv() 

# FastAPI app
app = FastAPI()

# Allow frontend calls from anywhere (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID")

WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama3-70b-8192"

print("✅ Loaded keys: GROQ:", bool(GROQ_API_KEY), "TTS:", bool(TTS_API_KEY), "VOICE_ID:", bool(TTS_VOICE_ID))

# Serve static files (make sure you have static/index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# ------------------------------
# 🧠 Detect speech using Silero VAD
# ------------------------------
@app.post("/detect-voice")
async def detect_voice(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    print(f"📥 Received chunk: {len(audio_bytes)} bytes")

    # Read waveform (expects 16 kHz mono wav)
    wav = read_audio(io.BytesIO(audio_bytes), sampling_rate=16000)

    # Detect speech timestamps
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
    print(f"🔍 speech_timestamps: {speech_timestamps}")

    speech_detected = len(speech_timestamps) > 0
    return {"speech_detected": speech_detected}

# ------------------------------
# 🎤 Transcribe + respond + TTS
# ------------------------------
@app.post("/transcribe-and-respond")
async def transcribe_and_respond(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    print("📦 Received file:", file.filename)

    # Step 1: Transcribe with Whisper
    whisper_resp = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (file.filename, content, "audio/wav")},
        data={"model": WHISPER_MODEL}
    )
    print("📝 Whisper status:", whisper_resp.status_code)
    transcription = whisper_resp.json().get("text", "")
    print("📝 Transcription:", transcription)

    # Step 2: Generate response from LLaMA
    system_prompt = (
        "آپ ایک مددگار، انسانی جیسے اسسٹنٹ ہیں۔ صرف اور صرف اردو زبان میں مکمل جواب دیں۔ "
        "جواب میں انگریزی الفاظ یا جملے استعمال نہ کریں۔"
    )
    llm_resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription}
            ]
        }
    )
    print("🤖 LLM status:", llm_resp.status_code)
    llm_json = llm_resp.json()
    print("🤖 LLM raw response:", llm_json)

    llm_output = llm_json.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not llm_output:
        print("⚠️ LLM returned empty or error.")
        llm_output = "معذرت، جواب تیار کرتے ہوئے ایک خامی پیش آئی۔"
    print("🤖 LLM output:", llm_output)

    # Step 3: Convert response to speech (ElevenLabs)
    tts_resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}",
        headers={
            "xi-api-key": TTS_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": llm_output,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        },
        stream=True
    )
    print("🔊 TTS status:", tts_resp.status_code)

    if tts_resp.status_code != 200:
        print("❌ TTS failed:", tts_resp.text)
        return {"error": "TTS failed", "details": tts_resp.text}

    return StreamingResponse(tts_resp.iter_content(1024), media_type="audio/mpeg")
