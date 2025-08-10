from fastapi import FastAPI, File, UploadFile, Form
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
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator,
 collect_chunks) = utils
print("✅ Silero VAD loaded.")

# 🔑 Load API keys (make sure you have a .env file or set env vars)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TTS_API_KEY = os.getenv("TTS_API_KEY")
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID")

WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama3-70b-8192"

print("✅ Loaded keys: GROQ:", bool(GROQ_API_KEY), "TTS:", bool(TTS_API_KEY),
      "VOICE_ID:", bool(TTS_VOICE_ID))

# FastAPI app
app = FastAPI()

# Allow frontend calls from anywhere (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (make sure you have static/index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🧠 Per-user conversation history (for demo, in memory)
conversation_histories = {}


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
# 🎤 Transcribe + respond + TTS with context retention
# ------------------------------
@app.post("/transcribe-and-respond")
async def transcribe_and_respond(
        file: UploadFile = File(...),
        language: str = Form("urdu"),
        user_id: str = Form("default")  # optional, defaults to "default"
):
    file_path = f"temp_{file.filename}"
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    print("📦 Received file:", file.filename)
    print("🌐 Language selected:", language)
    print("👤 User ID:", user_id)

    # Step 1: Transcribe with Whisper
    whisper_resp = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (file.filename, content, "audio/wav")},
        data={"model": WHISPER_MODEL})
    print("📝 Whisper status:", whisper_resp.status_code)
    transcription = whisper_resp.json().get("text", "")
    print("📝 Transcription:", transcription)

    # Step 2: Prepare conversation history
    history = conversation_histories.get(user_id, [])

    system_prompts = {
        "english":
        """
    You are an AI assistant developed by Finova Solutions. This is a demo to showcase Finova’s enterprise-grade AI capabilities. Your tone should be helpful, informative, and engaging.

Respond **only in English**.
**Limit your response to no more than 30 words.**

    When a user interacts with you, identify the industry they are referring to (e.g., healthcare, finance, real estate, hospitality, travel, education, retail, etc.) and dynamically tailor your response to showcase Finova’s AI/ML applications for that sector.

    In your introduction, include:
    1. A brief welcome from Finova Solutions.
    2. A mention that this is a demo of Finova’s AI-powered systems.
    3. A summary of Finova’s AI and ML service offerings: custom AI development, machine learning models, automation, and data intelligence solutions.
    4. Mention that Finova offers AI-powered SaaS products that streamline workflows and internal processes — including ERPs, CRMs, booking systems, and more — suitable for businesses of all sizes.

    Then, based on the industry detected or hinted at by the user, explain how Finova’s AI can help in that specific domain with a few relevant use cases or products.

    End by inviting the user to ask further questions or explore a personalized demo.
    Respond **only in English**.
    """,
        "urdu":
        """
    آپ Finova سولیوشنز کے تیار کردہ ایک AI اسسٹنٹ ہیں۔ یہ ڈیمو Finova کی انٹرپرائز سطح کی AI صلاحیتوں کو ظاہر کرنے کے لیے بنایا گیا ہے۔ آپ کا انداز مددگار، معلوماتی اور دوستانہ ہونا چاہیے۔

    براہ کرم صرف اردو زبان میں جواب دیں۔ ۔
    اپنے جواب کو 30 الفاظ سے زیادہ تک محدود نہ رکھیں۔

    جب کوئی صارف بات کرتا ہے تو اس بات کا اندازہ لگائیں کہ وہ کس صنعت کا ذکر کر رہا ہے (جیسے صحت، مالیات، رئیل اسٹیٹ، ہاسپیٹیلٹی، تعلیم، ریٹیل وغیرہ) اور اس کے مطابق Finova کی AI/ML ایپلی کیشنز اور حل دکھائیں۔

    اپنے تعارف میں شامل کریں:
    1. Finova سولیوشنز کی طرف سے خوش آمدید۔
    2. یہ بتائیں کہ یہ Finova کے AI سسٹمز کا ایک ڈیمو ہے۔
    3. Finova کی AI اور مشین لرننگ خدمات کا خلاصہ، جیسے کسٹم ماڈلز، آٹومیشن، اور ڈیٹا انٹیلیجنس۔
    4. یہ بتائیں کہ Finova AI سے چلنے والی SaaS مصنوعات بھی فراہم کرتا ہے، جیسے ERP، CRM، بکنگ سسٹمز وغیرہ، جو ہر سائز کے کاروبار کے لیے موزوں ہیں۔

    آخر میں صارف کو دعوت دیں کہ وہ مزید سوالات کرے یا ذاتی نوعیت کا ڈیمو دیکھے۔

    براہ کرم صرف اردو زبان میں جواب دیں۔۔
    """,
        "arabic":
        """
    أنت مساعد ذكي تم تطويره بواسطة Finova Solutions. هذا العرض التوضيحي يُظهر قدرات Finova في مجال الذكاء الاصطناعي على مستوى المؤسسات. يجب أن تكون نبرة صوتك ودودة، مفيدة، ومليئة بالمعلومات.
    يرجى الرد فقط باللغة العربية وعدم استخدام اللغة الإنجليزية.
    حدد إجابتك بما لا يزيد عن 30 كلمة.

    عند تفاعل المستخدم معك، حدّد القطاع أو الصناعة المشار إليها (مثل الرعاية الصحية، المالية، العقارات، السياحة، التعليم، البيع بالتجزئة، وغيرها) وخصص ردودك لعرض تطبيقات Finova في هذا القطاع.

    في مقدمتك، قم بما يلي:
    1. رحب بالمستخدم باسم Finova Solutions.
    2. اذكر أن هذا عرض توضيحي لنظام Finova المدعوم بالذكاء الاصطناعي.
    3. قدّم ملخصاً عن خدمات Finova مثل تطوير النماذج المخصصة، التعلم الآلي، الأتمتة، وحلول ذكاء البيانات.
    4. اشرح أن Finova تقدم منتجات SaaS ذكية مثل أنظمة الـ ERP، CRM، وأنظمة الحجز، والمزيد — مناسبة لجميع أحجام الشركات.

    اختم بدعوة المستخدم لطرح المزيد من الأسئلة أو تجربة عرض مخصص.
    يرجى التحدث باللغة العربية فقط
    """
    }

    system_prompt = system_prompts.get(language.lower(),
                                       system_prompts["urdu"])

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": transcription})

    # Step 3: Generate response from LLM
    llm_resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": messages
        })
    print("🤖 LLM status:", llm_resp.status_code)
    llm_json = llm_resp.json()
    print("🤖 LLM raw response:", llm_json)

    llm_output = llm_json.get("choices", [{}])[0].get("message",
                                                      {}).get("content", "")
    if not llm_output:
        print("⚠️ LLM returned empty or error.")
        llm_output = "معذرت، جواب تیار کرتے ہوئے ایک خامی پیش آئی۔"

    words = llm_output.split()
    if len(words) > 30:
        truncated = " ".join(words[:30])
        # Try to cut at the last sentence-ending punctuation
        for end_char in ["۔", ".", "!", "؟"]:
            if end_char in truncated:
                truncated = truncated.rsplit(end_char, 1)[0] + end_char
                break
        llm_output = truncated
    
    print("🤖 LLM output:", llm_output)

    # Update conversation history
    history.append({"role": "user", "content": transcription})
    history.append({"role": "assistant", "content": llm_output})
    conversation_histories[user_id] = history

    # Step 4: Convert response to speech (TTS)
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
        stream=True)
    print("🔊 TTS status:", tts_resp.status_code)

    if tts_resp.status_code != 200:
        print("❌ TTS failed:", tts_resp.text)
        return {"error": "TTS failed", "details": tts_resp.text}

    return StreamingResponse(tts_resp.iter_content(1024),
                             media_type="audio/mpeg")


# ------------------------------
# ✅ Run server locally (optional)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
