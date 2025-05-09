import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Inisialisasi Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

MODEL = "gemini-1.5-flash"  # Gunakan nama model yang tersedia
model = genai.GenerativeModel(model_name=MODEL)

app = FastAPI(title="Intelligent Email Writer API")

# Schema request
class EmailRequest(BaseModel):
    category: str
    recipient: str
    subject: str
    tone: str
    language: str
    urgency_level: Optional[str] = "Biasa"
    points: List[str]
    example_email: Optional[str] = None

# Schema response
class EmailResponse(BaseModel):
    generated_email: str

# Fungsi membentuk prompt
def build_prompt(body: EmailRequest) -> str:
    lines = [
        f"Tolong buatkan email dalam {body.language.lower()} yang {body.tone.lower()}",
        f"kepada {body.recipient}.",
        f"Subjek: {body.subject}.",
        f"Kategori email: {body.category}.",
        f"Tingkat urgensi: {body.urgency_level}.",
        "",
        "Isi email harus mencakup poin-poin berikut:",
    ]
    lines += [f"- {point}" for point in body.points]
    if body.example_email:
        lines += ["", "Contoh email sebelumnya:", body.example_email]
    lines.append("")
    lines.append("Buat email yang profesional, jelas, dan padat.")
    return "\n".join(lines)

# Endpoint untuk generate email
@app.post("/generate/", response_model=EmailResponse)
async def generate_email(req: EmailRequest):
    try:
        prompt = build_prompt(req)

        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=1.0,
            top_k=40,
            max_output_tokens=1024,
        )

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )

        generated = response.text.strip() if response.text else None

        if not generated:
            raise ValueError("Tidak ada hasil yang dihasilkan oleh Gemini API")

        return {"generated_email": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating email: {str(e)}")

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Intelligent Email Writer API running"}
