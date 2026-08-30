import os
import base64
import json
import re
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="PlantCare AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "PlantCare AI backend is running"
    }


@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = file.content_type

        response = client.chat.completions.create(
           model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are PlantCare AI, an assistant for identifying plants
and possible plant diseases from photographs.

Analyze the uploaded image carefully.

Return ONLY valid JSON in this exact structure, no markdown fences, no preamble, no reasoning text before or after:

{
  "plant": "plant name or Unknown",
  "condition": "Healthy or the specific disease/issue name",
  "confidence": "High, Medium, or Low",
  "symptoms": "brief description of visible symptoms",
  "care": "specific, actionable treatment or care recommendation",
  "warning": "short uncertainty warning if needed, otherwise empty string"
}

Important:
- Do not invent a disease if the image is unclear.
- If the image is not a plant or leaf, say plant is "Unknown" and condition is a brief description of what you see instead.
- If you cannot reliably identify the disease, say condition is "Uncertain" and explain in symptoms.
- Treat the result as an AI assessment, not a guaranteed diagnosis.
- Respond with ONLY the JSON object. Do not include any explanation, reasoning, or thinking process in your response.
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this plant image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            reasoning_effort="none",
            response_format={"type":"json_object"}
        )

        result_text = response.choices[0].message.content or ""

        print("=== RAW RESULT START ===")
        print(result_text)
        print("=== RAW RESULT END ===")

        # Parse the model response into a real JSON object.
        # The model is instructed to return JSON, but this also tolerates
        # accidental surrounding text.
        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\\{.*\\}", result_text, re.DOTALL)

            if not json_match:
                raise ValueError("AI returned invalid JSON.")

            result_data = json.loads(json_match.group(0))

        # Make sure the fields expected by the frontend always exist.
        result_data = {
            "plant": str(result_data.get("plant", "Unknown")),
            "condition": str(result_data.get("condition", "Uncertain")),
            "confidence": str(result_data.get("confidence", "Low")),
            "symptoms": str(result_data.get("symptoms", "")),
            "care": str(result_data.get("care", "")),
            "warning": str(result_data.get("warning", "")),
        }

        print("=== CLEANED RESULT START ===")
        print(json.dumps(result_data, ensure_ascii=False, indent=2))
        print("=== CLEANED RESULT END ===")

        return {
            "success": True,
            "result": result_data
        }

    except Exception as e:
        print("AI ERROR:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Unable to analyze the image: {str(e)}"
        )