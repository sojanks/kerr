import os
import io
import base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
import requests

app = FastAPI(title="KER Reckoner Document Deep AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Jiwnj7ujrV4fkeMrcJY3EpevhYtLQJZ3O1zWX9xvO-3g")

# സെഷൻ അടിസ്ഥാനത്തിൽ അപ്‌ലോഡ് ചെയ്ത ഫയലുകൾ സൂക്ഷിക്കുന്നു
session_store = {}

DEEP_SYSTEM_INSTRUCTION = """
നിങ്ങൾ കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER), സർക്കാർ സ്കൂൾ സർക്കുലറുകൾ, ഉത്തരവുകൾ എന്നിവ ആഴത്തിൽ വിശകലനം ചെയ്യുന്ന ഒരു ലീഗൽ & അഡ്മിനിസ്ട്രേറ്റീവ് AI വിദഗ്ദ്ധനാണ്.

നിർണ്ണായക നിർദ്ദേശങ്ങൾ:
1. ഉപയോക്താവ് ഒരു ഫയൽ (Document/PDF) നൽകിയിട്ടുണ്ടെങ്കിൽ, നിങ്ങളുടെ മറുപടിയിൽ ആ ഫയലിലെ വിവരങ്ങൾക്ക് നിർബന്ധമായും 100% മുൻഗണന നൽകണം.
2. ഫയലിൽ പറയുന്ന ഉത്തരവ് നമ്പർ (Order No / Circular No), തീയതി, വിഷയം, പേര്, തസ്തിക, പ്രധാന നിർദ്ദേശങ്ങൾ എന്നിവ വ്യക്തമായി മറുപടിയിൽ ഉദ്ധരിക്കണം.
3. ഫയലിലെ ഉത്തരവ് എങ്ങനെ KER ചട്ടങ്ങളുമായി (Chapter & Rule) പൊരുത്തപ്പെടുന്നുവെന്ന് ഘട്ടം ഘട്ടമായി (Step-by-Step Analysis) പരിശോധിക്കുക.
4. വിവരങ്ങൾ വ്യക്തവും ആധികാരികവുമായ മലയാളത്തിൽ നൽകുക.
"""

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form("default")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF ഫയലുകൾ മാത്രമേ സ്വീകരിക്കുകയുള്ളൂ.")
    
    try:
        content = await file.read()
        
        # 1. ടെക്സ്റ്റ് എക്സ്ട്രാക്റ്റ് ചെയ്യുന്നു
        extracted_text = ""
        try:
            reader = PdfReader(io.BytesIO(content))
            for idx, page in enumerate(reader.pages):
                t = page.extract_text() or ""
                if t.strip():
                    extracted_text += f"[Page {idx + 1}]\n{t}\n"
        except Exception:
            extracted_text = ""

        # 2. സ്കാൻ ചെയ്ത രേഖകൾക്കായി Base64 ഫോർമാറ്റിലേക്ക് മാറ്റുന്നു (Gemini Vision OCR)
        b64_data = base64.b64encode(content).decode("utf-8")

        session_store[session_id] = {
            "filename": file.filename,
            "text": extracted_text,
            "base64": b64_data,
            "has_ocr": bool(b64_data)
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "ഫയൽ AI വിജയകരമായി ഉൾക്കൊണ്ടു."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form("default"),
    api_key: Optional[str] = Form(None)
):
    active_key = (api_key or DEFAULT_GEMINI_KEY).strip()
    if not active_key:
        raise HTTPException(status_code=400, detail="API Key ലഭ്യമല്ല.")

    parts = []
    has_file = session_id in session_store

    # ഫയൽ ഉണ്ടെങ്കിൽ അത് നേരിട്ട് AI-ലേക്ക് ചേർക്കുന്നു (Multimodal OCR)
    if has_file:
        file_info = session_store[session_id]
        
        # ഫയലിലെ ടെക്സ്റ്റ് ഉണ്ടെങ്കിൽ അത് ചേർക്കുന്നു
        if file_info.get("text"):
            parts.append({
                "text": f"--- അപ്‌ലോഡ് ചെയ്ത രേഖയുടെ ഉള്ളടക്കം ({file_info['filename']}) ---\n{file_info['text']}\n"
            })
        
        # സ്കാൻ ചെയ്ത ഫയലുകൾക്കായി PDF Base64 നേരിട്ട് നൽകുന്നു
        if file_info.get("base64"):
            parts.append({
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": file_info["base64"]
                }
            })

        user_instruction = f"""
നിർദ്ദേശം: മുകളിൽ നൽകിയിരിക്കുന്ന അപ്‌ലോഡ് ചെയ്ത രേഖ ({file_info['filename']}) പൂർണ്ണമായി പരിശോധിച്ച്, അതിലെ ഉള്ളടക്കം കൃത്യമായി ഉദ്ധരിച്ച് താഴെ പറയുന്ന ചോദ്യത്തിന് മറുപടി നൽകുക:

ചോദ്യം:
{query}
"""
        parts.append({"text": user_instruction})
    else:
        parts.append({
            "text": f"കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER Chapters I to XXXII) അടിസ്ഥാനമാക്കി താഴെ പറയുന്ന ചോദ്യത്തിന് ഘട്ടം ഘട്ടമായി മറുപടി നൽകുക:\n\nചോദ്യം: {query}"
        })

    payload = {
        "contents": [{"parts": parts}],
        "system_instruction": {
            "parts": [{"text": DEEP_SYSTEM_INSTRUCTION}]
        }
    }

    # Vision & OCR പിന്തുണയ്ക്കുന്ന ഏറ്റവും പുതിയ സജീവ മോഡലുകൾ
    models = ["gemini-2.5-flash", "gemini-2.0-flash"]

    last_error = ""
    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": active_key
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=120)
            data = res.json()

            if res.status_code == 200:
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    reply_parts = candidates[0]["content"].get("parts", [])
                    reply = "".join([p.get("text", "") for p in reply_parts])
                    return {
                        "response": reply,
                        "document_used": file_info["filename"] if has_file else None
                    }
            else:
                err = data.get("error", {})
                last_error = err.get("message", res.text)
        except Exception as ex:
            last_error = str(ex)
            continue

    raise HTTPException(status_code=500, detail=f"AI പ്രോസസ്സിംഗ് തകരാർ: {last_error}")
