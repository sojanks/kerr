import os
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
import requests

app = FastAPI(title="KER Reckoner Deep AI")
app.mount("/static", StaticFiles(directory="static"), name="static")

# നിങ്ങളുടെ പുതിയ API കീ ഡിഫോൾട്ടായി നൽകുന്നു
DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Jiwnj7ujrV4fkeMrcJY3EpevhYtLQJZ3O1zWX9xvO-3g")

session_documents = {}

# ആഴത്തിലുള്ള നിയമ വിശകലനത്തിനായുള്ള സിസ്റ്റം നിർദ്ദേശങ്ങൾ
DEEP_SYSTEM_INSTRUCTION = """
നിങ്ങൾ കേരള എജ്യുക്കേഷൻ റൂൾസ് (Kerala Education Rules - KER), സ്കൂൾ വിദ്യാഭ്യാസ ചട്ടങ്ങൾ, സർവീസ് നിയമങ്ങൾ എന്നിവയിൽ അതീവ പ്രാവീണ്യമുള്ള ഒരു ലീഗൽ & അഡ്മിനിസ്ട്രേറ്റീവ് AI കൺസൾട്ടന്റാണ്.

ഒരു ചോദ്യം ലഭിക്കുമ്പോൾ സ്റ്റാറ്റിക് ഉത്തരങ്ങൾ നൽകാതെ, യഥാർത്ഥ AI ഇന്റലിജൻസ് ഉപയോഗിച്ച് താഴെ പറയുന്ന ഘട്ടങ്ങളിലൂടെ (Step-by-Step Deep Analysis) ആഴത്തിൽ അപഗ്രഥിച്ച് മാത്രം മറുപടി നൽകുക:

1. **വിഷയ സംഗ്രഹം (Core Legal Issue):** ചോദ്യകർത്താവ് ഉന്നയിച്ചിരിക്കുന്ന നിയമപരമായ പ്രശ്നം എന്താണെന്ന് വ്യക്തമാക്കുക.
2. **ബാധകമായ ചട്ടങ്ങൾ (Statutory Authority):** ബന്ധപ്പെട്ട KER അധ്യായം (Chapter I to XXXII), ചട്ടം (Rule), ഉപചട്ടം (Sub-rule), സർക്കാർ ഗസറ്റ് ഉത്തരവുകൾ എന്നിവ കൃത്യമായി ഉദ്ധരിക്കുക (ഉദാ: KER Chapter XIV-A Rule 43 / Rule 44A / Rule 51A).
3. **ആഴത്തിലുള്ള യുക്തിചിന്തയും വ്യാഖ്യാനവും (Step-by-Step Legal Analysis & Precedents):** 
   - തന്നിരിക്കുന്ന സാഹചര്യത്തിൽ നിയമം എങ്ങനെ പ്രയോഗിക്കപ്പെടുന്നു?
   - യോഗ്യത, സീനിയോറിറ്റി, മുൻഗണന, പ്രായപരിധി ഇളവുകൾ എന്നിവ തമ്മിലുള്ള താരതമ്യം.
   - അപ്‌ലോഡ് ചെയ്ത രേഖകൾ ഉണ്ടെങ്കിൽ അതിലെ നിബന്ധനകൾ കൂടി പരിശോധിച്ച് താരതമ്യം ചെയ്യുക.
4. **അന്തിമ തീരുമാനം (Conclusion & Actionable Advice):** ഫയൽ തീർപ്പാക്കലിനോ അപ്പീലിനോ ഔദ്യോഗികമായി സ്വീകരിക്കേണ്ട അന്തിമ തീരുമാനം നൽകുക.

ശ്രദ്ധിക്കുക: വ്യാജ വിവരങ്ങളോ ഇല്ലാത്ത ചട്ടങ്ങളോ ഉണ്ടാക്കരുത്. ഉയർന്ന നിലവാരമുള്ള ആധികാരിക മലയാളത്തിൽ മറുപടി നൽകുക.
"""

def extract_text_from_pdf(stream_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(stream_bytes))
    full_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            full_text.append(f"[പേജ് {idx + 1}]\n{text}")
    return "\n\n".join(full_text)

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
        extracted = extract_text_from_pdf(content)
        if not extracted.strip():
            raise HTTPException(status_code=400, detail="PDF-ൽ നിന്ന് ടെക്സ്റ്റ് കണ്ടെത്താൻ കഴിഞ്ഞില്ല.")
            
        session_documents[session_id] = {
            "filename": file.filename,
            "text": extracted[:150000]
        }
        return {"status": "success", "filename": file.filename}
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
        raise HTTPException(status_code=400, detail="Gemini API Key ലഭ്യമല്ല.")

    doc_context = ""
    if session_id in session_documents:
        doc = session_documents[session_id]
        doc_context = f"\n\n[ഉപയോക്താവ് അപ്‌ലോഡ് ചെയ്ത സർക്കുലർ/രേഖ: {doc['filename']}]\n{doc['text']}"

    prompt_content = f"""
{DEEP_SYSTEM_INSTRUCTION}

ഉപയോക്താവിന്റെ ചോദ്യം (User Query):
{query}

റഫറൻസ് രേഖകൾ (Reference Context):
{doc_context if doc_context else "പ്രത്യേകം ഫയൽ അപ്‌ലോഡ് ചെയ്തിട്ടില്ല. കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER Chapters I to XXXII) അടിസ്ഥാനമാക്കി ഘട്ടം ഘട്ടമായി ആഴത്തിൽ അപഗ്രഥിക്കുക."}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_content}
                ]
            }
        ]
    }

    # AQ. കീകളെയും AIza കീകളെയും ഒരുപോലെ സപ്പോർട്ട് ചെയ്യുന്ന ഗൂഗിളിന്റെ നിലവിലെ സജീവ മോഡലുകൾ
    active_models = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite"
    ]

    last_error = ""
    for model_name in active_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        # പുതിയ AQ. Auth കീകളെ കൃത്യമായി സ്വീകരിക്കുന്ന x-goog-api-key ഹെഡർ ഉപയോഗിക്കുന്നു
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": active_key
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=90)
            data = res.json()

            if res.status_code == 200:
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    reply = "".join([p.get("text", "") for p in parts])
                    return {
                        "response": reply,
                        "is_deep_ai": True,
                        "document": session_documents.get(session_id, {}).get("filename", None)
                    }
            else:
                err = data.get("error", {})
                last_error = err.get("message", res.text)
        except Exception as ex:
            last_error = str(ex)
            continue

    raise HTTPException(status_code=500, detail=f"AI അപഗ്രഥനത്തിൽ തകരാർ: {last_error}")
