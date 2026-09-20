import os
import io
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

app = FastAPI(
    title="KER Ready Reckoner AI",
    description="Kerala Education Rules AI Assistant"
)

# Static files folder
app.mount("/static", StaticFiles(directory="static"), name="static")

session_documents = {}

SYSTEM_INSTRUCTION = """
നിങ്ങൾ കേരള എജ്യുക്കേഷൻ റൂൾസ് (Kerala Education Rules - KER), സ്കൂൾ വിദ്യാഭ്യാസ ചട്ടങ്ങൾ എന്നിവയിൽ പ്രാവീണ്യമുള്ള ഒരു ഔദ്യോഗിക AI റെഡി റെക്കണർ അസിസ്റ്റന്റാണ്.

നിർദ്ദേശങ്ങൾ:
1. നൽകിയിരിക്കുന്ന വിവരങ്ങളും KER നിയമങ്ങളും അപഗ്രഥിച്ച് മാത്രം മറുപടി നൽകുക.
2. സാധ്യമാകുന്നിടത്തെല്ലാം ബന്ധപ്പെട്ട അധ്യായം (Chapter), ചട്ടം (Rule) എന്നിവ ഉദ്ധരിക്കുക (ഉദാ: KER Chapter XIV-A, Rule 43).
3. ചോദ്യത്തിന് ആധികാരികവും കൃത്യവുമായ ഉത്തരം നൽകുക. വ്യാജ വിവരങ്ങൾ നൽകരുത്.
4. ചോദ്യം മലയാളത്തിലാണെങ്കിൽ മലയാളത്തിൽ വ്യക്തമായി മറുപടി നൽകുക.
"""

def extract_text_from_pdf_stream(stream_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(stream_bytes))
    full_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            full_text.append(f"--- പേജ് {idx + 1} ---\n" + text)
    return "\n\n".join(full_text)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form("default")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="പിഡിഎഫ് (PDF) ഫയലുകൾ മാത്രമേ അപ്‌ലോഡ് ചെയ്യാൻ സാധിക്കൂ.")
    
    try:
        content = await file.read()
        extracted_text = extract_text_from_pdf_stream(content)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="പിഡിഎഫിൽ നിന്ന് ടെക്സ്റ്റ് ലഭ്യമായില്ല.")
            
        session_documents[session_id] = {
            "filename": file.filename,
            "text": extracted_text[:120000]
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"'{file.filename}' വിജയകരമായി അപഗ്രഥിച്ചു."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form("default"),
    api_key: Optional[str] = Form(None)
):
    active_key = (api_key or GEMINI_API_KEY).strip()
    if not active_key:
        raise HTTPException(
            status_code=400, 
            detail="Gemini API Key ലഭ്യമല്ല. ദയവായി സെറ്റിങ്സിൽ API Key നൽകുക."
        )
    
    doc_context = ""
    if session_id in session_documents:
        doc_info = session_documents[session_id]
        doc_context = f"\n\n[ഉപയോക്താവ് അപ്‌ലോഡ് ചെയ്ത ഫയൽ: {doc_info['filename']}]\n{doc_info['text']}"
    
    user_prompt = f"""
{SYSTEM_INSTRUCTION}

റഫറൻസ് വിവരങ്ങൾ:
{doc_context if doc_context else "പ്രത്യേകം ഫയൽ അപ്‌ലോഡ് ചെയ്തിട്ടില്ല. കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER) പൊതു നിയമങ്ങൾ അനുസരിച്ച് മറുപടി നൽകുക."}

ചോദ്യം:
{query}
"""

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro"
    ]

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": user_prompt}
                ]
            }
        ]
    }

    last_error_msg = ""
    # URL ക്വറി പാരാമീറ്റർ (?key=...) വഴി അയക്കുന്നു
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key}"
        try:
            res = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            data = res.json()
            
            if res.status_code == 200:
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    reply_text = "".join([p.get("text", "") for p in parts])
                    return {
                        "response": reply_text,
                        "has_document": bool(doc_context),
                        "document_name": session_documents.get(session_id, {}).get("filename", None)
                    }
            else:
                err = data.get("error", {})
                last_error_msg = err.get("message", res.text)
        except Exception as ex:
            last_error_msg = str(ex)
            continue

    raise HTTPException(status_code=500, detail=f"AI പ്രോസസ്സിംഗിൽ തകരാർ: {last_error_msg}")
