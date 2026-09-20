import os
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="KER Ready Reckoner AI",
    description="Kerala Education Rules & Document Analysis AI Assistant"
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for user session uploaded documents & preloaded KER
session_documents = {}

SYSTEM_PROMPT = """
നിങ്ങൾ കേരള എജ്യുക്കേഷൻ റൂൾസ് (Kerala Education Rules - KER), സ്കൂൾ വിദ്യാഭ്യാസ ചട്ടങ്ങൾ, അനുബന്ധ സർക്കാർ ഉത്തരവുകൾ എന്നിവയിൽ പ്രാവീണ്യമുള്ള ഒരു ഔദ്യോഗിക AI റെഡി റെക്കണർ അസിസ്റ്റന്റാണ് (KER Ready Reckoner AI).

നിർദ്ദേശങ്ങൾ:
1. നൽകിയിരിക്കുന്ന റഫറൻസ് വിവരങ്ങളും (KER Rules Context) അപ്‌ലോഡ് ചെയ്ത ഫയലുകളിലെ വിവരങ്ങളും കൃത്യമായി അപഗ്രഥിച്ച് മാത്രം മറുപടി നൽകുക.
2. മറുപടിയിൽ സാധ്യമാകുന്നിടത്തെല്ലാം ബന്ധപ്പെട്ട അധ്യായം (Chapter), ചട്ടം (Rule), ഉപചട്ടം (Sub-rule) എന്നിവ വ്യക്തമായി ഉദ്ധരിക്കുക (ഉദാ: KER Chapter XIV-A, Rule 43).
3. ചോദ്യത്തിന് വ്യക്തമായ ഉത്തരം നൽകുക. വിവരങ്ങൾ ലഭ്യമല്ലെങ്കിൽ വ്യക്തമായി "ഈ വിവരങ്ങൾ നിലവിലെ രേഖകളിൽ ലഭ്യമല്ല" എന്ന് പറയുക. തെറ്റായ ചട്ടങ്ങളോ വ്യാജ വിവരങ്ങളോ ഉണ്ടാക്കരുത് (No Hallucinations).
4. മലയാളത്തിലാണ് ചോദ്യമെങ്കിൽ വ്യക്തവും ആധികാരികവുമായ മലയാളത്തിലും, ഇംഗ്ലീഷിലാണെങ്കിൽ ഇംഗ്ലീഷിലും മറുപടി നൽകുക.
5. അന്തിമ തീരുമാനങ്ങൾക്ക് ഔദ്യോഗിക ഗസറ്റ് വിജ്ഞാപനങ്ങളും വിദ്യാഭ്യാസ വകുപ്പ് സർക്കുലറുകളും പരിശോധിക്കണമെന്ന് ഓർമ്മിപ്പിക്കുക.
"""

def extract_text_from_pdf_stream(stream_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(stream_bytes))
    full_text = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            full_text.append(f"--- പേജ് / Page {idx + 1} ---\n" + text)
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
            raise HTTPException(status_code=400, detail="പിഡിഎഫിൽ നിന്ന് ടെക്സ്റ്റ് ലഭ്യമായില്ല. സ്കാൻ ചെയ്ത ഇമേജ് PDF ആണെങ്കിൽ OCR ആവശ്യമാണ്.")
            
        session_documents[session_id] = {
            "filename": file.filename,
            "text": extracted_text[:120000] # Limiting context window for safety
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"'{file.filename}' വിജയകരമായി അപഗ്രഥിച്ചു. ഇനി ഇതുമായി ബന്ധപ്പെട്ട ചോദ്യങ്ങൾ ചോദിക്കാം."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form("default"),
    api_key: Optional[str] = Form(None)
):
    active_key = api_key or GEMINI_API_KEY
    if not active_key:
        raise HTTPException(
            status_code=400, 
            detail="Gemini API Key ലഭ്യമല്ല. ദയവായി സെറ്റിങ്സിൽ സ്വന്തം API Key നൽകുക അല്ലെങ്കിൽ .env ഫയലിൽ കോൺഫിഗർ ചെയ്യുക."
        )
    
    genai.configure(api_key=active_key)
    
    doc_context = ""
    if session_id in session_documents:
        doc_info = session_documents[session_id]
        doc_context = f"\n\n[ഉപയോക്താവ് അപ്‌ലോഡ് ചെയ്ത ഫയൽ: {doc_info['filename']}]\n{doc_info['text']}"
    
    user_prompt = f"""
ചോദ്യം (Question):
{query}

റഫറൻസ് ഡാറ്റ (Document Context):
{doc_context if doc_context else "പ്രത്യേകം ഫയൽ അപ്‌ലോഡ് ചെയ്തിട്ടില്ല. കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER) പൊതു നിയമങ്ങൾ അനുസരിച്ച് മറുപടി നൽകുക."}
"""

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(user_prompt)
        return {
            "response": response.text,
            "has_document": bool(doc_context),
            "document_name": session_documents.get(session_id, {}).get("filename", None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI പ്രോസസ്സിംഗിൽ തകരാർ: {str(e)}")
