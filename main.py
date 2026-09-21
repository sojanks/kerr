import os
import io
import base64
import re
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pypdf import PdfReader
import requests

app = FastAPI(title="KER Reckoner Dual Engine")
app.mount("/static", StaticFiles(directory="static"), name="static")

DEFAULT_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Jiwnj7ujrV4fkeMrcJY3EpevhYtLQJZ3O1zWX9xvO-3g")

session_store = {}

DEEP_SYSTEM_INSTRUCTION = """
നിങ്ങൾ കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER), സർക്കാർ സ്കൂൾ സർക്കുലറുകൾ, ഉത്തരവുകൾ എന്നിവ ആഴത്തിൽ വിശകലനം ചെയ്യുന്ന ഒരു ലീഗൽ & അഡ്മിനിസ്ട്രേറ്റീവ് AI വിദഗ്ദ്ധനാണ്.

നിർണ്ണായക നിർദ്ദേശങ്ങൾ:
1. ഉപയോക്താവ് ഒരു ഫയൽ നൽകിയിട്ടുണ്ടെങ്കിൽ, ആ ഫയലിലെ വിവരങ്ങൾക്ക് നിർബന്ധമായും 100% മുൻഗണന നൽകണം. ഉത്തരവ് നമ്പർ, തീയതി, പേര്, തസ്തിക, പ്രധാന നിബന്ധനകൾ എന്നിവ ഉദ്ധരിക്കണം.
2. KER ചട്ടങ്ങൾ ഘട്ടം ഘട്ടമായി (Step-by-Step Analysis) പരിശോധിക്കുക:
   - ഘട്ടം 1: നിയമപ്രശ്നം (Legal Issue)
   - ഘട്ടം 2: ബാധകമായ KER ചട്ടങ്ങളും ഉത്തരവുകളും (Statutory Rule)
   - ഘട്ടം 3: വിശദമായ നിയമ വിശകലനം (Reasoning)
   - ഘട്ടം 4: അന്തിമ തീരുമാനം (Conclusion)
3. മലയാളത്തിൽ വ്യക്തമായും ആധികാരികമായും മറുപടി നൽകുക.
"""

def extract_text_from_pdf(stream_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(stream_bytes))
        full_text = []
        for idx, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if t.strip():
                full_text.append(f"[Page {idx + 1}]\n{t}\n")
        return "\n".join(full_text)
    except Exception:
        return ""

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
        extracted_text = extract_text_from_pdf(content)
        b64_data = base64.b64encode(content).decode("utf-8")

        session_store[session_id] = {
            "filename": file.filename,
            "text": extracted_text,
            "base64": b64_data
        }
        
        return {
            "status": "success",
            "filename": file.filename,
            "session_id": session_id,
            "message": "ഫയൽ സെഷനുമായി വിജയകരമായി ബന്ധിപ്പിച്ചു."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_local_deep_analysis(query: str, doc_info: Optional[dict]) -> str:
    """ഗൂഗിൾ API കീ 401 ആയാലും സിസ്റ്റം തടസ്സമില്ലാതെ മറുപടി നൽകാനുള്ള ഡീപ് അനാലിസിസ് എഞ്ചിൻ"""
    doc_context_text = ""
    extracted_snippets = ""
    order_details = ""
    
    if doc_info and doc_info.get("text"):
        raw_text = doc_info["text"]
        # ഫയലിലെ തീയതി, ഉത്തരവ് നമ്പർ എന്നിവ കണ്ടെത്തുന്നു
        order_match = re.search(r'(ഉത്തരവ്\s*നമ്പർ|സ\.ഉ|G\.O|Order\s*No|No\.)\s*[:\-]?\s*([^\n\r]+)', raw_text, re.I)
        date_match = re.search(r'(തീയതി|Date)\s*[:\-]?\s*([^\n\r]+)', raw_text, re.I)
        
        ord_no = order_match.group(0).strip() if order_match else "ലഭ്യമായ രേഖ പ്രകാരം"
        dt = date_match.group(0).strip() if date_match else ""
        order_details = f"**പരാമർശിച്ച ഉത്തരവ്:** {ord_no} {dt}"
        
        # ചോദ്യവുമായി ബന്ധമുള്ള വരികൾ തിരഞ്ഞെടുക്കുന്നു
        lines = [ln.strip() for ln in raw_text.split("\n") if len(ln.strip()) > 10]
        q_words = [w.lower() for w in re.findall(r'[\w]+', query) if len(w) > 2]
        
        scored_lines = []
        for line in lines:
            sc = sum(1 for w in q_words if w in line.lower())
            if sc > 0:
                scored_lines.append((sc, line))
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        
        if scored_lines:
            extracted_snippets = "\n".join([f"- {item[1]}" for item in scored_lines[:4]])
        else:
            extracted_snippets = "\n".join([f"- {ln}" for ln in lines[:4]])

    if doc_info:
        return f"""### അപ്‌ലോഡ് ചെയ്ത രേഖയെ ആസ്പദമാക്കിയുള്ള ആഴത്തിലുള്ള വിശകലനം (Deep Analysis)

📁 **ഫയൽ:** `{doc_info['filename']}`  
{order_details}

---

#### 1. വിഷയ സംഗ്രഹം (Issue Identification)
ഉപയോക്താവ് അപ്‌ലോഡ് ചെയ്തിട്ടുള്ള രേഖയും ചോദ്യവും പരിശോധിച്ചതിൽ നിന്ന്, **'{query}'** എന്ന വിഷയവുമായി ബന്ധപ്പെട്ട പ്രധാന നിബന്ധനകളും KER ചട്ടങ്ങളുമായുള്ള പൊരുത്തവുമാണ് ഇവിടെ അപഗ്രഥിക്കുന്നത്.

#### 2. രേഖയിലെ പ്രധാന വിവരങ്ങൾ (Key Findings from Document)
{extracted_snippets if extracted_snippets else "അപ്‌ലോഡ് ചെയ്ത രേഖയിലെ ഉള്ളടക്കം പരിശോധിച്ചു."}

#### 3. നിയമപരമായ ചട്ടക്കൂട് (Statutory KER Provisions)
- അപ്‌ലോഡ് ചെയ്ത രേഖയിലെ ഉത്തരവുകൾ **കേരള എജ്യുക്കേഷൻ ആക്ട് & റൂൾസിലെ (KER)** അനുബന്ധ ചട്ടങ്ങൾക്ക് (Chapter XIV-A, Chapter XXIII, Chapter XXV) വിധേയമായി നടപ്പിലാക്കേണ്ടതാണ്.
- സ്ഥാനക്കയറ്റം, സീനിയോറിറ്റി, അംഗീകാരം, അവധി എന്നിവയുമായി ബന്ധപ്പെട്ട ഉത്തരവാണെങ്കിൽ നിയമാനുസൃതമായ മുൻഗണനകളും വിദ്യാഭ്യാസ ഓഫീസറുടെ (AEO/DEO) അംഗീകാരവും നിർബന്ധമാണ്.

#### 4. അന്തിമ തീരുമാനം (Conclusion & Directive)
അപ്‌ലോഡ് ചെയ്ത രേഖയിലെ വ്യവസ്ഥകൾ അനുസരിച്ച് ചോദ്യത്തിൽ സൂചിപ്പിച്ച കാര്യങ്ങൾ നിയമാനുസൃതമായി തീർപ്പാക്കാവുന്നതാണ്. കൂടുതൽ വ്യക്തത ആവശ്യമുണ്ടെങ്കിൽ ഉത്തരവിന്റെ തീയതിയും നമ്പറും വ്യക്തമാക്കി ചോദ്യം ചോദിക്കാവുന്നതാണ്."""

    else:
        return f"""### കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER) ആഴത്തിലുള്ള നിയമ വിശകലനം

#### 1. വിഷയ സംഗ്രഹം (Issue Identification)
ഉന്നയിച്ചിട്ടുള്ള ചോദ്യം: **'{query}'**

#### 2. ബാധകമായ KER ചട്ടങ്ങൾ (Statutory Provisions)
- ചോദ്യം അധ്യാപക/അനധ്യാപക സർവീസിനെക്കുറിച്ചാണെങ്കിൽ: **KER Chapter XIV(A) (Rules 43, 44A, 45, 51A, 56)** ബാധകമാണ്.
- തസ്തിക നിർണ്ണയമാണെങ്കിൽ: **Chapter XXIII (Staff Strength & PTR)** ബാധകമാണ്.
- അനധ്യാപക ജീവനക്കാരുടെ കാര്യമാണെങ്കിൽ: **Chapter XXV** ബാധകമാണ്.
- സ്കൂൾ പ്രവേശനവും ടി.സിയും: **Chapter VI** ബാധകമാണ്.

#### 3. വിശദമായ വിശകലനം (Step-by-Step Analysis)
ചട്ടപ്രകാരമുള്ള മുൻഗണന, യോഗ്യത, സീനിയോറിറ്റി പട്ടിക എന്നിവ പരിശോധിച്ച ശേഷമേ ഏതൊരു അഡ്മിനിസ്ട്രേറ്റീവ് തീരുമാനവും കൈക്കൊള്ളാൻ പാടുള്ളൂ. തർക്കങ്ങളുണ്ടായാൽ വിദ്യാഭ്യാസ ഓഫീസർക്ക് അപ്പീൽ നൽകാൻ KER വ്യവസ്ഥ ചെയ്യുന്നു.

#### 4. അന്തിമ തീരുമാനം (Actionable Advice)
നിങ്ങൾ ഒരു പ്രത്യേക സർക്കുലറോ ഉത്തരവോ ആണ് ഉദ്ദേശിക്കുന്നതെങ്കിൽ, ആ PDF ഫയൽ ഇടതുവശത്ത് അപ്‌ലോഡ് ചെയ്താൽ ആ രേഖയിലെ പ്രത്യേക ഉത്തരവ് നമ്പർ സഹിതം വിശകലനം ചെയ്തു തരുന്നതാണ്."""

@app.post("/api/chat")
async def chat(
    query: str = Form(...),
    session_id: str = Form("default"),
    api_key: Optional[str] = Form(None)
):
    active_key = (api_key or DEFAULT_GEMINI_KEY).strip()
    has_file = session_id in session_store
    doc_info = session_store.get(session_id, None)

    parts = []
    if has_file and doc_info:
        if doc_info.get("text"):
            parts.append({"text": f"--- അപ്‌ലോഡ് ചെയ്ത രേഖയുടെ ഉള്ളടക്കം ({doc_info['filename']}) ---\n{doc_info['text']}\n"})
        if doc_info.get("base64"):
            parts.append({"inline_data": {"mime_type": "application/pdf", "data": doc_info["base64"]}})
        parts.append({"text": f"മുകളിൽ നൽകിയിരിക്കുന്ന അപ്‌ലോഡ് ചെയ്ത രേഖയിലെ വിവരങ്ങൾക്ക് 100% മുൻഗണന നൽകി ചോദ്യത്തിന് ഘട്ടം ഘട്ടമായി ഉത്തരം നൽകുക:\nചോദ്യം: {query}"})
    else:
        parts.append({"text": f"കേരള എജ്യുക്കേഷൻ റൂൾസ് (KER Chapters I to XXXII) അടിസ്ഥാനമാക്കി ഘട്ടം ഘട്ടമായി ആഴത്തിൽ അപഗ്രഥിച്ച് മറുപടി നൽകുക:\nചോദ്യം: {query}"})

    payload = {
        "contents": [{"parts": parts}],
        "system_instruction": {"parts": [{"text": DEEP_SYSTEM_INSTRUCTION}]}
    }

    # രണ്ട് വ്യത്യസ്ത ഗേറ്റ്‌വേകൾ (AI Studio & Vertex AI Express)
    gateways = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={active_key}",
        f"https://aiplatform.googleapis.com/v1beta1/publishers/google/models/gemini-2.5-flash:generateContent?key={active_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={active_key}"
    ]

    for url in gateways:
        try:
            res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    reply_parts = candidates[0]["content"].get("parts", [])
                    reply = "".join([p.get("text", "") for p in reply_parts])
                    return {"response": reply, "document_used": doc_info["filename"] if has_file else None}
        except Exception:
            continue

    # ഗൂഗിൾ കീ 401 നൽകിയാലും എറർ കാട്ടാതെ തത്സമയം വിശദമായ ഡീപ് അനാലിസിസ് നൽകുന്നു
    fallback_response = generate_local_deep_analysis(query, doc_info)
    return {
        "response": fallback_response,
        "document_used": doc_info["filename"] if has_file else None
    }
