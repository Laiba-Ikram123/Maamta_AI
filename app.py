import os
import re
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ============================================================
# Maamta AI — Source-Grounded Maternal & Newborn Health MVP
# Pakistan-first, privacy-conscious, single-file Streamlit app
#
# Put approved official guideline PDFs in:
#   guidelines/
# ============================================================

st.set_page_config(
    page_title="Maamta AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "Maamta AI provides information grounded in selected clinical guidelines and is "
    "not a replacement for a qualified healthcare professional. It does not provide "
    "a confirmed diagnosis or personalized prescription. If you or a mother/newborn "
    "may be experiencing an emergency or serious danger sign, seek immediate "
    "professional medical care."
)

GUIDELINE_DIR = Path("guidelines")
TOP_K = 5
MIN_RELEVANCE = 0.08

PAKISTAN_PRIORITY = [
    "pcpnc",
    "mcpc",
    "emonc",
    "immunization",
    "newborn",
    "maternal",
    "national",
    "pakistan",
]

DANGER_PATTERNS = [
    (r"\bheavy bleeding\b|\bsevere bleeding\b|\bprofuse bleeding\b", "heavy/severe bleeding"),
    (r"\bbleeding\b", "bleeding"),
    (r"\bconvulsion(s)?\b|\bseizure(s)?\b", "convulsions/seizures"),
    (r"\bunconscious\b|\bloss of consciousness\b|\bunresponsive\b", "loss of consciousness/unresponsiveness"),
    (r"\bdifficulty breathing\b|\bdifficult breathing\b|\bbreathlessness\b|\bsevere breathing\b", "difficulty/severe breathing problem"),
    (r"\bsevere abdominal pain\b|\babdominal pain\b", "severe abdominal pain"),
    (r"\bsevere headache\b|\bheadache\b", "severe headache"),
    (r"\bblurred vision\b|\bvisual disturbance\b", "visual disturbance"),
    (r"\bhigh fever\b|\bsevere fever\b|\bfever\b", "high/severe fever"),
    (r"\bfoul[- ]smelling discharge\b|\bfoul smelling discharge\b", "foul-smelling discharge"),
    (r"\bnot breathing\b|\bno breathing\b", "not breathing"),
    (r"\bblue\b.*\b(lips|skin)\b|\bcyanosis\b", "blue/grey lips or skin"),
    (r"\bpoor feeding\b", "poor feeding"),
]

# Clinical term mapping to bridge lay symptoms with guideline terminology
SYMPTOM_EXPANSIONS = {
    r"\bheadache\b": "headache elevated blood pressure pre-eclampsia eclampsia hypertension danger signs",
    r"\bbleeding\b": "vaginal bleeding haemorrhage postpartum antepartum shock placenta",
    r"\bfever\b": "fever infection sepsis temperature danger signs",
    r"\bbreathing\b": "difficulty breathing respiratory distress cyanosis",
    r"\babdominal pain\b": "abdominal pain contractions labour ectopic abruptio",
    r"\bvision\b": "blurred vision visual disturbance pre-eclampsia",
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def source_priority(filename: str) -> int:
    name = filename.lower()
    for i, keyword in enumerate(PAKISTAN_PRIORITY):
        if keyword in name:
            return len(PAKISTAN_PRIORITY) - i
    return 0

def expand_query(query: str) -> str:
    expanded = query
    for pattern, extra_terms in SYMPTOM_EXPANSIONS.items():
        if re.search(pattern, query, flags=re.I):
            expanded += f" {extra_terms}"
    return expanded

@st.cache_data(show_spinner=False)
def load_guidelines():
    docs = []
    if not GUIDELINE_DIR.exists():
        return docs

    for pdf_path in sorted(GUIDELINE_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            continue

        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = normalize(page.extract_text() or "")
            except Exception:
                text = ""

            if not text:
                continue

            words = text.split()
            chunk_size = 180
            overlap = 35
            start = 0
            while start < len(words):
                chunk_words = words[start:start + chunk_size]
                chunk = " ".join(chunk_words).strip()
                if len(chunk) >= 80:
                    docs.append(
                        {
                            "text": chunk,
                            "document": pdf_path.name,
                            "page": page_no,
                            "section": "PDF page text",
                            "priority": source_priority(pdf_path.name),
                        }
                    )
                if start + chunk_size >= len(words):
                    break
                start += chunk_size - overlap

    return docs

@st.cache_resource(show_spinner=False)
def build_index(texts):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def retrieve(query, docs, top_k=TOP_K):
    if not docs:
        return []

    vectorizer, matrix = build_index([d["text"] for d in docs])
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()

    ranked = sorted(
        range(len(docs)),
        key=lambda i: (scores[i], docs[i]["priority"]),
        reverse=True,
    )

    results = []
    for i in ranked[:top_k]:
        if scores[i] >= MIN_RELEVANCE:
            item = dict(docs[i])
            item["score"] = float(scores[i])
            results.append(item)
    return results

def corpus_has_phrase(docs, pattern):
    combined = " ".join(d["text"].lower() for d in docs)
    return re.search(pattern, combined, flags=re.I) is not None

def assess_safety(user_text, docs):
    text = normalize(user_text).lower()
    matches = []

    for pattern, label in DANGER_PATTERNS:
        if re.search(pattern, text, flags=re.I) and corpus_has_phrase(docs, pattern):
            matches.append(label)

    matches = list(dict.fromkeys(matches))

    if matches:
        return {
            "urgent": True,
            "signs": matches,
            "message": (
                "The information reported includes a potential danger sign "
                "identified in the approved guideline material. This should be "
                "treated as potentially urgent and assessed by a qualified "
                "health professional without delay."
            ),
        }

    return {"urgent": False, "signs": [], "message": ""}

def format_sources(results):
    lines = []
    for r in results:
        lines.append(
            f"- **{r['document']}** — page {r['page']} "
            f"(relevance {r['score']:.2f})"
        )
    return "\n".join(lines)

def build_prompt(mode, question, health_summary, results, safety):
    evidence = "\n\n".join(
        [
            f"[SOURCE {i+1}]\n"
            f"Document: {r['document']}\n"
            f"Page: {r['page']}\n"
            f"Evidence:\n{r['text']}"
            for i, r in enumerate(results)
        ]
    )

    mode_instruction = (
        "Use simple, clear, empathetic, non-technical language suitable for a mother/family. "
        "Focus on safety, reassurance, and the next immediate steps."
        if mode == "Mother / Family"
        else
        "Use clinical, guideline-oriented terminology suitable for a healthcare worker. "
        "Include diagnostic criteria, danger sign assessments, and referral protocols."
    )

    safety_instruction = (
        "A critical obstetric danger sign was detected. State urgency first. "
        "Explain that symptoms like severe headaches can be linked with elevated blood pressure "
        "or pre-eclampsia and require urgent evaluation (checking blood pressure and urine protein). "
        "Advise visiting the nearest health facility immediately."
        if safety["urgent"]
        else
        "No explicit danger sign was triggered, but encourage standard antenatal/postnatal checks "
        "if symptoms persist."
    )

    return f"""
You are Maamta AI, a source-grounded maternal and newborn health guidance assistant for Pakistan.

NON-NEGOTIABLE GROUNDING RULES:
1. Answer strictly using the provided APPROVED SOURCE EVIDENCE.
2. If evidence touches on the symptom, provide immediate triage advice based on that evidence.
3. If the sources completely lack relevant information to guide the user safely, state:
   "I couldn't find sufficient information in the available guidelines to answer this safely."
4. Do not invent diagnoses, numerical thresholds, prescriptions, or dosages not found in sources.
5. Emphasize that a clinical assessment is necessary.
6. The response must follow this structure:
   🚨 Safety / Urgency
   🩺 Guideline-Based Guidance
   ➡️ Recommended Next Action

USER MODE:
{mode}
{mode_instruction}

SAFETY CONTEXT:
{safety_instruction}

STRUCTURED HEALTH INFORMATION:
{health_summary}

USER QUESTION:
{question}

APPROVED SOURCE EVIDENCE:
{evidence}

Write a concise, caring, and practical response.
""".strip()

def call_groq(prompt):
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    base_url = st.secrets.get("GROQ_BASE_URL", os.getenv("GROQ_BASE_URL", None))
    client = Groq(api_key=api_key, base_url=base_url) if base_url else Groq(api_key=api_key)
    
    # Updated to openai/gpt-oss-120b
    model = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": "You are a maternal and newborn health triage assistant. Adhere strictly to the evidence.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# ---------------- UI ----------------

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 2rem;}
    .maamta-title {font-size: 2.3rem; font-weight: 700; margin-bottom: 0.2rem;}
    .maamta-subtitle {color: #5f6368; margin-bottom: 1.5rem;}
    .notice {padding: 1rem; border: 1px solid #d9dee3; border-radius: 10px; background: #fafbfc;}
    .urgent {padding: 1rem; border: 1px solid #c62828; border-radius: 10px; background: #fff7f7;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="maamta-title">🩺 Maamta AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="maamta-subtitle">Source-grounded maternal & newborn health guidance for Pakistan</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Maamta AI")
    mode = st.radio(
        "User mode",
        ["Mother / Family", "Healthcare Worker"],
        index=0,
    )

    st.divider()
    st.subheader("Structured health information")

    weeks = st.number_input("Pregnancy weeks (optional)", min_value=0, max_value=45, value=0)
    age = st.number_input("Age (optional)", min_value=0, max_value=120, value=0)
    bp = st.text_input("Blood pressure (optional)", placeholder="e.g. 120/80")
    temperature = st.text_input("Temperature (optional)", placeholder="e.g. 38°C")
    pulse = st.text_input("Pulse (optional)", placeholder="e.g. 90 bpm")
    bleeding = st.selectbox(
        "Bleeding",
        ["Not reported", "No", "Yes", "Heavy / severe"],
    )
    symptoms = st.text_area(
        "Symptoms",
        placeholder="Describe symptoms relevant to the question.",
        height=110,
    )
    other = st.text_area(
        "Other relevant details",
        placeholder="Optional. Do not enter name, CNIC, phone, or address.",
        height=90,
    )

    st.divider()
    st.caption("Privacy-conscious MVP: data is used solely in memory for the active query.")

docs = load_guidelines()

with st.expander("Approved source status", expanded=not bool(docs)):
    if docs:
        documents = sorted({d["document"] for d in docs})
        st.success(f"Loaded {len(documents)} PDF document(s) and {len(docs)} searchable passages.")
        for doc in documents:
            st.write(f"• {doc}")
    else:
        st.warning(
            "No guideline PDFs were found. Add approved Pakistan maternal/newborn "
            "guideline PDFs to the `guidelines/` folder before using Maamta AI."
        )

health_summary = "\n".join(
    [
        f"Pregnancy weeks: {weeks if weeks else 'Not provided'}",
        f"Age: {age if age else 'Not provided'}",
        f"Blood pressure: {bp or 'Not provided'}",
        f"Temperature: {temperature or 'Not provided'}",
        f"Pulse: {pulse or 'Not provided'}",
        f"Bleeding: {bleeding}",
        f"Symptoms: {symptoms or 'Not provided'}",
        f"Other details: {other or 'Not provided'}",
    ]
)

question = st.chat_input("Ask a maternal or newborn health question...")

if question:
    if not docs:
        st.error("I couldn't find sufficient information in the available guidelines to answer this safely.")
        st.info("Add your approved Pakistan guideline PDFs to the `guidelines/` folder and reload the app.")
        st.stop()

    combined_input = f"{question}\n{health_summary}"
    safety = assess_safety(combined_input, docs)

    active_details = []
    if symptoms:
        active_details.append(symptoms)
    if bp:
        active_details.append(f"blood pressure {bp}")
    if bleeding in ["Yes", "Heavy / severe"]:
        active_details.append(f"bleeding {bleeding}")

    search_text = f"{question} {' '.join(active_details)}".strip()
    expanded_search_query = expand_query(search_text)

    results = retrieve(expanded_search_query, docs)

    if not results:
        vectorizer, matrix = build_index([d["text"] for d in docs])
        q_vec = vectorizer.transform([expanded_search_query])
        scores = cosine_similarity(q_vec, matrix).ravel()
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        results = [dict(docs[i], score=float(scores[i])) for i in ranked[:TOP_K] if scores[i] > 0.02]

    if not results:
        st.warning("I couldn't find sufficient information in the available guidelines to answer this safely.")
        st.stop()

    if safety["urgent"]:
        st.markdown(
            '<div class="urgent"><strong>🚨 Safety / Urgency</strong><br>'
            + safety["message"]
            + "<br><br><strong>Potential signs reported:</strong> "
            + ", ".join(safety["signs"])
            + "</div>",
            unsafe_allow_html=True,
        )

    with st.spinner("Reviewing approved guideline evidence..."):
        try:
            prompt = build_prompt(
                mode=mode,
                question=question,
                health_summary=health_summary,
                results=results,
                safety=safety,
            )
            answer = call_groq(prompt)
        except Exception as exc:
            st.error(f"Maamta AI could not generate a response: {exc}")
            st.stop()

    st.markdown("### 🩺 Guideline-Based Guidance")
    st.markdown(answer)

    st.markdown("### ➡️ Recommended Next Action")
    if safety["urgent"]:
        st.error(
            "Because a potential danger sign was reported, seek prompt professional "
            "medical assessment. If the situation appears life-threatening, go to the "
            "nearest emergency maternity facility immediately."
        )
    else:
        st.info(
            "Follow the source-grounded guidance above. If symptoms are severe, worsening, "
            "or concerning, seek assessment from a qualified healthcare professional."
        )

    st.markdown("### 📚 Evidence / Sources")
    st.markdown(format_sources(results))

    with st.expander("View retrieved guideline passages"):
        for i, r in enumerate(results, start=1):
            st.markdown(
                f"**Source {i}: {r['document']} — page {r['page']}**  \n"
                f"{r['text']}"
            )

st.divider()
st.markdown(
    f'<div class="notice"><strong>Medical disclaimer</strong><br>{DISCLAIMER}</div>',
    unsafe_allow_html=True,
)
