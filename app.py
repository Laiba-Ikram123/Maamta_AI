import os
import re
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

st.set_page_config(
    page_title="Maamta AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

GUIDELINE_DIR = Path("guidelines")
TOP_K = 5
MIN_RELEVANCE = 0.06

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
    (r"\b(heavy bleeding|severe bleeding|profuse bleeding|zyada khoon|bohot khoon|khoon beh raha|khoon ja raha|khoon ke lothray|clots)\b", "heavy/severe bleeding"),
    (r"\b(bleeding|khoon|rakht|lohu|daag|spotting)\b", "bleeding"),
    (r"\b(convulsion(s)?|seizure(s)?|jhatkay|jhatke|doray|daura|mirgi|fits)\b", "convulsions/seizures"),
    (r"\b(unconscious|loss of consciousness|unresponsive|behosh|behoshi|ghashi|hosh kho dena)\b", "loss of consciousness/unresponsiveness"),
    (r"\b(difficulty breathing|breathlessness|respiratory distress|saans phoolna|saans lene me dushwari|dam ghutna|saans tez)\b", "difficulty/severe breathing problem"),
    (r"\b(severe abdominal pain|pet dard|peit me dard|pait dard|shadeed dard|kokh me dard|talli me dard)\b", "severe abdominal pain"),
    (r"\b(severe headache|headache|sar dard|sar me shadeed dard|sar phat raha|sar ghoomna|chakkar)\b", "severe headache"),
    (r"\b(blurred vision|visual disturbance|dhundla nazar|dhundla dikhai|aankhon k samne andhera|chamkay)\b", "visual disturbance"),
    (r"\b(high fever|severe fever|fever|tez bukhar|bukhar|tap|hararat|kapkapi|shivering)\b", "high/severe fever"),
    (r"\b(foul[- ]smelling discharge|badboodar pani|kharab pani|badboodar discharge|peep)\b", "foul-smelling discharge"),
    (r"\b(not breathing|no breathing|saans nahi le raha|saans band|sans rukna)\b", "not breathing"),
    (r"\b(blue|cyanosis|neela par jana|neela rang|hont neelay)\b", "blue/grey lips or skin"),
    (r"\b(poor feeding|doodh nahi pee raha|feed nahi le raha|bacha sust hai|chhati nahi pakad raha)\b", "poor feeding"),
    (r"\b(epigastric pain|pasliyon me dard|pasli k neeche dard|seene k neeche dard)\b", "epigastric pain"),
    (r"\b(cord prolapse|naal bahar|naad bahar|umbilical cord outside)\b", "umbilical cord prolapse"),
    (r"\b(swelling|sujan|paon sooj jana|munh soojna|face swelling)\b", "severe swelling/edema"),
]

SYMPTOM_EXPANSIONS = {
    r"\b(headache|sar dard|sar me dard|chakkar|dhundla|vision|andhera|bp|blood pressure|pre-eclampsia|eclampsia)\b":
        "headache elevated blood pressure pre-eclampsia eclampsia hypertension danger signs proteinuria visual disturbance",
    r"\b(bleeding|khoon|rakht|lohu|haemorrhage|pph|aonwal|placenta|lothray|bachedani|atony)\b":
        "vaginal bleeding haemorrhage postpartum antepartum shock atonic uterus oxytocin tranexamic acid E-MOTIVE massage",
    r"\b(preterm|waqt se pehle|satwasa|athwasa|teeka|steroid|dexamethasone|betamethasone|lungs|phephray|tocolytic|nifedipine)\b":
        "preterm labour antenatal corticosteroids dexamethasone betamethasone nifedipine tocolysis fetal lung maturity gestational age",
    r"\b(shock|nabz tez|bp low|systolic|pulse|cannula|fluid|saline|ringers|clotting test|nasg|garment)\b":
        "shock hypovolaemic resuscitation IV fluids normal saline ringer lactate large-bore cannula bedside clotting test NASG",
    r"\b(labour|dard e zeh|delivery|contractions|paani chhootna|leaking|water break|meconium|sabz pani|kala pani|cord|naad|aonwal|stuck|kandha phansna)\b":
        "labour childbirth contractions cervical dilatation descent meconium amniotic fluid cord clamping shoulder dystocia breech",
    r"\b(vaccine|teekay|teeka|hifazati|bcg|polio|opv|ipv|pentavalent|pcv|rota|measles|khasra|typhoid|tcv|td|tetanus|vvm|mdvp|cold chain)\b":
        "immunization EPI vaccination schedule BCG OPV Pentavalent PCV Rotavirus IPV Measles Rubella TCV Td vaccine vial monitor MDVP",
    r"\b(nawzaida|newborn|bacha|neonatal|kam wazan|lbw|kmc|kangaroo|naaf|cord care|doodh|breastfeeding|resuscitation|apnoea|cyanosis|neela)\b":
        "newborn resuscitation skin to skin thermal care kangaroo mother care KMC exclusive breastfeeding vitamin K eye prophylaxis cyanosis",
    r"\b(infection|sepsis|bukhar|fever|badboo|pus|peep|discharge|antibiotic|ampicillin|gentamicin|erythromycin|amoxicillin|wound|tanka)\b":
        "infection sepsis puerperal endometritis amnionitis antibiotics ampicillin gentamicin erythromycin wound care prophylactic",
    r"\b(miscarriage|hamal zaya|abortion|safe|mva|safai|ectopic|bache dani se bahar)\b":
        "abortion threatened incomplete inevitable ectopic pregnancy manual vacuum aspiration misoprostol evacuation",
    r"\b(udasi|depression|rona|psychosis|wehshat|khayal|mental|dill ghabrana)\b":
        "postpartum blues depression psychosis maternal mental health anxiety supportive care",
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def is_roman_urdu(text: str) -> bool:
    urdu_markers = [
        r"\b(hai|hain|ki|ka|ke|ko|se|me|mein|par|kya|kyun|kab|kaise|karo|karein|raha|rahi|ho|tha|thi|the|nahi|aur|bhi|bohot|zyada|dard|sar|khoon|peit|pait|bacha|teeka|teekay|hidayat|ilaaj|doctor)\b"
    ]
    t = text.lower()
    for marker in urdu_markers:
        if re.search(marker, t):
            return True
    return False

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

def assess_safety(user_text, docs):
    text = normalize(user_text).lower()
    matches = []

    for pattern, label in DANGER_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            matches.append(label)

    matches = list(dict.fromkeys(matches))

    if matches:
        return {
            "urgent": True,
            "signs": matches,
            "message_en": (
                "The information reported includes a potential danger sign "
                "identified in official guidelines. This should be treated as urgent "
                "and assessed by a qualified healthcare professional without delay."
            ),
            "message_ur": (
                "Batai gayi alamat official guidelines ke mutabiq khatray ki nishani ho sakti hai. "
                "Isay fori aur ahem samjhein aur bila taa-kheer qareebi hospital ya lady doctor se check karwayen."
            ),
        }

    return {"urgent": False, "signs": [], "message_en": "", "message_ur": ""}

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
        "Use simple, polite, empathetic language suitable for a mother or family."
        if mode == "Mother / Family"
        else
        "Use clinical, guideline-oriented terminology suitable for a healthcare worker or doctor."
    )

    safety_instruction = (
        "CRITICAL: A potential danger sign was detected! Put urgency and safety guidance first. "
        "Instruct immediate facility evaluation."
        if safety["urgent"]
        else
        "No severe emergency danger sign was detected. Provide standard guideline recommendations."
    )

    return f"""
You are Maamta AI, an official source-grounded maternal and newborn health assistant for Pakistan.

STRICT LANGUAGE RULES (PAKISTANI ROMAN URDU ONLY):
1. When the user asks in Roman Urdu or Urdu:
   - You MUST reply strictly in authentic Pakistani Roman Urdu (the way people speak and text in Pakistan).
   - STRICTLY FORBIDDEN: NEVER use Roman Hindi words. Do NOT use:
     * "kripya" -> Use "baraye meharbani" or "aap"
     * "upchar" / "ilaj" -> Use "ilaj" or "medical dekh bhal"
     * "lakshan" -> Use "alamat" or "nishaniyan"
     * "samasya" -> Use "masla" or "takleef"
     * "turant" / "shighra" -> Use "foran" or "bila taa-kheer"
     * "chhatra" / "mahila" -> Use "aurat" or "khatoon" or "mareeza"
     * "prasav" -> Use "delivery" or "paidaish"
     * "garbhavastha" -> Use "hamal" ya "pregnancy"
     * "aspataal" -> Use "hospital" ya "qareebi health center"
     * "sujhaav" -> Use "mashwara" ya "hidayat"
   - Use natural Pakistani daily phrases: "foran doctor se ruju karein", "khatray ki alamat", "qareebi hospital jayen", "apna blood pressure check karwayen".
2. If the user asks in English, reply entirely in English.
3. If the user asks in Urdu script, reply in Urdu script.

NON-NEGOTIABLE GROUNDING RULES:
1. Base your answer STRICTLY on the supplied APPROVED SOURCE EVIDENCE.
2. If evidence touches on the condition/symptom, give complete guideline-supported advice.
3. If the evidence completely lacks relevant information to answer safely, reply in clean Roman Urdu:
   "Mojooda official guidelines mein is baray mein mukammal maloomat nahi mil sakeen." (Or in English: "I couldn't find sufficient information in the available guidelines to answer this safely.")
4. Never invent medications, dosages, or protocols not present in the sources.
5. Structure your response using these exact section headers:
   🚨 Safety / Urgency (Khatray Ki Alamat)
   🩺 Guideline-Based Guidance (Hidayat)
   ➡️ Recommended Next Action (Agla Zaroori Qadam)

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

Write a natural, respectful Pakistani Roman Urdu response without any Hindi vocabulary.
""".strip()

def call_groq(prompt):
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    base_url = st.secrets.get("GROQ_BASE_URL", os.getenv("GROQ_BASE_URL", None))
    client = Groq(api_key=api_key, base_url=base_url) if base_url else Groq(api_key=api_key)

    model = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": "You are Maamta AI, a maternal triage assistant for Pakistan. Adhere strictly to Pakistani Roman Urdu and evidence.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

st.markdown(
    """
    <style>
    .block-container {max-width: 1150px; padding-top: 2rem;}
    .maamta-title {font-size: 2.3rem; font-weight: 700; margin-bottom: 0.2rem;}
    .maamta-subtitle {color: #a0aab2; margin-bottom: 1.5rem;}
    
    .notice {
        padding: 1rem; 
        border: 1px solid #d9dee3; 
        border-radius: 10px; 
        background: #fafbfc;
        color: #1a1a1a !important;
        margin-top: 1.5rem;
    }
    .notice strong {
        color: #000000 !important;
    }
    
    .urgent {
        padding: 1rem; 
        border: 1px solid #c62828; 
        border-radius: 10px; 
        background: #fff7f7;
        color: #212121 !important;
        margin-bottom: 1rem;
    }
    .urgent strong {
        color: #b71c1c !important;
    }
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
    st.caption("Privacy-conscious MVP: data is processed in-memory and not stored permanently.")

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

question = st.chat_input("Ask in English ya Roman Urdu mein sawal poochein...")

if question:
    user_is_urdu = is_roman_urdu(question) or is_roman_urdu(symptoms)

    if not docs:
        if user_is_urdu:
            st.error("Mojooda guidelines mein is baray mein mukammal maloomat nahi mil sakeen.")
            st.info("Pehle guidelines folder mein Pakistan official PDFs upload karein.")
        else:
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
        results = [dict(docs[i], score=float(scores[i])) for i in ranked[:TOP_K] if scores[i] > 0.015]

    if not results:
        if user_is_urdu:
            st.warning("Mojooda guidelines mein is baray mein mukammal maloomat nahi mil sakeen.")
        else:
            st.warning("I couldn't find sufficient information in the available guidelines to answer this safely.")
        st.stop()

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

    st.markdown(answer)

    if user_is_urdu:
        st.markdown(
            """
            <div class="notice">
                <strong>Medical Disclaimer (Zaroori Wazahat):</strong><br>
                Maamta AI sirf official clinical guidelines par mabni maloomat faraham karta hai aur yeh kisi qualified doctor ya healthcare professional ka mutabadil nahi hai. Yeh koi pakka diagnosis ya prescription nahi deta. Agar aap ya bacha kisi emergency ya khatray ki alamat ka samna kar rahe hain, toh bila taa-kheer qareebi hospital ya doctor se ruju karein.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="notice">
                <strong>Medical Disclaimer:</strong><br>
                Maamta AI provides information grounded in selected clinical guidelines and is not a replacement for a qualified healthcare professional. It does not provide a confirmed diagnosis or personalized prescription. If you or a mother/newborn may be experiencing an emergency or serious danger sign, seek immediate professional medical care.
            </div>
            """,
            unsafe_allow_html=True,
        )
