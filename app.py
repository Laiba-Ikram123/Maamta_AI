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
    initial_sidebar_state="collapsed",
)

def load_css(file_name="style.css"):
    css_path = Path(file_name)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

GUIDELINE_DIR = Path("guidelines")
TOP_K = 3
MIN_RELEVANCE = 0.05

PAKISTAN_PRIORITY = [
    "nutrition",
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
    r"\b(bhuk|bhook|appetite|khana|khorak|diet|nutrition|kamzori|weakness|matli|vomiting|ulte|qay|heartburn|jalan|hazma|pait|peit)\b":
        "nutrition pregnancy diet meals calories protein iron folic acid nausea vomiting small meals heartburn constipation fluids",
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
        r"\b(hai|hain|ki|ka|ke|ko|se|me|mein|par|kya|kia|kyun|kyu|kab|kaise|kese|karo|karein|kare|karu|kroun|raha|rahi|rahe|ho|tha|thi|the|nahi|ni|na|aur|bhi|bohot|boht|zyada|zada|dard|sar|khoon|peit|pait|bacha|bache|teeka|teekay|hidayat|ilaaj|ilaj|hona|chahiye|chahye|khau|khao|khana|khorak|bhuk|bhook|kamzori|batao|bataen|mashwara|mujhe|muje|mera|meri|mere)\b"
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
                if len(chunk) >= 70:
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
                "Isay fori aur ahem samjhein aur bila taa-kheer qareebi hospital ya doctor se check karwayen."
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

    user_is_urdu = is_roman_urdu(question) or is_roman_urdu(health_summary)

    if user_is_urdu:
        mode_instruction = (
            "Aap aasan, ba-adab aur wazeh Roman Urdu mein baat karein jo mareeza ya uske ghar walon ko aasani se samajh aa sakay."
            if mode == "Mother / Family"
            else
            "Aap clinical aur guideline-oriented Roman Urdu istemal karein jo healthcare worker ya doctor ke liye munasib ho."
        )
        safety_instruction = (
            "ZAROORI: Khatray ki alamat report hui hai! Salamati aur fori hospital janay ki hidayat pehle dein."
            if safety["urgent"]
            else
            "Koi fori emergency alamat nahi mili, aam guideline ke mutabiq mashwara dein."
        )
        language_rule = """
CRITICAL LANGUAGE REQUIREMENT:
- The user is communicating in ROMAN URDU.
- You MUST answer 100% ENTIRELY in authentic Pakistani Roman Urdu.
- UNDER NO CIRCUMSTANCES should you reply in English prose.
- NEVER use Roman Hindi words (Do NOT use: kripya, upchar, lakshan, samasya, turant, chhatra, mahila, prasav, garbhavastha, aspataal, sujhaav). Use Pakistani Roman Urdu words (baraye meharbani, ilaj, alamat, masla, foran, khatoon/aurat, delivery/paidaish, hamal, hospital, mashwara, khorak).
- Section headers must be exactly:
  🚨 Khatray Ki Alamat
  🩺 Hidayat (Guideline Guidance)
  ➡️ Agla Zaroori Qadam (Recommended Next Action)
"""
        final_instruction = "Write a comprehensive, fully detailed, and caring response completely in Pakistani Roman Urdu. Do not write in English."
    else:
        mode_instruction = (
            "Use simple, clear, polite, empathetic English suitable for a mother or her family."
            if mode == "Mother / Family"
            else
            "Use formal clinical and guideline-oriented English suitable for a healthcare worker."
        )
        safety_instruction = (
            "CRITICAL: A potential danger sign was detected! Prioritize safety instructions and immediate facility assessment."
            if safety["urgent"]
            else
            "No severe danger sign detected. Provide standard guideline recommendations."
        )
        language_rule = """
CRITICAL LANGUAGE REQUIREMENT:
- The user is communicating in ENGLISH.
- You MUST answer 100% ENTIRELY in professional English.
- DO NOT use any Roman Urdu or Hindi words.
- Section headers must be exactly:
  🚨 Safety / Urgency
  🩺 Guideline-Based Guidance
  ➡️ Recommended Next Action
"""
        final_instruction = "Write a comprehensive, fully detailed, and professional response completely in English."

    return f"""
You are Maamta AI, an official source-grounded maternal and newborn health assistant for Pakistan.

{language_rule}

NON-NEGOTIABLE GROUNDING RULES:
1. Base your answer STRICTLY on the supplied APPROVED SOURCE EVIDENCE.
2. If evidence touches on diet, nutrition, appetite, or nausea, provide full guideline-supported guidance from the evidence.
3. If the evidence completely lacks relevant information to answer safely, state clearly that official guidelines lack sufficient information.
4. Never invent medications, dosages, or protocols not present in the sources.

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

{final_instruction}
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
        max_tokens=1800,
        messages=[
            {
                "role": "system",
                "content": "You are Maamta AI, a source-grounded maternal and newborn health assistant. You strictly match the user language (Roman Urdu vs English) and follow grounding rules.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# 1. Top Navigation Bar (High-Contrast Frosted Brand Logo)
st.markdown(
    """
    <div class="top-nav-brand">
        <div class="brand-logo-svg">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="46" height="46" fill="none">
                <defs>
                    <linearGradient id="maamtaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#ff3b4e"/>
                        <stop offset="100%" stop-color="#800a16"/>
                    </linearGradient>
                </defs>
                <rect width="48" height="48" rx="14" fill="url(#maamtaGrad)" />
                <path d="M24 38s-13-8.2-13-17.5C11 15.5 15 12 19.5 12c2.5 0 4.5 1.2 4.5 1.2s2-1.2 4.5-1.2c4.5 0 8.5 3.5 8.5 8.5C37 29.8 24 38 24 38z" 
                      fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.45"/>
                <circle cx="21" cy="18" r="3" fill="#ffffff"/>
                <path d="M16 29c0-3.5 2.5-6 6-6s6 2.5 6 6" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round"/>
                <circle cx="27.5" cy="22.5" r="2" fill="#ffccd0"/>
                <path d="M25 29c0-2 1.5-3.5 3.5-3.5" stroke="#ffccd0" stroke-width="2" stroke-linecap="round"/>
                <path d="M34 14l1 2 2 1-2 1-1 2-1-2-2-1 2-1 1-2z" fill="#ffffff"/>
            </svg>
        </div>
        <div class="brand-title-wrap">
            <div class="nav-title">Maamta <span>AI</span></div>
            <span class="nav-tagline">National Clinical Triage</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Pure Crimson Glassmorphism Hero Section
st.markdown(
    """
    <div class="hero-wrapper-red">
        <div class="hero-title-red">
            Empowering Every Mother With <span>Smarter Maternal Care</span>
        </div>
        <div class="hero-subtext-red">
            AI-powered maternal triage and danger sign detection, strictly grounded in Pakistan's official national health guidelines.
        </div>
        <div class="feature-grid-red">
            <div class="feature-card-red">
                <div class="feature-icon-box-red">🩺</div>
                <h4>Real-Time Triage</h4>
                <p>Instant danger detection</p>
            </div>
            <div class="feature-card-red">
                <div class="feature-icon-box-red">📚</div>
                <h4>Source Grounded</h4>
                <p>Zero hallucinations</p>
            </div>
            <div class="feature-card-red">
                <div class="feature-icon-box-red">🗣️</div>
                <h4>Bilingual Support</h4>
                <p>English & Roman Urdu</p>
            </div>
            <div class="feature-card-red">
                <div class="feature-icon-box-red">🏥</div>
                <h4>Facility Guidance</h4>
                <p>Urgent referral protocols</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. Patient Intake Section
st.markdown(
    """
    <div class="intake-header">
        <h3>📋 Patient Intake & Clinical Vitals</h3>
        <span>Compulsory for Safe Triaging</span>
    </div>
    """,
    unsafe_allow_html=True,
)

r1_col1, r1_col2, r1_col3, r1_col4 = st.columns(4)
with r1_col1:
    mode = st.selectbox("User mode", ["Mother / Family", "Healthcare Worker"])
with r1_col2:
    weeks = st.number_input("Pregnancy weeks", min_value=0, max_value=45, value=0)
with r1_col3:
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
with r1_col4:
    bleeding = st.selectbox("Bleeding status", ["Not reported", "No", "Yes", "Heavy / severe"])

r2_col1, r2_col2, r2_col3, r2_col4 = st.columns(4)
with r2_col1:
    bp = st.text_input("Blood pressure", placeholder="e.g. 120/80")
with r2_col2:
    temperature = st.text_input("Temperature", placeholder="e.g. 37°C")
with r2_col3:
    pulse = st.text_input("Pulse", placeholder="e.g. 80 bpm")
with r2_col4:
    other = st.text_input("Other medical history", placeholder="e.g. Diabetes, None")

symptoms = st.text_area(
    "Active symptoms & complaints",
    placeholder="Describe physical discomfort, pain, headache, nausea, or fever here...",
    height=85,
)

docs = load_guidelines()

health_summary = "\n".join(
    [
        f"User mode: {mode}",
        f"Pregnancy weeks: {weeks}",
        f"Age: {age}",
        f"Blood pressure: {bp}",
        f"Temperature: {temperature}",
        f"Pulse: {pulse}",
        f"Bleeding: {bleeding}",
        f"Symptoms: {symptoms}",
        f"Other details: {other or 'None'}",
    ]
)

# 4. Bottom Chat Input
question = st.chat_input("Ask Maamta AI in English or Roman Urdu...")

if question:
    user_is_urdu = is_roman_urdu(question) or is_roman_urdu(symptoms)

    missing_fields = []
    if weeks == 0:
        missing_fields.append("Pregnancy weeks" if not user_is_urdu else "Pregnancy weeks (hamal ke haftay)")
    if age == 0:
        missing_fields.append("Age" if not user_is_urdu else "Age (umar)")
    if not bp.strip():
        missing_fields.append("Blood pressure" if not user_is_urdu else "Blood pressure (BP)")
    if not pulse.strip():
        missing_fields.append("Pulse" if not user_is_urdu else "Pulse (nabz)")
    if not symptoms.strip():
        missing_fields.append("Active symptoms" if not user_is_urdu else "Active symptoms (alamat)")

    if missing_fields:
        if user_is_urdu:
            st.warning(f"Baraye meharbani pehle intake form mein yeh zaroori maloomat darj karein: **{', '.join(missing_fields)}**.")
        else:
            st.warning(f"Please fill in the required health vitals before asking: **{', '.join(missing_fields)}**.")
        st.stop()

    if not docs:
        if user_is_urdu:
            st.error("Mojooda guidelines mein is baray mein mukammal maloomat nahi mil sakeen.")
            st.info("Pehle guidelines folder mein Pakistan official PDFs upload karein.")
        else:
            st.error("I couldn't find sufficient information in the available guidelines to answer this safely.")
            st.info("Add approved Pakistan guideline PDFs to the `guidelines/` folder.")
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

    with st.spinner("Analyzing national clinical evidence..."):
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

    st.markdown(f'<div class="response-container">{answer}</div>', unsafe_allow_html=True)

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
