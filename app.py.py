import streamlit as st
from streamlit_chat import message
from PIL import Image, ImageDraw
import io, os, pickle, time, hashlib
import PyPDF2, docx, requests, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import faiss
import speech_recognition as sr
import numpy as np
import torch
from threading import Thread
import networkx as nx
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# CONFIG (USE ENV VARIABLES)
# -------------------------------
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]
GENAI_API_URL = "https://api.groq.com/openai/v1/chat/completions"
YOLO_MODEL = "yolov8s.pt"

EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# INITIALIZE MODELS
# -------------------------------
def load_yolo_model():
    try:
        model = YOLO(YOLO_MODEL)
        try:
            if device=="cuda" and hasattr(model, "to"): model.to(device)
        except: pass
        return model
    except Exception as e:
        st.error(f"YOLO load failed: {e}")
        return None

def load_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        return model
    except Exception as e:
        st.error(f"Embedding model load failed: {e}")
        return None

yolo_model = load_yolo_model()
embedding_model = load_embedding_model()

# -------------------------------
# SESSION STATE
# -------------------------------
if "stats" not in st.session_state: st.session_state.stats = {"queries":0, "files_uploaded":0, "images_detected":0}
if "history" not in st.session_state: st.session_state.history = []
if "rag_index" not in st.session_state: st.session_state.rag_index = None
if "rag_texts" not in st.session_state: st.session_state.rag_texts = []
if "rag_embeddings" not in st.session_state: st.session_state.rag_embeddings = None
if "voice_input" not in st.session_state: st.session_state.voice_input = ""
if "voice_input_text" not in st.session_state: st.session_state.voice_input_text = ""
if "uploaded_files_hash" not in st.session_state: st.session_state.uploaded_files_hash = ""
if "uploaded_files_session" not in st.session_state: st.session_state.uploaded_files_session = None
if "yolo_cache" not in st.session_state: st.session_state.yolo_cache = {}
if "graph" not in st.session_state: st.session_state.graph = None
if "voice_recording_active" not in st.session_state: st.session_state.voice_recording_active = False
if "pdf_buffer" not in st.session_state: st.session_state.pdf_buffer = None

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🚀 Ultimate Creation Chatbot", layout="wide")
st.title("🚀 Ultimate Creation Chatbot")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚡ Options")
st.sidebar.markdown("""
**Quick tips**
- Upload PDF/TXT/DOCX in *Upload documents for RAG*.  
- Use *Voice Input* to speak.  
- Use *Download Chat PDF* to export conversation.
""")

theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""<style>[data-testid="stAppViewContainer"] { background:#1e1e1e; color:#fff; }</style>""", unsafe_allow_html=True)

if st.sidebar.button("Clear Chat"): st.session_state.history = []
if st.sidebar.button("Clear Documents"):
    st.session_state.rag_index = None
    st.session_state.rag_texts = []
    st.session_state.rag_embeddings = None
    st.session_state.uploaded_files_hash = ""
    st.session_state.uploaded_files_session = None
    st.success("Cleared uploaded documents.")

uploaded_files = st.sidebar.file_uploader("Upload docs for RAG 📄", type=["pdf","txt","docx"], accept_multiple_files=True)
if uploaded_files: st.session_state.uploaded_files_session = uploaded_files
files_for_processing = st.session_state.uploaded_files_session or uploaded_files
st.session_state.stats["files_uploaded"] = len(files_for_processing) if files_for_processing else 0

language = st.sidebar.selectbox("Language 🌐", ["English","Urdu"])
tone = st.sidebar.selectbox("Tone ✍️", ["Formal","Casual","Funny","Charming","Sarcastic"])
recipient = st.sidebar.text_input("Recipient Email")
subject = st.sidebar.text_input("Subject")
confidence = st.sidebar.slider("YOLO Confidence",0.0,1.0,0.25,0.05)
if st.sidebar.button("🎤 Voice Input"): st.session_state.voice_input = "record"

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def process_file(file):
    texts = []
    try:
        if file.name.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip(): texts.append(t.strip())
        elif file.name.lower().endswith(".txt"):
            raw = file.read()
            text = raw.decode("utf-8", errors="ignore")
            if text.strip(): texts.append(text.strip())
        elif file.name.lower().endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                if para.text.strip(): texts.append(para.text.strip())
    except Exception as e:
        st.warning(f"Error reading {file.name}: {e}")
    return texts

def get_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()

def build_rag_index(all_texts, cache_path):
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                index, embeddings, texts = pickle.load(f)
            st.session_state.rag_index = index
            st.session_state.rag_embeddings = embeddings
            st.session_state.rag_texts = texts
            st.success("Loaded cached RAG index.")
            return

        embeddings_list = []
        batch_size = 64
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            emb = embedding_model.encode(batch, batch_size=len(batch), convert_to_numpy=True)
            embeddings_list.append(emb)
        embeddings = np.vstack(embeddings_list)

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))

        st.session_state.rag_index = index
        st.session_state.rag_embeddings = embeddings
        st.session_state.rag_texts = all_texts

        with open(cache_path, "wb") as f:
            pickle.dump((index, embeddings, all_texts), f)
        st.success("RAG index built and cached.")
    except Exception as e:
        st.warning(f"RAG indexing failed: {e}")

def rag_search(query, k=5):
    if st.session_state.rag_index:
        try:
            q_emb = embedding_model.encode([query], convert_to_numpy=True)
            D,I = st.session_state.rag_index.search(np.array(q_emb,dtype=np.float32), k=min(k,len(st.session_state.rag_texts)))
            return [st.session_state.rag_texts[i] for i in I[0]]
        except Exception as e:
            st.warning(f"RAG search failed: {e}")
    return []

def genai_request(prompt, placeholder):
    try:
        headers = {"Authorization": f"Bearer {GENAI_API_KEY}"}
        payload = {"model":"llama-3.3-70b-versatile","messages":[{"role":"system","content":"You are a helpful AI assistant."},{"role":"user","content": prompt}],"max_tokens":500}
        response = requests.post(GENAI_API_URL, json=payload, headers=headers, timeout=30)
        data = response.json()
        ai_text = data.get("choices",[{}])[0].get("message",{}).get("content","🤖 No response.")
    except Exception as e:
        ai_text = f"🤖 API error: {e}"
    st.session_state.history.append({"role":"ai","message":ai_text})
    placeholder.empty()
    return ai_text

def update_contextual_knowledge_graph(user_query, ai_response, top_rag_texts):
    G = nx.Graph()
    user_node = f"User: {user_query[:50]}"
    ai_node = f"AI: {ai_response[:50]}"
    G.add_node(user_node, type="user")
    G.add_node(ai_node, type="ai")
    
    ai_emb = embedding_model.encode([ai_response], convert_to_numpy=True)

    for text, emb in zip(st.session_state.rag_texts, st.session_state.rag_embeddings):
        if text in top_rag_texts:
            G.add_node(text, type="rag", emb=emb)
            sim = float(cosine_similarity(ai_emb, emb.reshape(1,-1))[0][0])
            G.add_edge(ai_node, text, weight=sim)

    for chat in st.session_state.history[-50:]:
        msg = chat["message"]
        emb = embedding_model.encode([msg], convert_to_numpy=True)
        G.add_node(msg, type="chat", emb=emb)
        sim = float(cosine_similarity(ai_emb, emb)[0][0])
        G.add_edge(ai_node, msg, weight=sim*0.7)

    return G

def process_yolo_image(img_file, confidence=0.25):
    img_hash = hash(img_file.getvalue())
    if img_hash in st.session_state.yolo_cache:
        return st.session_state.yolo_cache[img_hash]

    img = Image.open(img_file)
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width*ratio), int(img.height*ratio)))

    detected = []
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    if yolo_model:
        results = yolo_model(img)
        for r in results[0].boxes.data.tolist():
            x1,y1,x2,y2,conf,cls = r
            if conf < confidence: continue
            cls_name = yolo_model.names[int(cls)]
            detected.append(cls_name)
            draw.rectangle([int(x1),int(y1),int(x2),int(y2)], outline="red", width=3)
            draw.text((int(x1), max(int(y1)-15,0)), f"{cls_name} {conf:.2f}", fill="yellow")
    desc = "Detected: " + ", ".join(detected) if detected else "No objects detected."
    st.session_state.yolo_cache[img_hash] = (img_draw, desc)
    return img_draw, desc

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Object Detection", "📊 Analytics"])

# --- Chat Tab ---
with tab1:
    st.header("Chat with Smart AI 🤖")
    user_input = ""

    # Voice input
    if st.session_state.voice_input == "record" and not st.session_state.voice_recording_active:
        st.session_state.voice_recording_active = True
        def record_voice():
            try:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    st.info("🎤 Listening...")
                    audio = r.listen(source, timeout=5)
                    st.session_state.voice_input_text = r.recognize_google(audio)
            except Exception as e:
                st.session_state.voice_input_text = f"Error: {e}"
            finally:
                st.session_state.voice_input = ""
                st.session_state.voice_recording_active = False
        Thread(target=record_voice).start()
        st.info("Recording in background...")

    if st.session_state.voice_input_text:
        user_input = st.session_state.voice_input_text
        if user_input.startswith("Error:"): st.error(user_input); user_input=""

    if not user_input:
        user_input = st.text_input("Type your query... ✏️", key="chat_input")

    send_disabled = st.session_state.voice_input == "record"
    if st.button("Send 📨", disabled=send_disabled) and user_input:
        st.session_state.stats["queries"] += 1
        st.session_state.history.append({"role":"user","message":f"🧑 {user_input}"})

        top_rag_texts = []
        if files_for_processing:
            files_hash = "_".join([f.name+str(getattr(f,"size",0)) for f in files_for_processing])
            cache_path = f".cache_rag_{hash(files_hash)}.pkl"
            if files_hash != st.session_state.uploaded_files_hash:
                all_texts = []
                for f in files_for_processing: all_texts.extend(process_file(f))
                if all_texts: build_rag_index(all_texts, cache_path)
                st.session_state.uploaded_files_hash = files_hash
            elif os.path.exists(cache_path):
                with open(cache_path,"rb") as f: st.session_state.rag_index, st.session_state.rag_embeddings, st.session_state.rag_texts = pickle.load(f)
            top_rag_texts = rag_search(user_input)
        context_text = "\n".join(top_rag_texts)

        placeholder = st.empty()
        with st.spinner("🤖 AI is typing..."): time.sleep(0.5)
        final_prompt = f"{context_text}\n\nUser: {user_input}\nTone: {tone}\nLanguage: {language}" if context_text else f"User: {user_input}\nTone: {tone}\nLanguage: {language}"
        ai_response = genai_request(final_prompt, placeholder)

        if len(st.session_state.history) > 100: st.session_state.history = st.session_state.history[-100:]
        try: st.session_state.graph = update_contextual_knowledge_graph(user_input, ai_response, top_rag_texts)
        except: pass

    st.subheader("📜 Chat History")
    for i, chat in enumerate(st.session_state.history):
        message(chat["message"], is_user=(chat["role"]=="user"), key=f"{chat['role']}_{i}")

    # PDF download
    if st.button("Download Chat PDF 📄"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        content = [Paragraph("Chat History", styles["Title"]), Spacer(1,12)]
        for chat in st.session_state.history:
            content.append(Paragraph(f"{chat['role'].upper()}: {chat['message']}", styles["Normal"]))
            content.append(Spacer(1,12))
        try: doc.build(content); buffer.seek(0); st.download_button("Download PDF", data=buffer, file_name="chat_history.pdf", mime="application/pdf")
        except Exception as e: st.error(f"PDF build failed: {e}")

    # Email
    if st.button("Send Last Response via Email ✉️"):
     if recipient and subject and st.session_state.history:
        last_response = st.session_state.history[-1]["message"]
        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.attach(MIMEText(last_response, "plain"))

            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()

            st.success("Email sent ✅")
        except Exception as e:
            st.error(f"Email failed: {e}")
    else:
        st.warning("Enter recipient + subject + have chat!")

    # Knowledge Graph display
    if st.sidebar.button("Show Knowledge Graph"):
        if st.session_state.graph:
            net = Network(height="500px", width="100%", notebook=False, directed=False)
            for node, data in st.session_state.graph.nodes(data=True):
                color = {"user":"blue","ai":"green","rag":"orange","chat":"lightgray"}.get(data.get("type","chat"), "gray")
                net.add_node(node, label=node, color=color)
            for u,v,d in st.session_state.graph.edges(data=True):
                w = d.get("weight", 0.1)
                net.add_edge(u, v, value=w, width=1 + w*4, color=f"rgba(255,0,0,{w})")
            net.show_buttons(filter_=['physics'])
            path = "context_graph.html"
            net.save_graph(path)
            with open(path, "r", encoding="utf-8") as f: html = f.read()
            st.components.v1.html(html, height=500)

# --- Object Detection Tab ---
with tab2:
    st.header("Object Detection with YOLOv8 🖼️")
    uploaded_images = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="img_upload")
    st.session_state.stats["images_detected"] = len(uploaded_images) if uploaded_images else 0

    for img_file in uploaded_images or []:
        try:
            if img_file.size > 2*1024*1024: st.warning(f"{img_file.name} is >2MB, resizing...")
            img_draw, desc = process_yolo_image(img_file, confidence)
            st.image(img_draw, caption="Detections 🎯", use_column_width=True)
            st.info(desc)
            buf=io.BytesIO(); img_draw.save(buf, format="PNG"); buf.seek(0)
            st.download_button("Download Annotated", data=buf, file_name=f"annotated_{img_file.name}", mime="image/png")
        except Exception as e: st.error(f"Error processing {img_file.name}: {e}")

# --- Analytics Tab ---
with tab3:
    st.header("📊 Analytics Dashboard")
    st.write(f"Total Queries: {st.session_state.stats['queries']}")
    st.write(f"Files Uploaded: {st.session_state.stats['files_uploaded']}")
    st.write(f"Images Processed: {st.session_state.stats['images_detected']}")