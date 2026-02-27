import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import contractions
import pickle

# --- PAGE SETUP ---
st.set_page_config(page_title="NewtonAI Chat Moderator", page_icon="🛡️", layout="centered")

# --- LOAD MODELS ---
# @st.cache_resource ensures the models only load once, making the app much faster
@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/toxicity_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- PREPROCESSING & ENGINE LOGIC ---
def preprocess_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_chat_message(message, threshold=0.9999):
    clean_msg = preprocess_text(message)
    vec_msg = vectorizer.transform([clean_msg])
    
    probs = model.predict_proba(vec_msg)
    class_probs = {labels[i]: probs[i][0][1] for i in range(len(labels))}
    
    flags = []
    for label, prob in class_probs.items():
        if label == 'severe_toxic' and prob >= threshold:
            flags.append(label)
        elif label != 'severe_toxic' and prob >= 0.5:
            flags.append(label)
            
    if not flags:
        flags.append('non-toxic')
        
    max_prob = max(class_probs.values())
    if 'non-toxic' in flags:
        severity = 0
    else:
        severity = int(np.interp(max_prob, [0.5, 1.0], [1, 4]))
        if len(flags) > 1:
            severity += 1
        if 'severe_toxic' in flags:
            severity = 5
            
    action = "None"
    if severity == 0:
        action = "Allow"
    elif severity in [1, 2]:
        action = "Warning"
    elif severity in [3, 4]:
        action = "Mute (24h)"
    elif severity >= 5:
        action = "Auto-Ban"
        
    return flags, severity, action

# --- FRONT-END UI ---
st.title("🛡️ Live Chat Moderation Engine")
st.markdown("Test the multi-label toxicity classifier built for **NewtonAI Technologies**.")

# User Input
user_input = st.text_area("Enter a chat message to analyze:", placeholder="Type a dummy chat message here...")

# Trigger Analysis
if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        flags, severity, action = analyze_chat_message(user_input)
        
        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Severity Score", f"{severity} / 5")
        col2.metric("Recommended Action", action)
        
        # Format flags to look cleaner (e.g., 'severe_toxic' -> 'Severe Toxic')
        formatted_flags = ", ".join(flags).title().replace('_', ' ')
        col3.metric("Flags Detected", formatted_flags)
        
        # Visual color-coded feedback based on the moderation action
        if action == "Allow":
            st.success("✅ **Message Approved:** No toxic content detected.")
        elif action == "Warning":
            st.warning("⚠️ **Warning Issued:** Minor infractions detected.")
        elif action == "Mute (24h)":
            st.error("🔇 **User Muted:** Moderate toxicity or direct insults detected.")
        elif action == "Auto-Ban":
            st.error("🚫 **USER AUTO-BANNED:** Severe toxicity or threats detected!")