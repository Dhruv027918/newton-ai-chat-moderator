import streamlit as st
import pickle
import re
import emoji
import contractions
import numpy as np

# Load Models
@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/toxicity_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
    return vectorizer, model, threshold

vectorizer, model, optimal_threshold = load_models()
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Preprocessing & Spam Check
def analyze_message(message):
    # 1. Check for spam (URLs)
    contains_link = bool(re.search(r'https?://\S+|www\.\S+', message))
    
    # 2. Clean text for ML model
    clean_msg = str(message).lower()
    clean_msg = contractions.fix(clean_msg)
    clean_msg = emoji.replace_emoji(clean_msg, replace='')
    clean_msg = re.sub(r'https?://\S+|www\.\S+', '', clean_msg) 
    clean_msg = re.sub(r'[^a-zA-Z\s]', '', clean_msg)
    clean_msg = re.sub(r'\s+', ' ', clean_msg).strip()
    
    # 3. Predict
    vec_msg = vectorizer.transform([clean_msg])
    probs = model.predict_proba(vec_msg)
    scores = {labels[i]: probs[i][0][1] for i in range(len(labels))}
    
    # 4. Apply Flags
    flags = []
    for label, prob in scores.items():
        if label == 'severe_toxic' and prob >= optimal_threshold:
            flags.append(label)
        elif label != 'severe_toxic' and prob >= 0.5:
            flags.append(label)
            
    if contains_link:
        flags.append("spam/advertisement")
        
    if not flags:
        flags.append("non-toxic")
        
    # 5. Severity & Action Engine
    severity = 0
    if "non-toxic" not in flags:
        max_prob = max(scores.values())
        severity = 3 if contains_link else int(np.interp(max_prob, [0.5, 1.0], [1, 4]))
        if len(flags) > 1 and not contains_link:
            severity = min(severity + 1, 4)
        if 'severe_toxic' in flags or 'threat' in flags:
            severity = 5
            
    action = "Allow"
    if severity == 5:
        action = "Auto-Ban"
    elif severity >= 3:
        action = "Mute (24h)"
    elif severity >= 1:
        action = "Warning"
        
    return flags, severity, action

# Streamlit UI
st.title("🛡️ Advanced Moderation Engine")
st.markdown("Analyzes text for toxicity, specific threats, and spam/links.")

user_input = st.text_area("Enter a comment to test:", "")

if st.button("Analyze Text"):
    if user_input:
        flags, severity, action = analyze_message(user_input)
        
        st.subheader("Analysis Results:")
        st.write(f"**Flags Detected:** {', '.join(flags).title()}")
        st.write(f"**Severity Score:** {severity}/5")
        
        if action == "Allow":
            st.success(f"**Action:** {action}")
        elif action == "Warning":
            st.warning(f"**Action:** {action}")
        else:
            st.error(f"**Action:** {action}")
    else:
        st.warning("Please enter some text to analyze.")