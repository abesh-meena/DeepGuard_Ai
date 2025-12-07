"""
ui/streamlit_app.py

Professional Streamlit UI for DeepGuard AI - DeepFake Detection System
Run with: streamlit run ui/streamlit_app.py
"""
import streamlit as st
import requests
from io import BytesIO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="DeepGuard AI - DeepFake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding-top: 0rem;
            background-color: #f3f4f6; /* light gray for better contrast */
        }
        
        /* Header styling */
        .header {
            background: linear-gradient(135deg, #1f2937 0%, #4f46e5 100%);
            padding: 2rem 0;
            margin: -1rem -1rem 2rem -1rem;
            box-shadow: 0 4px 6px rgba(15, 23, 42, 0.35);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .header-title {
            color: #f9fafb;
            font-size: 3rem;
            font-weight: bold;
            margin: 0;
            text-align: center;
            text-shadow: 2px 2px 6px rgba(15, 23, 42, 0.6);
        }
        
        .header-subtitle {
            color: rgba(249, 250, 251, 0.9);
            font-size: 1.2rem;
            text-align: center;
            margin-top: 0.5rem;
        }
        
        /* Footer styling */
        .footer {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: #e5e7eb;
            padding: 2rem 0;
            margin: 3rem -1rem -1rem -1rem;
            text-align: center;
            box-shadow: 0 -4px 6px rgba(15, 23, 42, 0.4);
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .footer-text {
            color: rgba(229, 231, 235, 0.9);
            font-size: 1rem;
            margin: 0;
        }
        
        /* Card styling */
        .card {
            background: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.12);
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #f9fafb;
            border: none;
            border-radius: 6px;
            padding: 0.6rem 2.2rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            transition: all 0.2s ease-in-out;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(79, 70, 229, 0.35);
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #4f46e5;
            border-radius: 10px;
            padding: 1rem;
            background-color: #eef2ff;
        }
        
        /* Success/Error message styling */
        .stSuccess, .stError {
            border-radius: 6px;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #e5edff 0%, #d1fae5 100%);
            padding: 1.6rem;
            border-radius: 12px;
            margin: 1rem 0;
            border-left: 5px solid #4f46e5;
            color: #111827;
        }
        .info-box h3,
        .info-box p {
            color: #111827;
        }
        
        .feature-box {
            background: #ffffff;
            padding: 1.6rem;
            border-radius: 12px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
            color: #111827;
        }
        .feature-box h4,
        .feature-box li {
            color: #111827;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <div class="header-content">
            <h1 class="header-title">🛡️ DeepGuard AI</h1>
            <p class="header-subtitle">Advanced DeepFake Detection System | Protect What's Real</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    API_URL = st.text_input(
        "API Endpoint",
        value="http://localhost:8000/predict",
        help="Enter your FastAPI backend URL"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Supported Formats")
    st.markdown("""
    **Images:**
    - JPG, JPEG, PNG
    
    **Videos:**
    - MP4, AVI, MOV, MKV
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    DeepGuard AI uses advanced deep learning 
    models to detect deepfake content in 
    images and videos with high accuracy.
    """)

# Main content area
st.markdown("""
    <div class="info-box">
        <h3>🔍 How It Works</h3>
        <p>Upload an image or video file to analyze. Our AI-powered system will detect if the content is real or a deepfake using state-of-the-art neural networks.</p>
    </div>
""", unsafe_allow_html=True)

# File upload section
st.markdown("### 📤 Upload File for Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"],
        help="Upload an image or video file to detect deepfakes",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
    <div class="feature-box">
        <h4>✨ Features</h4>
        <ul>
            <li>95%+ Accuracy (Hybrid Model)</li>
            <li>Image & Video Support</li>
            <li>Real-time Analysis</li>
            <li>Fast Processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Display uploaded file and process
if uploaded_file is not None:
    st.markdown("---")
    
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_video:
            st.markdown("### 📹 Uploaded Video")
            # For videos, we'll show a placeholder or first frame if possible
            st.info(f"Video file: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
            st.markdown("**Note:** Video processing may take longer than images.")
        else:
            st.markdown("### 🖼️ Uploaded Image")
            # For older Streamlit versions, `use_container_width` may not exist.
            # Use `use_column_width` instead for similar behavior.
            st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
    
    with col2:
        st.markdown("### 📊 File Information")
        st.metric("File Name", uploaded_file.name)
        st.metric("File Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
        st.metric("File Type", "Video" if is_video else "Image")
    
    # Analyze button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze for DeepFake",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Prepare file for upload
            files = {
                "file": (
                    uploaded_file.name,
                    open(tmp_path, "rb"),
                    uploaded_file.type if hasattr(uploaded_file, 'type') else f"application/{file_extension}"
                )
            }
            
            # Show progress
            with st.spinner("🔄 Analyzing file... This may take a moment."):
                response = requests.post(API_URL, files=files, timeout=120)
            
            # Close file
            files["file"][1].close()
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results in a nice format
                st.markdown("---")
                st.markdown("### 📊 Analysis Results")
                
                # Prediction result
                prediction = result.get("prediction", "unknown")
                probabilities = result.get("probabilities", [[0, 0]])
                
                # Color coding
                if prediction.lower() == "fake":
                    prediction_color = "🔴"
                    prediction_bg = "#ffebee"
                else:
                    prediction_color = "🟢"
                    prediction_bg = "#e8f5e9"
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: {prediction_bg}; padding: 2rem; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h2 style="margin: 0; color: #333;">{prediction_color} Prediction</h2>
                        <h1 style="margin: 0.5rem 0; color: #333; text-transform: uppercase;">{prediction}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Probability display
                    if len(probabilities) > 0 and len(probabilities[0]) >= 2:
                        real_prob = probabilities[0][0] * 100
                        fake_prob = probabilities[0][1] * 100
                        
                        st.markdown("### 📈 Confidence Scores")
                        st.progress(real_prob / 100, text=f"Real: {real_prob:.2f}%")
                        st.progress(fake_prob / 100, text=f"Fake: {fake_prob:.2f}%")
                
                # Detailed results
                with st.expander("📋 View Detailed Results"):
                    st.json(result)
                
                # Success message
                st.success("✅ Analysis completed successfully!")
                
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The file might be too large or the server is busy.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection error. Please make sure the API server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

else:
    # Show placeholder when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #667eea;">👆 Upload a file to get started</h3>
        <p style="color: #666;">Select an image or video file from your device to analyze for deepfake content</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <p class="footer-text">This project made by <strong>ABESH MEENA</strong></p>
            <p class="footer-text" style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">
                DeepGuard AI - Advanced DeepFake Detection System | Powered by Deep Learning
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)
