import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cv2
import tempfile
import os
import pandas as pd
from datetime import datetime

# 🧠 Sayfa Konfigürasyonu
st.set_page_config(
    page_title="🧠 Brain Tumor Detection | AI Medical Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/casper',
        'Report a bug': 'mailto:casper@example.com',
        'About': """
        # 🧠 Brain Tumor Detection System
        
        **AI-Powered Medical Image Analysis**
        
        This application uses YOLO (You Only Look Once) deep learning model 
        for automated brain tumor detection in MRI scans.
        
        **Features:**
        - Real-time tumor detection
        - Confidence scoring
        - Interactive visualizations
        - Medical report generation
        
        **Developer:** Casper
        **Model:** YOLOv8s trained on brain MRI dataset
        **Performance:** 84% Recall, 47.6% mAP@50
        """
    }
)

# 🎨 Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 15px 0;
        color: white;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h4 {
        margin-bottom: 10px;
        font-size: 1.2em;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 5px 0;
        opacity: 0.9;
        font-size: 0.95em;
    }
    
    .detection-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 15px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 15px 0;
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(225, 112, 85, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# 🎯 Ana Başlık
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Brain Tumor Detection System</h1>
    <p>Advanced Medical Image Analysis with YOLOv8 Deep Learning</p>
    <p><strong>Real-time MRI Scan Analysis • Automated Detection • Clinical Decision Support</strong></p>
</div>
""", unsafe_allow_html=True)

# 🔧 Model Loading Function
@st.cache_resource
def load_model():
    """Model yükleme fonksiyonu"""
    try:
        model_path = "best.pt"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            return model, "Custom trained model loaded successfully! ✅"
        else:
            st.error("⚠️ Trained model not found! Please train the model first.")
            return None, "Model not found"
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None, f"Error: {str(e)}"

# 🖼️ Görüntü İşleme Fonksiyonu
def process_image(image, model):
    """Görüntü analizi ve tespit"""
    try:
        # Model prediction
        results = model(image)
        
        # Sonuçları parse et
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Koordinatlar ve güven skoru
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Sınıf ismi
                    class_name = "Brain Tumor" if class_id == 1 else "Normal"
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                    
                    # Görsel üzerine çizim
                    draw = ImageDraw.Draw(annotated_image)
                    
                    # Renk seçimi
                    color = "#FF4B4B" if class_name == "Brain Tumor" else "#00FF00"
                    
                    # Bounding box çiz
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Label metni
                    label = f"{class_name}: {confidence:.2%}"
                    
                    # Text background
                    bbox = draw.textbbox((x1, y1-25), label)
                    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=color)
                    draw.text((x1, y1-25), label, fill="white")
        
        return annotated_image, detections
        
    except Exception as e:
        st.error(f"❌ Image processing error: {str(e)}")
        return image, []

# 📊 Sonuç Analizi Fonksiyonu
def analyze_results(detections):
    """Tespit sonuçlarını analiz et"""
    if not detections:
        return {
            'status': 'No Detection',
            'risk_level': 'Low',
            'recommendation': 'No abnormalities detected. Regular check-ups recommended.',
            'tumor_count': 0,
            'max_confidence': 0.0
        }
    
    tumor_detections = [d for d in detections if d['class'] == 'Brain Tumor']
    
    if not tumor_detections:
        return {
            'status': 'Normal',
            'risk_level': 'Low',
            'recommendation': 'No tumors detected. Continue regular monitoring.',
            'tumor_count': 0,
            'max_confidence': max([d['confidence'] for d in detections])
        }
    
    max_confidence = max([d['confidence'] for d in tumor_detections])
    tumor_count = len(tumor_detections)
    
    # Risk seviyesi belirleme
    if max_confidence >= 0.8:
        risk_level = 'High'
        recommendation = 'High confidence tumor detection. Immediate medical consultation recommended.'
    elif max_confidence >= 0.6:
        risk_level = 'Medium'
        recommendation = 'Potential tumor detected. Further examination by specialist advised.'
    else:
        risk_level = 'Low'
        recommendation = 'Low confidence detection. Additional imaging may be needed for confirmation.'
    
    return {
        'status': 'Tumor Detected',
        'risk_level': risk_level,
        'recommendation': recommendation,
        'tumor_count': tumor_count,
        'max_confidence': max_confidence
    }

# 📱 Ana Uygulama
def main():
    # Sidebar - Model Info & Controls
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Model durumu
        model, model_status = load_model()
        
        if model:
            st.success(model_status)
            
            # Model bilgileri
            st.subheader("📊 Model Performance")
            
            # Güzel renkli metrikler
            st.markdown("""
            <div class="sidebar-metric">
                <h4>🎯 Recall</h4>
                <p>84.0%</p>
                <small>High Sensitivity</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sidebar-metric">
                <h4>⚡ Precision</h4>
                <p>45.7%</p>
                <small>Moderate</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sidebar-metric">
                <h4>📈 mAP@50</h4>
                <p>47.6%</p>
                <small>Good Performance</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="sidebar-metric">
                <h4>🔧 Model Size</h4>
                <p>11.1M</p>
                <small>Efficient</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Ayarlar
            st.subheader("⚙️ Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.25, 
                step=0.05,
                help="Minimum confidence level for detections"
            )
            
            show_all_detections = st.checkbox(
                "Show All Detections", 
                value=True,
                help="Display both tumor and normal tissue detections"
            )
            
        else:
            st.error("⚠️ Model not available")
            st.info("Please ensure the trained model exists in: runs/detect/brain_tumor_yolov8s/weights/best.pt")
            return
        
        # Medical Disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Medical Disclaimer</h4>
        <p><small>This AI system is designed for screening purposes only. 
        All results should be reviewed by qualified medical professionals. 
        This tool does not replace professional medical diagnosis.</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ana içerik alanı
    if model:
        # Tabs oluştur
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Image Analysis", 
            "📊 Batch Processing", 
            "📈 Model Info", 
            "📋 Medical Report"
        ])
        
        with tab1:
            st.header("🔍 Single Image Analysis")
            
            # Görüntü yükleme seçenekleri
            upload_option = st.radio(
                "Choose input method:",
                ["Upload Image", "Use Sample Images", "Camera Capture"]
            )
            
            uploaded_image = None
            
            if upload_option == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Choose a brain MRI image...",
                    type=['png', 'jpg', 'jpeg', 'webp', 'dcm'],
                    help="Upload MRI scan in PNG, JPG, JPEG, WebP, or DICOM format"
                )
                
                if uploaded_file:
                    uploaded_image = Image.open(uploaded_file).convert('RGB')
                    
            elif upload_option == "Use Sample Images":
                # Sample image paths (if available)
                sample_dir = "brain-tumor/valid/images"
                if os.path.exists(sample_dir):
                    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.webp'))]
                    
                    if sample_files:
                        selected_sample = st.selectbox("Select a sample image:", sample_files)
                        if selected_sample:
                            sample_path = os.path.join(sample_dir, selected_sample)
                            uploaded_image = Image.open(sample_path).convert('RGB')
                    else:
                        st.info("No sample images found in validation directory.")
                else:
                    st.info("Sample directory not found.")
                    
            elif upload_option == "Camera Capture":
                camera_image = st.camera_input("Take a picture of MRI scan")
                if camera_image:
                    uploaded_image = Image.open(camera_image).convert('RGB')
            
            # Görüntü analizi
            if uploaded_image:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Original Image")
                    st.image(uploaded_image, caption="Uploaded MRI Scan", use_column_width=True)
                    
                    # Görüntü bilgileri
                    st.info(f"""
                    **Image Information:**
                    - Size: {uploaded_image.size}
                    - Mode: {uploaded_image.mode}
                    - Format: {getattr(uploaded_image, 'format', 'Unknown')}
                    """)
                
                # Analiz butonu
                if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🧠 AI is analyzing the brain scan..."):
                        # Model ile tahmin
                        annotated_image, detections = process_image(uploaded_image, model)
                        
                        # Sonuçları analiz et
                        analysis = analyze_results(detections)
                    
                    with col2:
                        st.subheader("🎯 Detection Results")
                        st.image(annotated_image, caption="AI Analysis Results", use_column_width=True)
                    
                    # Sonuç özeti
                    st.markdown("---")
                    st.subheader("📋 Analysis Summary")
                    
                    # Ana sonuç kartı
                    if analysis['status'] == 'Tumor Detected':
                        st.markdown(f"""
                        <div class="detection-result">
                        <h3>🔴 TUMOR DETECTED</h3>
                        <p><strong>Confidence:</strong> {analysis['max_confidence']:.1%}</p>
                        <p><strong>Tumor Count:</strong> {analysis['tumor_count']}</p>
                        <p><strong>Risk Level:</strong> {analysis['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                        <h3>✅ NO TUMOR DETECTED</h3>
                        <p><strong>Status:</strong> {analysis['status']}</p>
                        <p><strong>Risk Level:</strong> {analysis['risk_level']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Öneriler
                    st.markdown(f"""
                    <div class="info-box">
                    <h4>💡 Medical Recommendation</h4>
                    <p>{analysis['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detaylı sonuçlar
                    if detections:
                        st.subheader("📊 Detailed Detection Results")
                        
                        # DataFrame oluştur
                        df_detections = pd.DataFrame(detections)
                        df_detections['confidence'] = df_detections['confidence'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(df_detections, use_container_width=True)
                        
                        # Güven skor dağılımı
                        fig = px.bar(
                            x=[d['class'] for d in detections],
                            y=[d['confidence'] for d in detections],
                            title="Detection Confidence Scores",
                            labels={'x': 'Detection Class', 'y': 'Confidence Score'},
                            color=[d['class'] for d in detections]
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("📊 Batch Processing")
            st.info("🚧 This feature will process multiple images simultaneously. Coming soon!")
            
            # Batch processing placeholder
            st.markdown("""
            **Planned Features:**
            - Upload multiple MRI scans
            - Automated batch analysis
            - Comparative results
            - Export summary reports
            - Statistical analysis
            """)
        
        with tab3:
            st.header("📈 Model Information & Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏗️ Model Architecture")
                st.markdown("""
                - **Model:** YOLOv8s (Small)
                - **Parameters:** 11.1M
                - **Input Size:** 640x640
                - **Classes:** 2 (Normal, Tumor)
                - **Training Epochs:** 63
                - **Early Stopping:** Yes (Patience: 20)
                """)
                
                st.subheader("📊 Performance Metrics")
                metrics_data = {
                    'Metric': ['Precision', 'Recall', 'mAP@50', 'mAP@50-95'],
                    'Score': [0.457, 0.840, 0.476, 0.346],
                    'Percentage': ['45.7%', '84.0%', '47.6%', '34.6%']
                }
                
                df_metrics = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics, use_container_width=True)
            
            with col2:
                st.subheader("📈 Performance Visualization")
                
                # Performance radar chart
                metrics = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
                values = [0.457, 0.840, 0.476, 0.346]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name='Model Performance'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title="Model Performance Metrics"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Training info
                st.subheader("🎯 Training Details")
                st.markdown("""
                - **Dataset:** Brain Tumor MRI Images
                - **Training Images:** 893
                - **Validation Images:** 223
                - **Optimizer:** AdamW
                - **Learning Rate:** 0.01
                - **Batch Size:** 16
                - **Device:** CPU
                """)
        
        with tab4:
            st.header("📋 Medical Report Generator")
            
            if 'analysis' in locals() and 'uploaded_image' in locals() and uploaded_image:
                # Rapor bilgileri
                report_data = {
                    'Patient ID': st.text_input("Patient ID", value="P001"),
                    'Date': datetime.now().strftime("%Y-%m-%d"),
                    'Time': datetime.now().strftime("%H:%M:%S"),
                    'Technician': st.text_input("Technician", value="AI System"),
                    'Analysis Status': analysis['status'],
                    'Risk Level': analysis['risk_level'],
                    'Confidence': f"{analysis['max_confidence']:.1%}",
                    'Recommendation': analysis['recommendation']
                }
                
                # Rapor görüntüleme
                st.subheader("📄 Generated Medical Report")
                
                report_content = f"""
                **BRAIN MRI ANALYSIS REPORT**
                
                **Patient Information:**
                - Patient ID: {report_data['Patient ID']}
                - Analysis Date: {report_data['Date']}
                - Analysis Time: {report_data['Time']}
                - Analyzing System: {report_data['Technician']}
                
                **Analysis Results:**
                - Status: {report_data['Analysis Status']}
                - Risk Level: {report_data['Risk Level']}
                - AI Confidence: {report_data['Confidence']}
                
                **Medical Recommendation:**
                {report_data['Recommendation']}
                
                **Technical Details:**
                - Model: YOLOv8s Brain Tumor Detection
                - Detection Threshold: {confidence_threshold}
                - Analysis Method: Deep Learning Object Detection
                
                **Disclaimer:**
                This automated analysis is for screening purposes only. 
                Results must be reviewed and validated by qualified medical professionals.
                """
                
                st.text_area("Report Content", report_content, height=400)
                
                # İndirme butonu
                if st.button("📥 Download Report", type="primary"):
                    st.download_button(
                        label="💾 Save Report as TXT",
                        data=report_content,
                        file_name=f"brain_tumor_report_{report_data['Patient ID']}_{report_data['Date']}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Please analyze an image first to generate a medical report.")

# 🚀 Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h4>🧠 AI Technology</h4>
    <p><strong>YOLOv8 Deep Learning</strong></p>
    <p>Real-time Detection</p>
    <p><small>11.1M Parameters</small></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h4>⚡ Performance</h4>
    <p><strong>84% Sensitivity</strong></p>
    <p>Sub-second Analysis</p>
    <p><small>47.6% mAP@50</small></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h4>🎯 Purpose</h4>
    <p><strong>Medical Screening</strong></p>
    <p>Decision Support</p>
    <p><small>Portfolio Ready</small></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; padding: 20px; color: #666;'>
<p>🧠 <strong>Brain Tumor Detection System</strong> | Developed with ❤️ for Medical AI | 
<a href='https://github.com/casper' target='_blank'>GitHub</a> | 
<a href='https://linkedin.com/in/casper' target='_blank'>LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
