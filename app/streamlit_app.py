import os
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import time
from PIL import Image
import cv2
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import plotly.io as pio

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Create videos directory if it doesn't exist
VIDEOS_DIR = os.path.join(project_root, 'videos')
Path(VIDEOS_DIR).mkdir(parents=True, exist_ok=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from models.emotion_model import (
    EmotionCNN,  # Original CNN
    EmotionCNNModern,  # Modern CNN
    EmotionCNNRegularized,  # Regularized CNN
    load_model
)
import torchvision.transforms as transforms
import time
from models.detect_emotions import SalesCallAnalyzer, process_video
from config import MODEL_TYPE_PROD, MODEL_PATH, CONFIDENCE_THRESHOLD, ANALYSIS_INTERVAL
import plotly.graph_objects as go

def main():
    # Set page config
    st.set_page_config(
        page_title="Sales Call Emotion Analyzer",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e88e5;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("Sales Call Emotion Analyzer")
    st.markdown("""
        Analyze emotions and positive engagement in sales calls using deep learning.
        Upload a video or image to get started.
    """)

    # Create tabs with custom styling
    tab1, tab2 = st.tabs(["📹 Video Analysis", "🖼️ Image Analysis"])

    with tab1:
        st.header("Video Analysis")
        st.markdown("""
            Upload a video file to analyze emotions and positive engagement throughout the call.
            The system will process the video and provide detailed analytics.
        """)

        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Process video
            if st.button("Analyze Video", type="primary"):
                with st.spinner("Processing video..."):
                    try:
                        # Process video
                        results = process_video(video_path)

                        if results is None:
                            st.error("Video processing failed")
                            return

                        # Display results
                        st.success("Video processing complete!")

                        # Show metrics
                        st.subheader("Analysis Results")

                        # Calculate total frames and percentages
                        total_frames = results['total_frames']
                        emotion_counts = results['emotion_counts']
                        face_detection_count = results['face_detection_count']
                        face_coverage_distribution = results['face_coverage_distribution']
                        emotion_percentages = {emotion: (count/total_frames)*100 for emotion, count in emotion_counts.items()}
                        face_detection_percentage = (face_detection_count/total_frames)*100

                        # Create main metrics row with custom styling
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric(
                                "Average Positive Engagement",
                                f"{results['avg_engagement']:.2f}",
                                f"{'↑' if results['engagement_trend'] > 0 else '↓'} {abs(results['engagement_trend']):.2f}"
                            )
                        with metrics_col2:
                            st.metric(
                                "Emotion Stability",
                                f"{results['emotion_stability']:.2f}",
                                "Higher is more stable"
                            )
                        with metrics_col3:
                            st.metric(
                                "Face Detection Rate",
                                f"{face_detection_percentage:.1f}%",
                                f"{face_detection_count} frames"
                            )

                        # Create two main columns for charts with equal width
                        main_col1, main_col2 = st.columns(2)

                        with main_col1:
                            # Emotion Frequencies
                            fig_emotions = go.Figure()
                            fig_emotions.add_trace(go.Bar(
                                y=list(emotion_counts.keys()),
                                x=list(emotion_counts.values()),
                                orientation='h',
                                text=[f'{v} frames\n({emotion_percentages[k]:.1f}%)' for k, v in emotion_counts.items()],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(
                                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
                                ),
                                name='Emotion Frequencies'
                            ))
                            fig_emotions.update_layout(
                                title='Emotion Distribution',
                                xaxis_title='Number of Frames',
                                showlegend=False,
                                height=400,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_emotions, use_container_width=True)

                        with main_col2:
                            # Face Detection Distribution
                            fig_faces = go.Figure()
                            fig_faces.add_trace(go.Pie(
                                labels=['Frames with Faces', 'Frames without Faces'],
                                values=[face_detection_count, total_frames - face_detection_count],
                                hole=.4,
                                marker=dict(
                                    colors=['#1e88e5', '#90caf9']
                                ),
                                textinfo='label+percent',
                                textfont_size=14
                            ))
                            fig_faces.update_layout(
                                title='Face Detection Distribution',
                                showlegend=False,
                                height=400,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_faces, use_container_width=True)

                        # Create two columns for emotion transitions and face coverage
                        trend_col1, trend_col2 = st.columns(2)

                        with trend_col1:
                            # Emotion Transitions Analysis
                            transitions = results['emotion_transitions']
                            top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:5]

                            fig_transitions = go.Figure()
                            fig_transitions.add_trace(go.Bar(
                                y=[t[0] for t in top_transitions],
                                x=[t[1] for t in top_transitions],
                                orientation='h',
                                text=[f'{t[1]} times' for t in top_transitions],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(
                                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5']
                                )
                            ))
                            fig_transitions.update_layout(
                                title='Top 5 Emotion Transitions',
                                xaxis_title='Number of Transitions',
                                showlegend=False,
                                height=300,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_transitions, use_container_width=True)

                        with trend_col2:
                            # Face Coverage Distribution
                            fig_coverage = go.Figure()
                            fig_coverage.add_trace(go.Bar(
                                y=list(face_coverage_distribution.keys()),
                                x=list(face_coverage_distribution.values()),
                                orientation='h',
                                text=[f'{v} frames' for v in face_coverage_distribution.values()],
                                textposition='auto',
                                textfont=dict(size=14),
                                marker=dict(
                                    color=['#1e88e5', '#42a5f5', '#90caf9']
                                )
                            ))
                            fig_coverage.update_layout(
                                title='Face Coverage Distribution',
                                xaxis_title='Number of Frames',
                                showlegend=False,
                                height=300,
                                font=dict(size=14),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_coverage, use_container_width=True)

                        # Positive Engagement Trend (full width at bottom)
                        fig_engagement = go.Figure()
                        fig_engagement.add_trace(go.Scatter(
                            x=results['time_segments'],
                            y=results['engagement_scores'],
                            mode='lines',
                            line=dict(color='#1e88e5', width=2),
                            name='Positive Engagement',
                            fill='tozeroy',
                            fillcolor='rgba(30, 136, 229, 0.1)',
                            hovertemplate='Time: %{x:.1f} seconds<br>Engagement: %{y:.2f}<extra></extra>'
                        ))
                        fig_engagement.update_layout(
                            title='Positive Engagement Trend Over Time',
                            xaxis_title='Time (seconds)',
                            yaxis_title='Positive Engagement Score',
                            showlegend=False,
                            height=300,
                            font=dict(size=14),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=40, b=20),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True)

                        # Video Summary in an expander
                        with st.expander("Video Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{results['duration']:.2f} seconds")
                            with col2:
                                st.metric("Total Frames", f"{total_frames:,}")
                            with col3:
                                st.metric("Frames with Faces", f"{face_detection_count:,}")

                        # Show processed video
                        st.subheader("Processed Video")
                        if os.path.exists(results['output_path']):
                            video_file = open(results['output_path'], 'rb')
                            video_bytes = video_file.read()
                            st.video(video_bytes)

                            # Add download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=video_bytes,
                                    file_name=os.path.basename(results['output_path']),
                                    mime="video/mp4"
                                )
                            with col2:
                                st.download_button(
                                    label="Download Analysis Report (PDF)",
                                    data=results['pdf_report'].getvalue(),
                                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                        else:
                            st.error(f"Could not find processed video at {results['output_path']}")

                    except Exception as e:
                        st.error(f"An error occurred during video processing: {str(e)}")
                    finally:
                        # Clean up temporary files
                        try:
                            if os.path.exists(video_path):
                                os.unlink(video_path)
                        except Exception as e:
                            st.warning(f"Could not clean up temporary file: {str(e)}")

    with tab2:
        st.header("Image Analysis")
        st.markdown("""
            Upload an image to analyze emotions and positive engagement.
            The system will detect faces and provide detailed emotion analysis.
        """)

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Add analyze button
            if st.button("Analyze Image", type="primary"):
                analyze_image(uploaded_file)
            else:
                # Show placeholder message
                st.info("Click 'Analyze Image' to start the analysis")

def display_results():
    """Display analysis results in Streamlit UI."""
    st.header("Analysis Results")

    # Calculate frequencies and percentages
    total_frames = st.session_state.results['total_frames']
    emotion_counts = st.session_state.results['emotion_counts']
    face_detection_count = st.session_state.results['face_detection_count']

    # Calculate percentages
    emotion_percentages = {emotion: (count/total_frames)*100 for emotion, count in emotion_counts.items()}
    face_detection_percentage = (face_detection_count/total_frames)*100

    # Create three columns for main KPIs
    col1, col2, col3 = st.columns(3)

    # KPI 1: Emotion Frequencies
    with col1:
        st.subheader("Emotion Frequencies")
        fig1 = go.Figure(data=[
            go.Bar(
                y=list(emotion_counts.keys()),
                x=list(emotion_counts.values()),
                orientation='h',
                text=[f'{v} frames\n({emotion_percentages[k]:.1f}%)' for k, v in emotion_counts.items()],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(
                    color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
                )
            )
        ])
        fig1.update_layout(
            xaxis_title='Number of Frames',
            showlegend=False,
            height=400,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    # KPI 2: Face Detection Frequency
    with col2:
        st.subheader("Face Detection")
        fig2 = go.Figure(data=[
            go.Bar(
                y=['Frames with Faces', 'Frames without Faces'],
                x=[face_detection_count, total_frames - face_detection_count],
                orientation='h',
                text=[f'{face_detection_count} frames\n({face_detection_percentage:.1f}%)',
                      f'{total_frames - face_detection_count} frames\n({100-face_detection_percentage:.1f}%)'],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(
                    color=['#1e88e5', '#90caf9']
                )
            )
        ])
        fig2.update_layout(
            xaxis_title='Number of Frames',
            showlegend=False,
            height=200,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # KPI 3: Face Coverage Distribution
    with col3:
        st.subheader("Face Coverage Distribution")
        coverage_data = st.session_state.results['face_coverage_distribution']
        fig3 = go.Figure(data=[
            go.Bar(
                y=list(coverage_data.keys()),
                x=list(coverage_data.values()),
                orientation='h',
                text=[f'{v} frames' for v in coverage_data.values()],
                textposition='auto',
                textfont=dict(size=14),
                marker=dict(
                    color=['#1e88e5', '#42a5f5', '#90caf9']
                )
            )
        ])
        fig3.update_layout(
            xaxis_title='Number of Frames',
            showlegend=False,
            height=200,
            font=dict(size=14),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Display processed video
    st.header("Processed Video")
    video_file = open(st.session_state.results['output_path'], 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

    # Download buttons
    st.download_button(
        label="Download Processed Video",
        data=video_bytes,
        file_name=os.path.basename(st.session_state.results['output_path']),
        mime="video/mp4"
    )

def analyze_image(uploaded_file):
    """Analyze emotions in an uploaded image file."""
    analyzer = SalesCallAnalyzer()
    st.session_state.analyzer = analyzer

    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    # Convert PIL Image to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces, gray = analyzer.detect_faces(image)

    if len(faces) == 0:
        st.warning("No faces detected in the image")
        return

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Create a copy of the image for drawing all face boxes
        image_with_boxes = image_rgb.copy()

        # Draw all face boxes first
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"Face {i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with all face boxes
        st.image(image_with_boxes, caption="Image with Face Detection", width=600)

    with col2:
        # Create a list of face options for the dropdown
        face_options = [f"Face {i+1}" for i in range(len(faces))]
        selected_face = st.selectbox("Select a face to analyze:", face_options)
        selected_idx = int(selected_face.split()[1]) - 1  # Convert "Face 1" to index 0

        # Get the selected face coordinates
        x, y, w, h = faces[selected_idx]

        # Show the cropped face image (larger size)
        face_roi = gray[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        st.image(face_rgb, caption=f"Cropped Face {selected_idx+1}", width=300)

        # Process the selected face
        face_tensor = analyzer.preprocess_face(face_roi)
        face_tensor = face_tensor.to(analyzer.device)

        # Get prediction
        with torch.no_grad():
            outputs = analyzer.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            emotion_idx = predicted.item()
            confidence = probabilities[0][emotion_idx].item()

            # Map emotion index to label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion = emotion_labels[emotion_idx]

            # Display results
            st.write(f"Detected Emotion: {emotion}")
            st.write(f"Confidence: {confidence:.2f}")

            # Create a horizontal bar chart of emotion probabilities
            probs = probabilities[0].cpu().numpy()
            fig = go.Figure(data=[
                go.Bar(
                    y=emotion_labels,
                    x=probs,
                    orientation='h',
                    text=[f'{p:.2f}' for p in probs],
                    textposition='auto',
                    textfont=dict(size=16),
                    marker=dict(
                        color=['#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#1abc9c']
                    )
                )
            ])
            fig.update_layout(
                title='Emotion Probabilities',
                xaxis_title='Probability',
                yaxis_title='Emotion',
                xaxis_range=[0, 1],
                height=400,
                showlegend=False,
                font=dict(size=16),
                yaxis=dict(tickfont=dict(size=16)),
                xaxis=dict(tickfont=dict(size=16)),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            st.plotly_chart(fig, use_container_width=True)

def generate_pdf_report(results, video_path, emotion_screenshots=None):
    """Generate a PDF report with analysis results."""
    # Create a PDF document
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Sales Call Analysis Report", title_style))
    story.append(Spacer(1, 20))

    # Report metadata
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Video source: {os.path.basename(video_path)}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics_data = [
        ["Metric", "Value"],
        ["Average Positive Engagement", f"{results['avg_engagement']:.2f}"],
        ["Emotion Stability", f"{results['emotion_stability']:.2f}"],
        ["Face Detection Rate", f"{(results['face_detection_count']/results['total_frames'])*100:.1f}%"],
        ["Video Duration", f"{results['duration']:.2f} seconds"],
        ["Total Frames", f"{results['total_frames']:,}"]
    ]
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),  # Blue header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),  # Light blue background
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))  # Blue grid lines
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))

    # Emotion Distribution Chart
    story.append(Paragraph("Emotion Distribution", styles['Heading2']))
    fig_emotions = go.Figure()
    fig_emotions.add_trace(go.Bar(
        y=list(results['emotion_counts'].keys()),
        x=list(results['emotion_counts'].values()),
        orientation='h',
        text=[f'{v} frames\n({(v/results["total_frames"])*100:.1f}%)' for k, v in results['emotion_counts'].items()],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(
            color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5', '#90caf9', '#bbdefb']
        )
    ))
    fig_emotions.update_layout(
        title='Emotion Distribution',
        xaxis_title='Number of Frames',
        showlegend=False,
        height=400,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Save emotion distribution chart at 85% size
    emotion_chart_bytes = pio.to_image(fig_emotions, format='png', width=680, height=510)  # 85% of 800x600
    emotion_chart_buffer = io.BytesIO(emotion_chart_bytes)
    story.append(RLImage(emotion_chart_buffer, width=5.1*inch, height=3.825*inch))  # 85% of 6x4.5
    story.append(Spacer(1, 20))

    # Face Detection Distribution Chart
    story.append(Paragraph("Face Detection Distribution", styles['Heading2']))
    fig_faces = go.Figure()
    fig_faces.add_trace(go.Pie(
        labels=['Frames with Faces', 'Frames without Faces'],
        values=[results['face_detection_count'], results['total_frames'] - results['face_detection_count']],
        hole=.4,
        marker=dict(
            colors=['#1e88e5', '#90caf9']
        ),
        textinfo='label+percent',
        textfont_size=14
    ))
    fig_faces.update_layout(
        title='Face Detection Distribution',
        showlegend=False,
        height=400,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Save face detection chart at 85% size
    faces_chart_bytes = pio.to_image(fig_faces, format='png', width=680, height=510)  # 85% of 800x600
    faces_chart_buffer = io.BytesIO(faces_chart_bytes)
    story.append(RLImage(faces_chart_buffer, width=5.1*inch, height=3.825*inch))  # 85% of 6x4.5
    story.append(Spacer(1, 20))

    # Emotion Transitions Chart
    story.append(Paragraph("Top Emotion Transitions", styles['Heading2']))
    transitions = sorted(results['emotion_transitions'].items(), key=lambda x: x[1], reverse=True)[:5]
    fig_transitions = go.Figure()
    fig_transitions.add_trace(go.Bar(
        y=[t[0] for t in transitions],
        x=[t[1] for t in transitions],
        orientation='h',
        text=[f'{t[1]} times' for t in transitions],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(
            color=['#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#42a5f5']
        )
    ))
    fig_transitions.update_layout(
        title='Top 5 Emotion Transitions',
        xaxis_title='Number of Transitions',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Save transitions chart at 85% size
    transitions_chart_bytes = pio.to_image(fig_transitions, format='png', width=680, height=340)  # 85% of 800x400
    transitions_chart_buffer = io.BytesIO(transitions_chart_bytes)
    story.append(RLImage(transitions_chart_buffer, width=5.1*inch, height=2.55*inch))  # 85% of 6x3
    story.append(Spacer(1, 20))

    # Face Coverage Distribution Chart
    story.append(Paragraph("Face Coverage Distribution", styles['Heading2']))
    fig_coverage = go.Figure()
    fig_coverage.add_trace(go.Bar(
        y=list(results['face_coverage_distribution'].keys()),
        x=list(results['face_coverage_distribution'].values()),
        orientation='h',
        text=[f'{v} frames' for v in results['face_coverage_distribution'].values()],
        textposition='auto',
        textfont=dict(size=14),
        marker=dict(
            color=['#1e88e5', '#42a5f5', '#90caf9']
        )
    ))
    fig_coverage.update_layout(
        title='Face Coverage Distribution',
        xaxis_title='Number of Frames',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    # Save coverage chart at 85% size
    coverage_chart_bytes = pio.to_image(fig_coverage, format='png', width=680, height=340)  # 85% of 800x400
    coverage_chart_buffer = io.BytesIO(coverage_chart_bytes)
    story.append(RLImage(coverage_chart_buffer, width=5.1*inch, height=2.55*inch))  # 85% of 6x3
    story.append(Spacer(1, 20))

    # Positive Engagement Trend Chart
    story.append(Paragraph("Positive Engagement Trend", styles['Heading2']))
    fig_engagement = go.Figure()
    fig_engagement.add_trace(go.Scatter(
        x=results['time_segments'],
        y=results['engagement_scores'],
        mode='lines',
        line=dict(color='#1e88e5', width=2),
        name='Positive Engagement',
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.1)',
        hovertemplate='Time: %{x:.1f} seconds<br>Engagement: %{y:.2f}<extra></extra>'
    ))
    fig_engagement.update_layout(
        title='Positive Engagement Trend Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Positive Engagement Score',
        showlegend=False,
        height=300,
        font=dict(size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    # Save engagement chart at 85% size
    engagement_chart_bytes = pio.to_image(fig_engagement, format='png', width=680, height=340)  # 85% of 800x400
    engagement_chart_buffer = io.BytesIO(engagement_chart_bytes)
    story.append(RLImage(engagement_chart_buffer, width=5.1*inch, height=2.55*inch))  # 85% of 6x3
    story.append(Spacer(1, 20))

    # Emotion Distribution Table
    story.append(Paragraph("Emotion Distribution Details", styles['Heading2']))
    emotion_data = [
        ["Emotion", "Count", "Percentage"],
        *[[emotion, str(count), f"{(count/results['total_frames'])*100:.1f}%"]
          for emotion, count in results['emotion_counts'].items()]
    ]
    emotion_table = Table(emotion_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    emotion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),  # Blue header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),  # Light blue background
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))  # Blue grid lines
    ]))
    story.append(emotion_table)
    story.append(Spacer(1, 20))

    # Top Transitions Table
    story.append(Paragraph("Top Emotion Transitions Details", styles['Heading2']))
    transition_data = [
        ["Transition", "Count"],
        *[[transition, str(count)] for transition, count in transitions]
    ]
    transition_table = Table(transition_data, colWidths=[3*inch, 2*inch])
    transition_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),  # Blue header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e3f2fd')),  # Light blue background
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e88e5'))  # Blue grid lines
    ]))
    story.append(transition_table)
    story.append(Spacer(1, 20))

    # Emotion Screenshots
    if emotion_screenshots:
        story.append(Paragraph("Emotion Screenshots", styles['Heading2']))
        for emotion, screenshot in emotion_screenshots.items():
            if screenshot is not None:
                # Convert screenshot to PIL Image
                img = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
                # Save to temporary buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                # Add to PDF
                story.append(Paragraph(f"{emotion} Emotion", styles['Heading3']))
                story.append(RLImage(img_buffer, width=3.4*inch, height=2.55*inch))  # 85% of 4x3
                story.append(Spacer(1, 20))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def process_video(video_path):
    """Process a video file for emotion analysis."""
    analyzer = SalesCallAnalyzer()
    st.session_state.analyzer = analyzer

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = width * height
    duration = total_frames / fps

    # Save output in videos directory
    output_filename = f"output_{os.path.basename(video_path)}"
    output_path = os.path.join(VIDEOS_DIR, output_filename)

    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or 'mp4v' if 'avc1' doesn't work
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        st.error(f"Error: Could not create output video file {output_path}")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize tracking variables
    processed_frames = 0
    emotion_counts = {
        'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0,
        'Sad': 0, 'Surprise': 0, 'Neutral': 0
    }
    face_detection_count = 0
    face_coverage_distribution = {
        'Low (<25%)': 0,
        'Medium (25-75%)': 0,
        'High (>75%)': 0
    }

    # New tracking variables for additional analytics
    emotion_sequence = []  # Track emotion sequence
    engagement_scores = []  # Track engagement scores
    time_segments = []  # Track time segments
    emotion_transitions = {}  # Track emotion transitions
    last_emotion = None

    # Store first occurrence of each emotion for screenshots
    emotion_screenshots = {
        'Angry': None, 'Disgust': None, 'Fear': None, 'Happy': None,
        'Sad': None, 'Surprise': None, 'Neutral': None
    }

    # Engagement score weights (can be adjusted)
    engagement_weights = {
        'Happy': 1.0,
        'Surprise': 0.8,
        'Neutral': 0.5,
        'Sad': 0.3,
        'Fear': 0.2,
        'Angry': 0.1,
        'Disgust': 0.1
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        progress_percentage = (processed_frames / total_frames) * 100
        current_time = processed_frames / fps

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = analyzer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        # Calculate face coverage for this frame
        frame_face_area = sum(w * h for (x, y, w, h) in faces)
        if len(faces) > 0:
            face_detection_count += 1
            face_coverage = (frame_face_area / frame_area) * 100

            # Update face coverage distribution
            if face_coverage < 25:
                face_coverage_distribution['Low (<25%)'] += 1
            elif face_coverage < 75:
                face_coverage_distribution['Medium (25-75%)'] += 1
            else:
                face_coverage_distribution['High (>75%)'] += 1

        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]

            # Preprocess face
            face_tensor = analyzer.preprocess_face(face_roi)
            face_tensor = face_tensor.to(analyzer.device)

            # Get prediction
            with torch.no_grad():
                outputs = analyzer.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                emotion_idx = predicted.item()
                confidence = probabilities[0][emotion_idx].item()

                # Map emotion index to label
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion = emotion_labels[emotion_idx]

                # Store first occurrence of each emotion
                if emotion_screenshots[emotion] is None and confidence > 0.7:
                    emotion_screenshots[emotion] = frame.copy()

                # Update emotion counts
                emotion_counts[emotion] += 1

                # Track emotion sequence and transitions
                emotion_sequence.append(emotion)
                if last_emotion is not None:
                    transition = f"{last_emotion}→{emotion}"
                    emotion_transitions[transition] = emotion_transitions.get(transition, 0) + 1
                last_emotion = emotion

                # Calculate engagement score
                engagement_score = engagement_weights[emotion] * confidence
                engagement_scores.append(engagement_score)

                # Track time segments
                time_segments.append(current_time)

                # Draw emotion label
                label = f"Face {i+1}: {emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Add FPS and progress percentage
        info_text = f"FPS: {fps:.1f} | Progress: {progress_percentage:.1f}%"
        cv2.putText(frame, info_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)
        processed_frames += 1
        progress = processed_frames / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: {processed_frames}/{total_frames} frames")

    cap.release()
    out.release()

    # Calculate additional metrics
    avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
    engagement_trend = np.polyfit(time_segments, engagement_scores, 1)[0]  # Slope of engagement over time

    # Calculate emotion stability (how often emotions change)
    emotion_changes = sum(1 for i in range(1, len(emotion_sequence)) if emotion_sequence[i] != emotion_sequence[i-1])
    emotion_stability = 1 - (emotion_changes / len(emotion_sequence)) if emotion_sequence else 0

    # Generate PDF report
    pdf_buffer = generate_pdf_report({
        'emotion_counts': emotion_counts,
        'total_frames': total_frames,
        'face_detection_count': face_detection_count,
        'face_coverage_distribution': face_coverage_distribution,
        'output_path': output_path,
        'duration': duration,
        'avg_engagement': avg_engagement,
        'engagement_trend': engagement_trend,
        'emotion_stability': emotion_stability,
        'emotion_transitions': emotion_transitions,
        'engagement_scores': engagement_scores,
        'time_segments': time_segments
    }, video_path, emotion_screenshots)

    return {
        'emotion_counts': emotion_counts,
        'total_frames': total_frames,
        'face_detection_count': face_detection_count,
        'face_coverage_distribution': face_coverage_distribution,
        'output_path': output_path,
        'duration': duration,
        'avg_engagement': avg_engagement,
        'engagement_trend': engagement_trend,
        'emotion_stability': emotion_stability,
        'emotion_transitions': emotion_transitions,
        'engagement_scores': engagement_scores,
        'time_segments': time_segments,
        'pdf_report': pdf_buffer
    }

if __name__ == '__main__':
    main()
