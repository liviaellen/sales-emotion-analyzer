import os
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import base64

def generate_pdf_report(df: pd.DataFrame, summary: dict, video_path: str) -> str:
    """
    Generate a PDF report with emotion analysis results.

    Args:
        df: DataFrame containing emotion analysis results
        summary: Dictionary containing emotion statistics
        video_path: Path to the analyzed video file

    Returns:
        Path to the generated PDF report
    """
    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    # Set up fonts
    pdf.set_font("Arial", "B", 16)

    # Title
    pdf.cell(0, 10, "Sales Call Emotion Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Add timestamp
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 10, f"Video file: {os.path.basename(video_path)}", ln=True)
    pdf.ln(10)

    # Summary statistics
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Total Frames Analyzed: {summary['total_frames']}", ln=True)
    pdf.ln(5)

    # Emotion distribution
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Emotion Distribution", ln=True)
    pdf.set_font("Arial", "", 12)

    for emotion, percentage in summary['emotion_percentages'].items():
        pdf.cell(0, 10, f"{emotion}: {percentage:.1f}%", ln=True)

    pdf.ln(10)

    # Create and save charts
    # Emotion timeline
    fig = px.line(df, x='timestamp', y='emotion',
                  title='Emotion Changes Over Time',
                  labels={'timestamp': 'Time (seconds)', 'emotion': 'Emotion'})

    # Save chart as image
    img_bytes = fig.to_image(format="png")
    img_path = "reports/timeline.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # Add chart to PDF
    pdf.image(img_path, x=10, y=pdf.get_y(), w=190)
    pdf.ln(150)  # Add space after chart

    # Emotion distribution bar chart
    emotion_df = pd.DataFrame({
        'Emotion': list(summary['emotion_percentages'].keys()),
        'Percentage': list(summary['emotion_percentages'].values())
    })
    fig = px.bar(emotion_df, x='Emotion', y='Percentage',
                 title='Emotion Distribution',
                 labels={'Percentage': 'Percentage (%)'})

    # Save chart as image
    img_bytes = fig.to_image(format="png")
    img_path = "reports/distribution.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    # Add chart to PDF
    pdf.image(img_path, x=10, y=pdf.get_y(), w=190)

    # Save PDF
    report_path = "reports/emotion_analysis_report.pdf"
    pdf.output(report_path)

    # Clean up temporary image files
    os.remove("reports/timeline.png")
    os.remove("reports/distribution.png")

    return report_path
