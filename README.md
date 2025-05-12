# Sales Call Emotion Analyzer

A deep learning-based application that analyzes emotions in sales call recordings to help understand customer engagement and sentiment.

## Features

- **Emotion Detection**: Uses a PyTorch-based CNN model trained on FER-2013 dataset
- **Video Analysis**: Processes Zoom recordings to track emotions over time
- **Image Analysis**: Analyzes emotions in individual images
- **Multiple Interfaces**:
  - FastAPI web interface with real-time analysis
  - Streamlit dashboard with interactive visualizations
- **PDF Reports**: Generates detailed reports with emotion analytics

## Project Structure

```
sales-emotion-analyzer/
├── models/
│   ├── emotion_model.py      # PyTorch model architecture
│   ├── train.py             # Training script
│   └── detect_emotions.py   # Emotion detection pipeline
├── utils/
│   ├── video_reader.py      # Video processing utilities
│   ├── face_detector.py     # Face detection utilities
│   └── emotion_predictor.py # Emotion prediction utilities
├── app/
│   ├── fastapi_app.py       # FastAPI web application
│   ├── streamlit_app.py     # Streamlit dashboard
│   └── generate_report.py   # PDF report generation
├── data/                    # Sample data and model weights
├── reports/                 # Generated reports
├── static/                  # Static files for web interface
├── templates/               # HTML templates
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── .python-version         # Python version (3.10.x)
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/liviaellen/engagementdetector.git
   cd engagementdetector
   ```
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained model:

   ```bash
   python models/download_model.py
   ```

## Model Training

The project uses a PyTorch-based CNN model trained on the FER-2013 dataset. You can train the model in two ways:

### 1. Using Default Settings (Recommended)

```bash
python models/train.py
```

This uses the following default settings from `config.py`:

- 50 epochs
- Batch size of 64
- Learning rate of 0.001
- Modern model architecture
- Automatic device selection (MPS for Mac, CUDA for NVIDIA GPUs, CPU as fallback)

### 2. With Custom Settings

```bash
python models/train.py --data_dir data/fer2013 --epochs 50 --batch_size 32
```

### Training Process

The training script will:

1. Load the FER-2013 dataset from `data/fer2013/fer2013.csv`
2. Split data into training (80%) and validation (20%) sets
3. Train the model with progress bars showing loss and accuracy
4. Save the best model and final model in `artifacts/saved_models/`
5. Save training logs in `artifacts/logs/`

### Configuration

You can customize training parameters in `config.py`:

- `NUM_EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size for training
- `LEARNING_RATE`: Initial learning rate
- `MODEL_TYPE`: Model architecture ('modern', 'regularized', or 'original')
- `DEVICE_TYPE`: Training device ('auto', 'mps', 'cuda', or 'cpu')

## Usage

### FastAPI Web Interface

Run the FastAPI server:

```bash
python app.py --mode fastapi
```

Then open http://localhost:8000 in your browser.

### Streamlit Dashboard

Run the Streamlit app:

```bash
python app.py --mode streamlit
```

Then open http://localhost:8501 in your browser.

## API Documentation

The FastAPI interface provides the following endpoints:

- `GET /`: Web interface
- `POST /analyze`: Analyze emotions in an image
- `POST /analyze-video`: Analyze emotions in a video

For detailed API documentation, visit http://localhost:8000/docs when running the FastAPI server.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER-2013 dataset for emotion recognition
- PyTorch team for the deep learning framework
- FastAPI and Streamlit teams for the web frameworks

**Student Engagement Detection System in E-Learning Environment using OpenCV and CNN**

The student engagement detection system works by detecting student eye gaze and facial expression using OpenCV technology, FER-2013 dataset and CNN (convolutional neural network) method, receiving input through video file input or real-time webcam feed. The system will report on
the student engagement level "engaged" if the student's eyes are staring at the screen
and student facial expression showing a neutral or positive impression.

Paper : https://bit.ly/thesis-paper

Project Link : https://github.com/liviaellen/engagementdetector

Video Presentation : https://bit.ly/ellenskripsi

The demo could be found here:
The demo video is in Indonesian language.

**How to Install the engagement detector:**

1. Install prerequisites: homebrew, pip, python3, and mkvirtualenv, GNU Parallel
2. Create and access a python virtual environment
3. Install the prequisited python library by typing this command
   ``pip3 install -r requirements.txt``

**How to Run the Engagement Detector**

**Input : Existing Video**

1. If you want to process an existing video, run this command on the root directory
   ``parallel ::: "python eyegaze.py" "python emotion.py"``
   The command will process the input.mov video on the root directory as the input, make sure you rename the video you want to process as input.mov. If you want to change the default input name, change it in the .py code.
2. After the process finished, it will give you 4 output, resultEyegaze.txt and resultEmotion.txt contaning the analyzed result and video files output_eyegaze.mp4 and output_emotion.mp4 containing the anotated video.

**Input : Real Time Webcam**

1. If you want to process a real time video, run this command on the ./cam directory
   ``parallel ::: "python eyegaze_cam.py" "python emotion_cam.py"``
   The program will open your webcam and analyze your engagement.
2. After the process finished, it will give you 4 output, resultEyegaze.txt and resultEmotion.txt contaning the analyzed result and video files output_eyegaze.mp4 and output_emotion.mp4 containing the anotated video.

---

Bahasa - Indonesian Language

SISTEM PENDETEKSI ENGAGEMENT SISWA DALAM LINGKUNGAN E-LEARNING DENGAN TEKNOLOGI OPENCV BERBASIS CNN
Livia Ellen

1. Sebelum menjalankan program, harap install prequisite berupa homebrew, pip, python3 dan mkvirtualenv, GNU Parallel
2. Masuk ke virtual environment python
3. Install library yang dibutuhhkan dengan command
   pip3 install -r requirements.txt
4. Setelah semua library di-install, maka program siap dijalankan

Langkah-langkah menjalankan program

1. Pastikan sudah ada file input.mov pada root directory sebagai input dari program
2. Jalankan program eyegaze.py dan emotion.py secara bersamaan menggunakan command
   parallel ::: "python eyegaze.py" "python emotion.py"
3. Setelah progream dijalankan, program akan memberikan output berupa file teks resultEyegaze.txt dan resultEmotion.txt berisi nilai hasil analisa python script serta file video output_eyegaze.mp4 dan output_emotion.mp4 berisi video yang telah dianotasi oleh sistem pendeteksi engagement siswa.

Notes: Jika anda menjalankan program untuk kedua kalinya, pastikan anda telah memindahkan file text dan video hasil output sebelumnya, jika tidak, program akan melakukan rewrite pada data tersebut.
