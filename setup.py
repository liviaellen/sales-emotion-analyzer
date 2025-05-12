from setuptools import setup, find_packages

setup(
    name="engagementdetector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "streamlit",
        "plotly",
        "fpdf2",
        "python-dotenv",
        "scikit-learn",
        "Pillow",
        "tqdm"
    ],
)
