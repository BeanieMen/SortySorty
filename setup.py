from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sorty-sorty",
    version="1.0.0",
    author="AJ",
    description="Local face recognition and photo organization tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beaniemen/sorty-sorty",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "face-recognition>=1.3.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "yolo": [
            "ultralytics>=8.0.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "timm>=0.9.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sorty-sorty=cli:main",
        ],
    },
)
