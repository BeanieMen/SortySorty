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
    url="https://github.com/yourusername/sorty-sorty",
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
    entry_points={
        "console_scripts": [
            "sorty-sorty=cli:main",
        ],
    },
)
