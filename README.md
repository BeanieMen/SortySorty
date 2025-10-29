# Photo Sorterrm -rf .venv
python -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt# Photo Sorter - Python Edition



# Sorty Sorty

Local face recognition and photo organization tool with intelligent caching and parallel processing.A production-ready CLI tool that automatically organizes photos using local face recognition, OCR for screenshots, and intelligent clustering. No cloud services required - everything runs locally on your machine!



## ✨ Features## ✨ Features



- 🎭 **Face Recognition** - Automatic face detection and matching using dlib (128-D embeddings)- 🎭 **Face Recognition** - Identifies and groups photos by people using 128-D face embeddings

- ⚡ **Intelligent Caching** - Profile embeddings cached for 94x faster subsequent scans- 📸 **Screenshot Detection** - Automatically detects screenshots and chat images using OCR

- 🚀 **Parallel Processing** - Multi-process support for faster scanning (optional)- 🔍 **Duplicate Detection** - SHA-1 hash-based duplicate prevention

- 📊 **Interactive Learning** - Learn from matched photos to improve accuracy- 🎯 **Auto-Learning** - Learns from high-confidence matches to improve accuracy

- 🎨 **Beautiful CLI** - Colored output with progress bars using Rich- 🔄 **Smart Clustering** - Groups similar unknown faces using DBSCAN algorithm

- 🔍 **OCR Support** - Screenshot detection and text extraction- 📊 **Detailed Reports** - JSON reports with match confidence and review suggestions

- 📁 **Smart Organization** - Automatically organize photos by person- 🎨 **Rich CLI** - Beautiful command-line interface with progress bars

- 🔄 **Duplicate Detection** - SHA-1 hash-based duplicate prevention

- 📈 **Detailed Reports** - JSON reports with similarity scores and match details## 🚀 Quick Start

- 🎯 **Smart Clustering** - Groups similar unknown faces using DBSCAN

### Installation

## 🚀 Installation

```bash

### Arch Linux# Clone the repository

cd sorter-py

```bash

# Install system dependencies# Install dependencies

sudo pacman -S python python-pip python-dlib python-pillow tesseractpip install -r requirements.txt



# Clone the repository# On macOS/Linux, you may also need tesseract for OCR:

git clone https://github.com/yourusername/sorty-sorty.git# macOS: brew install tesseract

cd sorty-sorty# Ubuntu: sudo apt-get install tesseract-ocr

# Fedora: sudo dnf install tesseract

# Create virtual environment and install Python packages```

python -m venv .venv

source .venv/bin/activate.fish  # for fish shell### Basic Usage

# OR: source .venv/bin/activate  # for bash

```bash

# Install Python packages (dlib will use system package)# 1. Initialize a new project

pip install -r requirements.txtpython cli.py init --dir ./my-photos

```

# 2. Add profile photos for people you want to recognize

### Other Linux Distributionspython cli.py add-person --name "John Doe" --images photo1.jpg photo2.jpg



```bash# 3. Place photos to sort in the photos/ directory

# Install system dependencies

sudo apt install python3 python3-pip python3-venv tesseract-ocr  # Debian/Ubuntu# 4. Run the scanner

# ORpython cli.py scan --input ./my-photos/photos --output ./my-photos/output

sudo dnf install python3 python3-pip tesseract  # Fedora

# 5. Learn from high-confidence matches

# Clone and setuppython cli.py learn --report ./my-photos/output/report.json --min-similarity 0.7

git clone https://github.com/yourusername/sorty-sorty.git

cd sorty-sorty# 6. View statistics

python3 -m venv .venvpython cli.py stats --output ./my-photos/output

source .venv/bin/activate```

pip install -r requirements.txt

```## 📋 Commands



## 📋 Quick Start### `init` - Initialize Project



### 1. Initialize ProjectCreates directory structure and config file:



```bash```bash

python cli.py initpython cli.py init --dir ./my-project

``````



This creates:Creates:

- `profiles/` - Directory for profile photos- `profiles/` - Store reference photos for known people

- `photos/` - Directory for photos to sort- `photos/` - Input directory for photos to sort

- `output/` - Directory for organized results- `output/` - Organized output directory

- `sorty-sorty.config.json` - Configuration file- `sorty-sorty.config.json` - Configuration file



### 2. Add Profile Photos### `scan` - Scan and Organize Photos



```bashMain command to process and organize photos:

# Add a person's profile photos

python cli.py add-person "John Doe" john1.jpg john2.jpg john3.jpg```bash

python cli.py scan \

# Or manually create directories  --input ./photos \

mkdir -p profiles/john-doe  --output ./sorted \

cp john*.jpg profiles/john-doe/  --config ./sorty-sorty.config.json \

```  --threshold 0.6

```

### 3. Scan and Organize Photos

Options:

```bash- `--input, -i` - Input directory with photos (required)

# Basic scan (sequential, default)- `--output, -o` - Output directory for sorted photos (required)

python cli.py scan -i photos -o output- `--config, -c` - Path to config file (optional)

- `--threshold, -t` - Face match threshold 0.0-1.0 (optional)

# Parallel processing (faster for many photos)- `--profiles, -p` - Profiles directory path (optional)

python cli.py scan -i photos -o output --parallel

### `learn` - Auto-Learn from Results

# Verbose mode with timing details

python cli.py scan -i photos -o output -vAdds high-confidence matches to profile directories:



# Combine options for best performance```bash

python cli.py scan -i photos -o output --parallel -vpython cli.py learn \

```  --report ./output/report.json \

  --min-similarity 0.7 \

### 4. Learn from Results  --dry-run

```

After scanning, you can add matched photos to profiles:

Options:

```bash- `--report, -r` - Path to scan report JSON (required)

# Interactive learning (prompts after scan)- `--min-similarity, -s` - Minimum similarity threshold (default: 0.6)

python cli.py scan -i photos -o output- `--profiles, -p` - Profiles directory (optional)

# Answer 'y' when prompted to learn- `--dry-run` - Preview without copying files



# Or manually from report### `add-person` - Add Profile

python cli.py learn output/report.json --min-similarity 0.6

```Manually add a person's profile:



## 💻 Commands```bash

python cli.py add-person \

### `init` - Initialize Project  --name "Jane Smith" \

  --images photo1.jpg photo2.jpg photo3.jpg \

```bash  --profiles ./profiles

python cli.py init [BASE_DIR]```

```

Options:

Creates directory structure and config file.- `--name, -n` - Person's name (required)

- `--images, -i` - Image files (can specify multiple, required)

### `scan` - Scan and Organize Photos- `--profiles, -p` - Profiles directory (optional)



```bash### `stats` - View Statistics

python cli.py scan -i INPUT_DIR -o OUTPUT_DIR [OPTIONS]

```Display scan report statistics:



**Options:**```bash

- `-i, --input` - Input directory with photos to scan (required)python cli.py stats --output ./sorted

- `-o, --output` - Output directory for organized photos (required)```

- `-p, --profiles` - Profiles directory (default: `profiles/`)

- `-t, --threshold` - Face matching threshold 0.0-1.0 (default: 0.6)## ⚙️ Configuration

- `-v, --verbose` - Show detailed timing information

- `--parallel` - Enable parallel processing with multiple workersConfiguration file (`sorty-sorty.config.json`):

- `-c, --config` - Path to config file

```json

### `learn` - Auto-Learn from Results{

  "version": 1,

```bash  "profilesDir": "/path/to/profiles",

python cli.py learn REPORT_PATH [OPTIONS]  "threshold": 0.6,

```  "lowConfidenceThreshold": 0.45,

  "maxDimension": 1600,

**Options:**  "concurrency": 3,

- `--min-similarity` - Minimum similarity for learning (default: 0.6)  "outputStructure": {

- `--profiles, -p` - Profiles directory    "people": "people",

- `--dry-run` - Show what would be learned without making changes    "unknownPeople": "people/unknown",

    "screenshots": "screenshots",

### `add-person` - Add Profile    "chats": "screenshots/chats",

    "others": "others"

```bash  },

python cli.py add-person NAME IMAGE1 [IMAGE2 ...] [OPTIONS]  "ocr": {

```    "textLengthThreshold": 80,

    "chatKeywords": ["whatsapp", "telegram", "messenger"],

**Options:**    "enableOnFaces": false

- `--profiles, -p` - Profiles directory  },

  "renameWithTimestamp": false,

### `stats` - Show Statistics  "deleteDuplicates": false,

  "fastMode": false,

```bash  "storeUnknownClusters": true

python cli.py stats -p PROFILES_DIR}

``````



## ⚙️ Configuration## 🎯 How It Works



Edit `sorty-sorty.config.json`:### Face Recognition Pipeline



```json1. **Profile Loading**: Loads reference photos from `profiles/` directory

{2. **Face Detection**: Uses dlib's face detector to find faces

  "version": 1,3. **Embedding Extraction**: Generates 128-D face descriptors using FaceNet

  "profilesDir": "profiles",4. **Matching**: Compares embeddings using cosine similarity

  "threshold": 0.6,5. **Classification**: 

  "lowConfidenceThreshold": 0.5,   - ≥ 0.6 similarity → Matched (configurable)

  "maxDimension": 1600,   - 0.45-0.6 → Ambiguous (review suggested)

  "concurrency": 3,   - < 0.45 → Unknown

  "outputStructure": {

    "people": "people",### Similarity Thresholds

    "unknownPeople": "people/unknown",

    "screenshots": "screenshots",- **0.9-1.0** - Nearly identical (excellent for auto-learning)

    "chats": "screenshots/chats",- **0.7-0.9** - High confidence matches

    "others": "others"- **0.6-0.7** - Moderate confidence (default threshold)

  },- **0.45-0.6** - Low confidence (ambiguous, needs review)

  "ocr": {- **< 0.45** - Unknown face

    "textLengthThreshold": 80,

    "chatKeywords": ["whatsapp", "telegram", "messenger"],### Output Structure

    "enableOnFaces": false

  },```

  "renameWithTimestamp": false,output/

  "deleteDuplicates": false,├── people/

  "storeUnknownClusters": true,│   ├── john-doe/

  "verbose": false│   │   ├── photo1.jpg

}│   │   └── photo2.jpg

```│   ├── jane-smith/

│   └── unknown/

**Key Settings:**│       └── photo3.jpg

- `threshold` - Minimum similarity for confident match (default: 0.6)├── screenshots/

- `lowConfidenceThreshold` - Minimum for ambiguous match (default: 0.5)│   ├── screenshot1.png

- `maxDimension` - Downscale images larger than this (default: 1600px)│   └── chats/

- `concurrency` - Number of parallel workers (default: 3)│       └── whatsapp_chat.jpg

- `verbose` - Show detailed timing (default: false)├── others/

│   └── landscape.jpg

## ⚡ Performance└── report.json

```

### Caching System

## 📊 Report Schema

Profile embeddings are automatically cached in `profiles/.cache/`:

- **First scan**: Computes embeddings (~6-7s for typical profiles)The scan generates a detailed JSON report:

- **Subsequent scans**: Loads from cache (~0.07s) - **94x faster!**

- Cache automatically invalidates when profile photos change (SHA256 hash-based)```json

- Cache is computed once in main process, workers load from disk{

  "processed": 10,

### Processing Modes  "copied": 8,

  "duplicates": 2,

**Sequential Mode (Default)**:  "ambiguous": 1,

- Single process  "unknownFaces": 3,

- Best for small photo sets (<100 photos)  "errors": 0,

- Lower memory usage  "processedFiles": [

- ~4s for 3 photos (~1.3s per image)    {

      "source": "/input/photo1.jpg",

**Parallel Mode (`--parallel`)**:      "destination": "/output/people/john-doe/photo1.jpg",

- Multi-process with 3 workers (configurable)      "action": "copied",

- Best for large photo sets (>100 photos)      "matchedPerson": "john-doe",

- 40% faster than sequential      "similarity": 0.95

- ~2.4s for 3 photos (~0.8s per image)    }

  ],

### Verbose Mode (`-v`)  "clusters": [...],

  "reviewEntries": [...],

Shows detailed timing for each step:  "timestamp": "2025-10-29T10:00:00.000Z"

}

```bash```

python cli.py scan -i photos -o output -v

```## 🏗️ Architecture



Output includes:```

- Profile loading time (computing vs cache)src/

- Per-image breakdown:├── types/           # Type definitions (dataclasses)

  - Load time│   ├── config.py    # Configuration types

  - Resize time (for large images)│   ├── face.py      # Face recognition types

  - Face detection time│   ├── ocr.py       # OCR types

  - Encoding time│   ├── scan.py      # Scan result types

  - Total time per image│   └── learn.py     # Learn database types

- Overall processing time├── helpers/         # Pure utility functions

│   ├── math.py      # Cosine similarity, distance

### Performance Optimizations│   ├── string.py    # Slugify, escape

│   ├── image.py     # Hash, resize, metadata

- **Downscaling**: Images >1600px automatically downscaled│   └── fs.py        # File operations

- **HOG Model**: Fast face detection (vs CNN)├── services/        # Business logic

- **No Upsampling**: `number_of_times_to_upsample=0`│   ├── face_service.py      # Face detection/matching

- **No Jitters**: `num_jitters=0` for speed│   ├── ocr_service.py       # OCR text extraction

- **Multiprocessing**: Bypasses Python GIL for true parallelism│   ├── file_service.py      # File copying/duplicates

│   ├── cluster_service.py   # DBSCAN clustering

## 📊 Output Structure│   └── scan_service.py      # Main orchestration

├── commands/        # CLI command implementations

```│   └── learn.py     # Auto-learning logic

output/└── config.py        # Config management

├── people/```

│   ├── john-doe/

│   │   ├── photo1.jpg## 🔧 Tech Stack

│   │   └── photo2.jpg

│   ├── jane-smith/- **Face Recognition**: `face_recognition` (dlib-based, 128-D embeddings)

│   │   └── photo3.jpg- **OCR**: `pytesseract` (Tesseract wrapper)

│   └── unknown/- **Image Processing**: `Pillow` (PIL)

│       └── photo4.jpg- **Clustering**: `scikit-learn` (DBSCAN algorithm)

├── screenshots/- **CLI**: `click` (command-line interface)

│   ├── screenshot1.png- **UI**: `rich` (beautiful terminal output)

│   └── chats/- **Arrays**: `numpy` (numerical operations)

│       └── whatsapp_chat.jpg

├── others/## 🐛 Troubleshooting

│   └── misc-photo.jpg

└── report.json### No faces detected

```

- Ensure profile photos have clear, front-facing faces

### Report Format- Try photos with different lighting/angles

- Check that face_recognition library is properly installed

`report.json` contains detailed scan information:

### OCR not working

```json

{- Install Tesseract OCR for your system

  "processed": 10,- Verify with: `tesseract --version`

  "copied": 10,

  "duplicates": 0,### Import errors

  "ambiguous": 1,

  "unknownFaces": 2,```bash

  "errors": 0,# Reinstall dependencies

  "processedFiles": [pip install -r requirements.txt --force-reinstall

    {```

      "source": "/path/to/photo.jpg",

      "destination": "/path/to/output/people/john-doe/photo.jpg",### All photos going to "unknown"

      "action": "copied",

      "matchedPerson": "john-doe",- Check `profilesDir` in config points to correct location

      "similarity": 0.95- Ensure profile directories contain valid image files

    }- Try lowering the threshold (e.g., 0.5)

  ],

  "reviewEntries": [],## 📝 License

  "clusters": [],

  "timestamp": "2025-10-29T12:00:00.000000"MIT License - See LICENSE file for details

}

```## 🙏 Credits



## 🎯 Matching ThresholdsBased on the original TypeScript sorty-sorty project, migrated to Python for better stability and ease of use.



- **≥60%** - Confident match (copied to person's folder)## 🤝 Contributing

- **50-60%** - Ambiguous match (flagged for review)

- **<50%** - No match (copied to unknown folder)Contributions welcome! Please feel free to submit a Pull Request.


### Similarity Ranges

- **0.9-1.0** - Nearly identical (excellent for auto-learning)
- **0.7-0.9** - High confidence matches
- **0.6-0.7** - Moderate confidence (default threshold)
- **0.5-0.6** - Low confidence (ambiguous, needs review)
- **<0.5** - Unknown face

## 💡 Tips

1. **Profile Photos**: Use 2-5 clear photos per person from different angles
2. **Quality**: Higher quality profile photos = better matching
3. **Large Images**: Automatically downscaled to 1600px for faster processing
4. **Cache**: Delete `profiles/.cache/` to force recomputation if needed
5. **Parallel Mode**: Enable with `--parallel` for 100+ photos (40% faster)
6. **Learning**: Regularly learn from matched photos to improve accuracy
7. **Verbose**: Use `-v` flag to diagnose slow processing

## 🏗️ Architecture

```
src/
├── types/                    # Type definitions (dataclasses)
│   ├── config.py            # Configuration types
│   ├── face.py              # Face recognition types
│   ├── ocr.py               # OCR types
│   ├── scan.py              # Scan result types
│   └── learn.py             # Learn database types
├── helpers/                 # Pure utility functions
│   ├── math.py              # Cosine similarity, distance
│   ├── string.py            # Slugify, escape
│   ├── image.py             # Hash, resize, metadata
│   └── fs.py                # File operations
├── services/                # Business logic
│   ├── face_service.py      # Face detection/matching
│   ├── embedding_cache.py   # Profile embedding cache
│   ├── ocr_service.py       # OCR text extraction
│   ├── file_service.py      # File copying/duplicates
│   ├── cluster_service.py   # DBSCAN clustering
│   └── scan_service.py      # Main orchestration
└── commands/                # CLI command implementations
    └── learn.py             # Auto-learning logic
```

## 🔧 Tech Stack

- **Face Recognition**: `face_recognition` 1.3.0 (dlib-based, 128-D embeddings)
- **OCR**: `pytesseract` (Tesseract wrapper)
- **Image Processing**: `Pillow` (PIL)
- **Clustering**: `scikit-learn` (DBSCAN algorithm)
- **CLI**: `click` (command-line interface)
- **UI**: `rich` (beautiful terminal output with colors)
- **Multiprocessing**: `ProcessPoolExecutor` (bypasses GIL)
- **Arrays**: `numpy` (numerical operations)

## 🐛 Troubleshooting

### No faces detected

- Ensure profile photos have clear, front-facing faces
- Try photos with different lighting/angles
- Images are automatically downscaled if too large
- Check verbose mode (`-v`) for detection timing

### OCR not working

- Install Tesseract OCR: `sudo pacman -S tesseract` (Arch)
- Verify with: `tesseract --version`

### Slow processing

- Enable parallel mode: `--parallel`
- Use verbose mode to identify bottlenecks: `-v`
- Ensure images aren't excessively large (>1600px)
- Check if cache is being used (0.07s vs 6s loading)

### All photos going to "unknown"

- Check `profilesDir` in config points to correct location
- Ensure profile directories contain valid image files
- Try lowering the threshold (e.g., 0.5)
- Use verbose mode to see face detection results

### dlib installation issues

- On Arch Linux: Use system package `sudo pacman -S python-dlib`
- On other systems: May need build tools (cmake, gcc)

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Credits

Migrated from TypeScript sorty-sorty to Python for better ML library support and improved performance.

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.
