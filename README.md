# 📸 Sorty Sorty

**Local face recognition and photo organization tool** — everything runs on your machine.  
No cloud services, no uploads — just intelligent photo sorting, OCR, and clustering.

---

## 🚀 Features

- 🎭 **Face Recognition** — Detects and matches faces using dlib (128-D embeddings)
- ⚡ **Embedding Cache** — 94× faster after first run
- 🧠 **Auto-Learning** — Improves accuracy from high-confidence matches
- 🔍 **Duplicate Detection** — SHA-1 hash-based duplicate prevention
- 🧩 **Smart Clustering** — Groups unknown faces using DBSCAN
- 🖼️ **OCR Support** — Detects and extracts text from screenshots
- 🚀 **Parallel Processing** — Multi-process scanning for large photo collections
- 🎨 **Rich CLI** — Beautiful terminal interface with progress bars
- 📊 **Detailed Reports** — JSON summaries with match confidence and stats

---

## 💾 Installation

### Arch Linux
```bash
sudo pacman -S python python-pip python-dlib python-pillow tesseract
```

### Ubuntu / Debian
```bash
sudo apt install python3 python3-pip python3-venv tesseract-ocr
```

### Fedora
```bash
sudo dnf install python3 python3-pip tesseract
```

### Clone & Setup
```bash
git clone https://github.com/BeanieMen/SortySorty.git
cd SortySorty
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

`sorty-sorty.config.json`:
```json
{
  "inputDir": "photos",
  "outputDir": "output",
  "threshold": 0.6,
  "lowConfidenceThreshold": 0.5,
  "maxDimension": 1600,
  "concurrency": 3,
  "deleteDuplicates": false,
  "storeUnknownClusters": true
}
```

---

## 🧰 Commands

### `init`
Initialize a new Sorty Sorty project.

```bash
python cli.py init [--dir ./my-photos]
```

Creates:
```
my-photos/
├── profiles/
├── photos/
├── output/
└── sorty-sorty.config.json
```

---

### `add-person`
Add profile photos for known people.

```bash
python cli.py add-person -n "Jane Doe" -i jane1.jpg -i jane2.jpg
```

**Options:**
- `-n, --name` — Person's name (required)
- `-i, --images` — Image files (specify multiple times, required)
- `-p, --profiles` — Profiles directory

---

### `scan`
Scan and organize photos. Reads input/output from config by default.

```bash
python cli.py scan [OPTIONS]
```

**Options:**
- `-i, --input` — Override input folder from config
- `-o, --output` — Override output folder from config
- `-c, --config` — Path to config file
- `-p, --profiles` — Profiles directory (default: `profiles/`)
- `-t, --threshold` — Match threshold (default: 0.6)
- `--parallel` — Enable parallel processing
- `--concurrency` — Worker count (default: 3)
- `--max-dimension` — Maximum image dimension in pixels (default: 1600)
- `--delete-duplicates` — Delete duplicate photos instead of skipping
- `--rename-timestamp` — Rename photos with timestamp
- `--no-clustering` — Skip DBSCAN grouping
- `-v, --verbose` — Show detailed timing

**Example:**
```bash
# Uses config for input/output
python cli.py scan --parallel -v

# Override input/output
python cli.py scan -i photos -o sorted --parallel
```

---

### `learn`
Auto-learn from high-confidence matches.

```bash
python cli.py learn -r output/report.json [OPTIONS]
```

**Options:**
- `-r, --report` — Path to scan report JSON (required)
- `-s, --min-similarity` — Minimum similarity for learning (default: 0.6)
- `-p, --profiles` — Profiles directory
- `--dry-run` — Show actions without applying changes

---

### `stats`
Show summary statistics from the last scan.

```bash
python cli.py stats -o ./sorted
```

**Options:**
- `-o, --output` — Output directory with report (required)

---

## 📂 Output Structure

```
output/
├── people/
│   ├── john-doe/
│   ├── jane-smith/
│   └── unknown/
├── screenshots/
│   └── chats/
├── others/
└── report.json
```

---

## 🧠 Matching Thresholds

| Similarity | Classification       | Action                     |
|-------------|---------------------|-----------------------------|
| ≥ 0.60      | Confident Match     | Moved to person folder      |
| 0.50–0.60   | Ambiguous           | Flagged for review          |
| < 0.50      | Unknown Face        | Moved to `unknown/` folder  |


---

## 🧱 Architecture

```
src/
├── types/            # Data models
│   ├── config.py
│   ├── face.py
│   ├── scan.py
│   └── learn.py
├── helpers/          # Utility functions
│   ├── fs.py
│   ├── image.py
│   ├── math.py
│   └── string.py
├── services/         # Core logic
│   ├── face_service.py
│   ├── embedding_cache.py
│   ├── ocr_service.py
│   ├── file_service.py
│   ├── cluster_service.py
│   └── scan_service.py
├── commands/         # CLI commands
│   └── learn.py
└── config.py         # Config manager
```

---

## 🧩 Tech Stack

| Area | Library |
|------|----------|
| Face Recognition | `face_recognition` (dlib) |
| OCR | `pytesseract` |
| Clustering | `scikit-learn` (DBSCAN) |
| Image Processing | `Pillow` |
| CLI | `click` + `rich` |
| Multiprocessing | `concurrent.futures` |
| Arrays | `numpy` |

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| **No faces detected** | Ensure profile photos are clear, front-facing |
| **OCR not working** | Install Tesseract: `sudo pacman -S tesseract` |
| **Slow scans** | Enable `--parallel` and check cache usage |
| **All photos in unknown** | Lower threshold to `0.5` or add more profile photos |
| **dlib build errors** | On Arch: `sudo pacman -S python-dlib` |

---

## 💡 Tips

1. Use 2–5 high-quality profile photos per person  
2. Keep lighting and angles varied for better recognition  
3. Delete `profiles/.cache/` to force embedding recompute  
4. Use `--parallel` for 100+ photos  
5. Run `learn` regularly to improve accuracy  

---

## 📝 License

**MIT License** — See `LICENSE` file for details.

---


## 🤝 Contributing

Pull requests and suggestions are welcome!  
Submit an issue or PR at: [github.com/BeanieMen/SortySorty](https://github.com/BeanieMen/SortySorty)
