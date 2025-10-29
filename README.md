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
git clone https://github.com/yourusername/sorty-sorty.git
cd sorty-sorty
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

`sorty-sorty.config.json`:
```json
{
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
python cli.py add-person "Jane Doe" jane1.jpg jane2.jpg
```

---

### `scan`
Scan and organize photos.

```bash
python cli.py scan -i photos -o output [OPTIONS]
```

**Options:**
- `-i, --input` — Input folder (required)
- `-o, --output` — Output folder (required)
- `-p, --profiles` — Profiles directory (default: `profiles/`)
- `-t, --threshold` — Match threshold (default: 0.6)
- `--parallel` — Enable parallel processing
- `--concurrency` — Worker count (default: 3)
- `--delete-duplicates` — Remove duplicates
- `--no-clustering` — Skip DBSCAN grouping
- `-v, --verbose` — Show detailed timing

**Example:**
```bash
python cli.py scan -i photos -o sorted --parallel -v
```

---

### `learn`
Auto-learn from high-confidence matches.

```bash
python cli.py learn output/report.json [OPTIONS]
```

**Options:**
- `--min-similarity` — Minimum similarity for learning (default: 0.6)
- `--dry-run` — Show actions without applying changes

---

### `stats`
Show summary statistics from the last scan.

```bash
python cli.py stats --output ./sorted
```

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

## ⚡ Performance

| Optimization | Description |
|---------------|-------------|
| **Embedding Cache** | 94× faster after first run |
| **Parallel Mode** | 40% faster for large sets |
| **Image Downscaling** | Max 1600px width/height |
| **No Cloud** | 100% local processing |

**Example:**
- First run (cache build): ~6s per profile  
- Cached run: ~0.07s load time

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
│   ├── ocr_service.py
│   ├── cluster_service.py
│   └── scan_service.py
└── commands/         # CLI commands
    ├── scan.py
    └── learn.py
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

## 🙏 Credits

Originally based on the TypeScript version of **Sorty Sorty**, now rewritten in **Python** for better performance and native ML support.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Submit an issue or PR at: [github.com/yourusername/sorty-sorty](https://github.com/yourusername/sorty-sorty)
