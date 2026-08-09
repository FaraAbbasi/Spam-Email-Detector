# Spam Email Detector — GRU-Powered Classifier

Postmark is a deep learning email spam classifier. A GRU (Gated Recurrent Unit)
neural network, trained on labeled spam/ham email data, is served through a
FastAPI backend and paired with a clean, interactive web frontend for pasting
or uploading emails and getting instant predictions.

---

## Features

- **GRU deep learning model** for spam/ham classification, built with TensorFlow/Keras
- **FastAPI backend** with a simple `/predict` endpoint and CORS enabled for browser access
- **Interactive frontend** (single HTML file, no build step) with:
  - Paste-to-analyze text input
  - Drag-and-drop `.txt` / `.eml` file upload
  - Confidence score display
  - Scan history (saved locally in your browser)
  - Configurable API endpoint and settings panel
  - "About Model" panel explaining the preprocessing pipeline
- **Dockerized** for consistent, portable deployment

---

## Project structure

```

.
├── app/
    ├── main.py                   # FastAPI backend — loads model + tokenizer, exposes /predict
├── templates/
    ├── index.html                # Frontend UI (open directly in a browser) 
└── models/
    ├── best_spam_model.keras     # Trained GRU model
    └── tokenizer.pickle          # Fitted Keras tokenizer used at training time             
├── requirements.txt              # Pinned, conflict-checked Python dependencies
└── experimentss/
    ├── train_model.ipynb         # Model training code

```

---

## Getting started (local)

### 1. Clone the repository

```bash
git clone https://github.com/FaraAbbasi/Spam-Email-Detector.git
cd Spam_Email_Detection
```

### 2. Install dependencies

Using `uv` (recommended — much faster than pip, especially for TensorFlow):

```bash
pip install uv
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your model files

Place your trained files here:

```

models/best_spam_model.keras
models/tokenizer.pickle
```

### 4. Run the app

```bash
uvicorn app:app --reload
```

Open **`http://127.0.0.1:8000`** in your browser — `app.py` serves the UI
directly at the `/` route, so the frontend loads automatically along with the
API. There's no separate file to open.

Interactive API docs (for the `/predict` endpoint itself): `http://127.0.0.1:8000/docs`


---

## API reference

### `GET /`

Health check.

```json
{ "status": "Ok", "message": "Spam/Ham classifier API is running" }
```

### `POST /predict`

**Request body:**

```json
{ "text": "Subject: ...\nEmail body here..." }
```

**Response:**

```json
{
  "Email": "Subject: ... (truncated preview)",
  "Label": "SPAM 🚨",
  "Confidence": 0.9421,
  "SpamProbability": 0.9421
}
```

- `Confidence` — how sure the model is in the label it picked (not always the
  spam probability directly; if the model predicts HAM, this is `1 - spam_prob`)
- `SpamProbability` — the raw, unadjusted probability that the email is spam

---

## How text is preprocessed

Before being tokenized, every email is cleaned using the same steps applied
during training:

1. Lowercased
2. URLs replaced with an `url` token
3. Email addresses replaced with an `email` token
4. Numbers replaced with a `num` token
5. Punctuation stripped
6. Extra whitespace collapsed

Sequences are then padded/truncated to a fixed length of **150 tokens** before
being passed to the model.

---

## License

This project is open-source and available under the MIT License.

---

## Contributing

Contributions are warmly welcomed! This project thrives on community input and collaboration. We encourage you to participate in making it better, whether through bug reports, feature suggestions, or direct code contributions.

---

## Support

For issues and questions:

1. Open an Issue on GitHub

2. Provide detailed information about your problem

----
<div style="text-align: center;">
  Made with ❤️ for the open-source community<br>
  Give a ⭐ if you find this project useful!
</div>
