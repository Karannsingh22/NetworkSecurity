# Network Security Threat Detection

Detect potentially malicious or phishing URLs using Machine Learning.

A fully local, end-to-end machine learning project that detects **phishing
websites**. Everything runs on a normal laptop — there is no AWS, no
MongoDB Atlas, no external experiment tracker, and no cloud account of
any kind required. Docker is supported as an optional, purely local
containerized way to run the same app.

---

## 1. Problem Statement

Phishing websites imitate legitimate ones to steal credentials or other
sensitive information. Many characteristics of a URL and the page it
serves (use of an IP address instead of a domain, SSL certificate state,
number of external links, domain age, etc.) tend to differ between
legitimate and phishing sites. This project trains a classifier that,
given those characteristics, predicts whether a site looks **Likely
Legitimate** or **Potential Phishing**.

The Streamlit app has three modes:

- **URL Scanner** (for normal users) — type a URL like
  `https://example.com`, click **Scan URL**, and the app extracts what it
  genuinely can from the URL and the live webpage, then runs it through
  the trained model. DNS resolution is purely informational here — a
  domain that fails to resolve (e.g. a synthetic/example phishing-style
  hostname) still gets classified from its URL-text features; it is never
  treated as evidence of malice by itself.
- **Bulk URL Scanner** (for normal users) — upload a CSV or Excel file
  containing a column of URLs (`url`, `link`, `website`, `domain`, or
  `webpage`, case-insensitive). Every URL goes through the exact same
  extraction pipeline as the single-URL scanner, and results can be
  downloaded as CSV.
- **Developer / Model Diagnostics** (optional, for testing/demoing the
  model directly) — the original form where you set each of the 30
  pre-extracted dataset features by hand, plus a raw 30-feature CSV
  upload. This bypasses URL feature extraction entirely and is not the
  primary way to use the app.

## 2. Dataset

- File: `Network_Data/phisingData.csv`
- 11,055 rows, 30 input features + 1 target column (`Result`)
- All features are already numerically encoded as `-1`, `0`, or `1`
  (roughly: legitimate / suspicious / phishing-like indicator) — no
  missing values in the raw file.
- Target `Result`: `1` = phishing, `-1` = legitimate (mapped to `1`/`0`
  during training).

The exact schema (all 30 column names, in the exact order the model
expects them) is enforced by `data_schema/schema.yaml` and mirrored in
`networksecurity.constant.training_pipeline.EXPECTED_FEATURE_COLUMNS`.

## 3. ML Approach

**Training pipeline:** local CSV → data ingestion → data validation
(schema check + KS-test data-drift report) → data transformation (KNN
imputation) → model training/selection → saved model.

**Inference (both UI modes):** feature vector → the *same*
`final_model/preprocessor.pkl` → the *same* `final_model/model.pkl` →
prediction (+ confidence, if the model supports `predict_proba`).

- **Preprocessing:** a `KNNImputer` (`k=3`) fitted on the training split
  only, then reused unchanged at inference time for every prediction
  path (manual form, batch CSV, and the URL scanner).
- **Models compared:** Logistic Regression, Decision Tree, Random Forest,
  Gradient Boosting, AdaBoost — each tuned with a small `GridSearchCV`
  (`cv=3`) over a handful of hyperparameters.
- **Model selection metric:** accuracy on a held-out 20% test split.
- **Final reported metrics** (from the most recent local training run —
  see `Artifacts/<timestamp>/model_trainer/model_report.yaml` for the
  full candidate comparison after you train):

  | Split | F1 | Precision | Recall |
  |---|---|---|---|
  | Train | 0.992 | 0.989 | 0.994 |
  | Test  | 0.972 | 0.965 | 0.980 |

  Best model on this run: **Random Forest**, which also supports
  `predict_proba`, so the app can show a confidence score.

## 4. How the URL Scanner Works

```
User enters URL
       │
       ▼
URL validation (scheme/format check only - rejects malformed input)
       │
       ▼
Lexical feature extraction (URL string only - always runs)
       │
       ▼
DNS resolution - informational only, never blocks classification:
  resolved -> fetch the live page  |  unresolved / private-address -> skip fetch
       │
       ▼
Webpage feature extraction (only if the page was fetched)
       │
       ▼
Same preprocessing pipeline (final_model/preprocessor.pkl - KNNImputer)
       │
       ▼
Same trained model (final_model/model.pkl)
       │
       ▼
Prediction + confidence + a plain-English "URL Analysis" breakdown,
including DNS status shown as information only
```

**This is the most important part to understand and to be honest about:**
the model needs exactly 30 features, but only some of them can actually
be derived from a URL/webpage without a paid third-party data source. The
extractor therefore classifies every feature into one of three buckets:

### A. Extracted directly from the URL string (no network call)
`having_IP_Address`, `URL_Length`, `Shortining_Service`,
`having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`,
`having_Sub_Domain`, `HTTPS_token`, `port`

### A2. Extracted from DNS resolution alone (no page fetch needed)
`DNSRecord` — this only needs to know whether the hostname resolves at
all, so it is computed even when the live webpage itself can't be
fetched. DNS resolution failing is *never* treated as a malicious signal
on its own — it's simply one more (mildly informative) input feature,
exactly as it was in the original training dataset.

### B. Extracted by fetching and parsing the live webpage
`SSLfinal_State` (approximate — see limitations), `Favicon`,
`Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`,
`Submitting_to_email`, `Redirect`, `on_mouseover`, `RightClick`,
`popUpWidnow`, `Iframe`

### C. Cannot be reliably obtained without a paid/third-party service
`Domain_registeration_length`, `Abnormal_URL`, `age_of_domain`,
`web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`,
`Statistical_report`

**These 8 features are never guessed or hardcoded.** They are left as
`NaN` and passed through the *same* `KNNImputer` that was fit during
training — which is exactly what that preprocessing step exists to do.
This is a legitimate use of the pipeline's own missing-value handling,
not a fabricated value invented by this code. The app's "Feature-by-
feature extraction status" panel shows exactly which bucket every one of
the 30 features fell into for the URL you scanned, so nothing is hidden.

If the webpage itself can't be fetched — because DNS resolution failed,
the domain resolved to a private/internal address (blocked for basic
SSRF protection), the site is down, it timed out, TLS failed, etc. — the
12 Category B features are left as `NaN` and imputed. The 9 Category A
lexical features and `DNSRecord` remain "real" either way, and the app
always shows a visible, clearly-worded status (not an error) explaining
why some features were unavailable. **A domain that fails to resolve
still gets a full prediction** — it just does so with fewer real
features, exactly like any other case of a webpage that can't be
fetched.

## 5. Project Structure

```
NetworkSecurity/
│
├── app.py                     # Streamlit app: URL Scanner + Advanced Feature Input
├── main.py                    # Runs the full local training pipeline
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── Dockerfile                 # Local Docker image for the Streamlit app (no AWS)
├── .dockerignore
├── docker-compose.yml         # Optional: `docker compose up --build`
├── .streamlit/
│   └── config.toml            # Headless server settings used by both run modes
│
├── Network_Data/
│   └── phisingData.csv        # Raw dataset
│
├── data_schema/
│   └── schema.yaml            # Expected columns / dtypes / order
│
├── valid_data/
│   └── test.csv                # Small sample file for trying batch prediction
│
├── final_model/
│   ├── model.pkl               # Best trained classifier (used by app.py)
│   └── preprocessor.pkl        # Fitted KNNImputer pipeline (used by app.py)
│
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── url_feature_extractor.py   # URL/webpage feature extraction (new)
│   ├── pipeline/                # training_pipeline.py, batch_prediction.py
│   ├── entity/                  # config/artifact dataclasses
│   ├── constant/                # all the paths/params/expected columns in one place
│   ├── utils/                   # I/O helpers, metrics, the NetworkModel wrapper
│   ├── exception/                # custom exception with file/line info
│   └── logging/                  # simple file logger
│
└── Artifacts/                  # created by `python main.py` (timestamped
                                 # run outputs: ingested data, drift report,
                                 # transformed arrays, trained model, metrics)
```

`Artifacts/`, `logs/`, and `prediction_output/` are regenerated every time
you run the pipeline / app, so they're excluded from the ZIP and from
version control (`.gitignore`).

## 6. Installation

Requires Python 3.10+ **or** Docker — pick whichever run option you want
from Section 7 (you don't need both).

```bash
# from the project root
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
pip install -e .                # installs the `networksecurity` package locally
```

## 7. How to Run

There are two ways to run this project locally. Both use the same code
and the same `final_model/` artifacts; neither requires AWS, MongoDB
Atlas, or any other cloud account.

### Option 1 — Run normally with Python

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints — usually `http://localhost:8501`.

### Option 2 — Run with Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/)
running on Windows (or Docker on macOS/Linux).

Build the image:

```bash
docker build -t network-security .
```

Run the container:

```bash
docker run -p 8501:8501 network-security
```

Then open `http://localhost:8501` in your browser.

Optionally, with [Docker Compose](https://docs.docker.com/compose/):

```bash
docker compose up --build
```

**What the image contains and does:**
- Base image: `python:3.10-slim` (matches the Python 3.10+ requirement above).
- Installs only what's in `requirements.txt` (including `requests` and
  `beautifulsoup4` for the URL scanner) — no AWS CLI, no cloud SDKs.
- Copies the whole project into the image, **including the already-trained
  `final_model/model.pkl` and `final_model/preprocessor.pkl`**, so the
  container can serve predictions immediately — it does not need to train
  on startup.
- Starts Streamlit on `0.0.0.0:8501` (`ENTRYPOINT` in the `Dockerfile`),
  matching the `-p 8501:8501` port mapping above.
- Includes a container `HEALTHCHECK` that curls Streamlit's own
  `/_stcore/health` endpoint.
- `.dockerignore` keeps the image lean while still including the dataset,
  schema, and trained model the app needs.
- Reads no AWS environment variables and makes no calls to S3, EC2, or ECR.
- The container **does** need outbound internet access at *runtime* if you
  want to use the URL Scanner mode against real external sites (to fetch
  the page being analyzed) — this is normal outbound HTTP(S) traffic
  initiated by the app itself, not a cloud-provider dependency. Advanced
  Feature Input mode works with no internet access at all.

To retrain inside the container instead of using the bundled model, override
the entrypoint:

```bash
docker run --rm -v "$(pwd)/final_model:/app/final_model" network-security python main.py
```

### Train the model (optional — a trained model is already included)

```bash
python main.py
```

This reads `Network_Data/phisingData.csv`, validates it, fits the
preprocessing pipeline, trains/compares the five models above, and writes:

- `final_model/model.pkl` and `final_model/preprocessor.pkl` (used by the app)
- `Artifacts/<timestamp>/...` (full run artifacts + `model_report.yaml`)

### Batch-predict a CSV from the command line

```bash
python -m networksecurity.pipeline.batch_prediction valid_data/test.csv
```

Writes `prediction_output/output.csv`.

## 8. Example Usage

**URL Scanner:** enter `https://example.com`, click **🔍 Scan URL**. The
app shows:

```
Prediction:
✅ Likely Legitimate Website
Model confidence: 92.3%
```

or

```
Prediction:
⚠️ Potential Phishing Website
Model confidence: 87.4%
```

followed by a **URL Analysis** panel (domain, URL length, HTTPS/IP/`@`
checks, SSL validity, iframe/pop-up/right-click-disable detection, etc.)
and an expandable **feature-by-feature extraction status** table so you
can see exactly which of the 30 features were really extracted vs. left
for the imputer.

**Bulk URL Scanner:** upload a CSV/Excel file with a `url`/`link`/
`website`/`domain`/`webpage` column (any other columns are ignored).
Every URL is scanned with the exact same pipeline as above; results
(including DNS status and any per-row scan errors) can be downloaded as
CSV.

**Developer / Model Diagnostics → Manual feature entry:** set each
dropdown by hand and click **Predict** — useful for demonstrating the
raw model behavior.

**Developer / Model Diagnostics → Raw 30-feature CSV upload:** upload a
CSV with the 30 pre-extracted feature columns (see `valid_data/test.csv`)
and download the scored results. This bypasses URL feature extraction
entirely — use the Bulk URL Scanner instead if you only have URLs.

## 9. Security / Robustness of the URL Scanner

Since this project fetches user-supplied URLs, `url_feature_extractor.py`
takes several defensive measures:

- **Input validation** — requires `http://`/`https://`, rejects malformed
  URLs, with friendly error messages (no stack traces shown to the user).
- **Basic SSRF protection** — before connecting, the hostname is resolved
  and the IP is checked against private/loopback/link-local/reserved/
  multicast ranges (blocks e.g. `http://127.0.0.1/`, `http://10.0.0.5/`,
  cloud metadata endpoints like `169.254.169.254`, etc.).
- **Timeouts** — 5s connect / 8s read timeout on every request.
- **Capped redirects** — a hard limit of 5 redirects (`TooManyRedirects`
  is caught and handled).
- **Capped download size** — the response body is streamed and cut off at
  2 MB to avoid downloading huge files.
- **No code execution** — the fetched HTML is only ever parsed as text
  with BeautifulSoup's `html.parser`; no JavaScript is executed and no
  downloaded file is ever run.
- **TLS verification stays on** — a certificate error is treated as a
  real, informative signal (`SSLfinal_State` = suspicious) rather than
  silently disabling verification to force a fetch through.
- **Graceful failure everywhere** — DNS failures, connection errors, SSL
  errors, and timeouts are all caught and reported as a friendly message;
  nothing crashes the Streamlit app or leaks a Python traceback to the user.

## 10. Technologies Used

- **Python**, **pandas**, **NumPy**, **SciPy** (KS-test for drift detection)
- **scikit-learn** — `KNNImputer`, `LogisticRegression`, `DecisionTreeClassifier`,
  `RandomForestClassifier`, `GradientBoostingClassifier`, `AdaBoostClassifier`,
  `GridSearchCV` (pinned to `1.8.0` in `requirements.txt` so a fresh
  install always matches the version the bundled `final_model/*.pkl`
  files were trained with)
- **PyYAML** — schema and run-report files
- **Streamlit** — the UI (URL Scanner + Advanced Feature Input)
- **requests** + **beautifulsoup4** — fetching and parsing the live
  webpage for the URL scanner
- **Docker** — optional local containerized run (`docker build` / `docker run`)

## 11. What Was Fixed / Changed From the Original Version

This project started as an AWS/MongoDB/MLflow-based template. Over
several rounds of cleanup:

- Removed all AWS code (S3 sync, ECR/EC2 GitHub Actions deployment) and
  all AWS environment variables. **Docker was kept** — the `Dockerfile`
  was rewritten to drop the AWS CLI install and just run
  `streamlit run app.py` locally on port `8501`.
- Removed the MongoDB Atlas data-ingestion path in favor of reading the
  local CSV directly, and removed MLflow/DagsHub tracking (which had
  contained a hardcoded tracking password) in favor of a local YAML report.
- Fixed several real bugs: a class-instead-of-instance pickling bug, use
  of `r2_score` (a regression metric) to rank classifiers, a schema-
  validation bug that always failed, a missing `return` that discarded
  drift-check status, and a `DataFrame.drop(..., axis=1)` call that
  raises on current pandas.
- Replaced manual "enter 30 numbers" UI with a **URL Scanner** that
  extracts real features from the URL/webpage
  (`networksecurity/components/url_feature_extractor.py`), while keeping
  the original manual-entry form as an **Advanced Feature Input** mode.
- Added `NetworkModel._align_columns()` so every prediction path (manual
  form, batch CSV with any column order, and the URL scanner) is
  reindexed to exactly the column names/order the fitted preprocessor
  expects — scikit-learn raises if these don't match exactly, so this is
  a correctness fix, not just a convenience.
- Added `predict_proba` support end-to-end so the UI can show a
  **model confidence** score (explicitly labeled as the model's own
  certainty, not a guaranteed probability of maliciousness).
- Pinned `scikit-learn==1.8.0` in `requirements.txt` after discovering
  that an unpinned install could pull a newer scikit-learn than the one
  used to train `final_model/*.pkl`, which triggers
  `InconsistentVersionWarning` and risks breaking changes.

## 12. Limitations

**Model / dataset:**
- Model selection uses accuracy on a single train/test split; a
  cross-validated comparison (and metrics like ROC-AUC) would be more robust.
- No automated unit tests; CI does an install → import → train → predict
  smoke test instead.

**URL scanner specifically — please read this before trusting a result:**
- **8 of the 30 features (Category C above) can never be extracted** by
  this project without a paid/third-party data source (traffic rank,
  PageRank, WHOIS-based domain age/registration length, backlink counts,
  phishing-blacklist membership). They are imputed, not measured — a
  prediction is still produced, but it is working with less information
  than the model saw during training on the original dataset.
- Several lexical thresholds (e.g. the URL-length bins, the "% external
  resources" cutoffs) are reasonable approximations of the published
  dataset methodology, not a byte-for-byte reproduction of the original
  authors' exact feature-engineering code — the original extraction logic
  was never published alongside the dataset.
- Subdomain counting uses a simple heuristic (dot-counting after
  stripping `www.`), not a public-suffix list, so multi-part TLDs like
  `.co.uk` can be slightly miscounted.
- `SSLfinal_State` is approximated as "did HTTPS + certificate
  verification succeed", not the original feature's full certificate-age/
  issuer-trust logic.
- Only the first `<form>` tag on the page is inspected for `SFH`/
  submission behavior; pages with multiple relevant forms only have the
  first one considered.
- The scanner reflects the page **at the moment you scan it** — it is not
  a historical or continuously monitored signal.
- If the target site blocks automated requests, is down, or the
  connection fails for any reason, Category B features are also imputed
  and the app shows a visible warning that the result is based on less
  information than usual.
- This is a research/demo tool, not a production security product — never
  treat its output as a definitive verdict on a URL's safety.
