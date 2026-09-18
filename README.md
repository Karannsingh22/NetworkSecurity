# Phishing URL Detection System

A machine learning-based web application that detects potentially phishing websites by analyzing URL and webpage characteristics.

The system provides a Streamlit interface where users can scan individual URLs, upload multiple URLs for batch analysis, or test the trained model using pre-extracted features.

---

## 1. Problem Statement

Phishing websites are designed to imitate legitimate websites and trick users into providing sensitive information such as usernames, passwords, or financial details.

This project uses machine learning to analyze characteristics of URLs and webpages and classify them as:

* **Likely Legitimate Website**
* **Potential Phishing Website**

The system combines URL-based feature extraction with a trained machine learning model to provide predictions along with a model confidence score.

---

## 2. Key Features

### URL Scanner

Users can enter a URL such as:

```text
https://example.com
```

The application:

1. Validates the URL.
2. Extracts features directly from the URL.
3. Resolves the domain using DNS.
4. Attempts to fetch the webpage when possible.
5. Extracts additional webpage-based features.
6. Applies the trained preprocessing pipeline.
7. Sends the features to the trained model.
8. Displays the prediction and model confidence.

DNS failure does not automatically mean that a URL is phishing. The URL can still be analyzed using the features that are available.

### Bulk URL Scanner

Users can upload a CSV or Excel file containing URLs.

Supported column names include:

```text
url
link
website
domain
webpage
```

Each URL is processed through the same extraction and prediction pipeline used by the single URL scanner.

The results can be downloaded as a CSV file.

### Developer / Model Diagnostics

An additional mode is available for testing the trained model directly.

It allows:

* Manual entry of the original 30 features.
* Uploading a CSV containing the 30 pre-extracted features.
* Viewing the model prediction and confidence.

This mode is mainly intended for development, testing, and demonstration.

---

## 3. Dataset

The project uses the `phisingData.csv` dataset.

* **Rows:** 11,055
* **Input features:** 30
* **Target column:** `Result`

The original dataset contains numerically encoded features with values such as `-1`, `0`, and `1`.

The target is converted during training to:

```text
1  → Phishing
0  → Legitimate
```

The expected feature names and ordering are defined in:

```text
data_schema/schema.yaml
```

and enforced by the training pipeline.

---

## 4. Machine Learning Pipeline

The project follows an end-to-end machine learning pipeline:

```text
Dataset
   ↓
Data Ingestion
   ↓
Data Validation
   ↓
Data Drift Check
   ↓
Data Transformation
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Saved Model + Preprocessor
   ↓
Streamlit Prediction
```

### Preprocessing

A `KNNImputer` with `k=3` is used to handle missing feature values.

The preprocessor is fitted during training and saved as:

```text
final_model/preprocessor.pkl
```

The same preprocessor is reused during prediction.

### Models Compared

The training pipeline compares:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost

`GridSearchCV` is used with a small hyperparameter grid for model tuning.

The model with the best performance on the held-out test set is selected as the final model.

The currently saved model is a **Random Forest classifier**.

---

## 5. URL Feature Extraction

The URL scanner extracts features from two main sources.

### Features Extracted From the URL

These can be calculated directly from the URL without fetching the webpage:

```text
having_IP_Address
URL_Length
Shortining_Service
having_At_Symbol
double_slash_redirecting
Prefix_Suffix
having_Sub_Domain
HTTPS_token
port
```

### Features Extracted From DNS

```text
DNSRecord
```

DNS resolution is used to determine whether the domain resolves.

A failed DNS lookup does not automatically classify a URL as phishing.

### Features Extracted From the Webpage

When the webpage can be safely fetched, the scanner extracts features such as:

```text
SSLfinal_State
Favicon
Request_URL
URL_of_Anchor
Links_in_tags
SFH
Submitting_to_email
Redirect
on_mouseover
RightClick
popUpWidnow
Iframe
```

### Features That Cannot Be Reliably Extracted

Some features require external services or historical information that is not available directly from a URL or webpage, such as:

```text
Domain_registeration_length
Abnormal_URL
age_of_domain
web_traffic
Page_Rank
Google_Index
Links_pointing_to_page
Statistical_report
```

These values are left as missing and handled by the trained `KNNImputer`.

The application also shows the feature extraction status so users can see which features were successfully extracted and which were unavailable.

---

## 6. Security Measures

Since the application processes user-provided URLs, the URL scanner includes several safety measures:

* URL format validation.
* DNS resolution before attempting a connection.
* Blocking private, loopback, link-local, reserved, and multicast IP addresses.
* Protection against common SSRF targets.
* Connection and read timeouts.
* Maximum of 5 redirects.
* Maximum response size of 2 MB.
* HTML parsing using BeautifulSoup.
* No JavaScript execution.
* No downloaded files are executed.
* TLS certificate verification remains enabled.
* Network and parsing errors are handled without exposing Python tracebacks.

The scanner is intended for educational and demonstration purposes and should not be considered a production-grade security system.

---

## 7. Model Results

The latest local training run produced the following results:

| Split | F1 Score | Precision | Recall |
| ----- | -------- | --------- | ------ |
| Train | 0.992    | 0.989     | 0.994  |
| Test  | 0.972    | 0.965     | 0.980  |

The final trained model is stored in:

```text
final_model/model.pkl
```

The corresponding preprocessing pipeline is stored in:

```text
final_model/preprocessor.pkl
```

The application uses these saved files for prediction.

---

## 8. Project Structure

```text
Phishing-URL-Detection-System/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── Network_Data/
│   └── phisingData.csv
│
├── data_schema/
│   └── schema.yaml
│
├── valid_data/
│   └── test.csv
│
├── final_model/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── url_feature_extractor.py
│   │
│   ├── pipeline/
│   ├── entity/
│   ├── constant/
│   ├── utils/
│   ├── exception/
│   └── logging/
│
└── Artifacts/
```

`Artifacts/`, `logs/`, and `prediction_output/` are generated during execution and are not required in the repository.

---

## 9. Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **SciPy**
* **Scikit-learn**
* **PyYAML**
* **Streamlit**
* **Requests**
* **BeautifulSoup**
* **Docker**

---

## 10. Running the Project

### Install Dependencies

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available through the local Streamlit server, normally at:

```text
http://localhost:8501
```

The Streamlit application itself runs locally when started this way, but the **URL Scanner can make outbound internet requests** to DNS servers and external websites because it needs to analyze live URLs.

---

## 11. Docker

Docker can also be used to run the application.

Build the image:

```bash
docker build -t phishing-url-detection .
```

Run it:

```bash
docker run -p 8501:8501 phishing-url-detection
```

Then open:

```text
http://localhost:8501
```

Docker is only used to package and run the application. The project does not require AWS, MongoDB Atlas, MLflow, or any other cloud platform.

---

## 12. Training the Model

A trained model is already included in:

```text
final_model/
```

To retrain the model using the dataset:

```bash
python main.py
```

The pipeline will:

* Load the dataset.
* Validate the data.
* Perform the data drift check.
* Transform the data.
* Train and compare multiple classifiers.
* Select the best model.
* Save the trained model and preprocessor.
* Generate training artifacts.

---

## 13. Command-Line Batch Prediction

A CSV containing the expected pre-extracted 30 features can also be processed from the command line:

```bash
python -m networksecurity.pipeline.batch_prediction valid_data/test.csv
```

The prediction results are written to:

```text
prediction_output/output.csv
```

---

## 14. Limitations

* The model was trained on the available phishing URL dataset and may not generalize perfectly to modern websites.
* Some original dataset features cannot be directly obtained from a URL and are therefore handled using the preprocessing pipeline.
* Webpage-based features depend on whether the target website can be reached and parsed.
* Feature extraction uses practical approximations for some of the original dataset features.
* Model confidence represents the classifier's output, not a guarantee that a website is safe or malicious.
* The scanner analyzes the website at the time of scanning and does not provide continuous monitoring.

This project is intended as a **machine learning and cybersecurity demonstration**, not as a replacement for professional phishing detection or browser security systems.
