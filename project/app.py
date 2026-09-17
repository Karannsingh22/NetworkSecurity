
"""
Phishing URL Detection System - Streamlit application.

Three input modes (sidebar), in order of intended use:
  1. URL Scanner (default) - a normal user types a URL like
     https://example.com, the app extracts everything it legitimately
     can from the URL and the live webpage, and sends it through the
     exact same preprocessing + model used at training time.
  2. Bulk URL Scanner - upload a CSV/Excel file containing a column of
     URLs (any common name: url, link, website, domain, webpage). Each
     URL is run through the SAME extraction pipeline as mode 1, features
     are fed to the model, and results can be downloaded.
  3. Developer / Model Diagnostics (optional, not the primary workflow) -
     the original pre-extracted-dataset-feature form (manual entry) plus
     a raw 30-feature CSV upload, kept for testing/demoing the underlying
     ML model directly, bypassing URL feature extraction entirely.

Nothing here retrains the model - final_model/model.pkl and
final_model/preprocessor.pkl are loaded once (cached) and reused for
every prediction.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from networksecurity.pipeline.batch_prediction import BatchPredictionPipeline
from networksecurity.components.url_feature_extractor import scan_url, URLValidationError
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constant.training_pipeline import (
    FINAL_MODEL_DIR,
    FINAL_MODEL_FILE_NAME,
    FINAL_PREPROCESSOR_FILE_NAME,
    EXPECTED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

st.set_page_config(
    page_title="Phishing URL Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Feature metadata for the Developer manual-entry form
# ---------------------------------------------------------------------------
FEATURES = [
    ("having_IP_Address", "URL uses a raw IP address instead of a domain name", [-1, 1], "Security & Address Bar"),
    ("URL_Length", "Overall length of the URL", [-1, 0, 1], "Security & Address Bar"),
    ("Shortining_Service", "URL was created with a link-shortening service (e.g. bit.ly)", [-1, 1], "Security & Address Bar"),
    ("having_At_Symbol", "URL contains an '@' symbol", [-1, 1], "Security & Address Bar"),
    ("double_slash_redirecting", "URL contains '//' used to redirect the browser", [-1, 1], "Security & Address Bar"),
    ("Prefix_Suffix", "Domain contains a '-' (dash) prefix/suffix", [-1, 1], "Security & Address Bar"),
    ("having_Sub_Domain", "Number of sub-domains in the URL", [-1, 0, 1], "Security & Address Bar"),
    ("SSLfinal_State", "Trustworthiness of the site's SSL certificate", [-1, 0, 1], "Security & Address Bar"),
    ("Domain_registeration_length", "How far in the future the domain registration expires", [-1, 1], "Security & Address Bar"),
    ("Favicon", "Favicon is loaded from an external domain", [-1, 1], "Security & Address Bar"),
    ("port", "Website uses non-standard ports", [-1, 1], "Security & Address Bar"),
    ("HTTPS_token", "Domain part of the URL contains the token 'https'", [-1, 1], "Security & Address Bar"),
    ("Request_URL", "Share of page objects (images, videos, etc.) loaded from another domain", [-1, 1], "Abnormal Behaviour"),
    ("URL_of_Anchor", "Share of anchor '<a>' tags pointing to a different domain", [-1, 0, 1], "Abnormal Behaviour"),
    ("Links_in_tags", "Share of <meta>/<script>/<link> tags pointing to a different domain", [-1, 0, 1], "Abnormal Behaviour"),
    ("SFH", "Server Form Handler - where form data is submitted to", [-1, 0, 1], "Abnormal Behaviour"),
    ("Submitting_to_email", "Form data is submitted directly to an email address", [-1, 1], "Abnormal Behaviour"),
    ("Abnormal_URL", "URL does not match the domain's WHOIS registrant info", [-1, 1], "Abnormal Behaviour"),
    ("Redirect", "Number of times the page redirects the visitor", [0, 1], "Abnormal Behaviour"),
    ("on_mouseover", "Status bar text is changed on mouseover (common phishing trick)", [-1, 1], "Abnormal Behaviour"),
    ("RightClick", "Right-click is disabled on the page", [-1, 1], "Abnormal Behaviour"),
    ("popUpWidnow", "Page shows a pop-up window asking for information", [-1, 1], "Abnormal Behaviour"),
    ("Iframe", "Page uses an invisible <iframe>", [-1, 1], "Abnormal Behaviour"),
    ("age_of_domain", "Age of the domain (older = more trustworthy)", [-1, 1], "Domain & Traffic"),
    ("DNSRecord", "Domain has a DNS record", [-1, 1], "Domain & Traffic"),
    ("web_traffic", "Website's traffic / popularity rank", [-1, 0, 1], "Domain & Traffic"),
    ("Page_Rank", "Google PageRank of the page", [-1, 1], "Domain & Traffic"),
    ("Google_Index", "Page is indexed by Google", [-1, 1], "Domain & Traffic"),
    ("Links_pointing_to_page", "Number of external links pointing to this page", [-1, 0, 1], "Domain & Traffic"),
    ("Statistical_report", "Domain/IP appears in known phishing statistical reports", [-1, 1], "Domain & Traffic"),
]

VALUE_LABELS = {
    -1: "-1  (Legitimate)",
    0: "0  (Suspicious)",
    1: "1  (Phishing indicator)",
}

SECTIONS = [
    "Security & Address Bar",
    "Abnormal Behaviour",
    "Domain & Traffic",
]

LABEL_TEXT = {
    1: "Potential Phishing Website",
    0: "Likely Legitimate Website",
}

URL_COLUMN_CANDIDATES = [
    "url",
    "link",
    "website",
    "domain",
    "webpage",
]


@st.cache_resource(show_spinner="Loading trained model...")
def get_pipeline() -> BatchPredictionPipeline:
    return BatchPredictionPipeline()


# ---------------------------------------------------------------------------
# FIXED MODEL PATH CHECK
# ---------------------------------------------------------------------------
def model_files_exist() -> bool:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "final_model"

    model_path = model_dir / FINAL_MODEL_FILE_NAME
    preprocessor_path = model_dir / FINAL_PREPROCESSOR_FILE_NAME

    return model_path.exists() and preprocessor_path.exists()


def render_prediction_banner(predicted_class: int, confidence):
    if predicted_class == 1:
        st.error(f"### ⚠️ {LABEL_TEXT[1]}")
    else:
        st.success(f"### ✅ {LABEL_TEXT[0]}")

    if confidence is not None:
        st.caption(
            f"**Model confidence:** {confidence * 100:.1f}% "
            "(this reflects the trained model's own certainty on this input - "
            "not a guaranteed probability that the site is malicious)."
        )
    else:
        st.caption(
            "This model does not expose a confidence score for this prediction."
        )


def dns_status_display(dns_status: str) -> str:
    return {
        "resolved": "✅ Resolved",
        "unresolved": "⚠️ Could not resolve",
        "blocked_private": "🔒 Resolves to a private/internal address (fetch skipped for safety)",
    }.get(dns_status, "Not checked")


# ---------------------------------------------------------------------------
# Mode 1: URL Scanner
# ---------------------------------------------------------------------------
def url_scanner_mode():
    st.subheader("Enter Website URL")

    col1, col2 = st.columns([4, 1])

    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com",
            label_visibility="collapsed",
        )

    with col2:
        scan_clicked = st.button(
            "🔍 Scan URL",
            type="primary",
            use_container_width=True,
        )

    if not scan_clicked:
        st.info(
            "Enter a full URL (including `http://` or `https://`) and click **Scan URL**. "
            "The app will extract what it genuinely can from the URL and the live webpage, "
            "then run it through the same model used elsewhere in this app. "
            "**Note:** if the domain cannot be resolved (e.g. it's a fake/example domain), "
            "the URL is still classified using its text-based features - DNS failure is "
            "shown as information only and never automatically means 'malicious'."
        )
        return

    try:
        with st.spinner("Analyzing the URL..."):
            result = scan_url(url_input)

    except URLValidationError as exc:
        st.error(f"⚠️ {exc}")
        return

    except NetworkSecurityException:
        st.error(
            "Something went wrong while scanning that URL. "
            "Please double-check it and try again."
        )
        return

    pipeline = get_pipeline()

    predicted_class, label, confidence = pipeline.predict_single(
        result.to_ordered_feature_dict()
    )

    st.markdown("### Prediction")

    render_prediction_banner(
        predicted_class,
        confidence,
    )

    dns_col, fetch_col = st.columns(2)

    dns_col.metric(
        "DNS status (info only)",
        dns_status_display(result.dns_status),
    )

    fetch_col.metric(
        "Live page fetched",
        "Yes" if result.fetch_succeeded else "No",
    )

    if result.dns_status == "unresolved":
        st.warning(
            "⚠️ This domain could not be resolved. That is treated purely as **missing "
            "information about the live webpage** - it does NOT by itself mean the URL is "
            "malicious. The prediction above is based on the URL's own text features (length, "
            "use of '@', hyphens, subdomains, IP-address usage, etc.); webpage-dependent "
            "features are safely left for the model's imputer to fill in."
        )

    elif result.dns_status == "blocked_private":
        st.warning(
            "🔒 This domain resolves to a private/internal network address, so the app did "
            "not connect to it (basic SSRF protection). The URL is still classified on its "
            "text-based features."
        )

    elif not result.fetch_succeeded:
        st.warning(
            f"⚠️ The domain resolved, but the live webpage could not be fetched "
            f"({result.fetch_error}). The prediction above is based on the URL text and DNS "
            "status only; webpage-dependent features are imputed - treat it with extra caution."
        )

    st.markdown("### URL Analysis")

    a = result.analysis

    info_cols = st.columns(4)

    info_cols[0].metric(
        "Domain",
        result.domain or "-",
    )

    info_cols[1].metric(
        "URL length",
        a.get("url_length", "-"),
    )

    info_cols[2].metric(
        "Subdomains",
        a.get("subdomain_count", "-"),
    )

    info_cols[3].metric(
        "Redirects",
        a.get(
            "redirect_count",
            0 if result.fetch_succeeded else "-",
        ),
    )

    def yes_no(v):
        if v is None:
            return "Unknown"
        return "Yes" if v else "No"

    detail_rows = [
        (
            "HTTPS used",
            yes_no(a.get("https_used")),
        ),
        (
            "IP address in URL",
            yes_no(a.get("ip_address_detected")),
        ),
        (
            "@ symbol detected",
            yes_no(a.get("at_symbol_detected")),
        ),
        (
            "URL shortening service",
            yes_no(a.get("shortening_service_detected")),
        ),
        (
            "Hyphen in domain",
            yes_no(a.get("hyphen_in_domain")),
        ),
        (
            "'https' token inside domain name",
            yes_no(a.get("https_token_in_domain")),
        ),
        (
            "DNS record found",
            yes_no(
                result.dns_status in (
                    "resolved",
                    "blocked_private",
                )
            ),
        ),
    ]

    if result.fetch_succeeded:
        detail_rows += [
            (
                "SSL certificate valid",
                yes_no(a.get("ssl_valid")),
            ),
            (
                "Favicon hosted externally",
                yes_no(a.get("favicon_external")),
            ),
            (
                "Iframe(s) present",
                yes_no(a.get("iframe_count", 0) > 0),
            ),
            (
                "Status-bar/mouseover trick detected",
                yes_no(a.get("status_bar_trick_detected")),
            ),
            (
                "Right-click disabled",
                yes_no(a.get("right_click_disabled")),
            ),
            (
                "Pop-up script detected",
                yes_no(a.get("popup_script_detected")),
            ),
            (
                "Submits to an email address",
                yes_no(a.get("submits_to_email")),
            ),
            (
                "External resources ratio",
                f"{a.get('external_resource_ratio', 0) * 100:.0f}%",
            ),
            (
                "Suspicious anchor-link ratio",
                f"{a.get('suspicious_anchor_ratio', 0) * 100:.0f}%",
            ),
        ]

    detail_df = pd.DataFrame(
        detail_rows,
        columns=["Indicator", "Value"],
    )

    st.dataframe(
        detail_df,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander(
        "Feature-by-feature extraction status (what was really used)"
    ):
        st.caption(
            "Every one of the model's 30 input features falls into one of these buckets. "
            "Features marked **unavailable** are NOT guessed - they are left blank and filled "
            "in by the same KNN-imputer preprocessing object used during training, which is a "
            "legitimate, intentional part of the pipeline's missing-value handling."
        )

        status_rows = [
            {
                "Feature": name,
                "Status": result.feature_status.get(
                    name,
                    "unknown",
                ),
            }
            for name in EXPECTED_FEATURE_COLUMNS
        ]

        st.dataframe(
            pd.DataFrame(status_rows),
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Mode 2: Bulk URL Scanner
# ---------------------------------------------------------------------------
def _find_url_column(columns) -> "str | None":
    lowered = {
        str(c).strip().lower(): c
        for c in columns
    }

    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    return None


def _read_uploaded_table(uploaded_file) -> "pd.DataFrame | None":
    name = uploaded_file.name.lower()

    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)

        return pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return None


def bulk_url_scanner_mode():
    st.subheader("Bulk URL Scanner")

    st.caption(
        "Upload a CSV or Excel file containing a column of URLs (any of these column names "
        "are recognised, case-insensitive: **url, link, website, domain, webpage**). Other "
        "columns in the file are ignored. Each URL is run through the exact same feature "
        "extraction and model used by the single-URL scanner above."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:
        return

    input_df = _read_uploaded_table(uploaded_file)

    if input_df is None or input_df.empty:
        if input_df is not None:
            st.error("The uploaded file has no rows.")
        return

    url_column = _find_url_column(input_df.columns)

    if url_column is None:
        st.error(
            "Couldn't find a URL column in this file. Please include a column named one of: "
            "**url, link, website, domain, webpage** (case-insensitive) containing the "
            "website addresses to scan. Found columns: "
            + ", ".join(map(str, input_df.columns))
        )
        return

    st.success(
        f"Found URL column: **{url_column}** "
        f"({len(input_df)} row(s))."
    )

    max_rows = 500

    if len(input_df) > max_rows:
        st.warning(
            f"Only the first {max_rows} rows will be scanned "
            "to keep this demo responsive."
        )

        input_df = input_df.head(max_rows)

    if not st.button(
        "🔍 Scan All URLs",
        type="primary",
    ):
        return

    pipeline = get_pipeline()

    feature_rows = []
    dns_statuses = []
    scan_errors = []

    progress = st.progress(
        0.0,
        text="Scanning URLs...",
    )

    total = len(input_df)

    for i, raw_url in enumerate(
        input_df[url_column].astype(str).tolist()
    ):
        try:
            result = scan_url(raw_url)

            feature_rows.append(
                result.to_ordered_feature_dict()
            )

            dns_statuses.append(
                dns_status_display(result.dns_status)
            )

            scan_errors.append(None)

        except URLValidationError as exc:
            feature_rows.append(
                {
                    col: float("nan")
                    for col in EXPECTED_FEATURE_COLUMNS
                }
            )

            dns_statuses.append("-")
            scan_errors.append(str(exc))

        except NetworkSecurityException:
            feature_rows.append(
                {
                    col: float("nan")
                    for col in EXPECTED_FEATURE_COLUMNS
                }
            )

            dns_statuses.append("-")
            scan_errors.append(
                "Unexpected error while scanning this URL."
            )

        progress.progress(
            (i + 1) / total,
            text=f"Scanning URLs... ({i + 1}/{total})",
        )

    progress.empty()

    features_df = pd.DataFrame(
        feature_rows,
        columns=EXPECTED_FEATURE_COLUMNS,
    )

    scored_df = pipeline.predict_dataframe(
        features_df
    )

    output_df = input_df.reset_index(
        drop=True
    ).copy()

    output_df["dns_status"] = dns_statuses
    output_df["scan_error"] = scan_errors
    output_df["predicted_result"] = scored_df[
        "predicted_result"
    ]
    output_df["prediction_label"] = scored_df[
        "prediction_label"
    ]
    output_df["confidence"] = scored_df[
        "confidence"
    ]

    invalid_mask = pd.Series(
        scan_errors
    ).notna().values

    output_df.loc[
        invalid_mask,
        "predicted_result",
    ] = pd.NA

    output_df.loc[
        invalid_mask,
        "prediction_label",
    ] = "Not scored (invalid URL)"

    output_df.loc[
        invalid_mask,
        "confidence",
    ] = pd.NA

    n_errors = sum(
        1
        for e in scan_errors
        if e
    )

    st.write(
        f"Scanned **{total}** URL(s) - "
        f"{total - n_errors} succeeded, "
        f"{n_errors} could not be parsed."
    )

    st.dataframe(
        output_df,
        use_container_width=True,
    )

    st.download_button(
        "Download results as CSV",
        data=output_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="bulk_url_scan_results.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Mode 3: Developer / Model Diagnostics
# ---------------------------------------------------------------------------
def developer_manual_entry():
    st.caption(
        "Set each pre-extracted dataset feature directly and get a prediction - useful for "
        "testing the model or demonstrating it without depending on a live website. This "
        "bypasses URL feature extraction entirely."
    )

    values = {}

    for section in SECTIONS:
        with st.expander(
            section,
            expanded=(section == SECTIONS[0]),
        ):
            section_features = [
                f for f in FEATURES
                if f[3] == section
            ]

            cols = st.columns(3)

            for i, (
                name,
                description,
                allowed_values,
                _,
            ) in enumerate(section_features):

                with cols[i % 3]:
                    values[name] = st.selectbox(
                        label=name.replace(
                            "_",
                            " ",
                        ),
                        options=allowed_values,
                        format_func=lambda v: VALUE_LABELS.get(
                            v,
                            str(v),
                        ),
                        help=description,
                        key=f"single_{name}",
                    )

    if st.button(
        "Predict",
        type="primary",
        key="developer_predict_btn",
    ):
        pipeline = get_pipeline()

        predicted_class, label, confidence = pipeline.predict_single(
            values
        )

        st.markdown("### Prediction")

        render_prediction_banner(
            predicted_class,
            confidence,
        )


def developer_raw_feature_csv():
    st.caption(
        "Upload a CSV file containing the same 30 pre-extracted feature columns used during "
        "training (a target/Result column, if present, is ignored). This is for testing the "
        "model directly - it does NOT do any URL feature extraction. Use the **Bulk URL "
        "Scanner** mode instead if you just have a list of URLs. A sample file is available in "
        "`valid_data/test.csv`."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="dev_raw_csv",
    )

    if uploaded_file is None:
        return

    try:
        input_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:
        st.error(
            f"Could not read the uploaded file: {exc}"
        )
        return

    expected_columns = set(
        EXPECTED_FEATURE_COLUMNS
    )

    missing_columns = (
        expected_columns
        - set(input_df.columns)
    )

    if missing_columns:
        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join(sorted(missing_columns))
            + ". If you only have URLs (not pre-extracted features), "
            "use the Bulk URL Scanner mode instead."
        )
        return

    pipeline = get_pipeline()

    with st.spinner("Scoring records..."):
        result_df = pipeline.predict_dataframe(
            input_df
        )

    st.write(
        f"Scored **{len(result_df)}** records."
    )

    st.dataframe(
        result_df,
        use_container_width=True,
    )

    st.download_button(
        "Download predictions as CSV",
        data=result_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def developer_diagnostics_mode():
    st.subheader(
        "Developer / Model Diagnostics"
    )

    st.info(
        "This section is for testing and demonstrating the underlying model directly. "
        "Ordinary users should use **URL Scanner** or **Bulk URL Scanner** instead - neither "
        "of those requires manually entering any of the 30 engineered features."
    )

    tab1, tab2 = st.tabs(
        [
            "Manual feature entry",
            "Raw 30-feature CSV upload",
        ]
    )

    with tab1:
        developer_manual_entry()

    with tab2:
        developer_raw_feature_csv()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title(
        "🛡️ Phishing URL Detection System"
    )

    st.markdown(
        "##### Detect potentially malicious or phishing URLs using Machine Learning"
    )

    st.write(
        "This project detects **phishing websites** using a classical machine learning model "
        "trained on a labelled phishing-website dataset. Everything runs **fully locally** - "
        "no cloud services, no external APIs, no AWS."
    )

    if not model_files_exist():
        st.warning(
            "No trained model was found under `final_model/`. Run `python main.py` from the "
            "project root first to train the model, then restart this app."
        )
        st.stop()

    with st.sidebar:
        st.header("Choose Input Mode")

        mode = st.radio(
            label="Input mode",
            options=[
                "URL Scanner",
                "Bulk URL Scanner",
                "Developer / Model Diagnostics",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.header("About this project")

        st.write(
            "- **Problem**: classify a website as legitimate or phishing\n"
            "- **Dataset**: `Network_Data/phisingData.csv` (30 features, binary target `Result`)\n"
            "- **Preprocessing**: KNN imputation (fit on the training split, reused at inference)\n"
            "- **Models compared**: Logistic Regression, Decision Tree, Random Forest, "
            "Gradient Boosting, AdaBoost - the best one by test accuracy is used here\n"
            "- **Artifacts used**: `final_model/model.pkl`, `final_model/preprocessor.pkl`"
        )

        st.caption(
            "URL Scanner / Bulk URL Scanner: extract what they can from the URL/webpage; DNS "
            "resolution is informational only and never blocks a prediction. Features that "
            "need third-party data (traffic rank, PageRank, WHOIS age, etc.) are left for the "
            "model's own imputer to fill in - see the README for details."
        )

    if mode == "URL Scanner":
        url_scanner_mode()

    elif mode == "Bulk URL Scanner":
        bulk_url_scanner_mode()

    else:
        developer_diagnostics_mode()


if __name__ == "__main__":
    main()

