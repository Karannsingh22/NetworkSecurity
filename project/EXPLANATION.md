# Technical Explanation — DNS Fix & Pipeline Audit

## 1. Why "Could not resolve domain ..." appeared

**File:** `networksecurity/components/url_feature_extractor.py`
**Function:** `_ensure_host_is_public(hostname)` (old name), called from `scan_url()`

The old code path was:

```
scan_url(raw_url)
  -> _validate_and_parse_url(raw_url)      # format check, fine
  -> _ensure_host_is_public(hostname)      # <-- called unconditionally, BEFORE
                                            #     any feature extraction happened
       -> socket.gethostbyname(hostname)
       -> on socket.gaierror: raise URLValidationError(
              f"Could not resolve domain '{hostname}'."
          )
```

`_ensure_host_is_public` was written as an **SSRF guard** — its real job
was to stop the app from being used to probe internal network addresses
(`127.0.0.1`, `169.254.169.254` cloud metadata, `10.x.x.x`, etc.) before
connecting to them. To do that job it needed to resolve the hostname
first. But it was written to `raise` on *any* resolution failure,
including the completely ordinary case of a domain that simply doesn't
exist (a synthetic/example phishing-style hostname like
`paypal-security-check.example.com`).

Because this call happened at the very top of `scan_url()`, a
`socket.gaierror` there aborted the *entire* function via an exception —
lexical feature extraction, webpage feature extraction, and the model
prediction never ran. `app.py` caught `URLValidationError` and displayed
it as the final result, so the user only ever saw "Could not resolve
domain ..." and no prediction at all.

**In short: a real security check (SSRF protection) and an unrelated
convenience check (DNS existence) were fused into one function that
`raise`s, and that one function's failure mode was wired directly into
"stop everything."**

## 2. Why DNS failure should not prevent URL-based ML classification

The trained model's 30 features (`data_schema/schema.yaml`) are a mix of:

- **Lexical features** derived purely from the URL string (length,
  `@` symbol, IP-address usage, hyphens, subdomain count, etc.) — these
  need *zero* network access and are exactly the kind of signal a
  phishing URL is designed to trip.
- **Webpage features** that need the live page fetched.
- **Third-party features** (WHOIS age, PageRank, traffic rank, etc.)
  that this project has never been able to obtain locally, and were
  already being safely left as `NaN` for the trained `KNNImputer` to
  fill in — the "Category C" section further down.

DNS resolution has no special status in this list. It is a precondition
for fetching the webpage, nothing more. Treating "DNS failed" as
"prediction impossible" was strictly worse than treating an unreachable
webpage as "prediction impossible" — and the code already handled the
latter gracefully (see `fetch_error` handling in the old `scan_url`).
The DNS check was just never given the same graceful path.

Practically: phishing demo/test URLs are *routinely* built on
`.example.com` or otherwise-unregistered domains specifically because
nobody wants to spin up a real phishing site for a class project. A
demo tool that can't classify those isn't usable for its stated purpose.

## 3. What was fixed

`networksecurity/components/url_feature_extractor.py`:

- `_ensure_host_is_public` (raises) → replaced with `_resolve_host_safely`
  (never raises for DNS/network reasons). It returns a status —
  `"resolved"`, `"unresolved"`, or `"blocked_private"` — plus the
  resolved IP if any. A 5-second timeout was added so a slow/broken
  resolver can't hang the app.
- `scan_url()` no longer calls anything that can raise for a DNS
  failure. It:
  1. Validates *only* the URL's format (still raises `URLValidationError`
     for genuinely malformed input — empty string, no scheme, no host).
  2. Always extracts the 9 lexical features.
  3. Resolves DNS informationally and always sets a `DNSRecord` feature
     value (`-1` if it resolved to anything, `1` if not) — this used to
     be silently tied to "did the page fetch succeed," which meant it
     was wrongly left as `NaN` whenever the page fetch failed for a
     reason unrelated to DNS. It's now a first-class, always-computed
     feature, matching what the original dataset's `DNSRecord` column
     actually measures.
  4. Only attempts to fetch the live page if DNS resolved to a **public**
     address. If DNS failed, or resolved to a private/loopback/reserved
     address, the fetch is skipped (the SSRF protection is preserved —
     the app still never connects to internal network addresses) and the
     12 webpage-dependent features are left as `NaN` for the same
     `KNNImputer` used at training time.
  5. Returns a `dns_status` field on the result so the UI can show it as
     information, never as an automatic "malicious" verdict.
- `app.py` (`url_scanner_mode`, and the new `bulk_url_scanner_mode`) now
  display DNS status as a metric/info panel, with distinct, honest
  wording for "could not resolve" vs. "resolves to a private address"
  vs. "resolved but page fetch failed" — none of which claim the URL is
  malicious by themselves.

No exception type, function signature used elsewhere, or feature
column changed in a way that affects the trained model or preprocessor —
`EXPECTED_FEATURE_COLUMNS` and their order are untouched.

## 4. Was the ML pipeline already automated?

**Yes, mostly — and this is important: it was NOT rebuilt.** Before
touching anything, the following was verified by reading the code (not
assumed):

- **Dataset:** `Network_Data/phisingData.csv`, 30 pre-engineered
  features + binary target `Result` (`schema.yaml`).
- **Training pipeline:** `networksecurity/pipeline/training_pipeline.py`
  → data ingestion → data validation (schema + drift check) → data
  transformation → model training/selection → saved artifacts.
  Preprocessing (`data_transformation.py`) is a single-step
  `sklearn.Pipeline` wrapping a `KNNImputer(n_neighbors=3)`, fit only on
  the training split, then saved to both
  `Artifacts/.../preprocessing.pkl` and `final_model/preprocessor.pkl`.
- **Model:** the best of Logistic Regression / Decision Tree / Random
  Forest / Gradient Boosting / AdaBoost by test-set accuracy
  (`model_trainer.py`), saved to `final_model/model.pkl`.
- **Inference:** `networksecurity/pipeline/batch_prediction.py` loads
  both artifacts once (`BatchPredictionPipeline.__init__`) and every
  prediction path — single URL scan, the new bulk CSV/Excel scan, the
  developer manual form, and the developer raw-CSV upload — goes through
  the *same* `predict_single` / `predict_dataframe` methods, which in
  turn call `NetworkModel` (`utils/ml_utils/model/estimator.py`).
  `NetworkModel._align_columns` reindexes any input to the exact column
  names/order the fitted preprocessor expects
  (`preprocessor.feature_names_in_`), so **training-time and
  inference-time feature order are guaranteed to match** regardless of
  which UI mode built the feature dict/dataframe.
- **Conclusion: training feature pipeline == inference feature
  pipeline**, exactly as required. There was no mismatch to fix here —
  the only actual bug was the DNS `raise` described above, which
  prevented the (correct) pipeline from ever being reached for
  unresolvable domains.

## 5. Was the 30-feature Advanced/manual section necessary?

Not as the primary interface — and it wasn't already being forced on
ordinary users; `app.py` already had a `URL Scanner` as the default
mode before this change. It has been kept, renamed to **"Developer /
Model Diagnostics"**, moved to the third (non-default) sidebar option,
and explicitly labeled as a diagnostics/demo tool rather than the normal
way to use the app. This satisfies your requirement to preserve it for
debugging/demo value without making it primary.

## 6. Was the CSV upload implementation too restrictive?

Yes. The old "Batch CSV upload" (inside the old Advanced mode) required
the uploaded file to already contain **all 30 exact pre-engineered
feature columns** — i.e. it assumed the user already had a
dataset-shaped file, not a plain list of URLs. That's unrealistic for
the stated goal ("someone gives me an arbitrary CSV/Excel dataset
containing URLs").

**Fix:** a new, separate **Bulk URL Scanner** mode was added that:

- Accepts `.csv`, `.xlsx`, and `.xls`.
- Auto-detects a URL column by name (`url`, `link`, `website`, `domain`,
  `webpage`, case-insensitive) among whatever columns the file has.
- Ignores unrelated columns.
- Runs every URL through the *identical* `scan_url()` extraction used by
  the single-URL scanner, then through the same
  `BatchPredictionPipeline.predict_dataframe`.
- Never uses a `Result`/target column (if present) as an input feature —
  features come only from the URL text/webpage, so there's no leakage
  path even if the file happens to include a label column; it's only
  carried through untouched in the output for reference.
- Shows a clear, specific error (listing the columns it did find) if no
  recognizable URL column exists, instead of crashing or silently
  misinterpreting the file.
- Rows whose URL text is unusable (empty, malformed, wrong scheme) are
  marked `"Not scored (invalid URL)"` rather than being fed an all-`NaN`
  row into the model, which would otherwise produce a meaningless
  imputed prediction.

The old raw 30-column CSV upload was **not removed** — it now lives
under Developer / Model Diagnostics, clearly labeled as bypassing URL
extraction entirely, for people who already have a dataset-shaped file
and want to test the model directly.

## 7. Other issues found (and their status)

- **`DNSRecord` was silently coupled to webpage-fetch success.** Fixed
  as described in §3 — it's now computed directly from DNS resolution,
  independent of whether the page itself could be fetched.
- **No timeout on the DNS lookup itself** (only the HTTP request had
  timeouts). Fixed — a 5-second timeout is now applied around
  `socket.gethostbyname`.
- **Feature/column alignment, preprocessing, and model loading**: no
  issues found. `NetworkModel._align_columns` already defends against
  column-order mismatches, and both artifacts are loaded once and cached
  (`st.cache_resource`) rather than reloaded per request.
- **Missing-value handling**: correct and already intentional — `NaN`
  values are only ever produced for features that are genuinely
  unknown, and are handled by the exact `KNNImputer` fit at training
  time. Nothing invents values.
- **No retraining was performed or needed.** The existing
  `final_model/model.pkl` and `final_model/preprocessor.pkl` are
  untouched; this was purely an inference-path/UI fix.

## 8. Test inputs used to verify the fix

Verified directly against `scan_url()` and the full
`BatchPredictionPipeline` (see the "Test inputs" list in the final
summary message) — all of them now reach a prediction, and the
previously-fatal "Could not resolve domain ..." message no longer
appears as a final result for any of them.
