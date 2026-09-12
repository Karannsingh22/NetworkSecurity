"""
url_feature_extractor.py

Turns a raw URL typed by a user into the same 30-column feature vector the
Network Security phishing-detection model was trained on
(`data_schema/schema.yaml`, `EXPECTED_FEATURE_COLUMNS`).

Design goals (see README "How the URL Scanner Works" section for the full
write-up):

  * Only produce a value for a feature when it was genuinely derived from
    the URL string or the fetched webpage. Nothing is guessed or
    hardcoded.
  * Features that fundamentally require third-party data the dataset
    creators used at collection time (search-engine indexing, PageRank,
    Alexa/traffic rank, WHOIS registration/age, backlink counts, phishing
    blacklists) are left as `numpy.nan`. They are NOT invented. The exact
    same `KNNImputer` preprocessing object used at training time is then
    responsible for filling them in - that is a legitimate, intentional
    use of the pipeline's own missing-value handling, not a fabricated
    value chosen by this code.
  * Network access is defensive: URL scheme validation, DNS-resolved IP
    checked against private/loopback/link-local ranges before connecting
    (basic SSRF protection), request timeouts, a capped response size, a
    capped redirect count, and no JavaScript execution (HTML is only
    parsed as text with BeautifulSoup, never run).
"""
from __future__ import annotations

import ipaddress
import re
import socket
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse, urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import EXPECTED_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Network / safety configuration
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = (5, 8)          # (connect timeout, read timeout) seconds
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024   # 2 MB cap on the downloaded page
USER_AGENT = "NetworkSecurityThreatDetector/1.0 (+local research/demo tool)"

KNOWN_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "lnkd.in", "rebrand.ly", "cutt.ly", "shorte.st",
    "tiny.cc", "soo.gd", "s2r.co", "clicky.me", "budurl.com", "rb.gy",
    "shorturl.at", "v.gd",
}

# Features the dataset requires that cannot legitimately be derived from a
# URL or the webpage it serves without a paid/third-party data source
# (search-engine index status, Alexa/traffic rank, PageRank, backlink
# counts, WHOIS registration length, phishing-blacklist membership, ...).
UNAVAILABLE_FEATURES = {
    "Domain_registeration_length": "Requires a WHOIS registration-length lookup (unreliable/rate-limited from a web app; not implemented).",
    "Abnormal_URL": "Requires matching the URL against WHOIS registrant data.",
    "age_of_domain": "Requires WHOIS domain-creation-date data.",
    "web_traffic": "Requires a third-party traffic-ranking service (e.g. Alexa/Tranco); no such service is used here.",
    "Page_Rank": "Requires Google PageRank data, which is no longer publicly available.",
    "Google_Index": "Requires querying Google's search index directly, which is not done here.",
    "Links_pointing_to_page": "Requires a backlink index (e.g. Moz/Ahrefs), which would need a paid API.",
    "Statistical_report": "Requires cross-referencing known phishing blacklists/statistical reports, not implemented locally.",
}

LEXICAL_FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
    "HTTPS_token", "port",
]

# DNSRecord only needs to know whether the hostname resolves at all - it
# does NOT require the webpage to be fetchable, so it is tracked
# separately from WEBPAGE_FEATURES below. This lets us report it even when
# the page itself can't be downloaded (blocked, timed out, 404, etc.).
DNS_FEATURES = ["DNSRecord"]

WEBPAGE_FEATURES = [
    "SSLfinal_State", "Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags",
    "SFH", "Submitting_to_email", "Redirect", "on_mouseover", "RightClick",
    "popUpWidnow", "Iframe",
]

DNS_TIMEOUT_SECONDS = 5


class URLValidationError(Exception):
    """Raised for any user-input problem: bad format, blocked/unsafe host, etc."""


@dataclass
class URLScanResult:
    input_url: str
    final_url: Optional[str] = None
    domain: Optional[str] = None
    fetch_succeeded: bool = False
    fetch_error: Optional[str] = None
    # DNS resolution is informational only and must never, by itself, cause
    # a URL to be treated as malicious or stop the prediction:
    #   "resolved"        - hostname resolved to a (public) IP address
    #   "unresolved"      - DNS lookup failed (NXDOMAIN, timeout, etc.)
    #   "blocked_private" - resolved, but to a private/loopback/reserved
    #                       address; the app will not connect to it (SSRF
    #                       protection), but the URL is still classified
    #                       on its lexical features.
    dns_status: str = "unresolved"
    features: Dict[str, float] = field(default_factory=dict)
    feature_status: Dict[str, str] = field(default_factory=dict)  # "extracted" | "unavailable"
    analysis: Dict[str, object] = field(default_factory=dict)

    def to_ordered_feature_dict(self) -> Dict[str, float]:
        """Return the features in the exact column order the model expects."""
        return {col: self.features.get(col, np.nan) for col in EXPECTED_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# URL validation + SSRF-safe host resolution
# ---------------------------------------------------------------------------
def _validate_and_parse_url(raw_url: str):
    if not raw_url or not isinstance(raw_url, str):
        raise URLValidationError("Please enter a URL.")

    raw_url = raw_url.strip()
    if len(raw_url) > 2048:
        raise URLValidationError("URL is too long.")

    parsed = urlparse(raw_url)

    if parsed.scheme not in ("http", "https"):
        raise URLValidationError("URL must start with http:// or https://")

    if not parsed.netloc:
        raise URLValidationError("That doesn't look like a valid URL (missing domain).")

    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("That doesn't look like a valid URL (missing domain).")

    return parsed


def _resolve_host_safely(hostname: str):
    """
    Attempts to resolve `hostname` and classify it as safe/unsafe to
    connect to. This is DELIBERATELY non-raising for ordinary DNS
    failures: a domain that does not resolve (e.g. a synthetic/example
    phishing-style hostname such as "paypal-security-check.example.com")
    is an extremely common and completely legitimate thing to ask this
    model to classify - the URL text itself is still real evidence, and
    the model must still get a chance to score it.

    Returns a tuple (dns_status, resolved_ip, detail):
      - dns_status: "resolved" | "unresolved" | "blocked_private"
      - resolved_ip: the resolved IP as a string, or None
      - detail: a short, user-safe explanation string

    "blocked_private" means the hostname DID resolve, but to a
    loopback/private/link-local/reserved/multicast address (e.g.
    127.0.0.1, 169.254.169.254 cloud metadata, 10.0.0.5, ...). The app
    will not connect to it (basic SSRF protection), but - just like an
    unresolved domain - this never stops URL classification; only the
    webpage-derived features become unavailable.
    """
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(DNS_TIMEOUT_SECONDS)
        resolved_ip = socket.gethostbyname(hostname)
    except (socket.gaierror, socket.timeout, UnicodeError, OSError):
        return "unresolved", None, f"Could not resolve domain '{hostname}'."
    except Exception:
        # Never let an unexpected DNS/resolver error crash the scan.
        return "unresolved", None, f"Could not resolve domain '{hostname}'."
    finally:
        socket.setdefaulttimeout(old_timeout)

    try:
        ip_obj = ipaddress.ip_address(resolved_ip)
    except ValueError:
        return "unresolved", None, "Could not validate the resolved IP address."

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        return (
            "blocked_private",
            resolved_ip,
            "Domain resolves to a private/internal address; skipping the live "
            "page fetch for safety (this app will not connect to internal "
            "network addresses).",
        )

    return "resolved", resolved_ip, None


# ---------------------------------------------------------------------------
# Category A: pure lexical (URL-string only) features - no network needed
# ---------------------------------------------------------------------------
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _extract_lexical_features(parsed_url, raw_url: str) -> (Dict[str, float], Dict[str, object]):
    hostname = parsed_url.hostname or ""
    features: Dict[str, float] = {}
    analysis: Dict[str, object] = {}

    # having_IP_Address: domain is a raw IPv4 address instead of a name
    is_ip = bool(_IPV4_RE.match(hostname))
    features["having_IP_Address"] = 1 if is_ip else -1
    analysis["ip_address_detected"] = is_ip

    # URL_Length: short/medium/long, thresholds approximate the commonly
    # published binning for this dataset (<54 / 54-75 / >75 chars).
    url_len = len(raw_url)
    if url_len < 54:
        features["URL_Length"] = -1
    elif url_len <= 75:
        features["URL_Length"] = 0
    else:
        features["URL_Length"] = 1
    analysis["url_length"] = url_len

    # Shortining_Service: hostname matches a known link-shortener domain
    is_shortened = hostname.lower() in KNOWN_URL_SHORTENERS
    features["Shortining_Service"] = 1 if is_shortened else -1
    analysis["shortening_service_detected"] = is_shortened

    # having_At_Symbol
    has_at = "@" in raw_url
    features["having_At_Symbol"] = 1 if has_at else -1
    analysis["at_symbol_detected"] = has_at

    # double_slash_redirecting: an extra "//" appearing after the protocol
    last_double_slash = raw_url.rfind("//")
    features["double_slash_redirecting"] = 1 if last_double_slash > 7 else -1

    # Prefix_Suffix: '-' in the domain (e.g. paypal-secure.com)
    has_dash = "-" in hostname
    features["Prefix_Suffix"] = 1 if has_dash else -1
    analysis["hyphen_in_domain"] = has_dash

    # having_Sub_Domain: count dot-separated labels in the hostname, minus
    # "www" and the last two labels (treated as domain + TLD). This is a
    # simple heuristic and does not use a public-suffix list, so results
    # for domains like "example.co.uk" may be slightly off - documented
    # in the README as a known limitation.
    labels = [l for l in hostname.lower().split(".") if l and l != "www"]
    subdomain_count = max(0, len(labels) - 2)
    if subdomain_count == 0:
        features["having_Sub_Domain"] = -1
    elif subdomain_count == 1:
        features["having_Sub_Domain"] = 0
    else:
        features["having_Sub_Domain"] = 1
    analysis["subdomain_count"] = subdomain_count

    # HTTPS_token: the literal token "https" appears inside the hostname
    # itself (a classic trick, e.g. "https-mybank.com")
    https_token_in_domain = "https" in hostname.lower()
    features["HTTPS_token"] = 1 if https_token_in_domain else -1
    analysis["https_token_in_domain"] = https_token_in_domain

    # port: URL explicitly specifies a non-default port for its scheme.
    # (We deliberately do NOT port-scan the host - that would be an
    # intrusive active-recon action against a third-party server.)
    default_port = 443 if parsed_url.scheme == "https" else 80
    explicit_port = parsed_url.port
    non_standard_port = explicit_port is not None and explicit_port != default_port
    features["port"] = 1 if non_standard_port else -1

    analysis["domain"] = hostname
    analysis["scheme"] = parsed_url.scheme
    analysis["https_used"] = parsed_url.scheme == "https"

    return features, analysis


# ---------------------------------------------------------------------------
# Category B: features derived from fetching + parsing the webpage
# ---------------------------------------------------------------------------
def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _fetch_page(url: str):
    """
    Fetches the URL defensively: capped redirects, capped response size,
    sane timeouts, a descriptive User-Agent, and TLS verification left ON
    (a certificate error is treated as a real signal, not silently
    ignored). Returns (response, html_text, ssl_ok, error_message).
    """
    session = requests.Session()
    session.max_redirects = MAX_REDIRECTS
    headers = {"User-Agent": USER_AGENT}

    ssl_ok = True
    try:
        response = session.get(
            url, headers=headers, timeout=REQUEST_TIMEOUT,
            allow_redirects=True, stream=True, verify=True,
        )
    except requests.exceptions.SSLError:
        ssl_ok = False
        return None, None, ssl_ok, "SSL certificate could not be verified."
    except requests.exceptions.TooManyRedirects:
        return None, None, ssl_ok, "Too many redirects."
    except requests.exceptions.ConnectTimeout:
        return None, None, ssl_ok, "Connection timed out."
    except requests.exceptions.ReadTimeout:
        return None, None, ssl_ok, "The server took too long to respond."
    except requests.exceptions.ConnectionError:
        return None, None, ssl_ok, "Could not connect to the website."
    except requests.exceptions.RequestException as exc:
        return None, None, ssl_ok, f"Request failed ({type(exc).__name__})."

    # Read the body with a hard size cap to avoid downloading huge files.
    content_chunks = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                break
            content_chunks.append(chunk)
    except requests.exceptions.RequestException:
        pass

    html_bytes = b"".join(content_chunks)
    try:
        html_text = html_bytes.decode(response.encoding or "utf-8", errors="ignore")
    except Exception:
        html_text = html_bytes.decode("utf-8", errors="ignore")

    return response, html_text, ssl_ok, None


_MOUSEOVER_RE = re.compile(r"onmouseover\s*=.{0,80}window\.status", re.IGNORECASE | re.DOTALL)
_RIGHTCLICK_RE = re.compile(r"(event\.button\s*==\s*2)|(oncontextmenu\s*=)", re.IGNORECASE)
_POPUP_RE = re.compile(r"(window\.open\s*\()|(prompt\s*\()", re.IGNORECASE)
_MAILTO_RE = re.compile(r"mailto:", re.IGNORECASE)


def _extract_webpage_features(
    final_url: str, response, html_text: str, ssl_ok: bool, base_domain: str
) -> (Dict[str, float], Dict[str, object]):
    features: Dict[str, float] = {}
    analysis: Dict[str, object] = {}

    parsed_final = urlparse(final_url)

    # SSLfinal_State: approximate. We only know "did HTTPS + certificate
    # verification succeed", not the certificate's issuer trust level or
    # age the way the original dataset feature was engineered.
    if parsed_final.scheme == "https" and ssl_ok:
        features["SSLfinal_State"] = -1
    else:
        features["SSLfinal_State"] = 1
    analysis["ssl_valid"] = parsed_final.scheme == "https" and ssl_ok

    soup = BeautifulSoup(html_text, "html.parser")

    # Favicon: does the <link rel="icon"...> point at a different domain?
    favicon_external = False
    favicon_tag = soup.find("link", rel=lambda v: v and "icon" in v.lower())
    if favicon_tag and favicon_tag.get("href"):
        favicon_url = urljoin(final_url, favicon_tag["href"])
        favicon_external = _domain_of(favicon_url) not in ("", base_domain)
    features["Favicon"] = 1 if favicon_external else -1
    analysis["favicon_external"] = favicon_external

    # Request_URL: % of img/script/audio/video/embed/source tags pointing
    # at a different domain than the page itself.
    resource_tags = soup.find_all(["img", "script", "audio", "video", "embed", "source"])
    total_resources, external_resources = 0, 0
    for tag in resource_tags:
        src = tag.get("src")
        if not src:
            continue
        total_resources += 1
        if _domain_of(urljoin(final_url, src)) not in ("", base_domain):
            external_resources += 1
    request_url_ratio = (external_resources / total_resources) if total_resources else 0.0
    features["Request_URL"] = _ratio_to_ternary(request_url_ratio, low=0.22, high=0.61)
    analysis["external_resource_ratio"] = round(request_url_ratio, 3)
    analysis["resource_tag_count"] = total_resources

    # URL_of_Anchor: % of <a href> that are external, empty, or javascript:void
    anchors = soup.find_all("a")
    total_anchors, suspicious_anchors = 0, 0
    for tag in anchors:
        href = tag.get("href")
        total_anchors += 1
        if not href or href.strip() in ("#", "") or href.strip().lower().startswith("javascript:void"):
            suspicious_anchors += 1
        elif _domain_of(urljoin(final_url, href)) not in ("", base_domain):
            suspicious_anchors += 1
    anchor_ratio = (suspicious_anchors / total_anchors) if total_anchors else 0.0
    features["URL_of_Anchor"] = _ratio_to_ternary(anchor_ratio, low=0.31, high=0.67)
    analysis["suspicious_anchor_ratio"] = round(anchor_ratio, 3)
    analysis["anchor_tag_count"] = total_anchors

    # Links_in_tags: % of <meta>/<script>/<link> referencing an external domain
    meta_script_link = soup.find_all(["meta", "script", "link"])
    total_msl, external_msl = 0, 0
    for tag in meta_script_link:
        target = tag.get("src") or tag.get("href") or tag.get("content")
        if not target or not str(target).strip().lower().startswith(("http://", "https://", "//")):
            continue
        total_msl += 1
        if _domain_of(urljoin(final_url, target)) not in ("", base_domain):
            external_msl += 1
    msl_ratio = (external_msl / total_msl) if total_msl else 0.0
    features["Links_in_tags"] = _ratio_to_ternary(msl_ratio, low=0.17, high=0.81)
    analysis["external_meta_script_link_ratio"] = round(msl_ratio, 3)

    # SFH (Server Form Handler): where does the first <form action=...> point?
    form = soup.find("form")
    if form is None:
        features["SFH"] = -1
        analysis["form_action"] = None
    else:
        action = (form.get("action") or "").strip()
        analysis["form_action"] = action or "(empty)"
        if action == "" or action.lower() == "about:blank":
            features["SFH"] = 1
        elif _domain_of(urljoin(final_url, action)) not in ("", base_domain):
            features["SFH"] = 0
        else:
            features["SFH"] = -1

    # Submitting_to_email: form action or page script uses mailto:
    submits_to_email = bool(_MAILTO_RE.search(html_text))
    features["Submitting_to_email"] = 1 if submits_to_email else -1
    analysis["submits_to_email"] = submits_to_email

    # Redirect: how many redirects happened while fetching this page
    redirect_count = len(response.history) if response is not None else 0
    features["Redirect"] = -1 if redirect_count <= 1 else 1
    analysis["redirect_count"] = redirect_count

    # on_mouseover: status-bar text changed via onmouseover (classic trick)
    mouseover_trick = bool(_MOUSEOVER_RE.search(html_text))
    features["on_mouseover"] = 1 if mouseover_trick else -1
    analysis["status_bar_trick_detected"] = mouseover_trick

    # RightClick: page disables the context menu / right click
    right_click_disabled = bool(_RIGHTCLICK_RE.search(html_text))
    features["RightClick"] = 1 if right_click_disabled else -1
    analysis["right_click_disabled"] = right_click_disabled

    # popUpWidnow: page uses window.open()/prompt() (heuristic for popups
    # that ask the visitor for information)
    popup_detected = bool(_POPUP_RE.search(html_text))
    features["popUpWidnow"] = 1 if popup_detected else -1
    analysis["popup_script_detected"] = popup_detected

    # Iframe: page uses at least one <iframe>
    iframe_count = len(soup.find_all("iframe"))
    features["Iframe"] = 1 if iframe_count > 0 else -1
    analysis["iframe_count"] = iframe_count

    # Note: DNSRecord is computed in scan_url() directly from the DNS
    # resolution step (see _resolve_host_safely), not here - it doesn't
    # depend on the webpage fetch succeeding.

    return features, analysis


def _ratio_to_ternary(ratio: float, low: float, high: float) -> int:
    if ratio < low:
        return -1
    if ratio <= high:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def scan_url(raw_url: str) -> URLScanResult:
    """
    Validates the URL, safely fetches the page (if reachable), extracts
    every feature that can legitimately be derived, and returns a
    URLScanResult with:
      - features / feature_status for all 30 model columns
      - a human-readable `analysis` dict for the UI's "URL Analysis" panel

    Raises URLValidationError for bad/unsafe input (caught by the caller
    and shown as a friendly message - never a raw stack trace).
    """
    try:
        # Only genuinely malformed input stops classification here (bad
        # scheme, missing domain, empty string, etc.) - never a DNS/network
        # condition. See _resolve_host_safely() below for why.
        parsed = _validate_and_parse_url(raw_url)
        hostname = parsed.hostname

        result = URLScanResult(input_url=raw_url, domain=hostname)

        # --- Category A: pure lexical (URL-string only) features ---------
        # Always computed. Never depends on the network in any way, so it
        # is unaffected by DNS failures, timeouts, or blocked hosts.
        lexical_features, lexical_analysis = _extract_lexical_features(parsed, raw_url)
        result.features.update(lexical_features)
        result.analysis.update(lexical_analysis)
        for name in LEXICAL_FEATURES:
            result.feature_status[name] = "extracted"

        # --- DNS resolution: informational, never blocking ---------------
        dns_status, resolved_ip, dns_detail = _resolve_host_safely(hostname)
        result.dns_status = dns_status
        result.analysis["dns_status"] = dns_status
        result.analysis["resolved_ip"] = resolved_ip

        # DNSRecord is a genuine signal we now know regardless of whether
        # the page itself can be fetched: -1 if a DNS record exists
        # (resolved, even if it's a private address), 1 if it does not.
        result.features["DNSRecord"] = -1 if resolved_ip is not None else 1
        result.feature_status["DNSRecord"] = "extracted"

        # --- Category B: features that need the live webpage -------------
        if dns_status == "resolved":
            response, html_text, ssl_ok, fetch_error = _fetch_page(parsed.geturl())
        else:
            # Either the domain didn't resolve, or it resolved to a
            # private/internal address we refuse to connect to. Either
            # way we simply skip the network fetch - we do NOT raise, and
            # we do NOT treat this as evidence of malice on its own.
            response, html_text, ssl_ok, fetch_error = None, None, True, dns_detail

        if response is not None and html_text is not None:
            result.fetch_succeeded = True
            result.final_url = response.url
            base_domain = _domain_of(response.url)
            webpage_features, webpage_analysis = _extract_webpage_features(
                response.url, response, html_text, ssl_ok, base_domain
            )
            result.features.update(webpage_features)
            result.analysis.update(webpage_analysis)
            for name in WEBPAGE_FEATURES:
                result.feature_status[name] = "extracted"
        else:
            # Could not fetch the page (unresolved domain, blocked private
            # address, timeout, connection error, ...). Lexical features
            # and DNSRecord are still valid; every webpage-dependent
            # feature is genuinely unknown - left as NaN (imputed later
            # by the same KNNImputer used at training time), never guessed
            # and never treated as a malicious signal.
            result.fetch_succeeded = False
            result.fetch_error = fetch_error
            result.final_url = parsed.geturl()
            for name in WEBPAGE_FEATURES:
                result.features[name] = np.nan
                result.feature_status[name] = f"unavailable (page fetch skipped/failed: {fetch_error})"
            logging.info(f"URL scan: could not fetch '{raw_url}': {fetch_error}")

        # Category C: never obtainable locally - always NaN, always labeled.
        for name, reason in UNAVAILABLE_FEATURES.items():
            result.features[name] = np.nan
            result.feature_status[name] = f"unavailable ({reason})"

        return result

    except URLValidationError:
        raise
    except Exception as e:
        # Never leak a raw stack trace to the UI; wrap and re-raise as our
        # own exception type so app.py can show a friendly message while
        # the details still land in the log file.
        raise NetworkSecurityException(e, sys)
