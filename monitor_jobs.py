import os
import json
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ----------------------------
# Configuration
# ----------------------------

STATE_FILE = "state.json"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Gemini settings
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_CONFIDENCE_THRESHOLD = float(os.getenv("GEMINI_CONFIDENCE_THRESHOLD", "0.8"))
GEMINI_MAX_CANDIDATES_PER_SITE = int(os.getenv("GEMINI_MAX_CANDIDATES_PER_SITE", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@dataclass(frozen=True)
class Job:
    id: str
    title: str
    url: str
    source: str
    confidence: Optional[float] = None  # Only present for Gemini-approved items


@dataclass(frozen=True)
class SiteConfig:
    site_key: str
    name: str
    source_url: str
    base_url: str
    parser_key: str  # key into PARSERS


# Initial set (extend later)
SITES: List[SiteConfig] = [
    SiteConfig(
        site_key="paa",
        name="Pakistan Airports Authority (PAA)",
        source_url="https://paa.gov.pk/media/job-opportunities",
        base_url="https://paa.gov.pk",
        parser_key="paa",
    ),
    SiteConfig(
        site_key="ptcl",
        name="PTCL",
        source_url="https://ptcl.com.pk/Jobs",
        base_url="https://ptcl.com.pk",
        parser_key="ptcl",
    ),
    SiteConfig(
        site_key="systems",
        name="Systems Limited",
        source_url="https://www.systemsltd.com/careers/",
        base_url="https://www.systemsltd.com",
        parser_key="systems",
    ),
    SiteConfig(
        site_key="netsol",
        name="NETSOL Technologies",
        source_url="https://www.netsoltech.com/careers/",
        base_url="https://www.netsoltech.com",
        parser_key="netsol",
    ),
]


# ----------------------------
# State / IDs / Fetching
# ----------------------------

def stable_job_id(site_key: str, title: str, url: str) -> str:
    base = f"{site_key}||{title.strip()}||{url.strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def load_state(path: str = STATE_FILE) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    for k, v in list(data.items()):
        if not isinstance(v, list):
            data[k] = []
    return data

def save_state(state: Dict[str, List[str]], path: str = STATE_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, sort_keys=True)

def fetch_html(url: str, timeout: int = 30) -> str:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


# ----------------------------
# Deterministic parsers (baseline)
# ----------------------------

def parse_generic_links(
    site_key: str,
    html: str,
    base_url: str,
    must_contain_any: Optional[List[str]] = None,
    exclude_any: Optional[List[str]] = None,
    limit: int = 200,
) -> List[Job]:
    """
    Extract candidate links and reduce obvious noise using keywords.
    This is used both as (a) baseline and (b) Gemini candidate generator.
    """
    must_contain_any = must_contain_any or []
    exclude_any = exclude_any or []

    soup = BeautifulSoup(html, "html.parser")
    jobs: List[Job] = []

    for a in soup.select("a[href]"):
        text = " ".join(a.get_text(" ", strip=True).split())
        href = (a.get("href") or "").strip()
        if not href:
            continue

        abs_url = urljoin(base_url, href)
        hay = (text + " " + href).lower()

        if any(x.lower() in hay for x in exclude_any):
            continue

        if must_contain_any and not any(k.lower() in hay for k in must_contain_any):
            continue

        # Avoid very short menu items
        if len(text) < 4:
            continue

        jid = stable_job_id(site_key, text, abs_url)
        jobs.append(Job(id=jid, title=text, url=abs_url, source=site_key))

        if len(jobs) >= limit:
            break

    # De-duplicate
    seen = set()
    unique: List[Job] = []
    for j in jobs:
        if j.id in seen:
            continue
        seen.add(j.id)
        unique.append(j)
    return unique

def parse_paa(html: str, base_url: str) -> List[Job]:
    return parse_generic_links(
        site_key="paa",
        html=html,
        base_url=base_url,
        must_contain_any=["job", "career", "vacan", "advert", "assistant", "officer", "recruit", ".pdf"],
        exclude_any=["facebook", "twitter", "linkedin", "youtube", "privacy", "sitemap"],
    )

def parse_ptcl(html: str, base_url: str) -> List[Job]:
    return parse_generic_links(
        site_key="ptcl",
        html=html,
        base_url=base_url,
        must_contain_any=["job", "career", "vacan", "position", "apply"],
        exclude_any=["facebook", "twitter", "linkedin", "youtube", "privacy", "contact"],
    )

def parse_systems(html: str, base_url: str) -> List[Job]:
    return parse_generic_links(
        site_key="systems",
        html=html,
        base_url=base_url,
        must_contain_any=["career", "job", "opening", "vacan", "apply", "position"],
        exclude_any=["facebook", "twitter", "linkedin", "youtube", "privacy", "contact"],
    )

def parse_netsol(html: str, base_url: str) -> List[Job]:
    return parse_generic_links(
        site_key="netsol",
        html=html,
        base_url=base_url,
        must_contain_any=["career", "job", "opening", "vacan", "apply", "position"],
        exclude_any=["facebook", "twitter", "linkedin", "youtube", "privacy", "contact"],
    )

PARSERS: Dict[str, Callable[[str, str], List[Job]]] = {
    "paa": parse_paa,
    "ptcl": parse_ptcl,
    "systems": parse_systems,
    "netsol": parse_netsol,
}


# ----------------------------
# Gemini: one call per run (grouped candidates)
# ----------------------------

def _gemini_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY")

def build_gemini_payload(candidates_by_site: Dict[str, List[Job]]) -> Dict:
    """
    Create a compact structured payload for Gemini.
    """
    grouped = []
    for site_key, jobs in candidates_by_site.items():
        grouped.append(
            {
                "site_key": site_key,
                "candidates": [{"title": j.title, "url": j.url} for j in jobs],
            }
        )
    return {"sites": grouped}

def gemini_classify_jobs(
    candidates_by_site: Dict[str, List[Job]],
    threshold: float,
    timeout: int = 60,
) -> Dict[str, List[Job]]:
    """
    Returns Gemini-approved jobs per site_key with confidence attached.
    If GEMINI_API_KEY is missing, raises RuntimeError.
    """
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var / GitHub secret.")

    system_instruction = (
        "You are a strict classifier for job posting links.\n"
        "Goal: From each site's candidate links, return ONLY real, specific job postings.\n"
        "Reject navigation links, social links, privacy/terms, generic 'Careers' homepages, and unrelated news.\n"
        "Prefer links that are clearly job ads or job detail pages (including PDFs containing job advertisements).\n"
        "Output MUST be valid JSON only, no markdown, no extra text."
    )

    user_instruction = (
        "Given the following grouped candidate links by site_key, return a JSON object with the shape:\n"
        "{\n"
        '  "results": [\n'
        '    {"site_key": "...", "title": "...", "url": "...", "confidence": 0.0}\n'
        "  ]\n"
        "}\n\n"
        f"Rules:\n"
        f"- Only include items that are clearly job postings.\n"
        f"- confidence must be a number between 0 and 1.\n"
        f"- If uncertain, exclude the item.\n"
        f"- Keep only items with confidence >= {threshold}.\n"
        f"- Preserve original url exactly.\n"
        f"Now classify these candidates:\n"
    )

    payload = build_gemini_payload(candidates_by_site)
    prompt_text = user_instruction + json.dumps(payload, ensure_ascii=False)

    # Gemini REST API (Generative Language API)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    req = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    resp = requests.post(url, headers=headers, json=req, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Extract model text response
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Unexpected Gemini response format: {e}; data keys={list(data.keys())}")

    # Parse JSON
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini did not return valid JSON. Error: {e}. Raw text: {text[:500]}")

    results = parsed.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Gemini JSON missing 'results' list.")

    approved: Dict[str, List[Job]] = {k: [] for k in candidates_by_site.keys()}

    for item in results:
        if not isinstance(item, dict):
            continue
        site_key = item.get("site_key")
        title = item.get("title")
        job_url = item.get("url")
        conf = item.get("confidence")

        if site_key not in approved:
            continue
        if not isinstance(title, str) or not isinstance(job_url, str):
            continue
        try:
            conf_val = float(conf)
        except Exception:
            continue
        if conf_val < threshold:
            continue

        jid = stable_job_id(site_key, title, job_url)
        approved[site_key].append(Job(id=jid, title=title.strip(), url=job_url.strip(), source=site_key, confidence=conf_val))

    # de-duplicate per site
    for site_key, jobs in list(approved.items()):
        seen = set()
        uniq = []
        for j in jobs:
            if j.id in seen:
                continue
            seen.add(j.id)
            uniq.append(j)
        approved[site_key] = uniq

    return approved


# ----------------------------
# Diff + Email
# ----------------------------

def diff_new_jobs(current: List[Job], seen_ids: List[str]) -> Tuple[List[Job], List[str]]:
    seen = set(seen_ids)
    new_jobs = [j for j in current if j.id not in seen]
    updated_ids = list(seen.union({j.id for j in current}))
    return new_jobs, updated_ids

def send_email_gmail(subject: str, body: str) -> None:
    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO")

    if not gmail_user or not gmail_app_password or not email_to:
        raise RuntimeError("Missing env vars: GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO")

    msg = MIMEMultipart()
    msg["From"] = gmail_user
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(gmail_user, [email_to], msg.as_string())

def format_email(
    new_by_site: Dict[str, List[Job]],
    sites_by_key: Dict[str, SiteConfig],
) -> Tuple[str, str]:
    now = datetime.now(timezone.utc).astimezone()
    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    total = sum(len(v) for v in new_by_site.values())
    subject = f"[Job Monitor] {total} new job(s) found"

    lines: List[str] = []
    lines.append("New job postings detected")
    lines.append(f"Timestamp: {ts}")
    lines.append("")

    for site_key, jobs in new_by_site.items():
        if not jobs:
            continue
        site = sites_by_key[site_key]
        lines.append(f"== {site.name} ==")
        lines.append(f"Source: {site.source_url}")
        for j in jobs:
            conf = f"{j.confidence:.2f}" if isinstance(j.confidence, float) else "n/a"
            lines.append(f"- {j.title}  [confidence: {conf}]")
            lines.append(f"  {j.url}")
        lines.append("")

    return subject, "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def run() -> int:
    state = load_state(STATE_FILE)
    sites_by_key = {s.site_key: s for s in SITES}

    # 1) scrape + deterministic candidates per site (always)
    deterministic_candidates: Dict[str, List[Job]] = {}
    html_cache: Dict[str, str] = {}

    for site in SITES:
        logging.info("Fetching and parsing (deterministic): %s (%s)", site.name, site.source_url)
        parser = PARSERS.get(site.parser_key)
        if not parser:
            logging.warning("No parser for site_key=%s", site.site_key)
            deterministic_candidates[site.site_key] = []
            continue

        try:
            html = fetch_html(site.source_url)
            html_cache[site.site_key] = html
            candidates = parser(html, site.base_url)
            deterministic_candidates[site.site_key] = candidates[:GEMINI_MAX_CANDIDATES_PER_SITE]
            logging.info("Deterministic candidates for %s: %d", site.site_key, len(deterministic_candidates[site.site_key]))
        except Exception as e:
            logging.exception("Failed deterministic fetch/parse for %s: %s", site.site_key, e)
            deterministic_candidates[site.site_key] = []

    # 2) Gemini classification (one call per run)
    gemini_approved: Dict[str, List[Job]] = {s.site_key: [] for s in SITES}
    gemini_ok = False
    if _gemini_api_key():
        try:
            logging.info("Calling Gemini once for all sites (threshold=%.2f, model=%s)", GEMINI_CONFIDENCE_THRESHOLD, GEMINI_MODEL)
            gemini_approved = gemini_classify_jobs(
                candidates_by_site=deterministic_candidates,
                threshold=GEMINI_CONFIDENCE_THRESHOLD,
            )
            gemini_ok = True
            total_approved = sum(len(v) for v in gemini_approved.values())
            logging.info("Gemini approved %d posting(s) total.", total_approved)
        except Exception as e:
            logging.exception("Gemini classification failed; falling back to deterministic-only. Error: %s", e)
    else:
        logging.warning("GEMINI_API_KEY not set; using deterministic-only.")

    # 3) Union strategy (your choice): use BOTH Gemini-approved + deterministic
    #    - When Gemini works: union reduces false negatives but may increase false positives.
    #    - When Gemini fails: only deterministic is available.
    final_current_by_site: Dict[str, List[Job]] = {}

    for site in SITES:
        det = deterministic_candidates.get(site.site_key, [])
        gem = gemini_approved.get(site.site_key, []) if gemini_ok else []
        by_id: Dict[str, Job] = {}

        for j in det:
            by_id[j.id] = j
        for j in gem:
            # if Gemini version exists, prefer it (adds confidence field)
            by_id[j.id] = j

        final_current_by_site[site.site_key] = list(by_id.values())
        logging.info(
            "Final postings for %s: %d (det=%d, gem=%d)",
            site.site_key,
            len(final_current_by_site[site.site_key]),
            len(det),
            len(gem),
        )

    # 4) diff + state update + email
    all_new_by_site: Dict[str, List[Job]] = {}
    state_changed = False

    for site in SITES:
        current = final_current_by_site.get(site.site_key, [])
        seen_ids = state.get(site.site_key, [])

        new_jobs, updated_ids = diff_new_jobs(current, seen_ids)
        if new_jobs:
            all_new_by_site[site.site_key] = new_jobs

        if set(updated_ids) != set(seen_ids):
            state[site.site_key] = updated_ids
            state_changed = True

    if all_new_by_site:
        subject, body = format_email(all_new_by_site, sites_by_key)
        send_email_gmail(subject, body)
        logging.info("Email sent.")
    else:
        logging.info("No new postings found; no email sent.")

    if state_changed:
        save_state(state, STATE_FILE)
        logging.info("State updated: %s", STATE_FILE)
    else:
        logging.info("No state changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())