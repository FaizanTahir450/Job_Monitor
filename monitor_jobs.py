import os
import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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

@dataclass(frozen=True)
class SiteConfig:
    site_key: str
    name: str
    source_url: str
    base_url: str
    parser_key: str  # key into PARSERS


# Initial set (you can add more later)
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
# Utilities: state, ids, fetching
# ----------------------------

def stable_job_id(site_key: str, title: str, url: str) -> str:
    """
    Prefer stable URL (best), else use hash(title+url+site_key).
    We still hash to keep IDs short and consistent even if URL is long.
    """
    base = f"{site_key}||{title.strip()}||{url.strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def load_state(path: str = STATE_FILE) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure dict[str, list[str]]
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
# Parsers (site-specific)
# ----------------------------
# Notes:
# - Because each website can change HTML, each parser should be tolerant.
# - Parsers try to extract links that look like job postings.
# - If a parser becomes unreliable, update only that parser function.

def parse_generic_links(
    site_key: str,
    html: str,
    base_url: str,
    must_contain_any: Optional[List[str]] = None,
    exclude_any: Optional[List[str]] = None,
) -> List[Job]:
    """
    Fallback parser: collect meaningful <a> links and filter by keywords.
    This is not perfect, but it reduces noise and provides a baseline.

    must_contain_any: list of keywords; link text or href must include at least one.
    exclude_any: list of keywords to exclude.
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

        # Basic exclusions
        if any(x.lower() in hay for x in exclude_any):
            continue

        if must_contain_any and not any(k.lower() in hay for k in must_contain_any):
            continue

        # Avoid extremely short/empty titles
        if len(text) < 4:
            continue

        jid = stable_job_id(site_key, text, abs_url)
        jobs.append(Job(id=jid, title=text, url=abs_url, source=site_key))

    # De-duplicate by id while keeping order
    seen = set()
    unique: List[Job] = []
    for j in jobs:
        if j.id in seen:
            continue
        seen.add(j.id)
        unique.append(j)
    return unique

def parse_paa(html: str, base_url: str) -> List[Job]:
    # PAA page often contains tables / PDF links for job ads.
    # Filter to typical job-related keywords and PDFs.
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
# Diff logic
# ----------------------------

def diff_new_jobs(current: List[Job], seen_ids: List[str]) -> Tuple[List[Job], List[str]]:
    seen = set(seen_ids)
    new_jobs = [j for j in current if j.id not in seen]
    updated_ids = list(seen.union({j.id for j in current}))
    return new_jobs, updated_ids


# ----------------------------
# Email
# ----------------------------

def send_email_gmail(subject: str, body: str) -> None:
    gmail_user = os.getenv("GMAIL_USER")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD")
    email_to = os.getenv("EMAIL_TO")

    if not gmail_user or not gmail_app_password or not email_to:
        raise RuntimeError(
            "Missing email configuration. Set env vars: GMAIL_USER, GMAIL_APP_PASSWORD, EMAIL_TO"
        )

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


def format_email(new_by_site: Dict[str, List[Job]], sites_by_key: Dict[str, SiteConfig]) -> Tuple[str, str]:
    now = datetime.now(timezone.utc).astimezone()
    ts = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    subject = f"[Job Monitor] {sum(len(v) for v in new_by_site.values())} new job(s) found"

    lines: List[str] = []
    lines.append("New job postings detected")
    lines.append(f"Timestamp: {ts}")
    lines.append("")

    for site_key, jobs in new_by_site.items():
        site = sites_by_key[site_key]
        lines.append(f"== {site.name} ==")
        lines.append(f"Source: {site.source_url}")
        for j in jobs:
            lines.append(f"- {j.title}")
            lines.append(f"  {j.url}")
        lines.append("")

    return subject, "\n".join(lines)


# ----------------------------
# Main runner
# ----------------------------

def run() -> int:
    state = load_state(STATE_FILE)
    sites_by_key = {s.site_key: s for s in SITES}

    all_new_by_site: Dict[str, List[Job]] = {}
    state_changed = False

    for site in SITES:
        logging.info("Checking site: %s (%s)", site.name, site.source_url)

        parser = PARSERS.get(site.parser_key)
        if not parser:
            logging.warning("No parser found for site_key=%s", site.site_key)
            continue

        try:
            html = fetch_html(site.source_url)
            current_jobs = parser(html, site.base_url)

            logging.info("Parsed %d potential postings for %s", len(current_jobs), site.site_key)

            seen_ids = state.get(site.site_key, [])
            new_jobs, updated_ids = diff_new_jobs(current_jobs, seen_ids)

            if new_jobs:
                logging.info("Found %d NEW postings for %s", len(new_jobs), site.site_key)
                all_new_by_site[site.site_key] = new_jobs

            if set(updated_ids) != set(seen_ids):
                state[site.site_key] = updated_ids
                state_changed = True

        except Exception as e:
            # Important: one site failing should not stop the run
            logging.exception("Error checking site %s: %s", site.site_key, e)
            continue

    # Send one combined email if anything new across all sites
    if all_new_by_site:
        subject, body = format_email(all_new_by_site, sites_by_key)
        send_email_gmail(subject, body)
        logging.info("Email sent.")

    if state_changed:
        save_state(state, STATE_FILE)
        logging.info("State updated: %s", STATE_FILE)
    else:
        logging.info("No state changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())