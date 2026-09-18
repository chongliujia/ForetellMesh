"""Read-only archives and primary-source parsers for historical market replay."""
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .data import sha256_bytes, strict_json
from .market_dataset import now
from .schema import ValidationError, iso, timestamp
from .synthetic_sft import canonical_hash

MAX_BYTES = 4_000_000
ALLOWED_HOSTS = {"gamma-api.polymarket.com", "clob.polymarket.com", "data-api.polymarket.com",
                 "external-api.kalshi.com", "www.federalreserve.gov", "assets.kalshi.com", "polygonscan.com"}


def allowed_url(url: str) -> bool:
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port or p.fragment:
        return False
    if p.hostname == "www.federalreserve.gov":
        return bool(re.fullmatch(r"/newsevents/pressreleases/monetary\d{8}a\.htm", p.path)) and not p.query
    if p.hostname == "assets.kalshi.com":
        return p.path in ("/contract_terms/FED.pdf", "/regulatory/product-certifications/FED.pdf") and not p.query
    if p.hostname == "polygonscan.com":
        return bool(re.fullmatch(r"/tx/0x[0-9a-fA-F]{64}", p.path)) and not p.query
    if p.hostname == "gamma-api.polymarket.com":
        return bool(re.fullmatch(r"/events/\d+", p.path)) and not p.query
    if p.hostname == "clob.polymarket.com":
        return bool(re.fullmatch(r"/clob-markets/0x[0-9a-fA-F]{64}", p.path)) and not p.query
    if p.hostname == "data-api.polymarket.com":
        return p.path in ("/v2/resolutions", "/v2/prices-history")
    return bool(re.fullmatch(r"/trade-api/v2/(historical/cutoff|events/[A-Z0-9-]+|series/[A-Z0-9-]+|"
                            r"(?:historical/markets|series/[A-Z0-9-]+/markets)/[A-Z0-9.\-]+/candlesticks)", p.path))


def fetch(url: str) -> tuple[int, bytes, str | None]:
    if not allowed_url(url):
        raise ValidationError("unsupported historical public URL")
    try:
        with urlopen(Request(url, headers={"User-Agent": "ForetellMesh/0.1 public-data-research"}), timeout=25) as response:
            if response.url != url:
                raise ValidationError("unexpected historical response redirect")
            status, raw, date = response.status, response.read(MAX_BYTES + 1), response.headers.get("Date")
    except HTTPError as exc:
        status, raw, date = exc.code, exc.read(MAX_BYTES + 1), exc.headers.get("Date")
    except (URLError, TimeoutError, ConnectionError) as exc:
        # Preserve failure coverage without retry loops or backdating a later fetch.
        status, raw, date = 0, str(exc).encode(), None
    if len(raw) > MAX_BYTES:
        raise ValidationError("historical response exceeds size limit")
    return status, raw, date


class Archive:
    def __init__(self, stage: Path):
        self.stage, self.requests, self.cache = stage, [], {}
        (stage / "raw").mkdir()

    def get(self, url: str, *, json_body: bool = True):
        if url in self.cache:
            return self.cache[url]
        started = now()
        status, raw, date = fetch(url)
        completed = now()
        if timestamp(started, "start") > timestamp(completed, "end"):
            raise ValidationError("backwards request clock")
        name = f"raw/{len(self.requests):04d}.bin"
        (self.stage / name).write_bytes(raw)
        ref = {"url": url, "file": name, "sha256": sha256_bytes(raw), "status": status,
               "started_at": started, "completed_at": completed, "server_date": date}
        self.requests.append(ref)
        obj = None if status != 200 else strict_json(raw.decode()) if json_body else raw
        self.cache[url] = obj
        return obj


def read_archive(root: Path) -> tuple[dict, dict]:
    m = strict_json((root / "manifest.json").read_text())
    if (m.get("schema_version") != "1" or m.get("kind") != "historical_market_capture"
            or canonical_hash(m["requests"]) != m["requests_sha256"]
            or canonical_hash(m["config"]) != m["config_sha256"]):
        raise ValidationError("historical archive manifest mismatch")
    result = {}
    for ref in m["requests"]:
        path = (root / ref["file"]).resolve()
        if not allowed_url(ref["url"]) or not path.is_relative_to(root.resolve()) or ref["url"] in result:
            raise ValidationError("invalid historical archive reference")
        raw = path.read_bytes()
        if sha256_bytes(raw) != ref["sha256"]:
            raise ValidationError("historical response hash mismatch")
        if timestamp(ref["started_at"], "start") > timestamp(ref["completed_at"], "end"):
            raise ValidationError("invalid archive request interval")
        result[ref["url"]] = (ref, raw)
    return m, result


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip and value.strip():
            self.parts.append(value.strip())


def html_text(value: str) -> str:
    parser = TextParser()
    parser.feed(value)
    return "\n".join(parser.parts)


def normalize(value: str) -> str:
    return " ".join(value.split())


def fed_statement(raw: bytes, url: str, expected_time: str) -> dict:
    """Extract the dated statement body, excluding navigation/current related links."""
    if not allowed_url(url) or urlparse(url).hostname != "www.federalreserve.gov":
        raise ValidationError("not an archived Federal Reserve statement")
    text = raw.decode("utf-8-sig")
    date = re.search(r'<p[^>]*class="article__time"[^>]*>(.*?)</p>', text, re.S)
    release = re.search(r'<p[^>]*class="releaseTime"[^>]*>([^<]*)', text, re.S)
    if not date or not release:
        raise ValidationError("official release time missing")
    day = datetime.strptime(normalize(html_text(date[1])), "%B %d, %Y")
    clock = re.search(r"For release at (\d+):(\d+) ([ap])\.m\. (EDT|EST)", normalize(html_text(release[1])))
    if not clock:
        raise ValidationError("unrecognized official release clock")
    hour = int(clock[1]) % 12 + (12 if clock[3] == "p" else 0)
    published = day.replace(hour=hour, minute=int(clock[2]), tzinfo=ZoneInfo("America/New_York"))
    if published.tzname() != clock[4] or iso(published.astimezone(timezone.utc)) != iso(timestamp(expected_time, "expected release")):
        raise ValidationError("official date/time does not match configured release")
    if day.strftime("%Y%m%d") not in url:
        raise ValidationError("release URL/date mismatch")
    updated = re.search(r'<div[^>]*id="lastUpdate"[^>]*>(.*?)</div>', text, re.S)
    if updated:
        last = normalize(html_text(updated[1])).removeprefix("Last Update:").strip()
        if datetime.strptime(last, "%B %d, %Y").date() != day.date():
            raise ValidationError("official archive reports a later revision; historical version required")
    marker = re.search(r'<div class="col-xs-12 col-sm-8 col-md-8">', text[release.end():])
    if not marker:
        raise ValidationError("statement body boundary missing")
    body = text[release.end() + marker.end():]
    body = body.split("For media inquiries", 1)[0]
    paragraphs = [normalize(html_text(x)) for x in re.findall(r"<p(?:\s[^>]*)?>(.*?)</p>", body, re.S)]
    paragraphs = [p for p in paragraphs if p]
    if not 2 <= len(paragraphs) <= 20 or not any("target range" in p for p in paragraphs):
        raise ValidationError("unexpected statement body")
    # A date-specific official release is the historical publication assertion;
    # local retrieval time is separately preserved and never rewritten.
    return {"text": "\n".join(paragraphs), "published_at": iso(published.astimezone(timezone.utc)),
            "source": url, "sha256": sha256_bytes(raw)}


def initialized_question(raw: bytes, *, tx_hash: str, request_id: str, adapter: str) -> dict:
    """Read one successful explorer receipt's ABI-encoded QuestionInitialized log.

This is an explorer attestation, not independent consensus verification. Never
interpret the Data API transaction_hash as a settlement transaction without
checking its actual event type and block time.
"""
    html = raw.decode()
    plain = html_text(html)
    tx = re.search(r"Transaction Hash:\s*(0x[0-9a-fA-F]{64})", plain)
    block = re.search(r"Block:\s*(\d+)", plain)
    clock = re.search(r"id=['\"]showUtcLocalDate['\"][^>]*data-timestamp=['\"](\d+)['\"]", html)
    if not tx or tx[1].lower() != tx_hash.lower() or not block or not clock or not re.search(r"Status:\s*Success", plain):
        raise ValidationError("invalid explorer receipt identity/status/time")
    matches = []
    for event in re.finditer(r"QuestionInitialized \(", plain):
        section = plain[event.start():]
        section = section.split("View Source", 1)[-1]
        qid = re.search(r"1: questionID\s*Dec\s*Decode\s*Hex\s*(?:0x)?([0-9A-Fa-f]{64})", section)
        request = re.search(r"2: requestTimestamp\s*Dec\s*Decode\s*Hex\s*(\d+)", section)
        encoded = re.search(r"Data\s*Dec\s*Hex\s*(0x[0-9a-fA-F]+)", section)
        if not qid or "0x" + qid[1].lower() != request_id.lower():
            continue
        prefix = plain[max(0, event.start() - 500):event.start()].lower()
        if adapter.lower() not in prefix or not request or not encoded:
            raise ValidationError("question log adapter/data mismatch")
        data = bytes.fromhex(encoded[1][2:])
        if len(data) < 160:
            raise ValidationError("truncated question log ABI")
        offset = int.from_bytes(data[:32], "big")
        if offset != 128:
            raise ValidationError("unsupported QuestionInitialized ABI")
        length = int.from_bytes(data[offset:offset + 32], "big")
        if not 1 <= length <= 50_000 or offset + 32 + length > len(data):
            raise ValidationError("invalid ancillary data length")
        ancillary = data[offset + 32:offset + 32 + length].decode("utf-8")
        if int(request[1]) != int(clock[1]):
            raise ValidationError("initialization timestamp differs from block time")
        matches.append({"question_id": request_id, "adapter": adapter.lower(), "transaction_hash": tx_hash,
                        "block_number": int(block[1]), "published_at": iso(datetime.fromtimestamp(int(clock[1]), timezone.utc)),
                        "ancillary_data": ancillary, "ancillary_sha256": sha256_bytes(ancillary.encode())})
    if len(matches) != 1:
        raise ValidationError("expected exactly one matching initialized question")
    return matches[0]
