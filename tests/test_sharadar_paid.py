"""Paid-month mechanics against a fake Sharadar API (sharadar.com/llms.txt, 2026-10-08):
full tables come from `GET /data/<table>?years=N`, which answers 302 to a pre-signed zip;
`tickers` is a full snapshot; 401 = key rejected; 403 "Exceeds free tier" = ran as free tier.
All numbers are invented - not Sharadar data.
"""
import io
import zipfile
from urllib.parse import urlparse

import pandas as pd
import pytest

from screener.diagnostics import universe_report
from screener.providers import base
from screener.providers import sharadar as sh
from screener.store import Store
from tests.test_sharadar_sample import _fake_tables

PAID = "paid-key-123456"
TICKERS = ["AAPL", "MSFT", "REIT", "OLDCO"]


def _zip(df: pd.DataFrame, name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"SHARADAR_{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


class _Resp:
    def __init__(self, status=200, text="", content=b"", headers=None):
        self.status_code, self.text, self._content, self.headers = status, text, content, headers or {}

    def iter_content(self, n):
        for i in range(0, len(self._content), n):
            yield self._content[i:i + n]

    def close(self):
        pass


def _tables():
    t = _fake_tables(TICKERS)
    tk = t["tickers"]
    tk.loc[tk["ticker"] == "OLDCO", "isdelisted"] = "Y"          # survivorship: a delisted company
    return t


@pytest.fixture
def paid_api(monkeypatch):
    calls = []
    tables = _tables()

    def fake_get(session, url, params=None, allow_redirects=True, **_):
        table = urlparse(url).path.rsplit("/", 1)[-1]
        p = dict(params or {})
        calls.append((table, p, allow_redirects))
        if p.get("api_key") != PAID:
            raise base.ApiError("HTTP 403 from x: Exceeds free tier", 403, "Exceeds free tier")
        if "years" in p:
            if not allow_redirects:
                return _Resp(302, headers={"Location": "https://files.example-bucket.com/x.zip?sig=abc"})
            return _Resp(200, content=_zip(tables[table], table.upper()))
        df = tables[table]
        if "ticker" in p:
            df = df[df["ticker"].isin(p["ticker"].split(","))]
        if "dimension" in p:
            df = df[df["dimension"] == p["dimension"]]
        if "fields" in p:
            df = df[p["fields"].split(",")]
        return _Resp(200, text=df.to_csv(index=False))

    monkeypatch.setattr(sh, "http_get", fake_get)
    return calls


def test_full_refresh_uses_bulk_route_for_every_table(project, paid_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", PAID)
    snap = sh.SharadarProvider(project, Store(project)).refresh()
    bulk = {t: p["years"] for t, p, _ in paid_api if "years" in p}
    assert set(bulk) == {"tickers", "fundamentals", "daily", "stocks"}    # nothing paged
    assert not [t for t, p, _ in paid_api if "years" not in p]
    assert bulk["fundamentals"] == 10 and bulk["daily"] == 10
    assert bulk["stocks"] == 5                                            # only 730 days are kept
    assert snap.manifest["mode"] == "full"
    sec = snap.read("securities").set_index("ticker")
    assert sec.at["OLDCO", "is_delisted"]
    fin = snap.read("financials")
    assert (fin[fin["ticker"] == "AAPL"]["period_type"] == "FY").sum() == 10
    assert fin.loc[(fin["ticker"] == "AAPL") & (fin["period_type"] == "FY"), "filing_date"].max() \
        == pd.Timestamp("2026-02-10")                                     # AR filing dates survive bulk


def test_universe_report_after_full_refresh(project, paid_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", PAID)
    sh.SharadarProvider(project, Store(project)).refresh()
    lines, checks = universe_report(project, "sharadar", expect={"AAPL": "default", "REIT": "reit",
                                                                  "OLDCO": None, "XOM": None})
    text = "\n".join(lines)
    assert "delisted" in text and "OLDCO" in text
    st = {name: status for name, status, _ in checks}
    assert st["full bulk download (not the sample)"] == "PASS"
    assert st["whole market downloaded"] == "FAIL"                        # 4 fake tickers, not ~15,000
    assert st["reference companies present"] == "FAIL"                    # XOM missing
    assert st["reference companies get the expected profile"] == "PASS"


def test_check_key_passes_with_a_paid_key_and_never_prints_it(project, paid_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", PAID)
    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-key")
    tables = _tables()
    tables_xom = tables["fundamentals"].assign(ticker="XOM")
    monkeypatch.setattr(sh, "http_get", _with_xom(sh.http_get, tables_xom))
    project.settings["providers"]["sharadar"]["min_free_disk_gb"] = 0
    rows = sh.SharadarProvider(project, Store(project)).check_key()
    assert all(ok for _, ok, _ in rows), rows
    assert PAID not in repr(rows)
    bulk = [d for n, ok, d in rows if n.startswith("bulk download")]
    assert len(bulk) == 4 and all("302" in d and "sig=" not in d for d in bulk)
    assert not Store(project).raw_dir("sharadar").exists()               # nothing downloaded


def _with_xom(inner, xom_rows):
    def get(session, url, params=None, **kw):
        if params and params.get("ticker") == "XOM":
            df = xom_rows[xom_rows["dimension"] == params["dimension"]][params["fields"].split(",")]
            return _Resp(200, text=df.to_csv(index=False))
        return inner(session, url, params=params, **kw)
    return get


def test_check_key_explains_free_tier_and_sample_key(project, paid_api, monkeypatch):
    monkeypatch.setenv("SHARADAR_API_KEY", "free-key")
    monkeypatch.setenv("SHARADAR_SAMPLE_API_KEY", "free-key")
    rows = sh.SharadarProvider(project, Store(project)).check_key()
    d = dict((n, (ok, det)) for n, ok, det in rows)
    # the same key is fine (Sharadar upgrades the account key); free-tier access is caught below
    assert d["paid key in .env"][0] is True and "same as the sample key" in d["paid key in .env"][1]
    assert d["paid coverage (XOM, not in free sample)"][0] is False
    assert "FREE tier" in d["paid coverage (XOM, not in free sample)"][1]


def test_http_errors_never_contain_the_key():
    class S:
        def get(self, url, params=None, **_):
            return _Resp(403, text=f"Forbidden for url {url}?api_key={params['api_key']}")

    with pytest.raises(base.ApiError) as e:
        base.http_get(S(), "https://api.sharadar.com/v1.0/data/daily", params={"api_key": "SECRET123"})
    assert "SECRET123" not in str(e.value) and "SECRET123" not in e.value.body
    assert e.value.status_code == 403


def test_bulk_file_that_is_an_error_page_is_rejected(tmp_path):
    p = tmp_path / "x.bulk"
    p.write_text('{"error": "Exceeds free tier"}')
    with pytest.raises(base.ApiError):
        sh.check_bulk_file(p, "SF1")
    z = tmp_path / "ok.bulk"
    z.write_bytes(_zip(pd.DataFrame({"a": [1]}), "T"))
    sh.check_bulk_file(z, "SF1")
