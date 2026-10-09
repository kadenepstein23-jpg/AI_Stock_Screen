"""Sharadar Core US Equities Bundle provider (licensed, personal use).

LICENSE NOTE — read docs/DATA_LICENSING.md. Under Sharadar's personal-use
license, raw data and any dataset that could reproduce Sharadar tables must be
deleted within 30 days after the subscription ends. This module therefore only
writes into data/ (git-ignored), and `python -m screener purge --provider sharadar`
removes everything it created.

Tables used (Sharadar names):
  TICKERS      security master (category, exchange, sector, SIC, delisted flag)
  SF1          fundamentals; dimension MRY (annual, restated) and MRT (TTM)
  DAILY        daily market cap / enterprise value
  SEP          daily prices (only the last `price_history_days` are kept)

Two download modes:
  sample : free data, used to verify this adapter BEFORE subscribing. The public
           `test-api-key` returns AAPL only; a free Sharadar account (no card)
           gives a key for the DJIA-30 sample — put it in .env as
           SHARADAR_SAMPLE_API_KEY.
  full   : bulk export of each table (requires SHARADAR_API_KEY, paid).

API facts this module relies on (sharadar.com/docs, checked 2026-10-08):
  * base https://api.sharadar.com/v1.0/data/<table>, ?api_key=..., tables
    tickers / fundamentals / daily / stocks; csv by default; limit (default
    10000) + skip/offset pagination for slices only; `years=5|10|full` on the same
    route = bulk download (HTTP 302 to a pre-signed zip, subscription-gated).
    Full tables must use the bulk route, not pagination. `tickers` only publishes
    a full snapshot (any `years` value returns it).
  * 401 = key missing/rejected; 403 "Exceeds free tier" = the request ran as
    free tier (the paid key is not active or not the one being sent).
  * `from`/`to` filter the FILING date (fundamentals) or trade date (daily,
    stocks) and default to ONE YEAR AGO -> every query passes `from`.
  * fundamentals key columns: ticker, dimension, calendardate, `date` (date key
    = filing date; older exports call it `datekey`), reportperiod, lastupdated.
    Dimensions MRY (annual, restated) and MRT (trailing twelve months).
  * daily `marketcap` and `ev` are in USD MILLIONS; fundamentals values in units
    of the reporting currency (USD for US filers).
"""
from __future__ import annotations

import gzip
import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .. import schema
from ..store import Snapshot
from .base import ApiError, Provider, RateLimiter, http_get, redact

log = logging.getLogger(__name__)

TABLE_ENDPOINTS = {"TICKERS": "tickers", "SF1": "fundamentals", "DAILY": "daily", "SEP": "stocks"}

DJIA_SAMPLE = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS", "HD",
               "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG",
               "SHW", "TRV", "UNH", "V", "VZ", "WMT"]

# Sharadar SF1 column -> (canonical column, sign multiplier)
SF1_MAP: Dict[str, tuple] = {
    "revenue": ("revenue", 1), "cor": ("cost_of_revenue", 1), "gp": ("gross_profit", 1),
    "opinc": ("operating_income", 1), "ebit": ("ebit", 1), "ebitda": ("ebitda", 1),
    "depamor": ("d_and_a", 1), "intexp": ("interest_expense", 1), "ebt": ("pretax_income", 1),
    "taxexp": ("income_tax", 1), "netinccmn": ("net_income", 1), "epsdil": ("eps_diluted", 1),
    "shareswadil": ("shares_diluted_wavg", 1), "sharesbas": ("shares_outstanding", 1),
    "cashneq": ("cash", 1), "assetsc": ("current_assets", 1), "liabilitiesc": ("current_liabilities", 1),
    "assets": ("total_assets", 1), "liabilities": ("total_liabilities", 1), "debt": ("total_debt", 1),
    "equity": ("equity", 1), "intangibles": ("intangibles", 1), "accoci": ("aoci", 1), "ncfo": ("operating_cash_flow", 1),
    "capex": ("capex", -1),            # Sharadar reports capex as a negative cash flow
    "fcf": ("free_cash_flow", 1), "sbcomp": ("sbc", 1),
    "ncfdiv": ("dividends_paid", -1),  # negative cash flow in Sharadar
    "ncfcommon": ("net_common_issuance", 1),  # issuance net of repurchases
    "dps": ("dps", 1),
}
SF1_ID_COLS = ["ticker", "dimension", "calendardate", "date", "datekey", "reportperiod", "fiscalperiod",
               "lastupdated"]
# fundamentals cash-flow items that Sharadar reports as negative outflows; the sign is checked
# against the data (see _outflow_sign) rather than assumed
OUTFLOW_FIELDS = {"capex", "ncfdiv"}
# Sharadar fills some items with 0 when they are not separately reported (stock-based compensation
# for JPM, CAT and TRV in the sample, which disclose $3.6B, $242M and $256M) -> missing, not zero
ZERO_MEANS_MISSING = {"sbcomp"}


# ---------------------------------------------------------------------------
# Pure normalization (unit-tested with fixtures; no network)
# ---------------------------------------------------------------------------
def classify_category(category: str) -> (str, str):
    """Map a Sharadar TICKERS `category` to (security_type, share_class)."""
    c = (category or "").lower()
    share_class = "secondary" if "secondary" in c else ("primary" if "primary" in c else "single")
    if "etf" in c or "etn" in c or "etd" in c:
        return "etf", share_class
    if "cef" in c or "fund" in c:
        return "fund", share_class
    if "preferred" in c:
        return "preferred", share_class
    if "warrant" in c:
        return "warrant", share_class
    if "right" in c:
        return "right", share_class
    if "unit" in c:
        return "unit", share_class
    if "adr" in c or "canadian" in c or "foreign" in c:
        return "adr", share_class
    if "common" in c or "stock" in c:
        return "common", share_class
    return "other", share_class


def normalize_tickers(tk: pd.DataFrame) -> pd.DataFrame:
    tk = tk.copy()
    if "table" in tk:
        # older exports label tables SF1/SEP, the current API fundamentals/stocks
        t = tk["table"].astype(str).str.upper()
        keep = t.isin(["SF1", "SEP", "FUNDAMENTALS", "STOCKS"])
        if keep.any():
            tk = tk[keep]
            t = t[keep]
        # prefer the fundamentals row when a ticker appears for several tables
        tk = tk.assign(_f=~t.isin(["SF1", "FUNDAMENTALS"])).sort_values("_f").drop(columns="_f")
        tk = tk.drop_duplicates("ticker", keep="first")
    types = tk.get("category", pd.Series("", index=tk.index)).astype(str).map(classify_category)
    out = pd.DataFrame({
        "ticker": tk["ticker"].astype(str).str.upper(),
        "name": tk.get("name"),
        "exchange": tk.get("exchange"),
        "security_type": [t[0] for t in types],
        "share_class": [t[1] for t in types],
        "is_delisted": tk.get("isdelisted", pd.Series("N", index=tk.index)).astype(str).str.upper().eq("Y"),
        "sector": tk.get("sector"),
        "industry": tk.get("industry"),
        "sic_code": tk.get("siccode"),
        "cik": None,
        "location": tk.get("location"),
        "currency": tk.get("currency", "USD"),
        "source_category": tk.get("category"),
    })
    return schema.conform(out, "securities")


def _datekey(sf1: pd.DataFrame) -> pd.Series:
    """Filing-date column: `date` in the current API, `datekey` in older exports."""
    if "datekey" in sf1:
        return sf1["datekey"]
    if "date" in sf1:
        return sf1["date"]
    return pd.Series(pd.NaT, index=sf1.index)


def _outflow_sign(vals: pd.Series) -> int:
    """-1 when the column is reported as negative outflows (Sharadar's convention), else +1."""
    v = vals.dropna()
    v = v[v != 0]
    if v.empty:
        return -1
    return -1 if (v < 0).mean() >= 0.5 else 1


def _as_reported_filing_dates(sf1: pd.DataFrame) -> pd.Series:
    """(ticker, AR-dimension, reportperiod) -> filing date. Sharadar indexes the restated MR
    dimensions by REPORT PERIOD; only the as-reported AR rows carry the real filing date."""
    ar = sf1[sf1["dimension"].isin(["ARY", "ART"])]
    if ar.empty or "reportperiod" not in ar:
        return pd.Series(dtype="datetime64[ns]")
    key = pd.MultiIndex.from_arrays([ar["ticker"].astype(str).str.upper(), ar["dimension"].str[-1],
                                     pd.to_datetime(ar["reportperiod"], errors="coerce")])
    return pd.Series(pd.to_datetime(_datekey(ar), errors="coerce").values, index=key).groupby(level=[0, 1, 2]).min()


def normalize_sf1(sf1: pd.DataFrame) -> pd.DataFrame:
    """SF1 (MRY + MRT rows) -> canonical financials. MRT keeps only the latest per ticker."""
    sf1 = sf1.copy()
    sf1["dimension"] = sf1["dimension"].astype(str).str.upper()
    filed_ar = _as_reported_filing_dates(sf1)
    sf1 = sf1[sf1["dimension"].isin(["MRY", "MRT"])]
    out = pd.DataFrame({
        "ticker": sf1["ticker"].astype(str).str.upper(),
        "period_type": np.where(sf1["dimension"] == "MRY", "FY", "TTM"),
        "period_end": pd.to_datetime(sf1.get("reportperiod", sf1.get("calendardate")), errors="coerce"),
        "filing_date": pd.to_datetime(_datekey(sf1), errors="coerce"),
    })
    if not filed_ar.empty:
        key = pd.MultiIndex.from_arrays([out["ticker"], sf1["dimension"].str[-1].values, out["period_end"]])
        ar_date = pd.Series(filed_ar.reindex(key).values, index=out.index)
        out["filing_date"] = ar_date.fillna(out["filing_date"])
    for src, (dst, sign) in SF1_MAP.items():
        if src in sf1:
            vals = pd.to_numeric(sf1[src], errors="coerce")
            if src in OUTFLOW_FIELDS:
                sign = _outflow_sign(vals)
            if src in ZERO_MEANS_MISSING:
                vals = vals.mask(vals == 0)
            out[dst] = vals.values * sign
    out = out.sort_values(["ticker", "period_type", "period_end", "filing_date"])
    out = out.drop_duplicates(["ticker", "period_type", "period_end"], keep="last")
    ttm = out[out["period_type"] == "TTM"].groupby("ticker").tail(1)
    out = pd.concat([out[out["period_type"] == "FY"], ttm], ignore_index=True)
    return schema.conform(out, "financials")


def normalize_daily(daily: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    out = pd.DataFrame({
        "ticker": daily["ticker"].astype(str).str.upper(),
        "date": pd.to_datetime(daily["date"], errors="coerce"),
        "market_cap": pd.to_numeric(daily.get("marketcap"), errors="coerce") * scale,
        "enterprise_value": pd.to_numeric(daily.get("ev"), errors="coerce") * scale,
    })
    return schema.conform(out, "market")


def normalize_sep(sep: pd.DataFrame) -> pd.DataFrame:
    close_unadj = sep["closeunadj"] if "closeunadj" in sep else sep["close"]
    out = pd.DataFrame({
        "ticker": sep["ticker"].astype(str).str.upper(),
        "date": pd.to_datetime(sep["date"], errors="coerce"),
        "close": pd.to_numeric(close_unadj, errors="coerce"),
        "close_adj": pd.to_numeric(sep.get("closeadj", sep["close"]), errors="coerce"),
        "close_split_adj": pd.to_numeric(sep["close"], errors="coerce"),
        # dollar volume uses split-adjusted close x split-adjusted volume (consistent units)
        "volume": pd.to_numeric(sep["volume"], errors="coerce")
        * pd.to_numeric(sep["close"], errors="coerce") / pd.to_numeric(close_unadj, errors="coerce"),
    })
    return schema.conform(out, "prices")


DOCUMENTED_DAILY_SCALE = 1e6


def detect_daily_scale(daily: pd.DataFrame, fin_mcap: pd.Series) -> float:
    """Sharadar DAILY marketcap/ev are documented in USD millions (SF1 in units). Confirmed from the
    data when SF1 market caps are available; otherwise the documented scale is used."""
    if daily.empty or fin_mcap.dropna().empty:
        return DOCUMENTED_DAILY_SCALE
    latest = daily.sort_values("date").groupby("ticker")["marketcap"].last()
    ratio = (fin_mcap.reindex(latest.index) / pd.to_numeric(latest, errors="coerce")).dropna()
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.empty:
        return DOCUMENTED_DAILY_SCALE
    med = float(ratio.median())
    return 1e6 if med > 1e4 else 1.0


def thin_daily(daily: pd.DataFrame, as_of: pd.Timestamp, keep_recent_days: int = 400) -> pd.DataFrame:
    """Keep recent rows plus the last trading day of each month (enough for FY-end valuation)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    recent = d["date"] >= as_of - pd.Timedelta(days=keep_recent_days)
    month = d["date"].dt.to_period("M")
    month_end = d.groupby([d["ticker"], month])["date"].transform("max") == d["date"]
    return d[recent | month_end]


# ---------------------------------------------------------------------------
# Provider (network)
# ---------------------------------------------------------------------------
def _read_any(path: Path, usecols: Optional[Iterable[str]] = None, chunksize: Optional[int] = None):
    with open(path, "rb") as f:
        magic = f.read(4)
    kwargs: Dict[str, Any] = dict(low_memory=False)
    if usecols is not None:
        cols = set(usecols)
        kwargs["usecols"] = lambda c: c in cols
    if chunksize:
        kwargs["chunksize"] = chunksize
    if magic[:2] == b"PK":
        zf = zipfile.ZipFile(path)
        member = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        return pd.read_csv(zf.open(member), **kwargs)
    if magic[:2] == b"\x1f\x8b":
        return pd.read_csv(gzip.open(path), **kwargs)
    return pd.read_csv(path, **kwargs)


def check_bulk_file(path: Path, table: str) -> None:
    """A bulk download must be a zip/gzip (or plain CSV). An error page saved to disk would otherwise
    surface later as a confusing parse error."""
    with open(path, "rb") as f:
        head = f.read(400)
    if head[:2] in (b"PK", b"\x1f\x8b"):
        return
    text = head.decode("utf-8", "replace").lstrip()
    if not text or text[0] in "{[<":
        raise ApiError(f"{table} bulk download returned no data file ({path.stat().st_size} bytes): "
                       f"{redact(text[:200])!r}")


def explain_api_error(e: "ApiError") -> str:
    body = (getattr(e, "body", "") or "").lower()
    code = getattr(e, "status_code", 0)
    if code == 401:
        return "HTTP 401: key missing or rejected - check SHARADAR_API_KEY in .env (no quotes or spaces)"
    if code == 403 and "free tier" in body:
        return ("HTTP 403 'Exceeds free tier': the request ran as FREE tier - the paid key is not active "
                "yet, or SHARADAR_API_KEY holds the sample key")
    if code == 403:
        return "HTTP 403: this table or bulk download is not included in the subscription"
    return redact(str(e))


class SharadarProvider(Provider):
    name = "sharadar"

    def _session(self):
        import requests

        return requests.Session()

    def _key(self, sample: bool) -> str:
        if sample:
            # a free-account key (DJIA 30) if configured, else the public key (AAPL only)
            return os.getenv(self.settings.get("sample_api_key_env", "SHARADAR_SAMPLE_API_KEY")) or \
                self.settings.get("sample_api_key", "test-api-key")
        key = os.getenv(self.settings.get("api_key_env", "SHARADAR_API_KEY"))
        if not key:
            raise RuntimeError("SHARADAR_API_KEY is not set. Add it to .env (never commit .env). "
                               "To test the adapter without a subscription use --sample.")
        return key

    def _paginate(self, session, table: str, key: str, params: Dict[str, Any], dest: Path,
                  limiter: RateLimiter, page_size: int = 10000) -> pd.DataFrame:
        """Paginated CSV query (used for TICKERS and for sample mode)."""
        url = f"{self.settings['base_url']}/{TABLE_ENDPOINTS[table]}"
        frames, offset = [], 0
        while True:
            q = dict(params, api_key=key, format="csv", limit=page_size)
            if offset:
                q["offset"] = offset
            r = http_get(session, url, params=q, limiter=limiter)
            from io import StringIO

            df = pd.read_csv(StringIO(r.text), low_memory=False)
            frames.append(df)
            if len(df) < page_size:
                break
            offset += page_size
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        return out

    def _bulk(self, session, table: str, key: str, dest: Path, limiter: RateLimiter,
              years: Optional[Any] = None) -> Path:
        """Bulk export: the API answers 302 with a pre-signed link to a zipped CSV."""
        url = f"{self.settings['base_url']}/{TABLE_ENDPOINTS[table]}"
        years = years or self.settings.get("bulk_years", 10)
        log.info("Downloading %s (years=%s)...", table, years)
        r = http_get(session, url, params={"api_key": key, "years": years}, limiter=limiter,
                     stream=True, timeout=1800)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        check_bulk_file(dest, table)
        log.info("Downloaded %s (%.1f MB)", table, dest.stat().st_size / 1e6)
        return dest

    def check_key(self) -> List[tuple]:
        """Cheap checks that the PAID key works, without downloading any table:
        (check, ok, detail) rows. Nothing printed here ever contains the key."""
        import shutil
        from urllib.parse import urlparse

        rows: List[tuple] = []
        try:
            key = self._key(sample=False)
        except RuntimeError as e:
            return [("paid key in .env", False, str(e))]
        sample_key = os.getenv(self.settings.get("sample_api_key_env", "SHARADAR_SAMPLE_API_KEY"))
        same = bool(sample_key) and sample_key.strip() == key.strip()
        # Sharadar upgrades the account's existing key when you subscribe, so the paid key may equal
        # the sample key. Paid access is decided by the coverage and bulk checks below, not by this.
        rows.append(("paid key in .env", True,
                     f"set ({len(key.strip())} characters" +
                     ("; same as the sample key - fine if the checks below pass)" if same else ")")))
        if key != key.strip():
            rows.append(("key formatting", False, "the key has leading/trailing spaces or quotes in .env"))
        session, limiter = self._session(), RateLimiter(2.0)
        years = int(self.settings.get("bulk_years", 10))
        today = pd.Timestamp.today().normalize()
        base = self.settings["base_url"]

        # 1) a company outside the free DJIA-30 sample, with the full history the tier should give
        try:
            r = http_get(session, f"{base}/fundamentals", limiter=limiter, retries=2, params={
                "api_key": key, "ticker": "XOM", "dimension": "MRY", "format": "csv",
                "from": (today - pd.DateOffset(years=years + 2)).date().isoformat(),
                "fields": "ticker,dimension,reportperiod,revenue"})
            from io import StringIO

            df = pd.read_csv(StringIO(r.text))
            per = pd.to_datetime(df.get("reportperiod"), errors="coerce").dropna()
            ok = len(per) >= years - 1
            rows.append(("paid coverage (XOM, not in free sample)", ok,
                         f"{len(per)} annual reports, FY{per.min().year}-FY{per.max().year}" if len(per)
                         else "no rows returned"))
        except ApiError as e:
            rows.append(("paid coverage (XOM, not in free sample)", False, explain_api_error(e)))

        # 2) bulk route for each table: expect a 302 to a pre-signed download (not followed here)
        for table in ("TICKERS", "SF1", "DAILY", "SEP"):
            y = self._years_for(table)
            label = f"bulk download: {TABLE_ENDPOINTS[table]} (years={y})"
            try:
                r = http_get(session, f"{base}/{TABLE_ENDPOINTS[table]}", limiter=limiter, retries=2,
                             params={"api_key": key, "years": y}, allow_redirects=False, stream=True)
                code = getattr(r, "status_code", 0)
                loc = (getattr(r, "headers", {}) or {}).get("Location", "")
                if code in (301, 302, 303, 307, 308) and loc:
                    rows.append((label, True, f"HTTP {code} -> download link on {urlparse(loc).netloc}"))
                elif code == 200:
                    ctype = (getattr(r, "headers", {}) or {}).get("Content-Type", "?")
                    rows.append((label, "zip" in ctype or "octet" in ctype or "csv" in ctype,
                                 f"HTTP 200, {ctype}"))
                else:
                    rows.append((label, False, f"unexpected HTTP {code}"))
                try:
                    r.close()
                except Exception:
                    pass
            except ApiError as e:
                rows.append((label, False, explain_api_error(e)))

        # 3) local disk space for the zipped downloads + parquet store
        root = self.cfg.data_dir
        while not root.exists() and root != root.parent:
            root = root.parent
        free_gb = shutil.disk_usage(root).free / 1e9
        need = float(self.settings.get("min_free_disk_gb", 8))
        rows.append(("free disk space", free_gb >= need, f"{free_gb:.0f} GB free (need about {need:.0f} GB)"))
        return rows

    def _years_for(self, table: str):
        if table == "SEP":   # only `price_history_days` of prices are kept: no need for 10 years
            return self.settings.get("price_bulk_years", 5)
        return self.settings.get("bulk_years", 10)

    def refresh(self, sample: bool = False, tickers: Optional[List[str]] = None, **_: Any) -> Snapshot:
        key = self._key(sample)
        session = self._session()
        limiter = RateLimiter(2.0)
        snap = self.store.new_snapshot(self.name, extra={
            "source": "Sharadar Core US Equities Bundle" + (" (sample)" if sample else ""),
            "mode": "sample" if sample else "full"})
        raw = self.store.raw_dir(self.name, snap.snapshot_id)
        as_of = snap.as_of
        px_days = int(self.settings.get("price_history_days", 730))

        if sample or tickers:
            if sample and key == self.settings.get("sample_api_key", "test-api-key") and not tickers:
                log.warning("Using the public test key: Sharadar returns AAPL only. For the DJIA-30 "
                            "sample create a free account and set SHARADAR_SAMPLE_API_KEY in .env.")
                tickers = ["AAPL"]
            tick = ",".join(tickers or DJIA_SAMPLE)
            # `from` defaults to one year ago on every table: ask for the full history explicitly
            hist_from = (as_of - pd.DateOffset(years=int(self.settings.get("bulk_years", 10)) + 2)).date().isoformat()
            px_from = (as_of - pd.Timedelta(days=px_days)).date().isoformat()
            tk = self._paginate(session, "TICKERS", key, {"ticker": tick}, raw / "tickers.csv", limiter)
            sf1 = self._paginate(session, "SF1", key, {"ticker": tick, "from": hist_from},
                                 raw / "sf1.csv", limiter)
            daily = self._paginate(session, "DAILY", key, {"ticker": tick, "from": hist_from},
                                   raw / "daily.csv", limiter)
            sep = self._paginate(session, "SEP", key, {"ticker": tick, "from": px_from}, raw / "sep.csv", limiter)
        else:
            # full tables come from the bulk route (Sharadar: do not page a full table)
            tk = _read_any(self._bulk(session, "TICKERS", key, raw / "tickers.bulk", limiter))
            sf1_path = self._bulk(session, "SF1", key, raw / "sf1.bulk", limiter)
            parts = []  # keep only the dimensions we use (MRY annual, MRT trailing) to limit memory
            for chunk in _read_any(sf1_path, usecols=SF1_ID_COLS + list(SF1_MAP) + ["marketcap"],
                                   chunksize=500_000):
                dim = chunk["dimension"].astype(str).str.upper()
                mr = chunk[dim.isin(["MRY", "MRT"])]
                ar = chunk.loc[dim.isin(["ARY", "ART"]), [c for c in SF1_ID_COLS if c in chunk]]
                parts.extend([mr, ar])                   # AR rows: filing dates only
            sf1 = pd.concat(parts, ignore_index=True)
            daily_parts = []
            for chunk in _read_any(self._bulk(session, "DAILY", key, raw / "daily.bulk", limiter),
                                   usecols=["ticker", "date", "marketcap", "ev"], chunksize=2_000_000):
                daily_parts.append(thin_daily(chunk, as_of))
            daily = pd.concat(daily_parts, ignore_index=True)
            sep_parts = []
            cutoff = as_of - pd.Timedelta(days=px_days)
            for chunk in _read_any(self._bulk(session, "SEP", key, raw / "sep.bulk", limiter,
                                                           years=self._years_for("SEP")),
                                   usecols=["ticker", "date", "close", "closeadj", "closeunadj", "volume"],
                                   chunksize=2_000_000):
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                sep_parts.append(chunk[chunk["date"] >= cutoff])
            sep = pd.concat(sep_parts, ignore_index=True)

        securities = normalize_tickers(tk)
        financials = normalize_sf1(sf1)
        mrt_mcap = None
        if "marketcap" in sf1:
            m = sf1[sf1["dimension"].astype(str).str.upper() == "MRT"].assign(_dk=_datekey(sf1)).sort_values("_dk")
            mrt_mcap = pd.to_numeric(m.groupby("ticker")["marketcap"].last(), errors="coerce")
        scale = detect_daily_scale(daily, mrt_mcap if mrt_mcap is not None else pd.Series(dtype=float))
        snap.manifest["daily_value_scale"] = scale
        if not (sample or tickers):
            daily = thin_daily(daily, as_of)
        snap.write("securities", securities)
        snap.write("financials", financials)
        snap.write("market", normalize_daily(daily, scale))
        snap.write("prices", normalize_sep(sep))
        if not daily.empty:
            snap.manifest["as_of"] = str(pd.to_datetime(daily["date"]).max().date())
            snap._save_manifest()
        self.store.activate(snap)
        return snap
