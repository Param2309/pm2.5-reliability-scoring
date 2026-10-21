#!/usr/bin/env python3
"""Pull hourly PurpleAir history (cf_1 and atm, channels A and B) for the three external sites.

Key: read ONLY from env var PURPLEAIR_KEY. It is sent as the X-API-Key header and is never printed or logged.
Usage:
  export PURPLEAIR_KEY=...            # your READ key
  python pull_purpleair_history.py --index BlueMtn=<n> --index FA-CTKo=<n> --index Huntington=<n> --plan      # offline, no key needed
  python pull_purpleair_history.py --index ... --dry-run     # one 1-day request per site, prints points cost and extrapolated total
  python pull_purpleair_history.py --index ...               # full pull (refuses if estimated cost > remaining balance)
Raw JSON: external_cf1/raw/<site>/<start>_<end>.json (one per chunk; existing chunks are skipped).
"""
import os, sys, json, time, argparse, datetime as dt
import urllib.request, urllib.parse, urllib.error

BASE = "https://api.purpleair.com/v1"
FIELDS = "pm2.5_cf_1_a,pm2.5_cf_1_b,pm2.5_atm_a,pm2.5_atm_b,humidity,temperature"
NF = len(FIELDS.split(","))
UTC = dt.timezone.utc
# name -> (site label, start UTC, end UTC inclusive, rows in existing file)
SITES = {
 "BlueMtn":   ("Missoula",  dt.datetime(2021,8,1,0,tzinfo=UTC),  dt.datetime(2021,9,25,0,tzinfo=UTC),   1312),
 "FA-CTKo":   ("NYC_Bronx", dt.datetime(2023,6,1,0,tzinfo=UTC),  dt.datetime(2023,6,14,23,tzinfo=UTC), 335),
 "Huntington":("LA",        dt.datetime(2025,1,1,8,tzinfo=UTC),  dt.datetime(2025,1,16,23,tzinfo=UTC),  375),
}
PAD = dt.timedelta(hours=1)   # 1 h pad each side so edge hours can be aligned later

def chunks(a, b, days):
    a -= PAD; b += PAD; step = dt.timedelta(days=days); out = []
    while a < b:
        e = min(a + step, b); out.append((a, e)); a = e
    return out

def key():
    k = os.environ.get("PURPLEAIR_KEY", "").strip()
    if not k: sys.exit("PURPLEAIR_KEY is not set in the environment.")
    return k

def get(path, params=None):
    url = BASE + path + (("?" + urllib.parse.urlencode(params)) if params else "")
    req = urllib.request.Request(url, headers={"X-API-Key": key()})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        sys.exit(f"HTTP {e.code} from PurpleAir: {body}")
    except Exception as e:
        sys.exit(f"Request failed: {type(e).__name__}")

def hist(idx, a, b):
    return get(f"/sensors/{idx}/history", {"start_timestamp": int(a.timestamp()), "end_timestamp": int(b.timestamp()),
                                           "average": 60, "fields": FIELDS})

def balance():
    j = get("/organization")
    return j.get("remaining_points")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="append", default=[], help="NAME=sensor_index (BlueMtn, FA-CTKo, Huntington)")
    ap.add_argument("--chunk-days", type=float, default=14, help="max span per request for average=60 (PurpleAir limit; lower it if the API rejects 14)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw"))
    ap.add_argument("--plan", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    idx = {}
    for s in a.index:
        n, _, v = s.partition("=")
        if n not in SITES or not v.strip().isdigit(): sys.exit(f"Bad --index {n}: need NAME=integer for one of {list(SITES)}")
        idx[n] = int(v)
    miss = [n for n in SITES if n not in idx]
    if miss: sys.exit(f"Missing sensor indices for: {miss}")
    total_rows = 0; plan = {}
    for n,(lab,s,e,rows) in SITES.items():
        cs = chunks(s, e, a.chunk_days); plan[n] = cs
        hrs = int((e-s).total_seconds()//3600)+1+2   # incl. padding
        total_rows += hrs
        print(f"{lab:10s} {n:11s} idx={idx[n]}  {s:%Y-%m-%d %H:%M} -> {e:%Y-%m-%d %H:%M} UTC  existing rows={rows}  chunks={len(cs)}  expected pulled rows~{hrs}")
    print(f"Expected total rows ~{total_rows} x {NF} fields")
    if a.plan: return
    if a.dry_run:
        b0 = balance(); print(f"Balance before: {b0}")
        pts = []; used_rows = 0
        for n,(lab,s,e,rows) in SITES.items():
            t0 = s; t1 = s + dt.timedelta(days=1)
            j = hist(idx[n], t0, t1); r = len(j.get("data", []))
            b1 = balance(); c = None if (b0 is None or b1 is None) else b0 - b1
            print(f"{lab}: 1-day request returned {r} rows, fields={len(j.get('fields',[]))}, points used={c}, balance now {b1}")
            if c is not None and r: pts.append(c); used_rows += r
            b0 = b1
        if pts and used_rows:
            per_row = sum(pts)/used_rows
            est = per_row*total_rows
            print(f"Points per returned row (all {NF} fields): {per_row:.3f}; estimated full-pull cost ~{est:.0f} points (+ one request base cost per chunk is included only in the per-row average)")
            print(f"Remaining balance: {b0}; " + ("COVERS the estimate." if b0 is not None and b0 >= est else "DOES NOT COVER the estimate. Stop."))
        else:
            print("Could not compute cost (no rows or no balance info).")
        return
    for n,(lab,s,e,rows) in SITES.items():
        d = os.path.join(a.outdir, lab); os.makedirs(d, exist_ok=True)
        for i,(t0,t1) in enumerate(plan[n],1):
            f = os.path.join(d, f"{t0:%Y%m%dT%H}_{t1:%Y%m%dT%H}.json")
            if os.path.exists(f): print(f"{lab} chunk {i}/{len(plan[n])}: exists, skipped"); continue
            j = hist(idx[n], t0, t1)
            with open(f, "w") as fh: json.dump(j, fh)
            print(f"{lab} chunk {i}/{len(plan[n])}: {len(j.get('data',[]))} rows saved"); time.sleep(1)
    print("Done. Upload the raw/ folder.")
main()
