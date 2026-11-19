"""Local analysis service for the gamaka dashboard. Run: /usr/bin/python3 dashboard/server.py
It listens only on this computer. Uploaded audio, pitch and results are stored in ~/.cache/gamaka_dashboard, outside the workspace, so Live Server does not reload for them."""
import hashlib, json, os, re, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote
sys.dont_write_bytecode = True
import soundfile as sf, uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

HERE = Path(__file__).resolve().parent; VD = HERE.parent; RF = VD / "results/raga_free"; CACHE = Path.home() / ".cache/gamaka_dashboard"
for d in ("audio", "pitch", "results", "jobs", "numba"): (CACHE / d).mkdir(parents=True, exist_ok=True)
ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".aif", ".aiff"}; MAX_BYTES = 500 * 1024 * 1024; ID_RE = re.compile(r"^[0-9a-f]{64}$"); PY = sys.executable
CODE = ["results/raga_free/rf_common.py", "results/raga_free/train_raga_free.py", "results/step3/train_variant.py", "results/two_stage/two_stage.py", "results/postproc/postprocess.py", "stage2_vocal.py", "stage1_vocal.py", "stage2_score.py", "vocal_pitch.py", "../sim_code_extracted/gamaka_sim/pipeline.py", "../sim_code_extracted/gamaka_sim/run_eval.py", "dashboard/analyze.py"]
DATA = ["results/raga_free/chosen_params.csv", "annotations/pass1_events.csv", "metadata.csv"]; TRAINING = CODE[:3] + ["stage2_vocal.py", "../sim_code_extracted/gamaka_sim/pipeline.py", "../sim_code_extracted/gamaka_sim/run_eval.py"]; PITCH_CODE = ["dashboard/extract_pitch.py", "run_crepe_vocal.py"]
HASH_CACHE, JOBS, META, LOCK = {}, {}, {}, threading.Lock(); POOL = ThreadPoolExecutor(1)
app = FastAPI(title="Gamaka dashboard service"); app.add_middleware(CORSMiddleware, allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$", allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "X-File-Name", "Range"], expose_headers=["Accept-Ranges", "Content-Range", "Content-Length"])

def file_hash(p):
    st = p.stat(); key = (str(p), st.st_mtime_ns, st.st_size); got = HASH_CACHE.get(key)
    if got is None:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""): h.update(block)
        got = HASH_CACHE[key] = h.hexdigest()
    return got
def rel(p): return os.path.relpath(p, VD)
def weight_files(): return sorted((RF / "weights").glob("W*/fold_*.pt"))
def manifest():
    out, missing = {}, []
    for r in CODE + DATA:
        p = (VD / r).resolve()
        if p.exists(): out[r] = file_hash(p)
        else: missing.append(r)
    for p in weight_files(): out[rel(p)] = file_hash(p)
    return out, missing
def fingerprint(m): return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
def pitch_fingerprint(): return hashlib.sha256("".join(file_hash(VD / r) for r in PITCH_CODE).encode()).hexdigest()[:16]
def evaluation():
    path = RF / "raga_free_report.txt"
    if not path.exists(): return None
    text = path.read_text(); pat = lambda name: re.search(re.escape(name) + r"\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)", text); out = {}
    for key, name in (("raga_free", "RAGA-FREE, tuned window"), ("raga_aware", "frozen raga-aware (Step 3)")):
        m = pat(name)
        if m: out[key] = dict(event_f1=float(m[1]), frame=float(m[2]), detections=int(m[3]), correct=int(m[4]))
    return dict(out, target=.6, report_modified=path.stat().st_mtime)
def stale_warning():
    w = weight_files()
    if not w: return ["No saved raga-free weights were found."]
    oldest = min(p.stat().st_mtime for p in w); return [f"{r} is newer than the saved weights; retrain to apply training changes." for r in TRAINING if (VD / r).resolve().exists() and (VD / r).resolve().stat().st_mtime > oldest]
def dataset_tracks():
    if not META.get("__tracks"):
        out = {}
        for p in sorted((VD / "audio").glob("*.mp3")): out[file_hash(p)] = dict(track=p.stem, path=str(p), ext=p.suffix)
        META["__tracks"] = out
    return META["__tracks"]
def info_for(aid):
    if not ID_RE.match(aid): raise HTTPException(400, "Bad audio id")
    known = dataset_tracks().get(aid)
    if known: return dict(id=aid, name=f"{known['track']}{known['ext']}", path=known["path"], ext=known["ext"], known=known["track"])
    p = CACHE / "audio" / f"{aid}.json"
    if not p.exists(): raise HTTPException(404, "Audio is no longer available; please add it again")
    return json.loads(p.read_text())
def paths(aid, fp): return CACHE / "results" / f"{aid}_{fp}.json", CACHE / "jobs" / f"{aid}_{fp}.progress.json", CACHE / "jobs" / f"{aid}_{fp}.log", CACHE / "jobs" / f"{aid}_{fp}.manifest.json"
def env():
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NUMBA_CACHE_DIR=str(CACHE / "numba"), TF_CPP_MIN_LOG_LEVEL="2", OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"); return e
def write_progress(p, stage, message, fraction): p.write_text(json.dumps(dict(stage=stage, message=message, fraction=fraction, time=time.time())))
def run_job(aid, fp, info, files):
    result, prog, log, mf = paths(aid, fp); job = JOBS[(aid, fp)]; job.update(state="running", started=time.time()); mf.write_text(json.dumps(files))
    try:
        pitch = ""
        if not info.get("known"):
            pitch = str(CACHE / "pitch" / f"{aid}_{pitch_fingerprint()}.npz")
            if not Path(pitch).exists():
                write_progress(prog, "pitch", "Measuring pitch with CREPE. This usually takes about 40% of the audio length.", .03)
                with open(log, "w") as fh: subprocess.run([PY, str(HERE / "extract_pitch.py"), info["path"], pitch], cwd=VD, env=env(), stdout=fh, stderr=subprocess.STDOUT, check=True)
        cmd = [PY, str(HERE / "analyze.py"), "--audio", info["path"], "--output", str(result), "--progress", str(prog), "--fingerprint", fp, "--manifest", str(mf), "--name", info["name"]] + (["--known", info["known"]] if info.get("known") else ["--pitch", pitch])
        with open(log, "a") as fh: subprocess.run(cmd, cwd=VD, env=env(), stdout=fh, stderr=subprocess.STDOUT, check=True)
        job.update(state="done", finished=time.time())
    except Exception as ex:
        tail = log.read_text(errors="replace").strip().splitlines()[-25:] if log.exists() else [str(ex)]; job.update(state="error", error="\n".join(tail), finished=time.time())
def start(aid):
    info = info_for(aid); files, missing = manifest()
    if missing: raise HTTPException(500, "Missing model files: " + ", ".join(missing))
    fp = fingerprint(files); result, *_ = paths(aid, fp)
    with LOCK:
        job = JOBS.get((aid, fp))
        if result.exists(): JOBS[(aid, fp)] = dict(state="done")
        elif not job or job["state"] == "error": JOBS[(aid, fp)] = dict(state="queued", queued=time.time()); POOL.submit(run_job, aid, fp, info, files)
    META[f"latest:{aid}"] = fp; return fp
@app.get("/api/status")
def status():
    files, missing = manifest(); busy = sum(j["state"] in ("queued", "running") for j in JOBS.values()); return dict(ok=not missing, fingerprint=fingerprint(files), files=files, missing=missing, warnings=stale_warning(), evaluation=evaluation(), busy_jobs=busy, time=time.time())
@app.get("/api/tracks")
def tracks(): return sorted([dict(id=k, name=v["track"]) for k, v in dataset_tracks().items()], key=lambda t: (t["name"] != "VARNAM", int(t["name"].split()[-1]) if t["name"] != "VARNAM" else 0))
@app.post("/api/upload")
async def upload(request: Request):
    name = Path(unquote(request.headers.get("x-file-name", "recording"))).name[:180] or "recording"; ext = Path(name).suffix.lower()
    if ext not in ALLOWED: raise HTTPException(415, "Please use WAV, MP3, FLAC, OGG or AIFF audio.")
    h, size = hashlib.sha256(), 0; fd, tmp = tempfile.mkstemp(dir=CACHE / "audio", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES: raise HTTPException(413, "The file is larger than 500 MB.")
                h.update(chunk); out.write(chunk)
        try: audio = sf.info(tmp)
        except Exception: raise HTTPException(415, "This audio file could not be read.")
        aid = h.hexdigest(); known = dataset_tracks().get(aid)
        if known: os.unlink(tmp); return dict(info_for(aid), duration=sf.info(known["path"]).duration)
        final = CACHE / "audio" / f"{aid}{ext}"; os.replace(tmp, final); (CACHE / "audio" / f"{aid}.json").write_text(json.dumps(dict(id=aid, name=name, path=str(final), ext=ext, known="", duration=audio.duration))); return info_for(aid)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
@app.get("/api/audio-info/{aid}")
def audio_info(aid): i = info_for(aid); return dict(id=aid, name=i["name"], known=i.get("known") or "")
@app.post("/api/analyze/{aid}")
def analyze(aid): return dict(fingerprint=start(aid), job=job_state(aid))
def job_state(aid, fp=None):
    fp = fp or META.get(f"latest:{aid}") or fingerprint(manifest()[0]); result, prog, _, _ = paths(aid, fp); j = JOBS.get((aid, fp)); state = "done" if result.exists() else (j or {}).get("state", "idle"); out = dict(state=state, fingerprint=fp, error=(j or {}).get("error"), started=(j or {}).get("started"))
    if prog.exists() and state in ("running", "queued"):
        try: out.update(json.loads(prog.read_text()))
        except Exception: pass
    return out
@app.get("/api/job/{aid}")
def job(aid): info_for(aid); return job_state(aid)
@app.get("/api/result/{aid}/{fp}")
def result(aid, fp):
    info_for(aid)
    if not re.match(r"^[0-9a-f]{16}$", fp): raise HTTPException(400, "Bad model fingerprint")
    p = paths(aid, fp)[0]
    if not p.exists(): raise HTTPException(404, "Result is not ready")
    return Response(p.read_bytes(), media_type="application/json")
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".aif": "audio/aiff", ".aiff": "audio/aiff"}
@app.get("/api/audio/{aid}")
def audio(aid, request: Request):
    i = info_for(aid); p = Path(i["path"]); size = p.stat().st_size; rng = request.headers.get("range"); start_b, end_b, code = 0, size - 1, 200; m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
    if m and (m[1] or m[2]):
        if m[1]: start_b = int(m[1]); end_b = min(int(m[2]) if m[2] else start_b + 2 ** 21 - 1, size - 1)
        else: start_b = max(0, size - int(m[2]))
        if start_b >= size: raise HTTPException(416, "Range not available")
        code = 206
    with open(p, "rb") as fh: fh.seek(start_b); data = fh.read(end_b - start_b + 1)
    h = {"Accept-Ranges": "bytes", "Content-Length": str(len(data))}
    if code == 206: h["Content-Range"] = f"bytes {start_b}-{start_b + len(data) - 1}/{size}"
    return Response(data, code, h, MIME.get(i["ext"], "application/octet-stream"))
# ------------------------------------------------------------ "Classify a selection" section
# A long-lived classify_worker.py process keeps the Stage 1 library and the detector in memory. It is restarted when any
# file it depends on changes, so this section follows code changes like the detector view. Nothing here writes annotations.
import io, select, uuid  # noqa: E402
from pydantic import BaseModel  # noqa: E402
CLASSIFY_CODE = ["dashboard/classify_core.py", "dashboard/classify_worker.py", "stage1_vocal.py", "vocal_pitch.py", "stage2_vocal.py", "stage2_score.py", "results/two_stage/two_stage.py", "results/postproc/postprocess.py", "results/raga_free/rf_common.py", "results/step3/train_variant.py", "../sim_code_extracted/gamaka_sim/pipeline.py", "../sim_code_extracted/gamaka_sim/run_eval.py", "annotations/pass1_events.csv", "metadata.csv", "results/raga_free/chosen_params.csv"]
CLASSIFY_LOG = CACHE / "jobs" / "classify_requests.log"; PITCH_POOL = ThreadPoolExecutor(1); PITCH_JOBS = {}
def classify_fingerprint():
    h = hashlib.sha256()
    for r in CLASSIFY_CODE:
        p = (VD / r).resolve(); h.update(f"{r}:{file_hash(p) if p.exists() else 'missing'};".encode())
    for p in weight_files(): h.update(f"{rel(p)}:{file_hash(p)};".encode())
    for p in sorted((VD / "results/two_stage/probs").glob("fold_*.npz")): h.update(f"{rel(p)}:{file_hash(p)};".encode())
    return h.hexdigest()[:16]
class ClassifyWorker:
    def __init__(self): self.proc, self.fp, self.lock, self.log = None, None, threading.Lock(), None
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill(); self.proc.wait()
        self.proc = None
    def _read(self, timeout):
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready: self.stop(); raise HTTPException(504, "The classifier did not answer in time; it has been restarted.")
        line = self.proc.stdout.readline()
        if not line: self.stop(); raise HTTPException(500, "The classifier stopped. See " + str(CACHE / "jobs/classify_worker.log"))
        return json.loads(line)
    def _start(self, fp):
        self.stop(); self.log = open(CACHE / "jobs" / "classify_worker.log", "a"); self.log.write(f"\n--- start {time.ctime()} fingerprint {fp}\n"); self.log.flush()
        self.proc = subprocess.Popen([PY, str(HERE / "classify_worker.py")], cwd=VD, env=env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        msg = self._read(300)
        if not msg.get("ready"): self.stop(); raise HTTPException(500, "The classifier could not start:\n" + msg.get("error", ""))
        self.fp = fp
    def call(self, req, timeout=120):
        with self.lock:
            fp = classify_fingerprint(); restarted = self.proc is None or self.proc.poll() is not None or fp != self.fp
            if restarted: self._start(fp)
            self.proc.stdin.write((json.dumps(req) + "\n").encode()); self.proc.stdin.flush(); out = self._read(timeout); out["fingerprint"], out["restarted"] = fp, restarted; return out
WORKER = ClassifyWorker()
def pitch_path(aid): return CACHE / "pitch" / f"{aid}_{pitch_fingerprint()}.npz"
def pitch_job(aid, info):
    final, log = pitch_path(aid), CACHE / "jobs" / f"{aid}.pitch.log"; tmp = final.with_name(f"{final.stem}.{uuid.uuid4().hex}.npz")
    try:
        with open(log, "w") as fh: subprocess.run([PY, str(HERE / "extract_pitch.py"), info["path"], str(tmp)], cwd=VD, env=env(), stdout=fh, stderr=subprocess.STDOUT, check=True)
        os.replace(tmp, final); PITCH_JOBS[aid] = dict(state="done")
    except Exception as ex:
        PITCH_JOBS[aid] = dict(state="error", error="\n".join(log.read_text(errors="replace").strip().splitlines()[-25:]) if log.exists() else str(ex))
    finally:
        if tmp.exists(): tmp.unlink()
def ensure_pitch(aid, info):
    """Dataset recordings already have CREPE pitch. For other audio, reuse the dashboard's cached pitch or measure it once."""
    if info.get("known"): return dict(state="ready")
    if pitch_path(aid).exists(): return dict(state="ready", pitch=str(pitch_path(aid)))
    with LOCK:
        j = PITCH_JOBS.get(aid)
        if j and j["state"] == "error": return j
        if not j or j["state"] == "done": PITCH_JOBS[aid] = j = dict(state="running", started=time.time()); PITCH_POOL.submit(pitch_job, aid, info)
    return dict(j, state="measuring")
def worker_req(info, extra):
    req = dict(extra, audio=info["path"]); req.update(known=info["known"]) if info.get("known") else req.update(pitch=str(pitch_path(info["id"]))); return req
def known_ids(): return {v["track"]: k for k, v in dataset_tracks().items()}
def resolve(audio_id, track):
    if track:
        aid = known_ids().get(track)
        if not aid: raise HTTPException(404, f"Unknown dataset track {track!r}")
        return info_for(aid)
    return info_for(audio_id or "")
@app.get("/api/classify/view/{aid}")
def classify_view(aid):
    info = info_for(aid); p = ensure_pitch(aid, info)
    if p["state"] == "error": raise HTTPException(500, "Pitch could not be measured:\n" + p.get("error", ""))
    if p["state"] != "ready": return dict(state="measuring", started=p.get("started"), message="Measuring pitch with CREPE. This usually takes about 40% of the audio length.")
    out = WORKER.call(worker_req(info, dict(cmd="view")))
    if not out["ok"]: raise HTTPException(422 if "reject" in out else 500, out.get("reject") or out.get("error"))
    return dict(out["result"], state="ready", id=aid, name=info["name"], classifier_fingerprint=out["fingerprint"])
class RegionIn(BaseModel):
    audio_id: str = ""
    track: str = ""
    start_s: float
    end_s: float
    subsequence: bool = False
@app.post("/api/classify_region")
def classify_region(body: RegionIn):
    t0 = time.perf_counter(); info = resolve(body.audio_id, body.track); p = ensure_pitch(info["id"], info)
    if p["state"] != "ready": raise HTTPException(409, "Pitch for this recording is still being measured. Try again when the timeline appears.")
    out = WORKER.call(worker_req(info, dict(cmd="classify", start=body.start_s, end=body.end_s, subsequence=body.subsequence)))
    ms = (time.perf_counter() - t0) * 1000
    with open(CLASSIFY_LOG, "a") as fh: fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{info['name']}\t{body.start_s:.3f}\t{body.end_s:.3f}\tsubsequence={body.subsequence}\t{'ok' if out['ok'] else 'rejected' if 'reject' in out else 'error'}\t{ms:.1f} ms\tworker {out.get('result', {}).get('timing_ms', {}).get('total_ms', '-')} ms\trestarted={out['restarted']}\n")
    print(f"classify_region {info['name']} {body.start_s:.2f}-{body.end_s:.2f} s: {ms:.0f} ms{' (classifier restarted)' if out['restarted'] else ''}", flush=True)
    if not out["ok"]: raise HTTPException(422 if "reject" in out else 500, out.get("reject") or out.get("error"))
    r = out["result"]; ids = known_ids()
    for n in r["nearest"]:
        a = ids.get(n["track"]); n["clip_url"] = f"/api/clip/{a}?start={max(0, n['start'] - .15):.3f}&end={n['end'] + .15:.3f}" if a else None
    r["timing_ms"]["request_ms"] = round(ms, 1); r["classifier_fingerprint"] = out["fingerprint"]; r["audio_id"] = info["id"]; return r
@app.get("/api/clip/{aid}")
def clip(aid, start: float, end: float):
    """A short WAV cut from a recording, for the example play buttons."""
    i = info_for(aid)
    if not (0 <= start < end) or end - start > 30: raise HTTPException(400, "Clips must be between 0 and 30 s long")
    with sf.SoundFile(i["path"]) as f:
        sr = f.samplerate; f.seek(min(int(start * sr), max(0, f.frames - 1))); y = f.read(int((end - start) * sr), dtype="float32", always_2d=True)
    buf = io.BytesIO(); sf.write(buf, y, sr, format="WAV", subtype="PCM_16"); return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})
if __name__ == "__main__":
    port = int(os.environ.get("GAMAKA_DASHBOARD_PORT", "8770")); print(f"Gamaka dashboard service: http://127.0.0.1:{port}", flush=True)
    def warm_classifier():
        try: WORKER.call(dict(cmd="ping"))
        except Exception as ex: print(f"Classifier could not start yet: {getattr(ex, 'detail', ex)}", flush=True)
    threading.Thread(target=warm_classifier, daemon=True).start()   # load the classifier in the background
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
