"""Long-lived classifier process for the "Classify a selection" section. Started and restarted by server.py.
Protocol: one JSON request per line on stdin, one JSON reply per line on the original stdout.
Everything else that is printed (library build, warnings) goes to stderr, which server.py writes to a log file."""
import json, os, sys, traceback
sys.dont_write_bytecode = True
proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1); sys.stdout = sys.stderr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def reply(obj):
    proto.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"); proto.flush()


try:
    import classify_core as C
    C.warm()
except Exception:
    reply(dict(ready=False, error=traceback.format_exc()[-4000:])); sys.exit(1)
reply(dict(ready=True, library_size=len(C.TS.LIB_CURVE)))
for line in sys.stdin:
    try:
        reply(dict(ok=True, result=C.handle(json.loads(line))))
    except C.Reject as ex:
        reply(dict(ok=False, reject=str(ex)))
    except Exception:
        reply(dict(ok=False, error=traceback.format_exc()[-4000:]))
