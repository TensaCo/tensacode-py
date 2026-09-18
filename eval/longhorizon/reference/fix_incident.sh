#!/bin/bash
# Reference fix for own/service_incident: the config key was renamed and points at a missing file.
set -e
python - <<'PY'
import json
cfg = json.load(open("/app/app/config.json"))
cfg["listen_port"] = cfg.pop("port", 8080)
cfg["data_file"] = "/app/data/orders.csv"
json.dump(cfg, open("/app/app/config.json", "w"), indent=1)
PY
cat > /app/postmortem.md <<'MD'
# Orders API outage

Two faults, both in `/app/app/config.json`:

1. The deploy renamed the port key to `port`, but `server.py` reads `listen_port`, so
   `load_config()` raised `KeyError: 'listen_port'` and the process exited (see logs/orders.log).
2. `data_file` pointed at `/app/data/orders_2026.csv`, which does not exist; the real export is
   `/app/data/orders.csv`. That caused the FileNotFoundError on the manual restart.

Fix: restored the `listen_port` key (8080) and pointed `data_file` at the existing CSV.
The service now answers on 127.0.0.1:8080 for /health, /orders and /orders?status=open.
MD
echo fixed
