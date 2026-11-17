"""Gunicorn settings for production (Render). Not used when you run `python app.py` locally."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
# One worker process: the models are loaded once and the heatmap cache is shared.
# Threads keep pages and the Digital Twin responsive while a scan runs (scans themselves are queued in app.py).
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
# Large histology slides can take a while on a small CPU.
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
