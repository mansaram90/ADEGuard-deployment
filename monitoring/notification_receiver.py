"""Local webhook observer; bounded in-memory history, not durable delivery."""

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
from uuid import uuid4


MAX_BODY_BYTES = 65536
MAX_ALERTS = 50
HISTORY_LIMIT = 100


def notification_records(payload):
    """Validate the complete group before recording any selected metadata."""
    if not isinstance(payload, dict) or payload.get("status") not in ("firing", "resolved"):
        raise ValueError("Expected an Alertmanager notification status")
    alerts = payload.get("alerts")
    if not isinstance(alerts, list) or not 1 <= len(alerts) <= MAX_ALERTS:
        raise ValueError("Expected between 1 and 50 alerts")
    records = []
    notification_id = uuid4().hex
    received_at = datetime.now(timezone.utc).isoformat()
    for alert in alerts:
        if not isinstance(alert, dict) or alert.get("status") not in ("firing", "resolved"):
            raise ValueError("Invalid alert status")
        labels = alert.get("labels")
        if not isinstance(labels, dict):
            raise ValueError("Expected alert labels")
        values = {}
        for key, limit in (("alertname", 128), ("job", 128), ("instance", 256)):
            value = labels.get(key, "")
            if not isinstance(value, str) or len(value) > limit or (key == "alertname" and not value):
                raise ValueError("Invalid alert metadata")
            values[key] = value
        records.append({
            "event": "alert_notification",
            "notification_id": notification_id,
            "received_at": received_at,
            "group_status": payload["status"],
            "status": alert["status"],
            "alert_name": values["alertname"],
            "job": values["job"],
            "instance": values["instance"],
        })
    return records


class NotificationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address):
        self.notifications = deque(maxlen=HISTORY_LIMIT)
        self.notifications_lock = Lock()
        super().__init__(address, NotificationHandler)


class NotificationHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *args):
        # Do not emit raw URLs or request bodies as access logs.
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok"})
        elif self.path == "/notifications":
            with self.server.notifications_lock:
                records = list(self.server.notifications)
            self.reply(200, {"notifications": records})
        else:
            self.reply(404, {"detail": "Not found"})

    def do_POST(self):
        if self.path != "/alerts":
            self.reply(404, {"detail": "Not found"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self.reply(415, {"detail": "Expected application/json"})
            return
        if self.headers.get("Transfer-Encoding"):
            self.reply(400, {"detail": "Chunked requests are not supported"})
            return
        if self.headers.get("Content-Length") is None:
            self.reply(411, {"detail": "Content-Length required"})
            return
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            self.reply(400, {"detail": "Invalid Content-Length"})
            return
        if length < 1:
            self.reply(400, {"detail": "Empty request body"})
            return
        if length > MAX_BODY_BYTES:
            self.reply(413, {"detail": "Request too large"})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete body")
            records = notification_records(json.loads(body))
        except TimeoutError:
            self.reply(408, {"detail": "Request body timed out"})
            return
        except (ValueError, UnicodeDecodeError):
            self.reply(400, {"detail": "Invalid notification"})
            return
        with self.server.notifications_lock:
            self.server.notifications.extend(records)
        for record in records:
            print(json.dumps(record), flush=True)
        self.reply(200, {"accepted": len(records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()
    server = NotificationServer((args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
