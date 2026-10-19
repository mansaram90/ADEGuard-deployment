"""Verify Alertmanager firing/resolved webhook delivery with a synthetic alert.

Use only a local/test Alertmanager routed to the local observer. This bypasses
Prometheus rule timing and never interrupts the application or contacts people.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import sys
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4


def request(url, payload=None):
    headers = {} if payload is None else {"Content-Type": "application/json"}
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    with urlopen(Request(url, data=body, headers=headers), timeout=10) as response:
        if response.status not in (200, 202):
            raise ValueError(f"Unexpected HTTP {response.status}: {url}")
        return response.read()


def wait_for_notification(receiver, instance, status, timeout):
    deadline = monotonic() + timeout
    while True:
        events = json.loads(request(receiver + "/notifications"))["notifications"]
        if any(event["instance"] == instance and event["alert_name"] == "ADEGuardNotificationSmokeTest"
               and event["status"] == status for event in events):
            return
        if monotonic() >= deadline:
            raise ValueError(f"No {status} notification received within {timeout}s")
        sleep(1)


def run_checks(alertmanager, receiver, timeout):
    request(alertmanager + "/-/ready")
    request(receiver + "/health")
    instance = "synthetic-" + uuid4().hex
    now = datetime.now(timezone.utc)
    alert = {
        "labels": {"alertname": "ADEGuardNotificationSmokeTest", "job": "adeguard-smoke", "instance": instance},
        "startsAt": (now - timedelta(seconds=5)).isoformat(),
        "endsAt": (now + timedelta(minutes=10)).isoformat(),
        "annotations": {"summary": "Local synthetic notification test"},
    }
    submitted = False
    resolved = False
    try:
        request(alertmanager + "/api/v2/alerts", [alert])
        submitted = True
        wait_for_notification(receiver, instance, "firing", timeout)
        print("PASS: synthetic firing alert delivered through Alertmanager to receiver", flush=True)
        alert["endsAt"] = datetime.now(timezone.utc).isoformat()
        request(alertmanager + "/api/v2/alerts", [alert])
        resolved = True
        wait_for_notification(receiver, instance, "resolved", timeout)
        print("PASS: synthetic recovery notification delivered to receiver", flush=True)
    finally:
        if submitted and not resolved:
            alert["endsAt"] = datetime.now(timezone.utc).isoformat()
            try:
                request(alertmanager + "/api/v2/alerts", [alert])
            except (URLError, OSError, ValueError) as error:
                print(f"Warning: synthetic alert cleanup failed: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alertmanager-url", default="http://127.0.0.1:9093")
    parser.add_argument("--receiver-url", default="http://127.0.0.1:9001")
    parser.add_argument("--timeout", type=float, default=60, help="Seconds allowed per notification")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        run_checks(args.alertmanager_url.rstrip("/"), args.receiver_url.rstrip("/"), args.timeout)
    except (URLError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
