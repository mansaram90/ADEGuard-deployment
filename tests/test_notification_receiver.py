"""Real local HTTP checks for the disposable notification observer."""

from copy import deepcopy
import http.client
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from monitoring.notification_receiver import HISTORY_LIMIT, MAX_BODY_BYTES, NotificationServer


@pytest.fixture
def receiver():
    server = NotificationServer(("127.0.0.1", 0))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def payload(status="firing"):
    return {
        "status": status,
        "alerts": [{"status": status, "labels": {
            "alertname": "ADEGuardAPIScrapeFailed", "job": "adeguard-api", "instance": "api:8000",
            "private_label": "private-label-marker",
        }, "annotations": {"description": "private-annotation-marker"}}],
    }


def post(base, data, content_type="application/json"):
    body = data if isinstance(data, bytes) else json.dumps(data).encode()
    return urlopen(Request(base + "/alerts", data=body, headers={"Content-Type": content_type}), timeout=5)


def history(base):
    with urlopen(base + "/notifications", timeout=5) as response:
        return json.load(response)["notifications"]


@pytest.mark.parametrize("status", ["firing", "resolved"])
def test_accepts_notifications_and_records_only_selected_metadata(receiver, status, capsys):
    base, _ = receiver
    with post(base, payload(status)) as response:
        assert response.status == 200
        assert json.load(response)["accepted"] == 1
    records = history(base)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == record["group_status"] == status
    assert record["alert_name"] == "ADEGuardAPIScrapeFailed"
    assert record["instance"] == "api:8000"
    assert record["notification_id"] and record["received_at"]
    assert "private-" not in json.dumps(records)
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == status
    assert "private-" not in output


@pytest.mark.parametrize("data", [[], {}, {"status": "unknown", "alerts": []},
    {"status": "firing", "alerts": []}, {"status": "firing", "alerts": [None]},
    {"status": "firing", "alerts": [{"status": "firing", "labels": {}}]}])
def test_rejects_invalid_notification_structure(receiver, data):
    base, _ = receiver
    with pytest.raises(HTTPError) as error:
        post(base, data)
    assert error.value.code == 400
    assert not history(base)


@pytest.mark.parametrize("body,content_type,expected", [
    (b"not-json", "application/json", 400),
    (b"\xff", "application/json", 400),
    (b"{}", "text/plain", 415),
    (b"x" * (MAX_BODY_BYTES + 1), "application/json", 413),
], ids=["invalid-json", "invalid-utf8", "wrong-content-type", "oversized-body"])
def test_rejects_bad_encoding_content_type_and_oversized_body(receiver, body, content_type, expected):
    base, _ = receiver
    with pytest.raises(HTTPError) as error:
        post(base, body, content_type)
    assert error.value.code == expected
    assert not history(base)


def test_group_is_validated_before_any_record_is_added(receiver):
    base, _ = receiver
    data = payload()
    invalid = deepcopy(data["alerts"][0])
    invalid["labels"]["instance"] = "x" * 257
    data["alerts"].append(invalid)
    with pytest.raises(HTTPError) as error:
        post(base, data)
    assert error.value.code == 400
    assert not history(base)


def test_history_is_bounded_and_individual_status_is_preserved(receiver):
    base, _ = receiver
    data = payload()
    data["alerts"] *= 50
    for _ in range(3):
        with post(base, data):
            pass
    assert len(history(base)) == HISTORY_LIMIT
    data = payload()
    data["alerts"][0]["status"] = "resolved"
    with post(base, data):
        pass
    assert history(base)[-1]["status"] == "resolved"
    assert history(base)[-1]["group_status"] == "firing"


def test_health_unknown_route_and_missing_length(receiver):
    base, server = receiver
    with urlopen(base + "/health") as response:
        assert json.load(response) == {"status": "ok"}
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/unknown")
    assert error.value.code == 404
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.putrequest("POST", "/alerts")
    connection.putheader("Content-Type", "application/json")
    connection.endheaders()
    assert connection.getresponse().status == 411
    connection.close()
