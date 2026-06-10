from app.services.events import MemoryEventBus, publish_event, recent_events, reset_event_bus


def test_memory_event_bus_records():
    bus = MemoryEventBus()
    bus.publish("document.test", {"document_id": "1"})
    recent = bus.recent(5)
    assert recent and recent[0]["type"] == "document.test"
    assert recent[0]["payload"]["document_id"] == "1"


def test_publish_event_records_and_never_raises(monkeypatch):
    reset_event_bus()
    publish_event("review.test", {"task_id": "t1"})
    events = recent_events(10)
    assert any(e["type"] == "review.test" for e in events)


def test_recent_events_is_newest_first():
    bus = MemoryEventBus()
    bus.publish("a", {})
    bus.publish("b", {})
    recent = bus.recent(2)
    assert recent[0]["type"] == "b"
