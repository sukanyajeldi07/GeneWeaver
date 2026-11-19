import pytest

import app as geneweaver


@pytest.fixture
def client():
    geneweaver.app.config["TESTING"] = True

    with geneweaver.app.test_client() as test_client:
        yield test_client


def test_benchmark_api_returns_success(client, monkeypatch):

    monkeypatch.setattr(
        geneweaver,
        "get_gpu_device",
        lambda: ("GPU unavailable for test", False)
    )

    response = client.get("/api/benchmark")

    assert response.status_code == 200

    data = response.get_json()

    assert data["success"] is True
    assert data["benchmark_status"] == "GPU_UNAVAILABLE_CPU_ONLY"
    assert data["gpu_available"] is False


def test_benchmark_runs_cpu_tests(client, monkeypatch):

    monkeypatch.setattr(
        geneweaver,
        "get_gpu_device",
        lambda: ("GPU unavailable for test", False)
    )

    response = client.get("/api/benchmark")
    data = response.get_json()

    assert data["total_tests"] == 4
    assert data["failed_tests"] == 0
    assert data["skipped_tests"] == 4

    for result in data["results"]:
        assert result["cpu_time_ms"] >= 0
        assert result["gpu_time_ms"] is None
        assert result["validation"] == "SKIPPED"


def test_benchmark_repetitions(client, monkeypatch):

    monkeypatch.setattr(
        geneweaver,
        "get_gpu_device",
        lambda: ("GPU unavailable for test", False)
    )

    response = client.get("/api/benchmark")
    data = response.get_json()

    assert data["repetitions"] == 5

    for result in data["results"]:
        assert result["repetitions"] == 5


def test_benchmark_contains_alignment_scores(client, monkeypatch):

    monkeypatch.setattr(
        geneweaver,
        "get_gpu_device",
        lambda: ("GPU unavailable for test", False)
    )

    response = client.get("/api/benchmark")
    data = response.get_json()

    for result in data["results"]:
        assert result["alignment_score"] is not None