import pytest
import time
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from caption_train.ratelimit import RateLimit, RateLimitContext, RateLimiter


@pytest.fixture
def temp_storage_file():
    """Create a temporary file for testing rate limit storage"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_file = f.name
    yield temp_file
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def rate_limiter(temp_storage_file):
    """Create a RateLimiter instance with temporary storage"""
    return RateLimiter("test_service", temp_storage_file)


def test_rate_limit_dataclass():
    # Test default values
    rate_limit = RateLimit()
    assert rate_limit.minute_limit == 0
    assert rate_limit.hour_limit == 0
    assert rate_limit.day_limit == 0

    # Test custom values
    rate_limit = RateLimit(minute_limit=10, hour_limit=100, day_limit=1000)
    assert rate_limit.minute_limit == 10
    assert rate_limit.hour_limit == 100
    assert rate_limit.day_limit == 1000


def test_rate_limit_context():
    # Create mock limiter
    mock_limiter = MagicMock()

    context = RateLimitContext(mock_limiter, "test_name")

    assert context.limiter == mock_limiter
    assert context.name == "test_name"
    assert not context.executed

    # Test record_usage
    context.record_usage()
    mock_limiter._record_usage.assert_called_once_with("test_name")
    assert context.executed

    # Test calling record_usage again doesn't call _record_usage again
    context.record_usage()
    mock_limiter._record_usage.assert_called_once()  # Still only called once


def test_rate_limit_context_manager():
    # Create mock limiter and context
    mock_limiter = MagicMock()
    mock_context = MagicMock()
    mock_limiter.limit.return_value = mock_context

    context = RateLimitContext(mock_limiter, "test_name")

    # Test __enter__
    result = context.__enter__()
    mock_limiter.limit.assert_called_once_with("test_name")
    assert result == mock_context

    # Test __exit__ with no exception
    context.__exit__(None, None, None)
    mock_context.record_usage.assert_called_once()

    # Test __exit__ with exception
    mock_context.reset_mock()
    context.__exit__(Exception, Exception("test"), None)
    mock_context.record_usage.assert_not_called()


def test_rate_limiter_init(temp_storage_file):
    limiter = RateLimiter("test_service", temp_storage_file)

    assert limiter.name == "test_service"
    assert limiter.storage_file == temp_storage_file
    assert isinstance(limiter.usage, dict)
    assert "test_service" in limiter.usage
    assert limiter.usage["test_service"] == []


def test_rate_limiter_load_usage_new_file(temp_storage_file):
    # Remove the file to test new file creation
    os.unlink(temp_storage_file)

    limiter = RateLimiter("test_service", temp_storage_file)

    assert limiter.usage == {"test_service": []}


def test_rate_limiter_load_usage_existing_file(temp_storage_file):
    # Create a file with existing data
    test_data = {"existing_service": [1234567890.0]}
    with open(temp_storage_file, "w") as f:
        json.dump(test_data, f)

    limiter = RateLimiter("test_service", temp_storage_file)

    assert limiter.usage == test_data


def test_rate_limiter_load_usage_corrupted_file(temp_storage_file):
    # Create a corrupted JSON file
    with open(temp_storage_file, "w") as f:
        f.write("invalid json")

    limiter = RateLimiter("test_service", temp_storage_file)

    assert limiter.usage == {"test_service": []}


def test_rate_limiter_save_usage(rate_limiter, temp_storage_file):
    rate_limiter.usage["test_service"] = [1234567890.0]
    rate_limiter._save_usage()

    # Verify the file was written correctly
    with open(temp_storage_file, "r") as f:
        data = json.load(f)

    assert data == {"test_service": [1234567890.0]}


def test_rate_limiter_cleanup_old_usage(rate_limiter):
    now = time.time()
    day_ago = now - 86400
    two_days_ago = now - 172800

    # Add timestamps: one recent, one old
    rate_limiter.usage["test_service"] = [two_days_ago, day_ago + 100, now]

    rate_limiter._cleanup_old_usage("test_service")

    # Should only keep timestamps from the last day
    assert len(rate_limiter.usage["test_service"]) == 2
    assert two_days_ago not in rate_limiter.usage["test_service"]


def test_rate_limiter_check_rate_limit_no_limits(rate_limiter):
    rate_limit = RateLimit()  # No limits set

    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is True


def test_rate_limiter_check_rate_limit_minute_limit(rate_limiter):
    rate_limit = RateLimit(minute_limit=2)

    # Add 2 recent timestamps (within the minute)
    now = time.time()
    rate_limiter.usage["test_service"] = [now - 30, now - 10]

    # Should pass with 2 requests
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is True

    # Add one more request to exceed the limit
    rate_limiter.usage["test_service"].append(now)
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is False


def test_rate_limiter_check_rate_limit_hour_limit(rate_limiter):
    rate_limit = RateLimit(hour_limit=2)

    # Add 2 recent timestamps (within the hour)
    now = time.time()
    rate_limiter.usage["test_service"] = [now - 1800, now - 900]  # 30min and 15min ago

    # Should pass with 2 requests
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is True

    # Add one more request to exceed the limit
    rate_limiter.usage["test_service"].append(now)
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is False


def test_rate_limiter_check_rate_limit_day_limit(rate_limiter):
    rate_limit = RateLimit(day_limit=2)

    # Add 2 recent timestamps (within the day)
    now = time.time()
    rate_limiter.usage["test_service"] = [now - 43200, now - 21600]  # 12h and 6h ago

    # Should pass with 2 requests
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is True

    # Add one more request to exceed the limit
    rate_limiter.usage["test_service"].append(now)
    result = rate_limiter._check_rate_limit("test_service", rate_limit)
    assert result is False


def test_rate_limiter_record_usage(rate_limiter):
    with patch("time.time", return_value=1234567890.0):
        rate_limiter._record_usage("test_service")

    assert 1234567890.0 in rate_limiter.usage["test_service"]


def test_rate_limiter_limit_success(rate_limiter):
    context = rate_limiter.limit("test_service", minute_limit=10)

    assert isinstance(context, RateLimitContext)
    assert context.limiter == rate_limiter
    assert context.name == "test_service"


def test_rate_limiter_limit_exceeded(rate_limiter):
    # Set up rate limiter to exceed minute limit
    now = time.time()
    rate_limiter.usage["test_service"] = [now - 30, now - 10, now]  # 3 requests in last minute

    with pytest.raises(Exception, match="Rate limit exceeded for test_service"):
        rate_limiter.limit("test_service", minute_limit=2)


def test_rate_limiter_limit_default_name(rate_limiter):
    context = rate_limiter.limit(minute_limit=10)  # No name specified

    assert context.name == "test_service"  # Should use the limiter's default name


def test_rate_limiter_context_manager_success(rate_limiter):
    with rate_limiter as context:
        assert isinstance(context, RateLimitContext)
        assert context.name == "test_service"

    # Should have recorded usage
    assert len(rate_limiter.usage["test_service"]) == 1


def test_rate_limiter_context_manager_with_exception(rate_limiter):
    try:
        with rate_limiter:
            raise ValueError("test exception")
    except ValueError:
        pass

    # Should not have recorded usage due to exception
    assert len(rate_limiter.usage["test_service"]) == 0


def test_rate_limiter_integration(rate_limiter):
    # Test the full workflow with context manager

    # First request should succeed
    with rate_limiter.limit("test_service", minute_limit=2) as context:
        context.record_usage()

    # Second request should succeed
    with rate_limiter.limit("test_service", minute_limit=2) as context:
        context.record_usage()

    # Third request should succeed as well (limit is 2, we've only made 2 requests)
    with rate_limiter.limit("test_service", minute_limit=2) as context:
        context.record_usage()

    # Fourth request should fail due to minute limit (now we have 3 requests > limit of 2)
    with pytest.raises(Exception, match="Rate limit exceeded"):
        with rate_limiter.limit("test_service", minute_limit=2):
            pass
