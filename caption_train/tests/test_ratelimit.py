import pytest
import time
import json
from caption_train.ratelimit import RateLimiter, RateLimit, RateLimitContext

@pytest.fixture
def temp_file(tmp_path):
    return str(tmp_path / "test_rate_limits.json")

@pytest.fixture
def limiter(temp_file):
    return RateLimiter("test", temp_file)

def test_init(temp_file):
    limiter = RateLimiter("test", temp_file)
    assert limiter.name == "test"
    assert limiter.storage_file == temp_file
    assert isinstance(limiter.usage, dict)
    assert "test" in limiter.usage

def test_load_usage_no_file(temp_file):
    limiter = RateLimiter("test", temp_file)
    assert limiter.usage == {"test": []}

def test_load_usage_with_file(temp_file):
    data = {"test": [1234567890.0]}
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    
    limiter = RateLimiter("test", temp_file)
    assert limiter.usage == {"test": [1234567890.0]}

def test_save_usage(limiter, temp_file):
    limiter.usage["test"] = [1234567890.0]
    limiter._save_usage()
    
    with open(temp_file, 'r') as f:
        data = json.load(f)
    
    assert data == {"test": [1234567890.0]}

def test_cleanup_old_usage(limiter):
    now = time.time()
    day_ago = now - 86400
    older = day_ago - 3600
    
    limiter.usage["test"] = [now, day_ago + 10, older]
    limiter._cleanup_old_usage("test")
    
    assert len(limiter.usage["test"]) == 2
    assert older not in limiter.usage["test"]

def test_check_rate_limit_under_limit(limiter: RateLimiter):
    now = time.time()
    limiter.usage["test"] = [now - 30, now - 40]
    
    result = limiter._check_rate_limit("test", RateLimit(minute_limit=3))
    assert result is True

def test_check_rate_limit_at_limit(limiter: RateLimiter):
    now = time.time()
    limiter.usage["test"] = [now - 10, now - 20, now - 30]
    
    result = limiter._check_rate_limit("test", RateLimit(minute_limit=3))
    assert result is True

def test_check_rate_limit_over_limit(limiter: RateLimiter):
    now = time.time()
    limiter.usage["test"] = [now - 10, now - 20, now - 30, now - 40]
    
    result = limiter._check_rate_limit("test", RateLimit(minute_limit=3))
    assert result is False

def test_check_rate_limit_hour(limiter: RateLimiter):
    now = time.time()
    hour_ago = now - 3600
    
    limiter.usage["test"] = [now - 10, hour_ago + 10, hour_ago - 10]
    
    result = limiter._check_rate_limit("test", RateLimit(hour_limit=2))
    assert result is True
    
    result = limiter._check_rate_limit("test", RateLimit(hour_limit=1))
    assert result is False

def test_check_rate_limit_day(limiter: RateLimiter):
    now = time.time()
    day_ago = now - 86400
    
    limiter.usage["test"] = [now - 10, day_ago + 10, day_ago - 10]
    
    result = limiter._check_rate_limit("test", RateLimit(day_limit=2))
    assert result is True
    
    result = limiter._check_rate_limit("test", RateLimit(day_limit=1))
    assert result is False

def test_record_usage(limiter: RateLimiter):
    limiter._record_usage("test")
    assert len(limiter.usage["test"]) == 1
    
    limiter._record_usage("new_key")
    assert "new_key" in limiter.usage
    assert len(limiter.usage["new_key"]) == 1

def test_limit_under_limit(limiter: RateLimiter):
    context = limiter.limit("test", minute_limit=10)
    assert isinstance(context, RateLimitContext)
    assert context.name == "test"
    assert context.executed is False

def test_limit_over_limit(limiter: RateLimiter):
    now = time.time()
    limiter.usage["test"] = [now - 10, now - 20, now - 30]
    
    with pytest.raises(Exception) as excinfo:
        limiter.limit("test", minute_limit=2)
    
    assert "Rate limit exceeded for test" in str(excinfo.value)

def test_context_manager_usage(limiter: RateLimiter):
    with limiter.limit("test_context", minute_limit=10) as context:
        assert context.executed is False
    
    assert context.executed is True
    assert len(limiter.usage["test_context"]) == 1

def test_rate_limiter_as_context_manager(limiter: RateLimiter):
    with limiter as context:
        assert isinstance(context, RateLimitContext)
        assert context.name == "test"
        assert context.executed is False
    
    assert context.executed is True
    assert len(limiter.usage["test"]) == 1

def test_multiple_limits(limiter: RateLimiter):
    now = time.time()
    minute_ago = now - 60
    hour_ago = now - 3600
    
    limiter.usage["multi"] = [
        now - 10, now - 20, now - 30,  # 3 in last minute
        minute_ago - 10, minute_ago - 20,  # 2 more in last hour
        hour_ago - 10  # 1 more in last day
    ]
    
    # All limits satisfied
    result = limiter._check_rate_limit("multi", RateLimit(minute_limit=5, hour_limit=10, day_limit=10))
    assert result is True
    
    # Minute limit exceeded
    result = limiter._check_rate_limit("multi", RateLimit(minute_limit=2, hour_limit=10, day_limit=10))
    assert result is False
    
    # Hour limit exceeded
    result = limiter._check_rate_limit("multi", RateLimit(minute_limit=5, hour_limit=4, day_limit=10))
    assert result is False
    
    # Day limit exceeded
    result = limiter._check_rate_limit("multi", RateLimit(minute_limit=5, hour_limit=10, day_limit=5))
    assert result is False
