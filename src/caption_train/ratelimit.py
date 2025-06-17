import time
import json
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RateLimit:
    minute_limit: int = 0
    hour_limit: int = 0
    day_limit: int = 0


class RateLimitContext:
    """Context manager for rate-limited operations.

    This class ensures that rate limit usage is only recorded when an operation
    completes successfully (no exception raised). It prevents double-counting
    of usage even if record_usage() is called multiple times.

    Attributes:
        limiter: RateLimiter instance that manages the rate limiting
        name: str - name/key for the rate limited operation
        executed: bool - tracks whether usage has been recorded to prevent double-counting
    """

    def __init__(self, limiter, name):
        self.limiter = limiter
        self.name = name
        self.executed = False

    def record_usage(self):
        """Record usage for this operation (only once per context).

        Safe to call multiple times - will only record once per context instance.
        """
        if not self.executed:
            self.limiter._record_usage(self.name)
            self.executed = True

    def __enter__(self):
        """Enter the context manager - creates nested rate limit context."""
        self.context = self.limiter.limit(self.name)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager - records usage only if no exception occurred.

        Args:
            exc_type: Exception type (None if no exception)
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if exc_type is None:
            self.context.record_usage()


class RateLimiter:
    def __init__(self, name: str, storage_file: str = "rate_limits.json"):
        self.name = name
        self.storage_file = storage_file
        self.usage = self._load_usage()

    def _load_usage(self) -> Dict[str, List[float]]:
        if not os.path.exists(self.storage_file):
            return {self.name: []}

        try:
            with open(self.storage_file, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {self.name: []}
        except (json.JSONDecodeError, IOError):
            return {self.name: []}

    def _save_usage(self) -> None:
        with open(self.storage_file, "w") as f:
            json.dump(self.usage, f)

    def _cleanup_old_usage(self, name: str) -> None:
        now = time.time()
        day_ago = now - 86400

        if name in self.usage:
            self.usage[name] = [t for t in self.usage[name] if t > day_ago]

    def _check_rate_limit(self, name: str, rate_limit: RateLimit) -> bool:
        if name not in self.usage:
            self.usage[name] = []

        self._cleanup_old_usage(name)

        now = time.time()
        timestamps = self.usage[name]

        if rate_limit.minute_limit > 0:
            minute_ago = now - 60
            if sum(1 for t in timestamps if t > minute_ago) > rate_limit.minute_limit:
                return False

        if rate_limit.hour_limit > 0:
            hour_ago = now - 3600
            if sum(1 for t in timestamps if t > hour_ago) > rate_limit.hour_limit:
                return False

        if rate_limit.day_limit > 0:
            day_ago = now - 86400
            if sum(1 for t in timestamps if t > day_ago) > rate_limit.day_limit:
                return False

        return True

    def _record_usage(self, name: str) -> None:
        if name not in self.usage:
            self.usage[name] = []

        self.usage[name].append(time.time())
        self._save_usage()

    def limit(self, name: str = None, minute_limit: int = 0, hour_limit: int = 0, day_limit: int = 0):
        limit_name = name if name is not None else self.name
        rate_limit = RateLimit(minute_limit, hour_limit, day_limit)

        if not self._check_rate_limit(limit_name, rate_limit):
            raise Exception(f"Rate limit exceeded for {limit_name}")

        return RateLimitContext(self, limit_name)

    def __enter__(self):
        self.context = self.limit(self.name)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.context.record_usage()
