"""Shared counting semaphore for concurrency control."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class Semaphore:
    """Classic counting semaphore for concurrency control.

    ``acquire()`` resolves immediately if a slot is free, otherwise queues the
    caller.  ``release()`` unblocks the next waiter in FIFO order.
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError(f"Semaphore max must be at least 1, got {max_concurrent}")
        self._max = max_concurrent
        self._current = 0
        self._queue: deque[asyncio.Future[None]] = deque()

    async def acquire(self) -> None:
        if self._current < self._max:
            self._current += 1
            return

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._queue.append(fut)
        await fut

    def release(self) -> None:
        if self._queue:
            fut = self._queue.popleft()
            if not fut.done():
                fut.set_result(None)
        else:
            self._current -= 1

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        await self.acquire()
        try:
            return await fn()
        finally:
            self.release()

    @property
    def active(self) -> int:
        return self._current

    @property
    def pending(self) -> int:
        return len(self._queue)
