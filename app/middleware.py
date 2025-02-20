import asyncio

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class RetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        retries = 3
        for _ in range(retries):
            response = await call_next(request)
            if response.status_code < 500:
                return response
            await asyncio.sleep(1)
        return response
