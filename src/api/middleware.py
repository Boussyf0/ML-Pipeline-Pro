"""Custom middleware for the FastAPI application."""
import logging
import time
import json
import uuid
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_401_UNAUTHORIZED
import redis


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                "content_length": request.headers.get("content-length", 0)
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration * 1000,
                    "error": str(e)
                }
            )
            raise
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
                "response_size": response.headers.get("content-length", 0)
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers first (load balancers, proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        # Fall back to direct connection
        if hasattr(request.client, "host"):
            return request.client.host
            
        return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting."""
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        """Initialize rate limiting middleware."""
        super().__init__(app)
        self.redis_client = redis_client
        self.memory_store = defaultdict(list)  # Fallback to memory if no Redis
        
        # Rate limit configuration
        self.rate_limits = {
            "/predict": {"requests": 100, "window_seconds": 60},
            "/models": {"requests": 10, "window_seconds": 60},
            "default": {"requests": 1000, "window_seconds": 3600}  # Default hourly limit
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits before processing request."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
            
        # Get client identifier
        client_id = self._get_client_identifier(request)
        
        # Get rate limit for endpoint
        endpoint_limit = self._get_endpoint_limit(request.url.path)
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, endpoint_limit):
            logger.warning(
                f"Rate limit exceeded for client {client_id} on {request.url.path}",
                extra={
                    "client_id": client_id,
                    "endpoint": request.url.path,
                    "limit": endpoint_limit
                }
            )
            
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "error_type": "rate_limit_exceeded",
                    "retry_after": endpoint_limit["window_seconds"]
                },
                headers={"Retry-After": str(endpoint_limit["window_seconds"])}
            )
            
        return await call_next(request)
        
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
            
        # Use user ID if available (from auth)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
            
        # Fall back to IP address
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip and hasattr(request.client, "host"):
            client_ip = request.client.host
            
        return f"ip:{client_ip or 'unknown'}"
        
    def _get_endpoint_limit(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint."""
        # Check for exact match
        if path in self.rate_limits:
            return self.rate_limits[path]
            
        # Check for prefix matches
        for pattern, limit in self.rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return limit
                
        return self.rate_limits["default"]
        
    async def _check_rate_limit(self, client_id: str, limit_config: Dict[str, int]) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        window_start = current_time - limit_config["window_seconds"]
        
        try:
            if self.redis_client:
                # Use Redis for distributed rate limiting
                key = f"rate_limit:{client_id}"
                
                # Get current request count in window
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
                pipe.zcard(key)  # Count current entries
                pipe.zadd(key, {str(current_time): current_time})  # Add current request
                pipe.expire(key, limit_config["window_seconds"])  # Set expiry
                
                results = pipe.execute()
                current_requests = results[1]
                
                return current_requests < limit_config["requests"]
                
            else:
                # Use memory store (not recommended for production)
                requests = self.memory_store[client_id]
                
                # Remove old requests
                self.memory_store[client_id] = [
                    req_time for req_time in requests 
                    if req_time > window_start
                ]
                
                # Check limit
                if len(self.memory_store[client_id]) >= limit_config["requests"]:
                    return False
                    
                # Add current request
                self.memory_store[client_id].append(current_time)
                return True
                
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return True


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API authentication."""
    
    def __init__(self, app):
        """Initialize authentication middleware."""
        super().__init__(app)
        
        # Configure API keys (in production, use environment variables or secure storage)
        self.valid_api_keys = {
            "admin_key_123": {"user_id": "admin", "permissions": ["admin"]},
            "service_key_456": {"user_id": "service", "permissions": ["predict", "read"]},
            "readonly_key_789": {"user_id": "readonly", "permissions": ["read"]}
        }
        
        # Endpoints that don't require authentication
        self.public_endpoints = {
            "/", "/health", "/ready", "/docs", "/redoc", "/openapi.json"
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Authenticate requests."""
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
            
        # Skip authentication for metrics (could be restricted in production)
        if request.url.path == "/metrics":
            return await call_next(request)
            
        # Get API key from header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            logger.warning(
                f"Missing API key for {request.url.path}",
                extra={
                    "client_ip": request.headers.get("X-Forwarded-For", "unknown"),
                    "endpoint": request.url.path
                }
            )
            
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "API key required",
                    "error_type": "authentication_required"
                },
                headers={"WWW-Authenticate": "X-API-Key"}
            )
            
        # Validate API key
        if api_key not in self.valid_api_keys:
            logger.warning(
                f"Invalid API key for {request.url.path}",
                extra={
                    "api_key": api_key[:8] + "..." if len(api_key) > 8 else api_key,
                    "endpoint": request.url.path
                }
            )
            
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Invalid API key",
                    "error_type": "authentication_failed"
                }
            )
            
        # Set user context
        user_info = self.valid_api_keys[api_key]
        request.state.user_id = user_info["user_id"]
        request.state.permissions = user_info["permissions"]
        
        # Check permissions for specific endpoints
        if not self._check_permissions(request):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Insufficient permissions",
                    "error_type": "permission_denied"
                }
            )
            
        return await call_next(request)
        
    def _check_permissions(self, request: Request) -> bool:
        """Check if user has required permissions for endpoint."""
        user_permissions = getattr(request.state, "permissions", [])
        
        # Admin can access everything
        if "admin" in user_permissions:
            return True
            
        # Check endpoint-specific permissions
        path = request.url.path
        method = request.method
        
        # Prediction endpoints
        if path.startswith("/predict"):
            return "predict" in user_permissions
            
        # Model management endpoints
        if path.startswith("/models") and method in ["POST", "DELETE"]:
            return "admin" in user_permissions
            
        # Read-only endpoints
        if method == "GET":
            return "read" in user_permissions or "predict" in user_permissions
            
        return False


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })
        
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for handling downstream failures."""
    
    def __init__(self, app):
        """Initialize circuit breaker."""
        super().__init__(app)
        self.failure_count = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.circuit_open = defaultdict(bool)
        
        # Circuit breaker configuration
        self.failure_threshold = 5
        self.timeout_seconds = 60
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle circuit breaker logic."""
        endpoint = request.url.path
        
        # Check if circuit is open
        if self._is_circuit_open(endpoint):
            logger.warning(f"Circuit breaker open for {endpoint}")
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Service temporarily unavailable",
                    "error_type": "circuit_breaker_open"
                }
            )
            
        try:
            response = await call_next(request)
            
            # Reset failure count on success
            if response.status_code < 500:
                self.failure_count[endpoint] = 0
                self.circuit_open[endpoint] = False
                
            return response
            
        except Exception as e:
            # Record failure
            self._record_failure(endpoint)
            raise
            
    def _is_circuit_open(self, endpoint: str) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open[endpoint]:
            return False
            
        # Check if timeout has passed
        if time.time() - self.last_failure_time[endpoint] > self.timeout_seconds:
            self.circuit_open[endpoint] = False
            self.failure_count[endpoint] = 0
            return False
            
        return True
        
    def _record_failure(self, endpoint: str) -> None:
        """Record a failure for the endpoint."""
        self.failure_count[endpoint] += 1
        self.last_failure_time[endpoint] = time.time()
        
        if self.failure_count[endpoint] >= self.failure_threshold:
            self.circuit_open[endpoint] = True
            logger.error(f"Circuit breaker opened for {endpoint} after {self.failure_count[endpoint]} failures")