import logging
import time
import asyncio
from typing import Callable, Any, Dict, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.state == "OPEN" and self.last_failure_time:
            # Check if reset timeout has passed
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
                logger.info("Circuit breaker moved to HALF-OPEN state")
                return False
            return True
        return False
    
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def record_success(self):
        """Record success and reset the circuit"""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED state")

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
):
    """Retry decorator with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Retry successful on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                        raise
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after error: {str(e)[:100]}... Waiting {delay:.1f}s"
                    )
                    
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception if last_exception else Exception("Unexpected error in retry logic")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Retry successful on attempt {attempt + 1}")
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}: {e}")
                        raise
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after error: {str(e)[:100]}... Waiting {delay:.1f}s"
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception if last_exception else Exception("Unexpected error in retry logic")
        
        # Return appropriate wrapper
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class ErrorHandler:
    """Main error handling class"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
            logger.debug(f"Created new circuit breaker for {name}")
        return self.circuit_breakers[name]
    
    async def execute_with_circuit_breaker(
        self, 
        name: str, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection"""
        breaker = self.get_circuit_breaker(name)
        
        # Check if circuit is open
        if breaker.is_open():
            logger.warning(f"Circuit breaker OPEN for {name}, skipping execution")
            raise CircuitOpenError(f"Circuit breaker open for {name}")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            breaker.record_success()
            return result
            
        except Exception as e:
            # Record failure
            breaker.record_failure()
            logger.error(f"Circuit breaker recorded failure for {name}: {str(e)[:200]}")
            raise
    
    def get_circuit_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers"""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                'state': breaker.state,
                'failure_count': breaker.failure_count,
                'last_failure_time': breaker.last_failure_time,
                'is_open': breaker.is_open()
            }
        return status

class CircuitOpenError(Exception):
    """Custom exception for circuit breaker open state"""
    pass

# Global instance for easy access
error_handler = ErrorHandler()
