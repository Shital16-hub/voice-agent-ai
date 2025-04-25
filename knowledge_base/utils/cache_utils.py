"""
Caching utilities for knowledge base queries.
"""
import os
import json
import time
import hashlib
from typing import Dict, Any, Optional

class StreamingResponseCache:
    """
    Cache for storing and retrieving responses to common queries.
    Optimized for low-latency responses in time-sensitive applications.
    """
    
    def __init__(self, cache_dir: str = "./cache", ttl_seconds: int = 86400):
        """
        Initialize the response cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_key(self, query: str) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: Query string
            
        Returns:
            Cache key
        """
        # Normalize query by lowercasing and removing extra whitespace
        normalized_query = " ".join(query.lower().split())
        
        # Create hash for the key
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache if available and not expired.
        
        Args:
            query: Query string
            
        Returns:
            Cached response or None if not found or expired
        """
        key = self._get_key(query)
        
        # Check memory cache first (fastest)
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if entry is expired
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["data"]
            
            # Remove expired entry
            del self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                # Check if entry is expired
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    # Add to memory cache for faster access next time
                    self.memory_cache[key] = entry
                    return entry["data"]
                
                # Remove expired cache file
                os.remove(cache_file)
            except Exception:
                # Ignore errors reading cache
                pass
        
        return None
    
    def set(self, query: str, response: Dict[str, Any]):
        """
        Store a response in the cache.
        
        Args:
            query: Query string
            response: Response data to cache
        """
        key = self._get_key(query)
        
        # Create cache entry
        entry = {
            "timestamp": time.time(),
            "data": response
        }
        
        # Store in memory cache
        self.memory_cache[key] = entry
        
        # Store on disk
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'w') as f:
                json.dump(entry, f)
        except Exception:
            # Ignore errors writing to disk
            pass
    
    def clear(self):
        """Clear the cache."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
        except Exception:
            # Ignore errors clearing cache
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count disk cache entries
        disk_count = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    disk_count += 1
        except Exception:
            pass
        
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": disk_count,
            "ttl_seconds": self.ttl_seconds,
            "cache_dir": self.cache_dir
        }