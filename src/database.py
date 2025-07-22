import os
import time
from typing import List, Dict, Any, Optional, Tuple
from config.settings import settings
import logging
from contextlib import contextmanager
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required libraries based on the database type
try:
    from databricks import sql
    from databricks.sql.client import Connection as DatabricksConnection
except ImportError:
    logger.error("Databricks SQL connector not installed. Run 'pip install databricks-sql-connector'")
    raise

class ConnectionPool:
    """Thread-safe connection pool for database connections"""
    def __init__(self, max_connections=8, timeout=30):  # Increased for parallel queries
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections = []
        self.in_use = set()
        self.last_connection_time = 0
        self._lock = threading.RLock()  # Use reentrant lock for nested calls

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool or create a new one"""
        conn = self._acquire_connection()
        try:
            yield conn
        finally:
            self._release_connection(conn)

    def _acquire_connection(self):
        """Acquire a connection from the pool or create a new one (thread-safe)"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            with self._lock:
                # Check if we have existing connections not in use
                for conn in self.connections:
                    if conn not in self.in_use:
                        self.in_use.add(conn)
                        return conn
                
                # If we have room for more connections, create a new one
                if len(self.connections) < self.max_connections:
                    # Rate limit connection creation
                    current_time = time.time()
                    if current_time - self.last_connection_time < 0.5:  # 0.5 second between new connections
                        time.sleep(0.1)
                        continue
                    
                    self.last_connection_time = current_time
                    
                    try:
                        conn = self._create_databricks_connection()
                        self.connections.append(conn)
                        self.in_use.add(conn)
                        logger.debug(f"Created new database connection. Pool size: {len(self.connections)}")
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create database connection: {e}")
                        time.sleep(1)
                        continue
            
            # If we're at max connections, wait a bit and try again
            time.sleep(0.1)
        
        raise TimeoutError("Timeout waiting for database connection")

    def _release_connection(self, conn):
        """Release a connection back to the pool (thread-safe)"""
        with self._lock:
            if conn in self.in_use:
                self.in_use.remove(conn)

    def _create_databricks_connection(self):
        """Create a new Databricks connection"""
        return sql.connect(
            server_hostname=settings.databricks_server_hostname,
            http_path=settings.databricks_http_path,
            access_token=settings.databricks_access_token
        )

    def close_all(self):
        """Close all connections in the pool (thread-safe)"""
        with self._lock:
            for conn in self.connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self.connections = []
            self.in_use = set()

class QueryCache:
    """Simple cache for query results"""
    def __init__(self, max_size=100, ttl=300):  # 5 minutes TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        """Get a value from the cache"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                logger.debug(f"Cache hit for key: {key}")
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key, value):
        """Set a value in the cache"""
        # If cache is full, remove the oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }

class DatabaseManager:
    def __init__(self):
        # Initialize connection pool
        self.pool = ConnectionPool(max_connections=10)
        
        # Initialize query cache
        self.cache = QueryCache()
    
    def _execute_query(self, query, params=None, cache_key=None):
        """Execute a query with connection pooling and optional caching"""
        # Check cache first if a cache key is provided
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        # Use the connection pool to execute the query
        with self.pool.get_connection() as conn:
            try:
                with conn.cursor() as cursor:
                    if params:
                        # Handle list of parameters for IN clause
                        if isinstance(params, list) or isinstance(params, tuple):
                            formatted_query = query
                            for param in params:
                                if isinstance(param, str):
                                    # Replace first ? with the parameter
                                    formatted_query = formatted_query.replace("?", f"'{param}'", 1)
                                else:
                                    # Replace first ? with the parameter
                                    formatted_query = formatted_query.replace("?", f"{param}", 1)
                            cursor.execute(formatted_query)
                        else:
                            # Single parameter
                            if isinstance(params, str):
                                query = query.replace("?", f"'{params}'")
                            else:
                                query = query.replace("?", f"{params}")
                            cursor.execute(query)
                    else:
                        cursor.execute(query)
                    
                    # Fetch results
                    columns = [col[0] for col in cursor.description] if cursor.description else []
                    results = []
                    for row in cursor:
                        results.append(dict(zip(columns, row)))
                
                # Cache results if a cache key is provided
                if cache_key:
                    self.cache.set(cache_key, results)
                
                return results
            
            except Exception as e:
                logger.error(f"Database query error: {e}")
                logger.error(f"Query: {query}")
                logger.error(f"Params: {params}")
                return []
    
    def search_by_spartnumber(self, spartnumber: str) -> List[Dict[str, Any]]:
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE SPARTNUMBER = ?
        """
        cache_key = f"spartnumber_{spartnumber}"
        return self._execute_query(query, (spartnumber,), cache_key)
    
    def search_by_partnumber_like(self, partnumber: str) -> List[Dict[str, Any]]:
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE PARTNUMBER LIKE ?
        """
        return self._execute_query(query, (f"%{partnumber}%",))
    
    def filter_by_class(self, records: List[Dict[str, Any]], class_filter: str = "M") -> List[Dict[str, Any]]:
        return [record for record in records if record['CLASS'] == class_filter]

    def search_parts_wildcard(self, pattern: str) -> List[Dict[str, Any]]:
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE PARTNUMBER LIKE ?
            ORDER BY PARTINDEX
        """
        return self._execute_query(query, (pattern,))
    
    def search_by_spartnumber_wildcard(self, pattern: str) -> List[Dict[str, Any]]:
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE SPARTNUMBER LIKE ?
            ORDER BY PARTINDEX
        """
        return self._execute_query(query, (pattern,))
    
    def get_all_parts(self) -> List[Dict[str, Any]]:
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name}
            ORDER BY PARTINDEX
            LIMIT 1000
        """
        cache_key = "all_parts"
        return self._execute_query(query, cache_key=cache_key)

    def batch_search_by_spartnumber(self, spartnumbers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for multiple part numbers in a single batch query for improved performance.
        
        Args:
            spartnumbers: List of standardized part numbers to search for
            
        Returns:
            Dictionary mapping each part number to its search results
        """
        if not spartnumbers:
            return {}
            
        # Deduplicate the part numbers
        unique_spartnumbers = list(set(spartnumbers))
        
        # Build a query with IN clause for better performance
        placeholders = ", ".join(["?"] * len(unique_spartnumbers))
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE SPARTNUMBER IN ({placeholders})
        """
        
        # Generate a cache key for the batch
        cache_key = f"batch_spartnumber_{'_'.join(sorted(unique_spartnumbers))}"
        
        # Execute the query
        all_results = self._execute_query(query, unique_spartnumbers, cache_key)
        
        # Organize results by SPARTNUMBER
        results_by_partnumber = {}
        for result in all_results:
            spartnumber = result.get('SPARTNUMBER', '')
            if spartnumber in unique_spartnumbers:
                if spartnumber not in results_by_partnumber:
                    results_by_partnumber[spartnumber] = []
                results_by_partnumber[spartnumber].append(result)
        
        # Ensure all requested part numbers have an entry (even if empty)
        for partnumber in unique_spartnumbers:
            if partnumber not in results_by_partnumber:
                results_by_partnumber[partnumber] = []
                
        return results_by_partnumber

    def add_test_part(self, spartnumber: str) -> bool:
        """
        Add a test part to the database for testing purposes.
        This is used in test scripts to ensure we have test data to match against.
        
        In a real production system, this would connect to the database and add a record.
        For our testing purposes, we'll use an in-memory cache to simulate adding test records.
        """
        logger.info(f"Adding test part to mock database: {spartnumber}")
        
        # Create a mock record
        mock_record = {
            "PARTINDEX": f"TEST_{spartnumber}",
            "PARTMFR": "TEST",
            "PARTNUMBER": spartnumber,
            "SPARTNUMBER": spartnumber,
            "partdesc": f"Test part {spartnumber}",
            "CLASS": "TEST"
        }
        
        # Store in cache as if it was retrieved from database
        cache_key = f"spartnumber_{spartnumber}"
        self.cache.set(cache_key, [mock_record])
        
        return True

    def close(self):
        """Close all connections in the pool"""
        self.pool.close_all()

    def search_similar_parts(self, spartnumber: str) -> List[Dict[str, Any]]:
        """
        Search for similar part numbers (variants) including those with different suffixes.
        This is crucial for finding variants like 155C, 155CE, 155CD for a base part number.
        
        Args:
            spartnumber: The base standardized part number to find variants for
            
        Returns:
            List of similar parts
        """
        # First try to find parts that start with the same base
        base_pattern = f"{spartnumber}%"
        base_variants = self.search_by_spartnumber_wildcard(base_pattern)
        
        # Then try to find parts where this part number might be a variant of
        # In case we're searching for a variant but need to find base parts too
        if len(spartnumber) > 3:
            # Only look for base parts if the spartnumber is reasonably long
            # Create patterns for potential base part numbers
            potential_base = spartnumber[:-1]  # Try removing last character
            base_matches = []
            
            # Iteratively try removing characters from the end
            # This helps find base parts when searching for variants
            # For example: searching for "155CE" should also find "155C"
            for i in range(1, min(3, len(spartnumber))):
                potential_base = spartnumber[:-i]
                if len(potential_base) >= 3:  # Ensure base is reasonable length
                    base_matches.extend(self.search_by_spartnumber_wildcard(f"{potential_base}%"))
        else:
            base_matches = []
        
        # Combine results
        all_results = base_variants + base_matches
        
        # Remove duplicates by PARTINDEX
        unique_results = {}
        for result in all_results:
            partindex = result.get('PARTINDEX')
            if partindex and partindex not in unique_results:
                unique_results[partindex] = result
                
        return list(unique_results.values())

    def search_parts_containing(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search for parts where SPARTNUMBER contains the given pattern.
        This supports the core pattern matching functionality.
        
        Args:
            pattern: The pattern to search for within part numbers
            
        Returns:
            List of matching parts
        """
        query = f"""
            SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
            FROM {settings.databricks_table_name} 
            WHERE SPARTNUMBER LIKE ?
            ORDER BY LENGTH(SPARTNUMBER), PARTINDEX
            LIMIT 100
        """
        return self._execute_query(query, (f"%{pattern}%",))

    def bulk_search_multiple_patterns(self, patterns: List[str], limit_per_pattern: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for multiple patterns in a single bulk operation for better performance.
        
        Args:
            patterns: List of patterns to search for
            limit_per_pattern: Maximum results per pattern
            
        Returns:
            Dictionary mapping pattern to results
        """
        if not patterns:
            return {}
        
        # Split patterns into exact and partial searches for optimization
        exact_patterns = [p for p in patterns if len(p) >= 5]  # Longer patterns for exact search
        partial_patterns = [p for p in patterns if len(p) >= 3]  # Shorter patterns for partial search
        
        results = {}
        
        if exact_patterns:
            # Bulk exact search with prioritized ordering
            placeholders = ','.join(['?' for _ in exact_patterns])
            query = f"""
                SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS, 
                       ? as search_pattern, 'exact' as match_type
                FROM {settings.databricks_table_name} 
                WHERE SPARTNUMBER IN ({placeholders})
                ORDER BY 
                    CASE 
                        WHEN class = 'M' THEN 1 
                        WHEN class = 'O' THEN 2 
                        WHEN class = 'V' THEN 3 
                        ELSE 4 
                    END,
                    LENGTH(SPARTNUMBER),
                    PARTINDEX
                LIMIT {limit_per_pattern * len(exact_patterns)}
            """
            
            for pattern in exact_patterns:
                pattern_results = self._execute_query(query, [pattern] + exact_patterns)
                results[pattern] = [r for r in pattern_results if r.get('search_pattern') == pattern]
        
        if partial_patterns:
            # Bulk partial search with LIKE operations
            for pattern in partial_patterns:
                if pattern not in results:
                    query = f"""
                        SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
                        FROM {settings.databricks_table_name} 
                        WHERE SPARTNUMBER LIKE ?
                        ORDER BY 
                            CASE 
                                WHEN class = 'M' THEN 1 
                                WHEN class = 'O' THEN 2 
                                WHEN class = 'V' THEN 3 
                                ELSE 4 
                            END,
                            LENGTH(SPARTNUMBER),
                            PARTINDEX
                        LIMIT {limit_per_pattern}
                    """
                    results[pattern] = self._execute_query(query, (f"%{pattern}%",))
        
        return results

    def bulk_pattern_search_optimized(self, search_tasks: List[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]], str]]:
        """
        Optimized bulk search for multiple pattern tasks with single database operation.
        
        Args:
            search_tasks: List of search task dictionaries
            
        Returns:
            List of tuples: (pattern, results, match_type)
        """
        if not search_tasks:
            return []
        
        # Group tasks by type for efficient bulk processing
        exact_tasks = [task for task in search_tasks if task['type'] == 'exact']
        partial_tasks = [task for task in search_tasks if task['type'] == 'partial']
        variation_tasks = [task for task in search_tasks if task['type'] == 'variation']
        
        results = []
        
        # Process exact matches in bulk
        if exact_tasks:
            exact_patterns = [task['pattern'] for task in exact_tasks]
            if exact_patterns:
                placeholders = ','.join(['?' for _ in exact_patterns])
                query = f"""
                    SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
                    FROM {settings.databricks_table_name} 
                    WHERE SPARTNUMBER IN ({placeholders})
                    ORDER BY 
                        CASE 
                            WHEN class = 'M' THEN 1 
                            WHEN class = 'O' THEN 2 
                            WHEN class = 'V' THEN 3 
                            ELSE 4 
                        END,
                        LENGTH(SPARTNUMBER)
                    LIMIT 200
                """
                
                bulk_exact_results = self._execute_query(query, exact_patterns)
                
                # Map results back to individual tasks
                for task in exact_tasks:
                    pattern = task['pattern']
                    pattern_results = [r for r in bulk_exact_results if r.get('SPARTNUMBER') == pattern]
                    results.append((pattern, pattern_results, task['match_type']))
        
        # Process partial matches efficiently
        for task in partial_tasks:
            pattern = task['pattern']
            query = f"""
                SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
                FROM {settings.databricks_table_name} 
                WHERE SPARTNUMBER LIKE ?
                ORDER BY 
                    CASE 
                        WHEN class = 'M' THEN 1 
                        WHEN class = 'O' THEN 2 
                        WHEN class = 'V' THEN 3 
                        ELSE 4 
                    END,
                    LENGTH(SPARTNUMBER)
                LIMIT 50
            """
            pattern_results = self._execute_query(query, (f"%{pattern}%",))
            results.append((pattern, pattern_results, task['match_type']))
        
        # Process variations in bulk
        if variation_tasks:
            variation_patterns = [task['pattern'] for task in variation_tasks]
            if variation_patterns:
                placeholders = ','.join(['?' for _ in variation_patterns])
                query = f"""
                    SELECT PARTINDEX, PARTMFR, PARTNUMBER, SPARTNUMBER, partdesc, class as CLASS
                    FROM {settings.databricks_table_name} 
                    WHERE SPARTNUMBER IN ({placeholders})
                    ORDER BY 
                        CASE 
                            WHEN class = 'M' THEN 1 
                            WHEN class = 'O' THEN 2 
                            WHEN class = 'V' THEN 3 
                            ELSE 4 
                        END,
                        LENGTH(SPARTNUMBER)
                    LIMIT 150
                """
                
                bulk_variation_results = self._execute_query(query, variation_patterns)
                
                # Map results back to individual tasks
                for task in variation_tasks:
                    pattern = task['pattern']
                    pattern_results = [r for r in bulk_variation_results if r.get('SPARTNUMBER') == pattern]
                    results.append((pattern, pattern_results, task['match_type']))
        
        return results

# Create a singleton instance
db_manager = DatabaseManager()
