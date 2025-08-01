# Motor Parts Processing System with Bidirectional Sliding Window

A system for processing motor part numbers using bidirectional sliding window search, cleaning them to match database formats, and finding parts information based on various classification criteria.

## 🚀 Live Demo

**[View Live Demo on GitHub Pages](https://gunjansarkar.github.io/partmfr/)**

The live demo shows the interface and provides setup instructions. To use the full functionality, clone and run locally as described below.

## Features

- **Bidirectional Sliding Window**: Improved part number matching by generating candidates from both the beginning and end of the input
- **Unified API**: Single API endpoint that handles both single part processing and batch operations
- **Advanced Scoring System**: Combination of part number and description matching scores
- **Interactive Web UI**: User-friendly interface for searching parts and viewing results
- **Batch Processing**: Support for processing multiple parts at once
- **Early Termination**: Performance optimization for high-confidence matches

## Architecture

This implementation uses a pure Python procedural approach with a bidirectional sliding window search algorithm.

## Scoring System

The part matching system uses a scoring algorithm to evaluate the quality of matches:

### Part Number Score (0-5)
- Measures how well the input part number matches the database part number
- Based on matching characters / total input characters
- Example: "BBM1693" vs "1693" = 4/7 = 57.14% = 3/5 (decimal part ≥ 0.8 rounds up)

### Description Score (0-5)
- Measures how many words from the input description match the database description
- Based on matching words / total input words
- Example: "HOSE-COOLANT SLEEVE" vs "HOSE" = 1/3 = 33.33% = 1/5 (decimal part < 0.8 rounds down)

### Confidence Code
- A two-digit code combining the part number score and description score
- Example: "51" means 5/5 part number match, 1/5 description match

### Noise Detection
- Indicates if the matched part number contains extra characters beyond the input
- Used to identify potentially inaccurate matches
- Example: Input "1693" vs Match "1693B" → Noise = 1 (extra "B")

## Getting Started

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the API server: `python start_api.py` or use the `run_unified_api.bat` script
4. Access the web interface at: `http://localhost:8000`
5. API documentation available at: `http://localhost:8000/docs`

## Testing

You can test the system using the provided scripts:

- Single part test: `python test_bidirectional_search.py`
- API endpoint test: `python test_api_request.py`
- Unified API test: `python test_unified_api.py`

## Examples

### Finding "A4710140022" from input "DDE-A471014002"

The bidirectional sliding window search algorithm successfully finds the target part with high confidence (97.7%) and identifies it as a bidirectional match.

### Key Components

1. **PartProcessor** (`src/processor.py`): 
   - Main processing class that handles the entire workflow
   - Uses bidirectional sliding window for improved part matching
   - Maintains all functionality with simplified implementation

2. **Unified API** (`api/unified_endpoint.py`):
   - Single endpoint for all operations
   - Supports both single part and batch processing
   - Consistent response format for all operations

3. **Pattern Optimizer** (`tools/pattern_optimizer.py`):
   - Tracks and prioritizes pattern matching strategies
   - Uses historical success rates to optimize search order
   - Supports early termination for high-confidence matches

4. **Database Manager** (`src/database.py`):
   - Handles database connections and queries
   - Supports caching for better performance
   - Implements efficient LIKE queries for pattern matching

## Bidirectional Sliding Window Search

The bidirectional sliding window search algorithm generates candidates by removing characters from both the beginning (forward) and end (backward) of the cleaned part number. This improves search accuracy by finding patterns that might be in the middle, beginning, or end of the part number.

### Example:

For input "DDE-A471014002":

1. Forward candidates (left to right):
   - DDEA471014002
   - DEA471014002
   - EA471014002
   - A471014002
   - ...etc.

2. Backward candidates (right to left):
   - DDEA471014002
   - DDEA47101400
   - DDEA4710140
   - ...etc.

This helps find matches where either the prefix or suffix might be different, such as finding "A4710140022" from input "DDE-A471014002".

## Unified API

The unified API provides a single endpoint for all operations:

```json
POST /api/unified
```

### Single Part Processing:

```json
{
    "operation_type": "single_part",
    "part_data": {
        "part_number": "DDE-A471014002",
        "part_description": "Optional description"
    }
}
```

### Batch Processing:

```json
{
    "operation_type": "batch",
    "part_data": [
        {"part_number": "DDE-A471014002"},
        {"part_number": "HENS-22137-1", "part_description": "Some description"}
    ]
}
```
   - Supports batch queries for better performance
   - Includes connection pooling for better performance

4. **API** (`api/main.py`):
   - FastAPI-based interface for processing part numbers
   - Supports single part processing and batch processing
   - Provides health check and statistics endpoints

## Features

- Cleans complex part numbers by removing prefixes/suffixes
- Matches cleaned part numbers against database records
- Filters parts by CLASS (M > O > V priority)
- Validates part descriptions using semantic matching
- Accurate description scoring based on input word matches
- RESTful API interface for integration
- Optimized query patterns for performance

## Setup

1. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

2. Configure environment variables in the `.env` file:
   ```
   # OpenAI API key for agent reasoning
   OPENAI_API_KEY=your_api_key_here
   
   # Database connection parameters (if using Databricks)
   server_hostname=your_databricks_hostname
   http_path=your_databricks_http_path
   access_token=your_databricks_access_token
   databricks_table_name=your_catalog.your_schema.your_table
   ```

3. Run the application:
   ```
   python -m start_api
   ```

4. Access the web interface at http://localhost:8000

## Usage

Submit part numbers through the API or web interface. The system will:

1. Clean the part number by removing prefixes, suffixes, and separators
2. Match against standardized part numbers in the database
3. Filter results based on CLASS priority (M > O > V)
4. Return matching parts with confidence scores

For more details, see the DOCUMENTATION.md file.

1. Set up a Databricks workspace and create a table with the required schema:
   ```sql
   CREATE TABLE your_catalog.your_schema.your_table (
     PARTINDEX BIGINT,
     PARTMFR STRING,
     PARTNUMBER STRING,
     SPARTNUMBER STRING,
     partdesc STRING,
     CLASS STRING
   )
   ```

2. Generate a Databricks access token and configure it in your `.env` file along with other connection parameters.

3. Update the `databricks_table_name` setting in your `.env` file.

4. Run the migration script to transfer data from SQLite to Databricks (if needed):
   ```
   python -m migrate_to_databricks
   ```

5. Verify the Databricks connection works:
   ```
   python -m test_databricks
   ```

### Using SQLite (Legacy)

To use SQLite instead of Databricks, set the following in `config/settings.py`:
```python
use_databricks = False
database_path = "path_to_your_sqlite_db.db"
```

## Performance Optimizations

This project includes several performance optimizations:

1. **Batch Processing**: Processes multiple queries in optimized batches
2. **Connection Pooling**: Reuses database connections for better performance
3. **Query Optimization**: Uses indexed columns and limiting results early
4. **Early Termination**: Stops processing when high-confidence matches are found

To improve performance:
```python
# Performance optimizations are built into the processor
# No additional setup required
```

## API Endpoints

### POST /api/process-part

Process a part number to find matching records.

**Request:**
```json
{
  "part_number": "TYTK09812001BA",
  "part_description": "Optional part description"
}
```

**Response:**
```json
{
  "status": "success",
  "cleaned_part": "TK0981200",
  "search_results": [...],
  "filtered_results": [...],
  "remanufacturer_variants": [...],
  "agent_messages": [...],
  "execution_time": 1.23
}
```

## Testing

To test the application:
```
python -m test_databricks
```

## Testing with Postman

To test the API using Postman:

### API Endpoint
```
http://127.0.0.1:8000/api/process-part
```

### Headers
```
Content-Type: application/json
```

### Request Body (POST)
```json
{
  "part_number": "TYTK09812001BA",
  "part_description": "Alternator"
}
```

### Testing without Description
To test how the system handles requests without a description, simply omit the part_description field:
```json
{
  "part_number": "TYTK09812001BA"
}
```

### Example with BRAB531
To test the specific BRAB531 case:
```json
{
  "part_number": "BRAB531",
  "part_description": "SENSOR"
}
```

### Getting Database Information
You can also use the database info endpoint to check the structure:
```
GET http://127.0.0.1:8000/api/database/info
```

## Migrating from SQLite to Databricks

To migrate existing data from SQLite to Databricks:
```
python -m migrate_to_databricks
```

This will:
1. Replace the SQLite database implementation with Databricks
2. Transfer data from your SQLite database to Databricks
3. Update performance optimizations for Databricks

## Troubleshooting

If you encounter connection issues with Databricks:

1. Verify your connection parameters in the `.env` file
2. Ensure your access token has the necessary permissions
3. Check that your Databricks table exists and has the correct schema
4. Run the test script to diagnose connection issues: `python -m test_databricks`
    {
      "PARTINDEX": 123,
      "CLASS": "M",
      "PARTMFR": "Toyota",
      "PARTNUMBER": "TYTK09812001BA",
      "SPARTNUMBER": "K098120",
      "partdesc": "Alternator - Remanufactured"
    },
    {
      "PARTINDEX": 124,
      "CLASS": "M",
      "PARTMFR": "Toyota",
      "PARTNUMBER": "TYTK09812001BB",
      "SPARTNUMBER": "K098120",
      "partdesc": "Alternator - Remanufactured V2"
    }
  ],
  "input": {
    "partnumber": "TYTK09812001BA",
    "part_description": "Optional part description"
  },
  "cleaning_info": {
    "message": "Found match using strategy: TYTK09812001BA -> K098120"
  }
```

### GET /api/database/info

Get information about the database structure.

## Cleaning Strategies

The system applies multiple cleaning strategies to part numbers:

1. Remove separators (-, _, ., spaces) and leading zeros
2. Extract meaningful patterns (letter + digits, digit sequences)
3. Remove common prefixes: TY, TYT, TYTK, OEM, MFG, etc.
4. Remove common suffixes: BA, NEW, OLD, REV, etc.
5. Look for embedded part numbers within the input
6. Try progressive character removal
7. Extract sequences that match database patterns
8. Handle corrupted data with special characters

## Part Description Matching

When a part description is provided:
1. The system first filters for CLASS = 'M' records
2. It then matches the description contextually with database descriptions
3. Returns records with matching descriptions if found
4. If no matches, returns all CLASS = 'M' records

## UI Troubleshooting

If you can't see any results in the UI after submitting a part number:

1. **Check Browser Console**: 
   - Open browser developer tools (F12 or right-click > Inspect)
   - Go to the Console tab
   - Look for any JavaScript errors or network request failures

2. **Verify API Response**: 
   - In the Network tab of developer tools
   - Find the request to `/api/process-part`
   - Check if the response contains data
   - If the response shows "status": "no_results", the system found no matches based on your criteria

3. **Test Direct API Call**:
   Use this PowerShell command to directly test the API:
   ```powershell
   $headers = @{"Content-Type"="application/json"}
   $body = @{part_number="3304"; part_description="ENGINE MOUNT"} | ConvertTo-Json
   Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/process-part" -Headers $headers -Body $body
   ```

4. **Known UI Display Issue**:
   - The UI may not display results if the API returns status "no_results" instead of "success"
   - This has been fixed in the latest version
   - If you still see this issue, check your browser console for more details
   - Try different part numbers or descriptions

5. **Check OpenAI API Key**:
   - Ensure your `.env` file contains a valid OpenAI API key
   - The system will return an error if the key is missing or invalid

## Testing the API

Use the provided `api_test.py` script to test the API:

```
python -m api_test --all        # Test both single and batch processing
python -m api_test --single     # Test only single part processing
python -m api_test --batch      # Test only batch processing
python -m api_test --verbose    # Show more detailed output
```

The test script includes several test cases that demonstrate the scoring system:

1. Part "3535843501" with description "HOSE-COOLANT SLEEVE":
   - Part Number Score: 5/5 (perfect match)
   - Description Score: 1/5 (only 1 out of 3 input words match)
   - Confidence Code: 51

2. Part "639127" with description "OIL FILTER":
   - Part Number Score: 5/5 (perfect match)
   - Description Score: 0/5 (no words match)
   - Confidence Code: 50
