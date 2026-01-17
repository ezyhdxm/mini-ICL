# Logging Guide

This project now uses Python's `logging` module instead of `print()` statements for better control over output.

## Benefits of Logging

- **Severity levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Easy to configure**: Change log level without modifying code
- **Better for production**: Can route to files, cloud services, etc.
- **Thread-safe**: Works correctly in multi-threaded applications

## Basic Usage

### In Your Module

```python
from icl.utils.logger import setup_logger

# At module level
logger = setup_logger(__name__)

# In your functions
def my_function(verbose=False):
    logger.info("Starting process...")  # Always shows at INFO level
    logger.debug("Detailed information")  # Only shows at DEBUG level
    logger.warning("Something might be wrong")
    logger.error("An error occurred")
```

### Controlling Verbosity

Instead of `if verbose: print(...)`, use log levels:

```python
# Old way
if verbose:
    print("Computing hiddens...")

# New way
logger.info("Computing hiddens...")  # Shows at INFO level

# Or for detailed debug info:
logger.debug("Processing batch 5/10")  # Only shows when DEBUG enabled
```

## Configuring Log Levels

### Default Behavior

By default, loggers are set to `INFO` level, which shows:
- ✅ INFO messages
- ✅ WARNING messages  
- ✅ ERROR messages
- ✅ CRITICAL messages
- ❌ DEBUG messages (hidden)

### Change Log Level Programmatically

```python
import logging
from icl.utils.logger import setup_logger

# Create logger with DEBUG level
logger = setup_logger(__name__, level=logging.DEBUG)

# Now debug messages will show
logger.debug("This will appear")
```

### Change Log Level Globally

```python
import logging

# Set root logger level
logging.basicConfig(level=logging.DEBUG)

# Or set for specific logger
logging.getLogger('icl.utils.train').setLevel(logging.DEBUG)
```

### Environment Variable (Recommended)

You can also set log level via environment variable:

```bash
# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"
python your_script.py

# Linux/Mac
export LOG_LEVEL=DEBUG
python your_script.py
```

Then in your code:

```python
import os
import logging
from icl.utils.logger import setup_logger

log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
logger = setup_logger(__name__, level=log_level)
```

## Migration from Print Statements

### Before
```python
print("Loading cached hiddens...")
if verbose:
    print(f"Step: {self.step}")
```

### After
```python
logger.info("Loading cached hiddens")
logger.info(f"Step: {self.step}")
```

## Log Levels Reference

| Level    | When to Use                                   | Example                                    |
|----------|-----------------------------------------------|--------------------------------------------|
| DEBUG    | Detailed diagnostic information               | `logger.debug("Processing item 5/100")`    |
| INFO     | General informational messages                | `logger.info("Training started")`          |
| WARNING  | Something unexpected but not an error         | `logger.warning("Using default value")`    |
| ERROR    | An error occurred, but app continues          | `logger.error("Failed to load file")`      |
| CRITICAL | Serious error, app may not continue           | `logger.critical("Out of memory")`         |

## Custom Formatting

You can customize the log format:

```python
from icl.utils.logger import setup_logger

custom_format = '%(asctime)s [%(levelname)s] %(message)s'
logger = setup_logger(__name__, format_string=custom_format)
```

## Files Updated

The following files have been migrated to use logging:
- [src/icl/utils/unified_interface.py](../src/icl/utils/unified_interface.py)
- [src/icl/utils/train.py](../src/icl/utils/train.py)
- [src/icl/utils/notebook_utils.py](../src/icl/utils/notebook_utils.py)
