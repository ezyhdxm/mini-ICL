# Experiment Browser Usage Guide

## Troubleshooting

If experiments are not displayed after opening `experiment_browser.html`, it may be due to browser security restrictions.

### Method 1: Use a Local Web Server (Recommended)

1. Open a terminal and navigate to the `results` folder:
   ```bash
   cd results
   ```

2. Start a Python HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Or Python 2
   python -m SimpleHTTPServer 8000
   ```

3. Open in your browser:
   ```
   http://localhost:8000/experiment_browser.html
   ```

### Method 2: Use VS Code Live Server

If you're using VS Code:
1. Install the "Live Server" extension
2. Right-click on `experiment_browser.html`
3. Select "Open with Live Server"

### Method 3: Use Other Local Servers

```bash
# Node.js (if installed)
npx http-server -p 8000

# Or use other tools
```

## Running the Index

After adding new experiments, remember to re-run the indexing:

```bash
python -m icl.utils.experiment_index
```

This will generate the `results/experiment_index.json` file for the browser to read.

## Verification

1. Confirm that `experiment_index.json` exists in the `results/` folder
2. Open browser developer tools (F12) to check console errors
3. Make sure to use an HTTP server instead of opening the file directly (avoid `file://` protocol)

