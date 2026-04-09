# /arksim ui

Launch the arksim web dashboard for visual exploration of results.

## When to use

- Browsing conversation transcripts visually
- Exploring evaluation results in a dashboard
- Sharing results with teammates who prefer a GUI over CLI output

## Flow

Call the `launch_ui` MCP tool:

```
launch_ui(port=8080)
```

Report the URL to the user:

```
arksim UI is running at http://localhost:8080
```

## Notes

- The UI runs locally. No data leaves the machine.
- The default port is 8080. If that port is in use, pass a different port number.
- The UI reads results from the same output directories configured in `config.yaml`.
- To stop the UI, the user can press Ctrl+C in the terminal or close the process.
