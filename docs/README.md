# ArkSim Docs

Documentation for ArkSim, built with [Mintlify](https://mintlify.com).

## Running locally

Install the Mintlify CLI and start the dev server:

```bash
npm i -g mint
mint dev
```

The docs will be available at `http://localhost:3000`.

## Doc versioning

All pages live in their respective version folder:

```
docs/
  main/        ← unreleased, active development
  v0.3.x/      ← latest release
  v0.2.0/      ← previous release
```

### Adding a new page

Add the page to `docs/main/` and reference it in the `main` version entry in `docs.json`.
