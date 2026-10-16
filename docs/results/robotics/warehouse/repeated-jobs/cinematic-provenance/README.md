# Cinematic execution provenance

Large text artifacts are stored as lossless gzip files. This changes storage only;
rendering inputs, execution receipts and reported results are unchanged.

`execution-files-sha256.json` is the exact original manifest, preserved byte-for-byte.
Its paths and hashes describe the files at execution time. `compression-map.json`
maps original paths and decompressed hashes to current stored paths and hashes.
`manifest.json` hashes the files actually present in this checkout, including the
original manifest and compression map. Execution receipts intentionally retain
their historical paths; decompress files to restore those paths.

To restore the two large text files, run from this directory in a writable copy:

```bash
gzip -dk tensorcode-six-cycles-final-budget/render-inputs/geometry.json.gz
gzip -dk tensorcode-six-cycles-final-budget/six-cycles-final/stdout.log.gz
```

`-k` keeps the compressed evidence. The commands refuse to overwrite existing
outputs. Verify each restored file against its `decompressed_sha256` in the
compression map or its entry in the original execution manifest. Restored files
are additional working files and are not entries in the current storage manifest.
