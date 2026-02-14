# Architecture encoding

We represent an architecture as a JSON-serializable dict:

Example:
{
  "stem_channels": 16,
  "stages": [
    {"depth": 2, "out_ch": 16, "block": "mbconv", "k": 3, "exp": 2, "se": 0},
    {"depth": 2, "out_ch": 24, "block": "mbconv", "k": 3, "exp": 4, "se": 1},
    {"depth": 3, "out_ch": 40, "block": "mbconv", "k": 5, "exp": 4, "se": 1}
  ],
  "head_channels": 128,
  "num_classes": 10
}

Why this matters:
- sampling is easy
- logging is easy
- later predictors can ingest it
