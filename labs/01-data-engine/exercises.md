# Exercises — Lab 01

Work these in order; each one deepens a stage of the pipeline.

## 1. Tune the LSH S-curve
`near_dedup` uses `num_perm=64` split into `bands=16, rows=4`. The probability that two docs of similarity *s* become candidate pairs is `P(s) = 1 - (1 - s**r)**b`.

- Plot `P(s)` for `(b=16, r=4)` vs `(b=8, r=8)` vs `(b=32, r=2)` across `s` from 0 to 1.
- Where does each configuration put the ~50% threshold? Which would you pick to catch reposts (s≈0.8) without collapsing merely-similar docs (s≈0.4)?
- Change the split in `near_dedup` and confirm the behavior on the sample corpus matches your prediction.

## 2. Try to beat decontamination
The decontaminator flags a training doc if ≥30% of its shingles appear in the eval index.

- Add a new doc to `RAW_TEXTS` that paraphrases the eval question heavily. Does it slip through? At what paraphrase level?
- This is exactly the adversary's move. What would you add to catch it — smaller shingles, embedding similarity, canary strings? Note the trade-offs (false positives cost you good data).

## 3. Add a real quality classifier
Replace one heuristic in `quality_ok` with a tiny learned classifier: label ~20 docs good/bad by hand, extract simple features (avg word length, stopword ratio, punctuation ratio), and train a logistic regression *from scratch* (no sklearn — it's ~20 lines). Compare its rejections to the heuristics.

## 4. Swap in real BPE tokenization
The `Tokenizer` here is word/punctuation level. Implement a minimal **byte-pair encoding**: start from characters, repeatedly merge the most frequent adjacent pair for N merges. Re-run the pipeline and compare `tokens_out` and `vocab_size` — BPE should shrink the vocab and change the token count.

## 5. Make the datasheet a trust anchor
Right now the datasheet stores a SHA-256 per shard. Extend it to also hash every *source document* and record which shards each contributed to. Then simulate an attacker editing one source doc after ingestion — show that the datasheet detects the change. (This is the seed of the supply-chain defense in Lab 11.)

## Stretch
Point the pipeline at a real corpus (e.g. a folder of `.txt` files, or a public-domain book split into paragraphs). Watch the dedup and quality numbers on real-world messiness — that's where the pipeline earns its keep.
