# checklists

## A) DARTS search checklist
- [ ] train/val split exists
- [ ] w params and α params separated
- [ ] alternate updates implemented
- [ ] α logs recorded
- [ ] genotype export exists

## B) Latency-aware DARTS checklist
- [ ] expected latency computed from softmax(α)
- [ ] λ configurable
- [ ] logs include val_loss and latency_total
- [ ] best genotype tracked

## C) Discrete rebuild checklist
- [ ] genotype → fixed ops
- [ ] no mixed ops remain
- [ ] model runs forward pass

## D) Benchmark checklist
- [ ] warmup
- [ ] measure loop
- [ ] p50/p99 reported
- [ ] device recorded

## E) FPGA validation checklist
- [ ] measurement protocol documented
- [ ] template JSONL exists
- [ ] validation script computes Spearman + error
