# Lab 06 — Exercises

Work through them in order. Each one pushes on a different part of the layer. Solution sketches are at the bottom.

### 1. Speak OpenTelemetry (⚙)
Rename the span attributes in `app.py` to OpenTelemetry's generative-AI semantic convention names (for example `model` → `gen_ai.request.model`, `input_tokens` → `gen_ai.usage.input_tokens`, `output_tokens` → `gen_ai.usage.output_tokens`). Check the current convention on opentelemetry.io first, because it is still evolving. Update the tests that read those attributes.
*Hint:* only `LLMApp._call_model` sets them. Keep the old names as well for one release if you want to show a non-breaking migration.

### 2. Per-tenant budgets (⚙⚙)
Right now there is one `BudgetBreaker` for the whole app. Give each tenant (pass a `tenant` argument to `handle()`) its own breaker plus a global one, so one noisy customer can't use up everyone's budget. Add a test where tenant A trips its breaker and tenant B keeps being served.
*Hint:* `GuardrailLayer.check_budget` can take a list of breakers and require all of them to allow the call. Charge all of them afterwards.

### 3. Fix Lab 01's ordering upstream (⚙⚙)
Lab 06 works around the fact that Lab 01's PHONE pattern runs before its APIKEY pattern (see "Logs" in the README). Fix it at the source in your own fork: reorder `_PII_PATTERNS` in `labs/01-data-engine/pipeline.py` so the most specific patterns run first, and widen APIKEY to match `sk-test-...` style keys. Run *both* labs' test suites. Does `SECRET_PATTERNS` in this lab still earn its place?
*Hint:* Python dicts keep insertion order, so the order you write them in is the order they run. A card number is currently tagged `[PHONE]` for the same reason.

### 4. Trace-based anomaly alert (⚙⚙)
Write `detect(tracer, logger) -> list[str]` that reads finished traces and logs and raises alerts for: any trace where `check_trace` finds problems; any request with more than N `model.generate` spans (a looping agent); any request where the output filter blocked something the input filter passed (an injection that got past the first layer). Run it over the demo's before/after traffic.
*Hint:* the third alert is the most valuable one in Act II. Use the `guardrail.decision` events on spans and the `guardrail.block` log lines; both carry the guard name.

### 5. A semantic input classifier, and its bill (⚙⚙⚙)
Add a second input guard that catches paraphrased injections. Offline, a bag-of-words cosine similarity against a small set of known injection examples is enough (Lab 09's `embeddings.py` has one you can load through `labs_path`). Make it catch `"Pretend the earlier guidance never existed..."` while still passing every request in `NORMAL_REQUESTS` and the benign cases in `test_input_filter_allows_benign_text_and_admits_its_limits`. Then measure what it costs: add its own span, give it a simulated latency, and report the false-positive rate on 50 benign requests you write.
*Hint:* the threshold is the whole game. Plot detections against false positives as you sweep it, and write down where you'd set it and why.

### 6. Automatic rollback (⚙⚙⚙)
Turn the canary into a closed loop: after every K canary requests, compare the canary's cost per request and block rate against stable's. If either is more than X% worse, set `router.rollback = True` and log a `canary.rollback` event with both numbers. Test that the verbose v2 prompt is rolled back automatically and that a v2 identical to v1 is not.
*Hint:* with small samples, noise can trigger a rollback by itself. Require a minimum number of canary requests before deciding, and say in a comment why you picked that number.

---

## Solution sketches

1. Mechanical rename; the lesson is that conventional names let any OpenTelemetry-compatible backend chart your token usage without custom parsing.
2. `GuardrailLayer(budget=[global_breaker, tenant_breakers[tenant]])`; `allow` = all allow; `charge` = charge all. Tenant A's sticky trip leaves B untouched.
3. Put APIKEY (widened to `(?:sk|api|key|token)[-_][A-Za-z0-9_-]{12,}`) first and CARD before PHONE. `SECRET_PATTERNS` still adds the AWS, GitHub, bearer and PEM formats, so it stays, but the ordering hazard goes away.
4. Rule three fires on two requests in the protected before/after run: the paraphrased injection and the API-key question, both blocked by `output_filter` after `input_filter` allowed them. That tells you what the rule really detects: leaks that got past the first layer, whatever caused them.
5. Cosine against about 10 injection exemplars can separate the paraphrase from the benign set at a threshold you have to tune by hand. Expect some benign requests to sit near the boundary. That is the real trade-off of semantic guardrails, now in numbers.
6. Track running means per variant from `Response.cost_usd`; because the mock is deterministic, a `min_requests` of about 20 should be enough to roll back the verbose v2 (+46% cost) and keep an identical v2. Real traffic is noisier and needs more.
