# Paid API Node Safety Rules

These rules apply to every remote API node in this plugin, including new nodes and changes to existing nodes.

## Cancellation and progress

- Every node that waits for a remote request, polling job, upload, or download must accept ComfyUI's hidden `UNIQUE_ID` and display a frontend status with elapsed seconds at least once per second while it is waiting.
- Every blocking wait must call `comfy.model_management.throw_exception_if_processing_interrupted()` at least every 0.2 seconds. This includes generation requests, uploads, polling sleeps, retries, and result downloads.
- A blocking HTTP call must not run directly on the ComfyUI execution thread. Run it in an interruptible wrapper and, on cancellation, close the HTTP session, cancel work that has not started, call executor shutdown with `wait=False`, and immediately re-raise `InterruptProcessingException`.
- Never catch or convert `InterruptProcessingException` into a normal node result. Do not use a broad `except BaseException` that swallows cancellation.
- After cancellation, do not submit another request, retry, poll, download, decode, or return a partial success result. Show a short `Cancelled` frontend status including elapsed seconds.
- Clear or replace the running status with `Completed`, `Failed`, `Timed out`, or `Cancelled` when execution ends.

## Billing safety

- Call `throw_exception_if_processing_interrupted()` immediately before every request that may create a billable task.
- Submit at most one billable generation request per node execution unless the user explicitly requested multiple jobs.
- Do not automatically retry a billable `POST` unless the provider supports an idempotency key and the same key is reused for every attempt. Connection ambiguity is not proof that a paid request failed.
- Use finite connection and response timeouts. Polling must also have a finite attempt or elapsed-time limit.
- If an API returns a task ID, preserve it in status or node output when practical so charges can be audited.
- Show actual usage, credits, or price returned by the service. If only an estimate is available, label it clearly as estimated.
- Never claim that local cancellation refunds or cancels a provider charge after the remote service has accepted the request.

## Verification

- Tests for paid API nodes must simulate a slow request and verify that ComfyUI cancellation returns control within 0.5 seconds without submitting a second billable request.
- Verify that elapsed time is visible through the frontend progress-text mechanism during the simulated wait.
