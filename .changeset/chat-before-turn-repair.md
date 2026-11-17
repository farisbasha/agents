---
"@cloudflare/ai-chat": patch
"@cloudflare/think": patch
"agents": patch
---

`AIChatAgent` and `Think` settle the transcript before a turn the same way: every turn fails the client tool calls, approval requests and approved calls left pending in earlier replies (with an error text that says which, passed through `repairInterruptedToolPart`), a skipped turn says why in the new `reason` (`continueLastTurn()` returns `"awaiting-client"` and changes nothing while the reply still waits on a client), a reply never reaches the model ending in reasoning, a tool result replaces the stored `clientTools` only when it settles its call, a `Think` request that sends no `clientTools` clears them, and `AIChatAgent` emits `chat:transcript:repaired`. See [Client-side tools](https://github.com/cloudflare/agents/blob/main/docs/agents/client-tools-continuation.md#when-the-user-moves-on-without-answering).
