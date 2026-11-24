# Demo video script (≤ 3 min)

A shot list with exact commands and suggested narration, so recording
this is a ~10-minute mechanical task rather than something to improvise
live on camera. Each shot includes a line or two you can say out loud —
treat them as a starting point, not a script to read verbatim.

Suggested recording tool: `obs-studio`, `ffmpeg -f x11grab`, or the OS's
built-in screen recorder (Ubuntu: Settings → Screen Recording, or
Ctrl+Shift+Alt+R in GNOME). A terminal window plus a browser window
(for `scripts/websocket_test_client.html`) is all you need — no extra
software to install for the demo itself.

## Shot list

**1. Server startup (~15s)**

```sh
HAMS_MODEL_DIR=data/processed/checkpoint PYTHONPATH=src \
  uv run uvicorn hams_tts.server.main:app --host 127.0.0.1 --port 8765
```

Narration: "This is the fine-tuned model running behind a FastAPI
streaming server. It loads the checkpoint on startup — you can see the
log confirming the model's loaded and which device it's running on."

Then, in a second terminal: `curl http://127.0.0.1:8765/health` — point
out the response shows `model_loaded: true` and the sampling rate, not
just that the process is up.

**2. Pure Arabic streaming (~25s)**

Open `scripts/websocket_test_client.html` in a browser. In the
streaming section, enter:

```
مرحبا، كيفك اليوم؟ شو أخبارك؟
```

with `lang_hint = ar`, and click "Speak (streaming)."

Narration: "This is pure Levantine Arabic — notice it includes شو and
كيفك, which are dialect words, not Modern Standard Arabic. I'll click
speak, and you'll see the TTFA number appear almost immediately — that's
the time to the *first* audio chunk, not the whole response. The audio
starts playing before the rest of the sentence has even finished
generating."

**3. Pure English streaming (~25s)**

```
Good morning! The meeting is scheduled for 3 o'clock.
```

with `lang_hint = en`.

Narration: "Same server, same model, switched to English. Same low
latency — this isn't two separate models glued together, it's one
checkpoint that handles both languages through a shared phoneme
vocabulary."

**4. Code-switched example (~30s)** — the one the brief explicitly
requires.

```
بدي أعمل schedule لـ meeting بكرا الساعة 3
```

Narration: "And here's the interesting one — Arabic and English mixed
in the same sentence, the way people actually talk. This exact kind of
sentence never appeared as audio during training — the training data
was purely monolingual Arabic or purely monolingual English. What's
being tested here is whether the shared phoneme vocabulary and the
language-marker tokens let the model generalize to a mix it never heard
paired with audio. [Pause while it plays.] You can judge for yourself
how smooth the transition sounds between the two languages."

**5. Pipecat pipeline (~45s)**

```sh
uv run python3 scripts/demo_pipecat_pipeline.py
```

(Requires the `local` extra for audio playback — see that script's
docstring for the one-time setup command.)

Narration: "Beyond the raw server, there's also a Pipecat integration —
a subclass of Pipecat's TTSService that any Pipecat pipeline can drop in
like any other TTS provider. This script feeds it a few lines of text,
one of them code-switched, and plays them back through Pipecat's own
pipeline machinery rather than calling the server directly — so this is
testing the actual integration point a conversational-agent pipeline
would use, not just the HTTP/WebSocket API."

**6. Benchmark output (~20s)**

```sh
PYTHONPATH=src uv run python3 scripts/benchmark.py --mode server --host 127.0.0.1 --port 8765
```

Narration: "And finally, the benchmark script — it drives the actual
running server end to end, the same way a real client would, and
measures time-to-first-audio, real-time factor, and peak VRAM against
the targets from the spec. [Point at the printed table.] All four
numbers are comfortably inside the targets."

**Total: ~2:40**, leaving margin under the 3-minute cap.

## Recording checklist

- [ ] Server running against the **final** trained checkpoint (not a
      smoke-test checkpoint) — confirm via `/health` before recording.
- [ ] Terminal font size large enough to read in a recording.
- [ ] One take per shot is fine — doesn't need to be a single unbroken
      recording; cut together afterward.
- [ ] If voice-over is easier to add after the fact than narrating live,
      that's fine too — the narration lines above work either way.
