# llama.cpp patches

## `pre-trigger-grammar-78433f6.patch`

Source: https://github.com/ggml-org/llama.cpp/discussions/22408#discussioncomment-16728485

This experimental patch lets `llama-server` accept OpenAI-style `tools` and a
custom `grammar` in the same request. The custom grammar is applied during the
reasoning phase, then llama.cpp's generated tool grammar takes over.

Use it through the repo build wrapper:

```bash
./scripts/build_llama_cpp.sh pre-trigger-grammar
SERVER_MODE=pre-trigger-grammar BACKGROUND=1 ./run_server.sh
```

The patch is version-sensitive. If it stops applying, set `LLAMA_CPP_REF` to a
compatible upstream llama.cpp revision or refresh the patch from the discussion.
