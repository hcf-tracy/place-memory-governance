#!/usr/bin/env bash
# Shared HF13 model-token normalization. Public scripts accept both the concise
# HF13 token (7b/3b) and the historical/full Ollama model name.
hf13_normalize_model() {
  case "${1:-}" in
    7b|qwen2.5:7b) printf '%s\n' 7b ;;
    3b|qwen2.5:3b) printf '%s\n' 3b ;;
    *)
      echo "Unsupported HF13 model token: ${1:-<empty>} (use 7b, 3b, qwen2.5:7b, or qwen2.5:3b)" >&2
      return 2
      ;;
  esac
}

hf13_model_config() {
  case "$(hf13_normalize_model "${1:-}")" in
    7b) printf '%s\n' configs/hf13_mem0_qwen7b.yaml ;;
    3b) printf '%s\n' configs/hf13_mem0_qwen3b.yaml ;;
  esac
}

hf13_ollama_model_name() {
  case "$(hf13_normalize_model "${1:-}")" in
    7b) printf '%s\n' qwen2.5:7b ;;
    3b) printf '%s\n' qwen2.5:3b ;;
  esac
}
