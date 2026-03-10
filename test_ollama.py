from llm_router import LLMRouter


def main() -> None:
    router = LLMRouter()
    result = router.call_local_chat(
        model="qwen3:8b",
        system_prompt='Return exactly this JSON object: {"ok": true}',
        user_prompt='Return exactly this JSON object: {"ok": true}',
        temperature=0,
    )
    print(result)


if __name__ == "__main__":
    main()
