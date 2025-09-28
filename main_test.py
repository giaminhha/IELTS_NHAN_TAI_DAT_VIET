from pipeline.llm import call_llm
def test_call_llm_basic():
    prompt = "Write a short passage about the benefits of reading books."
    result = call_llm(prompt)   # adjust model if needed

    print("=== Raw result ===")
    print(result)
    print("=== Word count ===", len(result.split()))

    # Basic sanity check
    assert isinstance(result, str), "call_llm should return a string"
    assert len(result.split()) > 5, "Result seems too short (maybe cached incorrectly?)"


if __name__ == "__main__":
    test_call_llm_basic()
