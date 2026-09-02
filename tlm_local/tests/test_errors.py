from tlm_local.errors import ModelNotPulledError, OllamaUnavailableError, translate_ollama_error


class _FakeAPIError(Exception):
    """Stand-in for openai.APIError - translate_ollama_error only inspects
    str(error), so any exception with the right message works for these
    tests without needing a real litellm/openai failure.
    """


class TestTranslateOllamaError:
    def test_returns_model_not_pulled_error_when_message_mentions_not_found(self):
        # given - the exact message litellm raises when Ollama can't find a model
        error = _FakeAPIError("litellm.NotFoundError: model 'ollama/foo' not found")

        # when
        result = translate_ollama_error(error, "ollama/foo")

        # then
        assert isinstance(result, ModelNotPulledError)
        assert result.model == "ollama/foo"

    def test_returns_ollama_unavailable_error_when_message_is_a_connection_failure(self):
        # given - the exact message litellm raises when Ollama isn't running
        error = _FakeAPIError("litellm.InternalServerError: OpenAIException - Connection error.")

        # when
        result = translate_ollama_error(error, "ollama/foo")

        # then
        assert isinstance(result, OllamaUnavailableError)

    def test_returns_none_when_the_failure_cannot_be_named(self):
        """The reason this returns None rather than guessing: a request Ollama
        understood and refused is not the server being down, and saying so
        sends the reader to check something that was never broken.
        """
        # given - a context longer than the model's window: Ollama is up and
        # answering, it just refused this request
        error = _FakeAPIError("litellm.BadRequestError: OpenAIException - context length exceeded")

        # when / then
        assert translate_ollama_error(error, "ollama/foo") is None

    def test_a_timeout_counts_as_unavailable(self):
        # given
        error = _FakeAPIError("litellm.APIConnectionError: Request timed out.")

        # when / then
        assert isinstance(translate_ollama_error(error, "ollama/foo"), OllamaUnavailableError)

    def test_not_found_check_is_case_insensitive(self):
        # given
        error = _FakeAPIError("Model 'foo' Not Found")

        # when
        result = translate_ollama_error(error, "ollama/foo")

        # then
        assert isinstance(result, ModelNotPulledError)


class TestModelNotPulledError:
    def test_message_strips_ollama_prefix_for_the_pull_command(self):
        # given / when
        error = ModelNotPulledError("ollama/qwen2.5:7b")

        # then
        assert "ollama pull qwen2.5:7b" in str(error)
        assert error.model == "ollama/qwen2.5:7b"

    def test_message_handles_a_model_string_with_no_prefix(self):
        # given / when
        error = ModelNotPulledError("qwen2.5:7b")

        # then
        assert "ollama pull qwen2.5:7b" in str(error)
