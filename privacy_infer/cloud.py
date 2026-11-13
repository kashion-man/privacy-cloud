"""Single fixed HTTPS destination; no redirects, ambient proxies or raw prompt interface."""
import json
import urllib.error
import urllib.request

from .credentials import get_api_key
from .errors import PrivacyError
from .policy import SanitizedRequest

ENDPOINT = "https://api.deepseek.com/chat/completions"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_body(request, model, max_tokens):
    return json.dumps({"model": model, "messages": request.messages(),
                       "max_tokens": max_tokens, "stream": False,
                       "thinking": {"type": "disabled"}}, ensure_ascii=False).encode("utf-8")


class DeepSeekClient:
    def __init__(self, model="deepseek-flash", timeout=60, max_tokens=1024, *, transport=None):
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self._transport = transport  # Dependency injection for network-free tests only.

    def complete(self, request, *, approved_body=None):
        if not isinstance(request, SanitizedRequest):
            raise PrivacyError("Cloud client requires a sanitized request")
        key = get_api_key()
        if not key or any(not c.isprintable() or c.isspace() for c in key) or not key.isascii():
            raise PrivacyError("Set a valid DEEPSEEK_API_KEY environment variable")
        body = request_body(request, self.model, self.max_tokens)
        if approved_body is not None and approved_body != body:
            raise PrivacyError("Cloud payload differs from the approved request")
        req = urllib.request.Request(ENDPOINT, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        try:
            if self._transport is not None:
                raw = self._transport(req, self.timeout)
            else:
                with self._opener.open(req, timeout=self.timeout) as response:
                    raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError()
            data = json.loads(raw)
            choice = data["choices"][0]
            result = choice["message"]["content"]
            if not isinstance(result, str) or not result.strip() or choice.get("finish_reason") != "stop":
                raise ValueError()
            return result
        except urllib.error.HTTPError as exc:
            # No response body, URL, headers or original exceptions in errors.
            raise PrivacyError(f"DeepSeek request failed (HTTP {exc.code})") from None
        except Exception:
            raise PrivacyError("DeepSeek request failed or returned an incomplete response") from None
