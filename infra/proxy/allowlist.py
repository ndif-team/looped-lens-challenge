"""The runner task's only way out: a mitmproxy addon that enforces the
competition's egress policy.

The runner's subnet has no route to the internet; its security group lets it
talk to this proxy and to nothing else. Three kinds of destination exist:

    ALLOW       tunnelled untouched (NDIF). The proxy sees only the host name.
    READ_ONLY   decrypted, and only GET and HEAD are forwarded (Hugging Face).
                Enough to fetch a model's config and tokenizer, not enough to
                upload the private inputs anywhere.
    anything else is refused.

A host matches an entry when it equals it or is a subdomain of it. Both lists
come from competition.yaml (`runner.egress`), as comma-separated environment
variables. `mitm.it` is mitmproxy's own page for handing out its certificate
authority, which the runner fetches once at start so it can trust READ_ONLY
connections.
"""

import logging
import os
import re

from mitmproxy import ctx, http

ALLOW = [h for h in os.environ.get("EGRESS_ALLOW", "").split(",") if h]
READ_ONLY = [h for h in os.environ.get("EGRESS_READ_ONLY", "").split(",") if h]
PORTS = {80, 443}
CERTIFICATE_HOST = "mitm.it"


def matches(host: str, entries: list[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == entry or host.endswith("." + entry) for entry in entries)


def refuse(flow: http.HTTPFlow, why: str) -> None:
    logging.info("refused %s %s: %s", flow.request.method, flow.request.url, why)
    if flow.request.stream:
        # A body over the streaming threshold (see the Dockerfile) is not read
        # by the proxy, so the connection is dropped without an answer.
        flow.kill()
        return
    flow.response = http.Response.make(403, f"blocked by the competition's egress policy: {why}\n", {"Content-Type": "text/plain"})


def load(loader) -> None:
    # Hosts in ALLOW are passed through as opaque TLS: no decryption, so NDIF's
    # traffic (including its websocket) is not touched.
    ctx.options.ignore_hosts = [rf"^(.+\.)?{re.escape(entry)}:443$" for entry in ALLOW]


def http_connect(flow: http.HTTPFlow) -> None:
    """A request to open a tunnel. Refusing here means a blocked host is never
    even connected to."""
    host, port = flow.request.host, flow.request.port
    if port not in PORTS or not matches(host, ALLOW + READ_ONLY):
        refuse(flow, f"{host}:{port} is not an allowed destination")


def requestheaders(flow: http.HTTPFlow) -> None:
    """A decrypted (or plain HTTP) request, decided when its headers arrive.
    The `request` hook runs after the body, and a body over the streaming
    threshold has been passed on to the server by then."""
    request = flow.request
    # `host` is where the proxy connects. `pretty_host` is the name the request
    # asks that server for, which the client writes freely. Each has to pass.
    names = (request.host, request.pretty_host)
    if names == (CERTIFICATE_HOST, CERTIFICATE_HOST):
        return
    for host in names:
        if request.port not in PORTS or not matches(host, ALLOW + READ_ONLY):
            return refuse(flow, f"{host}:{request.port} is not an allowed destination")
        if not matches(host, ALLOW) and request.method not in ("GET", "HEAD"):
            return refuse(flow, f"{host} is read-only; {request.method} is not allowed")
