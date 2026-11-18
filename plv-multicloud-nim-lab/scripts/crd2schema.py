#!/usr/bin/env python3
"""Read multi-document YAML on stdin, write <kind>_<version>.json for every CustomResourceDefinition (kubeconform layout).
Mirrors yannh/kubeconform's openapi2jsonschema: int-or-string becomes oneOf[string, integer]."""
import json
import sys
from pathlib import Path

import yaml


class Loader(yaml.SafeLoader):
    """YAML 1.1 reads a bare '=' as the !!value tag; CRDs use it in enums (Prometheus matchers: =, !=, =~, !~)."""


Loader.add_constructor("tag:yaml.org,2002:value", lambda loader, node: loader.construct_scalar(node))


def fix(node):
    if isinstance(node, dict):
        if node.get("x-kubernetes-int-or-string"):
            return {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        return {k: fix(v) for k, v in node.items()}
    if isinstance(node, list):
        return [fix(v) for v in node]
    return node


def crd_docs(text):
    """Split the stream first: chart output also carries dashboards/configs that PyYAML cannot parse (e.g. '=' scalars)."""
    chunk = []
    for line in text.splitlines():
        if line.startswith("---"):
            yield "\n".join(chunk)
            chunk = []
        else:
            chunk.append(line)
    yield "\n".join(chunk)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for raw in crd_docs(sys.stdin.read()):
        if "kind: CustomResourceDefinition" not in raw:
            continue
        doc = yaml.load(raw, Loader=Loader)  # noqa: S506 - SafeLoader subclass
        if not isinstance(doc, dict) or doc.get("kind") != "CustomResourceDefinition":
            continue
        kind = doc["spec"]["names"]["kind"].lower()
        for v in doc["spec"].get("versions", []):
            schema = (v.get("schema") or {}).get("openAPIV3Schema")
            if schema:
                (out / f"{kind}_{v['name']}.json").write_text(json.dumps(fix(schema)))
                n += 1
    print(f"wrote {n} CRD schemas to {out}")


if __name__ == "__main__":
    main()
