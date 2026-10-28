"""Helpers of the batch2 collection modules (CELT, ...). Additive only; common.py is not touched."""
from __future__ import annotations

import html.entities
import re

_ENT = re.compile(rb'&([A-Za-z][A-Za-z0-9]*);')
_XML5 = {b'amp', b'lt', b'gt', b'quot', b'apos'}
_EXTRA = {b'ampersir': '⁊', b'ampersand': '&#38;'}      # CELT: Tironian et
_DOCTYPE = re.compile(rb'<!DOCTYPE[^\[>]*(\[.*?\])?\s*>', re.S)
_XMLDECL = re.compile(rb'^\s*<\?xml[^>]*\?>', re.S)


def sgml_to_xml_bytes(data: bytes, encoding: str = 'iso-8859-1') -> bytes:
    """Make SGML-derived TEI P4 (CELT) parseable: drop the DOCTYPE, replace named character entities by their
    characters (the five XML entities are kept), re-encode as UTF-8. Never loads anything."""
    m = _XMLDECL.match(data)
    if m:
        em = re.search(rb'encoding=["\']([^"\']+)["\']', m.group(0))
        if em:
            encoding = em.group(1).decode('ascii', 'replace')
        data = data[m.end():]
    data = _DOCTYPE.sub(b'', data, count=1)

    def rep(mo):
        name = mo.group(1)
        if name in _XML5:
            return mo.group(0)
        if name in _EXTRA:
            return _EXTRA[name].encode('ascii', 'xmlcharrefreplace')
        cp = html.entities.name2codepoint.get(name.decode('ascii'))
        if cp is None:
            return b'?'
        return f'&#{cp};'.encode('ascii')

    data = _ENT.sub(rep, data)
    try:
        text = data.decode(encoding, errors='replace')
    except LookupError:
        text = data.decode('iso-8859-1', errors='replace')
    return b'<?xml version="1.0" encoding="UTF-8"?>' + text.encode('utf-8')
