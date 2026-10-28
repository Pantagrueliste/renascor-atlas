"""CreNum - Guillaume Cretin, Chronique francaise (ELAN, 5 books in verse, c. 1515; Grenoble).

Running-text <persName ref="#slug">, <placeName ref="#slug">, <orgName> (no ref). The slugs are descriptive French
names (#charlemagne, #apollon, #troie, #france, #jesus_christ) and mix historical, mythological and biblical figures;
the register (listPerson / listPlace) and the shared header (xi:include CreNum_header_common.xml) are NOT in the
harvest, so there is no way to flag mythological or biblical entries (note_type) and no authority ids or
coordinates: the slugs become opaque keys 'elan_crenum:<slug>' (in_register False). Tags without ref (45 persName,
775 placeName, all orgName) are kept with local_key None. Some refs lack '#' (ref="france"): treated the same.

v2 (issue 154): half of the surfaces ('f rance', 'd ieu', 'p aris'; 3,149 of 6,327) were corrupted by a detached initial: the source writes
the initial as <choice><orig>f</orig><reg>F</reg></choice> followed by the rest of the word, with pretty-printing whitespace between
the elements. The surface now takes the regularised reading (<reg>) of every <choice> and ignores the whitespace inside it:
'France'. The register and the shared header are on gricad-gitlab (robots.txt: Disallow /), not fetched: still opaque slugs.
Dating: the files carry no date (header is an xi:include); collection date 1515 from the catalogue entry
(date_source 'collection').
Licence: LICENSE_TIERS gives T1 LicenseRef-CC-BY-unversioned; the catalogue entry says 'Mixed (includes CC-BY-4.0)'
and the shipped files carry no statement; T1 kept, to be re-checked before any redistribution.
"""
SLUG = 'elan_crenum'
COLLS = ['elan/elan__crenum']
RENASCOR_IDS = ['elan-crenum']
LICENCE = 'LicenseRef-CC-BY-unversioned'
TIER = 'T1'
DESCRIPTION = 'Cretin, Chronique francaise (c. 1515): persons, places and orgs tagged in verse, ids as local slugs'
OPTIONS = {'opaque_keys': True, 'default_years': (1515, 1515)}

from .. import helpers_batch1 as H  # noqa: E402


def _reg_surface(el) -> str:
    """Text of a tag reading the regularised form of each <choice> (whitespace inside a choice is layout)."""
    from .. import common as C
    out = []

    def rec(e, in_choice):
        if e.text and (not in_choice or e.text.strip()):
            out.append(e.text.strip() if in_choice else e.text)
        for c in e:
            if not isinstance(c.tag, str):
                continue
            n = C.ln(c.tag)
            if n == 'choice':
                reg = [x for x in c if C.ln(x.tag) == 'reg']
                if reg:
                    rec(reg[0], True)
                else:
                    rec(c, True)
            elif n in ('orig', 'sic', 'abbr') and in_choice and e.tag and C.ln(e.tag) == 'choice':
                pass                                        # the diplomatic alternative is dropped when a <reg> exists
            elif n == 'note':
                pass
            else:
                rec(c, in_choice)
            if c.tail and (not in_choice or c.tail.strip()):
                out.append(c.tail.strip() if in_choice else c.tail)

    rec(el, False)
    return C.norm(''.join(out))


class Extractor(H.BaseTEI):
    def scan(self, root, rec, ctx, counters):
        from .. import common as C
        hits = super().scan(root, rec, ctx, counters)
        for h in hits:
            if any(C.ln(c.tag) == 'choice' for c in h.el.iter() if isinstance(c.tag, str)):
                h.surface = _reg_surface(h.el)
        return hits


EXTRACTOR = Extractor()
