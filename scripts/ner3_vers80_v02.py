"""Map the 80 Latin verse/drama passages annotated on 2026-10-25 (silver/verse_dev/ann_verse80.txt, guide v0.1) to the owner rules of
2026-10-25 (guide v0.2). Only my annotations and the passage text are read; no v3 / v4 prediction is opened.
Each 'conv' zone that a decision settles becomes firm (label kept or changed); zones still open (theatre speakers, families, other-world
places, metonymy, planet or god, antonomasia, Q15-Q18) stay 'conv'. Firm O entries are dropped (not an entity, no longer a zone).
    python scripts/ner3_vers80_v02.py ann_verse80.txt vers80_v02.txt changes.tsv
"""
import sys

# (rank, surface) -> (new label, still open?, rule, reason); applies to every occurrence of that surface in that passage
D = {}
def put(rank, surfs, lab, conv, rule, why):
    for s in surfs.split(' ; '):
        D[(rank, s)] = (lab, conv, rule, why)

PER_GOD = ('PER', False, '5', 'dieu antique, muse, figure mythologique : PER')
for r, s in [(0, 'Pegasidum'), (3, 'Venus ; Iuno ; Iovis ; Mars ; Pallas ; Flora'), (5, 'Iovis'), (7, 'Minerua'), (8, 'Leucothoë'),
             (12, 'Musarum ; Phoebi ; Martis ; Musis'), (14, 'Atlas'), (18, 'Priapo'), (26, 'Megaera ; Ioue ; Osirim ; Musae ; Apollo ; Charites ; Venus'),
             (27, 'Iovis ; Phoebo ; Iovisque ; Phoebum ; Hymen ; Phoebus'), (29, 'Musas'), (30, 'Saturnia ; Iovi'), (31, 'Mavors ; Pallas'),
             (35, 'Gorgonis ; Medusae'), (37, 'Musas ; Camenas ; Musarum ; Pieridas'), (40, 'Mercurius ; Iuppiter'), (41, 'Aonidum'),
             (44, 'Charites ; Cypris'), (48, 'Aonidum'), (56, 'Phoebus ; Tonanti'), (58, 'Nereus ; Neptuno'), (62, 'Phaetonta'),
             (65, 'Calliope ; Phoebus'), (66, 'Iuppiter'), (68, 'Gratiae ; Latonae'), (78, 'Musa ; Clio ; Pieridum')]:
    put(r, s, *PER_GOD)
put(24, 'Antichristum', 'PER', False, '5', 'figure biblique : PER')
put(51, 'Beli', 'PER', False, '5', 'Belus, roi ou dieu antique : PER')
O_PERS = ('O', False, '5', 'personnification : rien')
for r, s in [(3, 'Elegia'), (7, 'Sophia ; Suadâ ; Charis'), (8, 'Fortunae'), (27, 'Lux ; Luci ; Lucemque'), (31, 'Fortunae'), (33, 'Amor'),
             (66, 'Amor'), (0, 'Minervam'), (47, 'Veneris')]:
    put(r, s, *O_PERS)
O_GOD = ('O', False, '5', 'Dieu : rien')
put(2, 'Tonantis ; Patris', *O_GOD)
O_ASTRE = ('O', False, '5', 'astre, vent : rien')
for r, s in [(25, 'Caeciarum'), (48, 'Delius'), (58, 'Erigone ; Orion'), (59, 'Zephyro'), (62, 'Notis'), (68, 'Phoebe ; Nox'),
             (76, 'Tauri ; Zephyris ; Austris ; Aquilonibus ; Euris')]:
    put(r, s, *O_ASTRE)
O_PEOPLE = ('O', False, '2', 'peuple, gentilé, adjectif de lieu : rien')
for r, s in [(0, 'Adriaci'), (10, 'Cressa'), (11, 'Brachmannos'), (22, 'Arabes ; Graecos ; Graeci'), (29, 'Cambros'), (33, 'Colchos'),
             (35, 'erythraeo ; Herculeoque ; ibero'), (41, 'Germanos ; Graecis ; Latinus'), (44, 'Elysiis ; Elysios'), (45, 'Elyſios'),
             (49, 'Indos'), (56, 'Trinacrio'), (59, 'Erasiniis'), (66, 'Tarpei ; Trinacria'), (67, 'Turcarum'),
             (75, 'Aethiopum ; Nilotica ; Bethygenas ; Mauros'), (77, 'elysios'), (79, 'Illyricos ; Istros')]:
    put(r, s, *O_PEOPLE)
put(34, 'Agrippa', 'O', False, '-', 'nom commun (naissance par les pieds) : rien')
put(5, 'Bacchov ; Treutler ; Carnin', 'PER', False, '1', 'abréviation : point final hors du nom, PER ferme')
put(51, 'Herodot ; Darii Hystasp', 'PER', False, '1', 'abréviation : point final hors du nom, PER ferme')
put(12, 'Melphi', 'LOC', False, '3', 'lieu dans un titre (Melphi Princeps) : LOC')
put(37, 'Forilivii', 'LOC', False, '3', 'lieu dans un titre (Forilivii principem) : LOC')
put(52, 'Magdeburgensis', 'LOC', False, '2/3', '-ensis dans un titre : LOC')
put(37, 'Georgii Foriliviensis', 'PER', False, '2', '-ensis comme surnom sans mot de fonction : un seul PER (§3.6)')
put(41, 'Ioannem Reuchlin Phorcensem', 'PER', False, '2', '-ensis comme surnom sans mot de fonction : un seul PER (§3.6)')
ADD = {26: ('Roma', 'Pierides', 'PER', '5', 'Muses (Pierides) oubliées en v0.1 : PER')}


def main(src, dst, log):
    out, rows, used = [], [], set()
    for line in open(src, encoding='utf8'):
        line = line.rstrip('\n')
        if line.startswith('#') or not line.strip():
            out.append(line)
            continue
        head, _, body = line.partition(':')
        r = int(head.split()[1])
        ents = [] if body.strip() == '@' else [x.strip() for x in body.split(' ; ')]
        new = []
        for x in ents:
            parts = x.split('|')
            conv = parts[-1] == 'conv'
            surf, lab = ('|'.join(parts[:-2]), parts[-2]) if conv else ('|'.join(parts[:-1]), parts[-1])
            if r in ADD and surf == ADD[r][0] and (r, 'add') not in used:
                _, s2, l2, rule, why = ADD[r]
                new.append(f'{s2}|{l2}')
                rows.append((r, s2, '-', l2, rule, why))
                used.add((r, 'add'))
            if conv and (r, surf) in D:
                l2, c2, rule, why = D[(r, surf)]
                used.add((r, surf))
                rows.append((r, surf, f'{lab}|conv', l2 + ('|conv' if c2 else ''), rule, why))
                if l2 == 'O' and not c2:
                    continue
                new.append(f'{surf}|{l2}' + ('|conv' if c2 else ''))
            else:
                new.append(x)
        out.append(f'vd {r}: ' + (' ; '.join(new) if new else '@'))
    miss = set(D) - used
    assert not miss, miss
    out[0] = ('# vers80 under the owner rules of 2026-10-25 (guide v0.2), mapped from ann_verse80.txt (md5 1c16232f...) by '
              'scripts/ner3_vers80_v02.py, before any score. Same format; conv = zone still open.')
    open(dst, 'w', encoding='utf8').write('\n'.join(out) + '\n')
    with open(log, 'w', encoding='utf8') as f:
        f.write('vd\tsurface\tavant\taprès\trègle\traison\n')
        for row in rows:
            f.write('\t'.join(map(str, row)) + '\n')
    print(len(rows), 'changes')


if __name__ == '__main__':
    main(*sys.argv[1:4])
