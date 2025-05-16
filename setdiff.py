from pathlib import Path


def set_diff(
    i1: str | Path,
    i2: str | Path,
    o: str | Path
):
    i1, i2, o = Path(i1), Path(i2), Path(o)

    i1_set = set(i1.read_text().splitlines())
    i2_set = set(i2.read_text().splitlines())
    o_set = i1_set - i2_set
    o.write_text('\n'.join(list(sorted(o_set))))


if __name__ == '__main__':
    from fire import Fire
    Fire(set_diff)