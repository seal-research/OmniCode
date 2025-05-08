from pathlib import Path
import shutil


def clean(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)

    for idir in results_dir.iterdir():
        if not idir.is_dir():
            continue

        ilog_dir = idir / "logs"

        if not ilog_dir.exists():
            continue

        for i in idir.iterdir():
            if i.name != "logs":
                if i.is_dir():
                    shutil.rmtree(i)
                else:
                    i.unlink()

        for item in ilog_dir.iterdir():
            shutil.move(str(item), str(idir / item.name))
    
        ilog_dir.rmdir()
    



if __name__ == '__main__':
    import fire
    fire.Fire(clean)