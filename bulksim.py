"""
Run simulations for all tradesimlog folders that have tradesimparams.json
but are missing sim_log.csv (i.e. simulation hasn't run yet).
"""
from pathlib import Path
from lib.tradeparams import TradeSimParams
from simulator import run_simulation, _SIM_LOCK_FILE

TRADESIMLOG = Path("tradesimlog")

folders = sorted(TRADESIMLOG.iterdir())
pending = [
    f for f in folders
    if f.is_dir()
    and (f / "tradesimparams.json").exists()
    and not (f / "sim_log.csv").exists()
    and not (f / _SIM_LOCK_FILE).exists()
]

print(f"Found {len(pending)} pending simulation(s).\n")

for folder in pending:
    print(f"{'='*60}")
    print(f"Simulating {folder.name} ...")
    try:
        params = TradeSimParams.load_from_folder(folder)
        # model_storage_folder is init=False — not restored by load_from_folder
        object.__setattr__(params, 'model_storage_folder', Path(params.model_path) / "model")

        if not params.is_model_available():
            print(f"  Skipping — model not available at {params.model_storage_folder}")
            continue

        run_simulation(params)

    except Exception as ex:
        print(f"  ERROR: {ex}")
        try:
            params.log_error(str(ex))
        except Exception:
            pass
