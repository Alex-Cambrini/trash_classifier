RUN_NAME = None

def set_run_name(name: str):
    global RUN_NAME
    RUN_NAME = name

def get_run_name() -> str:
    if RUN_NAME is None:
        raise ValueError("RUN_NAME non è ancora stato settato")
    return RUN_NAME
