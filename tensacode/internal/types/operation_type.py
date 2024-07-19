from typing import Literal

OPERATION_TYPE = (
    Literal[
        "encode",
        "decode",
        "decide",
        "call",
        "choice",
        "combine",
        "convert",
        "correct",
        "modify",
        "predict",
        "program",
        "query",
        "retrieve",
        "run",
        "semantictransfer",
        "similarity",
        "split",
        "store",
        "styletransfer",
    ]
    | str
)
