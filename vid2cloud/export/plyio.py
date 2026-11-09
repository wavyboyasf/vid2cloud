"""PLY bez open3d: odczyt i zapis elementu `vertex` jako tablicy strukturalnej numpy.

Obsługiwane: `binary_little_endian`, `binary_big_endian` i `ascii`, właściwości skalarne
(char … double, także nazwy int8 … float64). Listy (np. `face`) nie są obsługiwane — chmury
z MASt3R-SLAM i z CloudCompare ich nie mają; plik z listą to błąd, nie cicha utrata danych.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_NAMES = {"i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort", "i4": "int", "u4": "uint",
          "f4": "float", "f8": "double"}


def _parse_header(f) -> tuple[str, list[tuple[str, int, list[tuple[str, str]]]], list[str]]:
    if f.readline().strip() != b"ply":
        raise ValueError("brak sygnatury 'ply'")
    fmt, elements, comments = None, [], []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("nagłówek PLY bez 'end_header'")
        tok = line.decode("ascii", "replace").split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] in ("comment", "obj_info"):
            comments.append(line.decode("utf-8", "replace").rstrip("\n")[len(tok[0]) + 1:])
        elif tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1][2].append((tok[-1], "list"))
            else:
                elements[-1][2].append((tok[2], _TYPES[tok[1]]))
        elif tok[0] == "end_header":
            return fmt, elements, comments


def read_ply(path, with_comments: bool = False):
    """Element `vertex` jako tablica strukturalna (nazwy właściwości jak w pliku)."""
    with open(path, "rb") as f:
        fmt, elements, comments = _parse_header(f)
        if not elements or elements[0][0] != "vertex":
            raise ValueError(f"{path}: pierwszy element to nie 'vertex'")
        name, n, props = elements[0]
        if any(t == "list" for _, t in props):
            raise ValueError(f"{path}: właściwość listowa w 'vertex' — nieobsługiwana")
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            out = np.zeros(n, dtype=[(p, t) for p, t in props])
            for i, (p, _) in enumerate(props):
                out[p] = data[:, i]
        else:
            order = "<" if fmt == "binary_little_endian" else ">"
            dt = np.dtype([(p, order + t) for p, t in props])
            out = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n).astype(dt.newbyteorder("="))
    return (out, comments) if with_comments else out


def xyz(vertices) -> np.ndarray:
    return np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64)


def write_ply(path, vertices: np.ndarray, comments: tuple[str, ...] = ()) -> None:
    """Binarny PLY (little endian) z tablicy strukturalnej; typy zachowane."""
    vertices = np.ascontiguousarray(vertices)
    props = []
    for p in vertices.dtype.names:
        t = vertices.dtype[p]
        props.append(f"property {_NAMES[t.kind + str(t.itemsize)]} {p}")
    header = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in comments),
              f"element vertex {len(vertices)}", *props, "end_header"]
    le = np.dtype([(p, vertices.dtype[p].newbyteorder("<")) for p in vertices.dtype.names])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vertices.astype(le).tobytes())


def xyz_vertices(points: np.ndarray, **fields) -> np.ndarray:
    """Tablica strukturalna x, y, z (float32) + dodatkowe pola o tej samej długości."""
    points = np.asarray(points)
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4")] + [(k, np.asarray(v).dtype.str) for k, v in fields.items()]
    out = np.zeros(len(points), dtype=dt)
    out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
    for k, v in fields.items():
        out[k] = v
    return out
