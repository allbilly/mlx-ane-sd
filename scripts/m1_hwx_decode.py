"""Retain bounded H13G container/task data and ordered register writes.

Format references (no source dependency): allbilly/ane at
9c5d3c21ee941a6bb6b2440ef86c0494cfd698cf, gpt2/hwx.py and
experimental/m1_register_map.md. Unknown words remain in the original HWX,
raw load commands, task headers and packet stream. This does not read MMIO
or establish that an offline export was the executable evaluated by macOS.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def name(data):
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def bounds(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(f"extent outside payload: {offset:#x}+{size:#x}/{len(data):#x}")
    return data[offset:offset + size]


def container(data):
    header = struct.unpack("<8I", bounds(data, 0, 32))
    if header[:3] != (0xBEEFFACE, 128, 4):
        raise ValueError(f"Expected H13G/M1 container, found {header[:3]}")
    report = {"architecture": "H13G", "isa_version": 7, "header_words": list(header),
              "sha256": digest(data), "bytes": len(data), "load_commands": [],
              "segments": [], "sections": [], "threads": [], "symbols": []}
    cursor, limit = 32, 32 + header[5]
    bounds(data, 32, header[5])
    symtab = None
    for index in range(header[4]):
        command, size = struct.unpack("<2I", bounds(data, cursor, 8))
        if size < 8 or cursor + size > limit:
            raise ValueError("Invalid load command size")
        raw = bounds(data, cursor, size)
        report["load_commands"].append({"index": index, "offset": cursor,
            "command": command, "size": size, "raw_hex": raw.hex()})
        if command == 0x19:
            fields = struct.unpack("<2I16s4Q4I", bounds(raw, 0, 72))
            segment = dict(name=name(fields[2]), vmaddr=fields[3], vmsize=fields[4],
                           fileoff=fields[5], filesize=fields[6], maxprot=fields[7],
                           initprot=fields[8], nsects=fields[9], flags=fields[10])
            payload = bounds(data, segment["fileoff"], segment["filesize"])
            segment["payload_sha256"] = digest(payload)
            report["segments"].append(segment)
            for section_index in range(segment["nsects"]):
                s = struct.unpack("<16s16s2Q8I", bounds(raw, 72 + section_index * 80, 80))
                section = dict(name=name(s[0]), segment=name(s[1]), addr=s[2], size=s[3],
                    offset=s[4], align_exponent=s[5], reloff=s[6], nreloc=s[7], flags=s[8],
                    reserved_words=list(s[9:]), relocations=[])
                for ri in range(s[7]):
                    address, info = struct.unpack("<2I", bounds(data, s[6] + ri * 8, 8))
                    section["relocations"].append({"address_raw": address, "info_raw": info})
                report["sections"].append(section)
        elif command == 4 and size >= 12:
            flavor = struct.unpack_from("<I", raw, 8)[0]
            thread = {"flavor": flavor, "raw_hex": raw.hex()}
            if flavor == 1 and size >= 0x820:
                entry, first_words_minus_one, count = struct.unpack_from("<QII", raw, 0x810)
                thread.update(bars=list(struct.unpack_from("<32Q", raw, 16)), entry=entry,
                              first_td_size=(first_words_minus_one + 1) * 4, td_count=count)
            report["threads"].append(thread)
        elif command == 2:
            symtab = struct.unpack("<4I", bounds(raw, 8, 16))
        cursor += size
    if cursor != limit:
        raise ValueError("Load command total differs from header")
    if symtab:
        symoff, count, stroff, strsize = symtab
        strings = bounds(data, stroff, strsize)
        for i in range(count):
            offset, kind, section, desc, value = struct.unpack("<IBBHQ", bounds(data, symoff + i * 16, 16))
            if offset >= len(strings):
                raise ValueError("Symbol string outside table")
            report["symbols"].append(dict(name=name(strings[offset:]), type=kind,
                                         section=section, desc=desc, value=value))
    return report


BLOCKS = [(0, "Common"), (0x4800, "L2"), (0x8800, "PE"), (0xC800, "NE"),
          (0x13800, "TileDMA-source"), (0x17800, "TileDMA-destination"), (0x1F800, "CoefficientDMA")]
NAMES = {0: "InDim", 8: "ChCfg", 12: "Cin", 16: "Cout", 20: "OutDim", 28: "ConvCfg",
         36: "GroupConvCfg", 40: "TileCfg", 52: "Cfg", 56: "TaskInfo", 60: "DPE"}
for base, labels in ((0x8800, ["Cfg", "BiasScale", "PreScale", "FinalScale"]),
                     (0xC800, ["KernelCfg", "MACCfg", "MatrixVectorBias", "AccBias", "PostScale"]),
                     (0x17800, ["DMAConfig", "BaseAddr", "RowStride", "PlaneStride", "DepthStride", "GroupStride", "Fmt"])):
    NAMES.update({base + 4 * i: label for i, label in enumerate(labels)})
for i in range(16):
    for base, label in ((0x1F808, "CoeffDMAConfig"), (0x1F848, "CoeffBaseAddr"), (0x1F888, "CoeffBfrSize")):
        NAMES[base + i * 4] = f"{label}[{i}]"


def tasks(text, entry, size, count):
    decoded, writes, seen = [], [], set()
    offset = entry
    while True:
        if offset in seen or len(decoded) >= count or size < 40 or size % 4:
            raise ValueError("Invalid/cyclic task chain")
        seen.add(offset)
        raw = bounds(text, offset, size)
        h = list(struct.unpack_from("<10I", raw))
        header_size = 44 if h[6] & (1 << 24) else 40
        bounds(raw, 0, header_size)
        selectors = []
        for word, labels in ((h[8], ["RBase0", "RBase1", "WBase", "TBase"]),
                             (h[9], ["KBase0", "KBase1", "KBase2", "KBase3"])):
            for i, label in enumerate(labels):
                selectors.append(dict(name=label, enabled=bool(word & (1 << (i * 6 + 5))),
                                      bank=(word >> (i * 6)) & 31))
        task = {"task_id": h[0] & 0xFFFF, "offset": offset, "size": size,
                "sha256": digest(raw), "header_size": header_size,
                "header_words": list(struct.unpack_from(f"<{header_size // 4}I", raw)),
                "network_id": (h[0] >> 16) & 255, "last_network_id": bool(h[0] & (1 << 24)),
                "end_of_network": bool(h[0] & (1 << 25)), "log_events_raw": h[2],
                "exceptions_raw": h[3], "debug_log_events_raw": h[4], "debug_exceptions_raw": h[5],
                "task_flags_raw": h[6], "next_pointer": h[7],
                "next_size": (((h[1] >> 16) & 0x1FF) + 1) * 4,
                "base_selectors": selectors, "packets": []}
        cursor = header_size
        while cursor < len(raw):
            word = struct.unpack_from("<I", raw, cursor)[0]
            packet = {"order": len(task["packets"]), "offset": offset + cursor, "header_raw": word}
            cursor += 4
            if word == 0:
                packet["kind"] = "padding"
            else:
                address, words = word & 0x3FFFFFF, (word >> 26) + 1
                if address % 4 or address + words * 4 > 0x20000:
                    raise ValueError("Register packet outside H13 engine range")
                values = list(struct.unpack(f"<{words}I", bounds(raw, cursor, words * 4)))
                packet.update(kind="register-write", engine_offset=address, values=values)
                for i, value in enumerate(values):
                    addr = address + i * 4
                    block = next(label for base, label in reversed(BLOCKS) if addr >= base)
                    writes.append(dict(task_id=task["task_id"], packet_order=packet["order"],
                        word_order=i, stream_offset=offset + cursor + i * 4,
                        engine_offset=addr, value=value, block=block, name=NAMES.get(addr)))
                cursor += words * 4
            task["packets"].append(packet)
        decoded.append(task)
        if not h[7]:
            break
        if h[7] < offset + size or h[7] % 4:
            raise ValueError("Overlapping/misaligned tasks")
        offset, size = h[7], task["next_size"]
    if len(decoded) != count or not decoded[-1]["end_of_network"]:
        raise ValueError("Task count/end flag differs from entry descriptor")
    return decoded, writes


def decode(path, destination):
    data = path.read_bytes()
    info = container(data)
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "container.json", info)
    threads = [t for t in info["threads"] if "td_count" in t]
    if len(threads) != 1:
        raise ValueError("Expected one H13 entry thread")
    thread = threads[0]
    section = next(s for s in info["sections"] if s["segment"] == "__TEXT" and s["name"] == "__text")
    text = bounds(data, section["offset"], section["size"])
    (destination / "task-descriptors.bin").write_bytes(text)
    decoded, writes = tasks(text, thread["entry"] - section["addr"], thread["first_td_size"], thread["td_count"])
    save(destination / "tasks.json", decoded)
    save(destination / "registers.json", {"kind": "compiler command stream, not live MMIO",
         "architecture": "H13G", "ordered_writes": writes})
    bindings = {"compiler_bars": thread["bars"], "entry": thread["entry"],
                "first_td_size": thread["first_td_size"], "td_count": thread["td_count"],
                "symbols": info["symbols"], "sections": info["sections"],
                "unresolved": ["runtime IOVAs and relocation", "request indices to compiler BARs",
                               "scratch initial contents and aliases", "Linux handles[32] and btsp_handle"]}
    bindings["tensor_descriptors"] = []
    for t in info["threads"]:
        raw = bytes.fromhex(t["raw_hex"])
        if t["flavor"] == 3 and len(raw) >= 0x78:
            bindings["tensor_descriptors"].append({
                "direction_code": struct.unpack_from("<I", raw, 0x14)[0],
                "element_code": struct.unpack_from("<I", raw, 0x24)[0],
                "shape": list(struct.unpack_from("<4I", raw, 0x28)),
                "strides_bytes": list(struct.unpack_from("<4Q", raw, 0x50)),
                "bytes": struct.unpack_from("<Q", raw, 0x70)[0],
                "raw_hex": t["raw_hex"]})
    bindings["bar_owners"] = []
    for bank, address in enumerate(thread["bars"]):
        owners = [{"segment_index": i, "name": s["name"], "relative_offset": address - s["vmaddr"]}
                  for i, s in enumerate(info["segments"])
                  if address and s["vmaddr"] <= address < s["vmaddr"] + s["vmsize"]]
        bindings["bar_owners"].append({"bank": bank, "original_address": address, "owners": owners})
    payloads = destination / "payloads"
    payloads.mkdir()
    bindings["segment_payloads"] = []
    for i, segment in enumerate(info["segments"]):
        file = f"segment-{i}.bin"
        payload = bounds(data, segment["fileoff"], segment["filesize"])
        (payloads / file).write_bytes(payload)
        bindings["segment_payloads"].append({**segment, "file": "payloads/" + file})
    save(destination / "bindings.json", bindings)
    return {"source": str(path), "sha256": info["sha256"], "tasks": len(decoded),
            "register_writes": len(writes), "decoded": str(destination)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hwx", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    print(json.dumps(decode(args.hwx, args.out), indent=2))


if __name__ == "__main__":
    main()
