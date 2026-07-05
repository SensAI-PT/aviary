"""
Stadio 0 - Cost model + benchmark del disco per lo streaming degli expert MoE.

Domanda: e' FISICAMENTE possibile fare streaming degli expert da disco
e generare a velocita' usabile, su QUESTA macchina?

Due numeri che servono:
  1. Banda effettiva del disco in lettura RANDOM, a blocchi grossi quanto un expert.
  2. Quanti byte/token dobbiamo leggere -> da cui il tetto di token/sec.

Nessun modello richiesto. Gira in secondi.
"""
import os, sys, time, mmap, random, argparse

MB = 1024 * 1024
GB = 1024 * MB


def bench_disk(path_dir, expert_mb=12.0, total_mb=2048, n_reads=200):
    """Crea un file, poi misura lettura sequenziale e random a chunk = un expert."""
    os.makedirs(path_dir, exist_ok=True)
    fpath = os.path.join(path_dir, "_bench.bin")
    chunk = int(expert_mb * MB)
    total = int(total_mb * MB)
    total = (total // chunk) * chunk
    n_chunks = total // chunk

    # scrittura
    t = time.time()
    with open(fpath, "wb") as f:
        buf = os.urandom(chunk)
        for _ in range(n_chunks):
            f.write(buf)
        f.flush(); os.fsync(f.fileno())
    write_bw = total / (time.time() - t) / GB

    # prova a buttare via la page cache (best effort, serve permessi su Linux nativo)
    try:
        os.system("sync")
        with open("/proc/sys/vm/drop_caches", "w") as c:
            c.write("3")
    except Exception:
        pass  # su /mnt/c o senza root non si puo': il numero sara' ottimistico

    f = open(fpath, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

    # random reads a chunk di un expert
    idxs = [random.randrange(n_chunks) for _ in range(n_reads)]
    s = 0
    t = time.time()
    for i in idxs:
        off = i * chunk
        s += mm[off]          # tocca la prima pagina
        s += mm[off + chunk - 1]  # e l'ultima -> forza il caricamento del range
        _ = bytes(mm[off:off + chunk])  # legge davvero l'intero expert
    rand_bw = (n_reads * chunk) / (time.time() - t) / GB

    mm.close(); f.close()
    os.remove(fpath)
    return write_bw, rand_bw


def cost_model(name, n_layers, n_active, expert_mb, disk_bw_gbs, ram_resident_gb):
    """Stampa il tetto di token/sec in funzione dell'hit-rate della cache."""
    bytes_cold = n_layers * n_active * expert_mb / 1024  # GB letti per token se 0 cache
    print(f"\n--- {name} ---")
    print(f"  layer={n_layers}  expert_attivi/layer={n_active}  expert={expert_mb:.1f} MB")
    print(f"  parte densa residente in RAM stimata: ~{ram_resident_gb:.1f} GB")
    print(f"  byte da streammare per token (cache fredda): {bytes_cold*1024:.0f} MB")
    print(f"  tetto token/sec @ banda {disk_bw_gbs:.2f} GB/s, al variare dell'hit-rate cache:")
    for hit in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        eff = bytes_cold * (1 - hit)
        tps = disk_bw_gbs / eff if eff > 0 else float("inf")
        print(f"      hit {hit*100:5.1f}%  ->  {tps:6.2f} tok/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="cartella su cui benchmarkare il disco")
    ap.add_argument("--expert-mb", type=float, default=12.0)
    args = ap.parse_args()

    print(f"Benchmark disco su: {os.path.abspath(args.dir)}  (chunk={args.expert_mb} MB)")
    wbw, rbw = bench_disk(args.dir, expert_mb=args.expert_mb)
    print(f"  scrittura seq : {wbw:.2f} GB/s")
    print(f"  lettura random: {rbw:.2f} GB/s   <-- numero che conta per lo streaming")

    # Scenari. expert_mb a Q4 ~ (hidden*inter*3)*0.5B.
    # OLMoE 1B-7B: 16 layer, 8 attivi, hidden 2048 inter 1024 -> ~3 MB Q4
    cost_model("OLMoE 1B-7B (piccolo, lo useremo allo Stadio 1)",
               n_layers=16, n_active=8, expert_mb=3.0,
               disk_bw_gbs=rbw, ram_resident_gb=1.0)
    # DeepSeek-V3/V4 class: ~60 layer MoE, 8 attivi, expert ~6 MB Q2
    cost_model("DeepSeek/GLM class @ Q2 (il sogno finale)",
               n_layers=60, n_active=8, expert_mb=6.0,
               disk_bw_gbs=rbw, ram_resident_gb=10.0)
