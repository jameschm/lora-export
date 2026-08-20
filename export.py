"""Export d'un adaptateur LoRA (PEFT/Unsloth) vers GGUF ou safetensors quantifies.

Marche aussi quand le modele de base a du code distant (trust_remote_code), cas ou
l'export integre d'Unsloth Studio echoue avec "Cannot determine model type: None".

Deux commandes, pas une de plus :

    python export.py --setup     # installe tout : .venv (exports) + .venv-abl (abliterix)
    python export.py --web       # ouvre l'interface, tout le travail se pilote de la

(--selftest lance les verifications internes ; --worker est l'appel interne du serveur.)

MIT. Conversion GGUF par https://github.com/ggml-org/llama.cpp
Abliteration par https://github.com/wuwangzhang1216/abliterix (AGPL, venv separe).
"""

import argparse, json, os, shutil, subprocess, sys, threading, venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTROOT = HERE / "out"
MERGED = OUTROOT / "merged-bf16"
ABLITERATED = OUTROOT / "abliterated-bf16"
ADAPTER = os.environ.get("LORA_ADAPTER", "")
LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"
RELEASES = "https://github.com/ggml-org/llama.cpp/releases"

# GGUF ecrits directement par convert_hf_to_gguf.py (aucun binaire a compiler)
CONVERT = ["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0"]
# GGUF necessitant le binaire llama-quantize (on convertit en f16 puis on requantifie)
QUANTIZE = [
    "q2_k", "q2_k_s", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q1_0", "q2_0",
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_xs", "iq3_s", "iq3_m", "iq4_nl", "iq4_xs", "mxfp4_moe",
]
# safetensors quantifies via torchao (relus par transformers/vLLM)
TORCHAO = {
    "fp8": "Float8DynamicActivationFloat8WeightConfig",
    "fp8wo": "Float8WeightOnlyConfig",
    "int8": "Int8DynamicActivationInt8WeightConfig",
    "int8wo": "Int8WeightOnlyConfig",
    "int4wo": "Int4WeightOnlyConfig",
}
# quantification bitsandbytes
BNB = {
    "nf4": dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    "fp4": dict(load_in_4bit=True, bnb_4bit_quant_type="fp4"),
    "int8bnb": dict(load_in_8bit=True),
}
GPU_ONLY = set(BNB) | {"fp8", "fp8wo"}  # ces quantifications n'ont pas de chemin CPU
BEST = ["gguf:q4_k_m", "gguf:q8_0", "hf:fp8"]  # mis en avant dans l'UI
ABL_VENV = HERE / ".venv-abl"  # abliterix pince des versions incompatibles avec le venv principal
ABL_BIN = ABL_VENV / ("Scripts" if os.name == "nt" else "bin") / ("abliterix.exe" if os.name == "nt" else "abliterix")
ABL_TRIALS = 20  # ponytail: 200 par defaut chez abliterix, trop pour du CPU
# sur Windows, `pip install torch` depuis PyPI donne la roue CPU : sans cet index, pas de GPU
CUDA_INDEX = "https://download.pytorch.org/whl/cu130"

REQUIREMENTS = """torch
transformers>=4.45
peft
accelerate
einops
safetensors
huggingface_hub
gguf
torchao
bitsandbytes
fastapi
uvicorn
"""


def methods():
    """Toutes les cibles d'export disponibles -> description."""
    m = {"bf16": "safetensors bf16 fusionne, non quantifie (reference qualite)"}
    for t in CONVERT:
        m[f"gguf:{t}"] = f"GGUF {t.upper()} - conversion directe, aucun binaire requis"
    for t in QUANTIZE:
        m[f"gguf:{t}"] = f"GGUF {t.upper()} - requantification, necessite llama-quantize"
    for k, v in TORCHAO.items():
        m[f"hf:{k}"] = f"safetensors torchao {v}"
    for k in BNB:
        m[f"hf:{k}"] = f"safetensors bitsandbytes {k}"
    m["adapter"] = "adaptateur LoRA seul, sans fusion (vLLM --enable-lora)"
    return m


def log(msg):
    print(msg, flush=True)


def human(n):
    for u in ("o", "Ko", "Mo", "Go"):
        if n < 1024 or u == "Go":
            return f"{n:.1f} {u}" if u != "o" else f"{n} o"
        n /= 1024


def weight(p: Path) -> int:
    """Taille d'un fichier ou d'un dossier de modele."""
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def report(out: Path) -> Path:
    """Etape finale : on regarde ce qu'on vient d'ecrire, au lieu de l'annoncer sur parole."""
    size = weight(out)
    if not size:
        raise SystemExit(f"sortie vide ou absente : {out}")
    log(f"OK -> {out} ({human(size)})")
    return out


def has_nvidia() -> bool:
    """GPU NVIDIA physiquement present (independant de la build de torch)."""
    return shutil.which("nvidia-smi") is not None


def cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pip_torch(py: Path):
    """Installe torch dans un venv, avec la roue CUDA si la machine a un GPU NVIDIA."""
    cmd = [str(py), "-m", "pip", "install", "torch"]
    if has_nvidia():
        cmd += ["--index-url", CUDA_INDEX]
        log(f"[setup] GPU NVIDIA detecte -> roue CUDA ({CUDA_INDEX})")
    else:
        log("[setup] pas de GPU NVIDIA -> torch CPU")
    subprocess.run(cmd, check=True)


def pick_device(device: str) -> str:
    """Repli sur le CPU si torch n'a pas CUDA : la fusion et la conversion GGUF n'en ont pas
    besoin (juste de la RAM). Pour du GPU : pip install torch --index-url <roue cu12x/cu13x>."""
    if device.startswith("cuda") and not cuda_ok():
        log("[device] torch sans CUDA -> repli sur cpu")
        return "cpu"
    return device


def dest_path(dest, default: Path, gguf=False) -> Path:
    """Destination finale : le champ Destination si rempli, sinon out/<nom par defaut>.
    Une destination qui designe un dossier (pas de suffixe .gguf) recoit le nom par defaut."""
    p = Path(dest).expanduser() if dest else default
    if gguf and p.suffix.lower() != ".gguf":
        p = p / default.name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def base_model_of(adapter: Path) -> str:
    return json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]


def base_dir(base: str):
    """Dossier local du modele de base, ou None s'il n'est pas encore en cache."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(base, local_files_only=True))
    except Exception:
        return Path(base) if Path(base).is_dir() else None


# --------------------------------------------------------------------------- preflight


def preflight(action, adapter, method="bf16", src_kind="auto", device="cuda:0"):
    """Tout ce qui peut echouer, verifie AVANT la fusion de 15 Go — pas apres.
    Renvoie {errors, warnings, info} ; le worker refuse de demarrer s'il y a une erreur."""
    err, warn, info = [], [], []
    adapter = Path(adapter) if adapter else None

    if not adapter or not (adapter / "adapter_config.json").exists():
        err.append(f"adaptateur introuvable : {adapter or '(vide)'} — il faut un dossier LoRA "
                   "contenant adapter_config.json")
        return {"errors": err, "warnings": warn, "info": info}

    base = base_model_of(adapter)
    info.append(f"modele de base : {base}")
    bdir = base_dir(base)
    if bdir:
        info.append(f"poids de base en cache : {human(weight(bdir))}")
    elif not MERGED.exists():
        warn.append(f"{base} n'est pas en cache local : il sera telecharge (plusieurs Go)")

    gpu = cuda_ok()
    if gpu:
        info.append("GPU CUDA disponible")
    elif has_nvidia():  # le piege : GPU present mais roue torch sans CUDA
        warn.append(f"GPU NVIDIA present mais torch est une build CPU — reinstalle avec "
                    f"pip install torch --index-url {CUDA_INDEX} (ou relance python export.py --setup)")
    else:
        info.append("pas de GPU NVIDIA : tout tourne en CPU (plus lent)")

    if action == "abliterate":
        if not ABL_BIN.exists():
            err.append(f"abliterix absent ({ABL_BIN}) — lance d'abord : python export.py --setup")
        if not gpu:
            warn.append("abliteration en CPU : compte plusieurs heures, et le reseau est requis "
                        "(datasets HF)")
    else:
        family, _, kind = method.partition(":")
        if family == "hf" and kind in GPU_ONLY and not gpu:
            err.append(f"hf:{kind} exige un GPU CUDA — choisis un type gguf: (100% CPU) ou "
                       "installe une roue torch CUDA")
        if family == "gguf":
            try:
                repo = llama_repo(quiet=True)
                info.append(f"llama.cpp : {repo}")
                if kind in QUANTIZE:
                    info.append(f"llama-quantize : {llama_quantize(repo)}")
            except SystemExit as e:
                err.append(str(e).splitlines()[0])
        if src_kind == "abliterated" and not (ABLITERATED / "config.json").exists():
            err.append("aucun modele abliterate : lance l'etape 2 avant d'exporter depuis lui")

    # place disque : fusion (~poids de base) + eventuel intermediaire f16 + sortie
    need = weight(bdir) if bdir and not MERGED.exists() else weight(MERGED)
    if need:
        factor = 2.4 if method.partition(":")[2] in QUANTIZE else 1.4
        free = shutil.disk_usage(OUTROOT if OUTROOT.exists() else HERE).free
        info.append(f"disque libre : {human(free)}, besoin estime : {human(need * factor)}")
        if free < need * factor:
            warn.append("place disque probablement insuffisante pour la chaine complete")
    return {"errors": err, "warnings": warn, "info": info}


# --------------------------------------------------------------------------- phases


def merge(adapter: Path, device: str) -> Path:
    """Fusionne le LoRA dans le modele de base -> out/merged-bf16, mis en cache.
    Le cache porte l'empreinte de l'adaptateur : changer de LoRA le reconstruit."""
    stamp = MERGED / ".source.json"
    ident = {"adapter": str(adapter.resolve()),
             "mtime": (adapter / "adapter_model.safetensors").stat().st_mtime
             if (adapter / "adapter_model.safetensors").exists() else 0}
    if (MERGED / "config.json").exists():
        if not stamp.exists():  # cache d'une version anterieure : on l'adopte, on ne le detruit pas
            log(f"[merge] cache sans empreinte, origine inconnue -> adopte tel quel ({MERGED})")
            stamp.write_text(json.dumps(ident))
            return MERGED
        if json.loads(stamp.read_text()) == ident:
            log(f"[merge] deja fait -> {MERGED}")
            return MERGED
        log("[merge] le cache vient d'un autre adaptateur (ou il a change) -> refusion")
        shutil.rmtree(MERGED)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    base = base_model_of(adapter)
    log(f"[merge] chargement de {base} (trust_remote_code=True, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True
    )
    log("[merge] application de l'adaptateur")
    model = PeftModel.from_pretrained(model, str(adapter), trust_remote_code=True).merge_and_unload()

    log(f"[merge] sauvegarde -> {MERGED}")
    MERGED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MERGED, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter), trust_remote_code=True).save_pretrained(MERGED)
    bdir = base_dir(base)  # code distant recopie : le dossier de sortie est autonome
    for py in (bdir.glob("*.py") if bdir else []):
        shutil.copy2(py, MERGED / py.name)
    stamp.write_text(json.dumps(ident))
    del model
    return report(MERGED)


def source(adapter: Path, device: str, which="auto") -> Path:
    """Modele que l'export quantifie. Explicite : un abliterated-bf16 oublie ne doit pas
    detourner silencieusement tous les exports suivants."""
    ready = (ABLITERATED / "config.json").exists()
    if which == "abliterated" or (which == "auto" and ready):
        log(f"[source] modele abliterate : {ABLITERATED}")
        return ABLITERATED
    log("[source] fusion LoRA simple (pas d'abliteration)")
    return merge(adapter, device)


def abliterate(adapter: Path, device: str, trials=ABL_TRIALS) -> Path:
    """Etape 2 : decensoring par abliterix (AGPL, appele comme programme externe)."""
    src = merge(adapter, device)
    if ABLITERATED.exists() and any(ABLITERATED.iterdir()):
        log(f"[abl] {ABLITERATED} n'est pas vide -> suppression (abliterix exige un dossier vide)")
        shutil.rmtree(ABLITERATED)
    log(f"[abl] abliteration, {trials} trials - lent sur CPU, reseau requis (datasets HF)")
    subprocess.run(
        [str(ABL_BIN),
         "--model.model-id", str(src),
         # bool | None cote abliterix : le flag nu ne suffit pas, il attend une valeur
         "--model.trust-remote-code", "True",
         "--non-interactive",
         "--non-interactive-output-dir", str(ABLITERATED),
         "--optimization.num-trials", str(trials),
         "--optimization.checkpoint-dir", str(OUTROOT / "abl-checkpoints")],
        check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    log("l'etape 3 peut maintenant exporter ce modele (source = auto ou abliterated)")
    return report(ABLITERATED)


# --------------------------------------------------------------------------- llama.cpp


def llama_dirs():
    """Emplacements ou chercher llama.cpp, du plus explicite au plus generique."""
    env = os.environ.get("LLAMA_CPP")
    home = Path.home()
    dirs = [Path(env)] if env else []
    dirs += [HERE / "llama.cpp", home / "llama.cpp", home / "src" / "llama.cpp",
             home / ".unsloth" / "llama.cpp",  # celui qu'installe Unsloth
             home / ".cache" / "llama.cpp", Path("C:/llama.cpp"),
             Path("/opt/llama.cpp"), Path("/usr/local/share/llama.cpp")]
    for exe in ("llama-quantize", "llama-cli", "llama-server"):
        found = shutil.which(exe)
        if found:  # .../repo/build/bin/llama-cli -> on remonte jusqu'a la racine du depot
            dirs += list(Path(found).resolve().parents)[:4]
    return dirs


def llama_repo(quiet=False) -> Path:
    """Depot llama.cpp deja present sur le poste (le convertisseur est du Python pur)."""
    for p in llama_dirs():
        if (p / "convert_hf_to_gguf.py").exists():
            if not quiet:
                log(f"[gguf] llama.cpp trouve : {p}")
            return p
    raise SystemExit(
        f"llama.cpp introuvable sur ce poste (git clone --depth 1 {LLAMA_CPP}, "
        "puis variable d'environnement LLAMA_CPP)\n"
        f"    binaires prets a l'emploi : {RELEASES}\n"
        "    cherche dans : " + ", ".join(str(p) for p in llama_dirs())
    )


def llama_quantize(repo: Path) -> str:
    exe = shutil.which("llama-quantize")
    if exe:
        return exe
    for d in [repo, *llama_dirs()]:
        for c in d.glob("build/bin/**/llama-quantize*"):
            if c.suffix in ("", ".exe"):  # sinon on attrape llama-quantize-impl.dll
                return str(c)
    raise SystemExit(
        f"binaire llama-quantize introuvable (compile llama.cpp ou prends une release {RELEASES})\n"
        "    ou choisis un type GGUF direct : " + ", ".join(CONVERT)
    )


# --------------------------------------------------------------------------- exports


def do_gguf(adapter, device, kind, dest, src_kind):
    repo = llama_repo()  # verifie avant la fusion, pas apres
    binary = llama_quantize(repo) if kind in QUANTIZE else None
    src = source(adapter, device, src_kind)
    conv = [sys.executable, str(repo / "convert_hf_to_gguf.py"), str(src)]
    out = dest_path(dest, OUTROOT / f"model-{kind}.gguf", gguf=True)

    if kind in CONVERT:
        subprocess.run(conv + ["--outtype", kind, "--outfile", str(out)], check=True)
    else:
        mid = OUTROOT / "model-f16.gguf"
        if not mid.exists():
            # le convertisseur n'ecrit que CONVERT ; le reste passe par llama-quantize
            log(f"[gguf] etape 1/2 : intermediaire f16 (obligatoire pour obtenir {kind.upper()})")
            subprocess.run(conv + ["--outtype", "f16", "--outfile", str(mid)], check=True)
        log(f"[gguf] etape 2/2 : requantification {kind.upper()} ({mid.name} est conserve)")
        subprocess.run([binary, str(mid), str(out), kind], check=True)

    report(out)
    log(f"llama-server -m {out} -ngl 99 -c 32768")
    return out


def do_hf(adapter, device, kind, dest, src_kind):
    import torch
    from transformers import AutoModelForCausalLM

    device = pick_device(device)
    if device == "cpu" and kind in GPU_ONLY:
        raise SystemExit(f"hf:{kind} exige un GPU CUDA (voir la verification de l'etape 1)")
    if kind in TORCHAO:
        import torchao.quantization as q
        from transformers import TorchAoConfig

        qc = TorchAoConfig(getattr(q, TORCHAO[kind])())
    else:
        from transformers import BitsAndBytesConfig

        cfg = dict(BNB[kind])
        if cfg.get("load_in_4bit"):
            cfg["bnb_4bit_compute_dtype"] = torch.bfloat16
        qc = BitsAndBytesConfig(**cfg)

    src = source(adapter, device, src_kind)
    out = dest_path(dest, OUTROOT / f"model-{kind}")
    log(f"[hf] quantification {kind}")
    model = AutoModelForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, device_map={"": device}, trust_remote_code=True,
        quantization_config=qc)
    out.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(out, safe_serialization=True)
    except Exception as e:  # certains tensor subclasses ne passent pas en safetensors
        log(f"[hf] safetensors refuse ({e}); repli sur le format torch")
        model.save_pretrained(out, safe_serialization=False)
    for f in src.iterdir():  # tokenizer + code distant
        if f.suffix in (".py", ".jinja") or f.name.startswith("tokenizer"):
            shutil.copy2(f, out / f.name)
    return report(out)


def do_adapter(adapter, dest):
    out = dest_path(dest, OUTROOT / "lora-adapter")
    out.mkdir(parents=True, exist_ok=True)
    for f in adapter.iterdir():
        if f.is_file():  # sans les checkpoint-*/
            shutil.copy2(f, out / f.name)
    report(out)
    log(f"vllm serve {base_model_of(adapter)} --trust-remote-code --enable-lora --lora-modules lora={out}")
    return out


def export(method, adapter: Path, device: str, dest=None, src_kind="auto"):
    family, _, kind = method.partition(":")
    if family == "gguf":
        return do_gguf(adapter, device, kind, dest, src_kind)
    if family == "hf":
        return do_hf(adapter, device, kind, dest, src_kind)
    if family == "adapter":
        return do_adapter(adapter, dest)
    src = source(adapter, device, src_kind)  # bf16 = le modele non quantifie lui-meme
    if not dest:
        return report(src)
    out = dest_path(dest, src)
    shutil.copytree(src, out, dirs_exist_ok=True)
    return report(out)


# --------------------------------------------------------------------------- interface web

PAGE = r"""<!doctype html><meta charset=utf-8><title>Export LoRA</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#d8d8d8;--card:#fafafa;--acc:#1a56db;--ok:#0a7a3d;--warn:#8a6d00;--err:#b42318}
@media(prefers-color-scheme:dark){
 :root{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2c3038;--card:#1b1e24;--acc:#5b8cff;--ok:#4ade80;--warn:#e3b341;--err:#ff7b72}}
*{box-sizing:border-box}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);
 max-width:900px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:22px;margin:0 0 2px}
.sub{color:var(--mut);font-size:13px;margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.card h2{font-size:14px;letter-spacing:.03em;text-transform:uppercase;color:var(--mut);margin:0 0 12px}
.card h2 span{color:var(--acc);font-weight:700;margin-right:8px}
.card h2 em{text-transform:none;font-style:normal;font-weight:400;color:var(--mut)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
.grid .wide{grid-column:1/-1}
@media(max-width:640px){.grid{grid-template-columns:1fr}}
label{display:block;font-size:12px;font-weight:600;color:var(--mut);margin-bottom:4px}
select,input{width:100%;font:inherit;padding:8px 10px;border:1px solid var(--line);
 border-radius:8px;background:var(--bg);color:var(--fg)}
button{font:inherit;font-weight:600;padding:9px 18px;border-radius:8px;border:1px solid var(--line);
 background:var(--bg);color:var(--fg);cursor:pointer}
button.p{background:var(--acc);border-color:var(--acc);color:#fff}
button:disabled{opacity:.5;cursor:not-allowed}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.hint{font-size:12.5px;color:var(--mut);margin:10px 0 0}
.tag{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;border:1px solid var(--line)}
.tag.y{color:var(--ok);border-color:var(--ok)} .tag.n{color:var(--warn);border-color:var(--warn)}
#chk{list-style:none;padding:0;margin:12px 0 0;font-size:13px}
#chk li{padding:2px 0} #chk li:before{margin-right:8px}
#chk li.e{color:var(--err)} #chk li.e:before{content:"X"}
#chk li.w{color:var(--warn)} #chk li.w:before{content:"!"}
#chk li.i{color:var(--mut)} #chk li.i:before{content:"-"}
#s{font-size:13px;color:var(--mut);margin-left:auto}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--acc);
 margin-right:6px;animation:b 1s infinite}
@keyframes b{50%{opacity:.25}}
pre{background:#0d1117;color:#c9d1d9;border-radius:10px;padding:12px;margin:0;
 height:320px;overflow:auto;white-space:pre-wrap;font:12.5px/1.5 ui-monospace,Consolas,monospace}
pre b{color:#4ade80}
</style>
<h1>Export du fine-tune</h1>
<p class=sub>Installation : <code>python export.py --setup</code> en ligne de commande. Le reste se passe ici.</p>

<div class=card><h2><span>1</span>Adaptateur</h2>
 <div class=grid><div class=wide><label for=ad>Dossier LoRA</label>
  <input id=ad value="ADAPTER" placeholder="C:\chemin\vers\mon-lora"></div></div>
 <div class=row><button id=ck>Verifier</button></div>
 <ul id=chk></ul></div>

<div class=card><h2><span>2</span>Abliteration <em>— optionnelle</em></h2>
 <div class=grid><div><label for=tr>Trials</label>
  <input id=tr type=number min=1 value="TRIALS"></div></div>
 <div class=row><button id=ab>Ablitérer</button> <span id=ast class=tag>...</span></div>
 <p class=hint>Fusionne le LoRA puis optimise le decensoring vers out/abliterated-bf16.
  Tres lent sans GPU, reseau requis.</p></div>

<div class=card><h2><span>3</span>Export</h2>
 <div class=grid>
  <div class=wide><label for=m>Format</label><select id=m>OPTIONS</select></div>
  <div class=wide><label for=out>Destination</label>
   <input id=out placeholder="vide = out/model-&lt;type&gt; — fichier .gguf ou dossier"></div>
  <div class=wide><label for=src>Modele a exporter</label><select id=src>
   <option value=auto>auto — abliterate si l'etape 2 a tourne</option>
   <option value=merged>fusion LoRA simple</option>
   <option value=abliterated>modele abliterate uniquement</option></select></div>
 </div>
 <div class=row><button id=b class=p>Exporter</button></div></div>

<div class=card><div class="row" style="margin-bottom:10px">
 <strong style="font-size:13px">Journal</strong><span id=s></span></div><pre id=o></pre></div>
<script>
let n=0,t=null,t0=0;
const ctl=[b,ab,ck,m,out,src,tr,ad];
function fmt(ms){const s=Math.round(ms/1000);return (s/60|0)+'m'+String(s%60).padStart(2,'0')+'s'}
function esc(x){return x.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function q(){return 'adapter='+encodeURIComponent(ad.value)+'&m='+encodeURIComponent(m.value)
 +'&src='+src.value}
async function check(act){const j=await(await fetch('/check?act='+act+'&'+q())).json();
 chk.innerHTML=j.errors.map(x=>'<li class=e>'+esc(x)+'</li>').join('')
  +j.warnings.map(x=>'<li class=w>'+esc(x)+'</li>').join('')
  +j.info.map(x=>'<li class=i>'+esc(x)+'</li>').join('');
 ast.textContent=j.abliterated?'modele abliterate present':'pas encore ablitere';
 ast.className='tag '+(j.abliterated?'y':'n');
 return j.errors.length==0}
ck.onclick=()=>check('export');
async function go(url,act){if(t)return;
 if(!await check(act)){s.textContent='verification en echec, voir l etape 1';return}
 o.textContent='';n=0;t0=Date.now();
 const j=await(await fetch(url,{method:'POST'})).json();
 if(j.ok){ctl.forEach(e=>e.disabled=true);t=setInterval(poll,600);poll()}
 else{s.textContent='refuse : '+(j.why||'une tache tourne deja')}}
ab.onclick=()=>go('/abliterate?adapter='+encodeURIComponent(ad.value)+'&trials='+tr.value,'abliterate');
b.onclick=()=>go('/run?'+q()+'&dest='+encodeURIComponent(out.value),'export');
async function poll(){const j=await(await fetch('/log?o='+n)).json();
 if(j.lines.length){const stick=o.scrollTop+o.clientHeight>=o.scrollHeight-40;
  o.innerHTML+=j.lines.map(l=>l.startsWith('OK -> ')?'<b>'+esc(l)+'</b>':esc(l)).join('\n')+'\n';
  n=j.next;if(stick)o.scrollTop=o.scrollHeight}
 s.innerHTML=j.running?'<span class=dot></span>en cours — '+fmt(Date.now()-t0)
  :(t0?'termine en '+fmt(Date.now()-t0):'');
 if(!j.running&&t){clearInterval(t);t=null;ctl.forEach(e=>e.disabled=false);check('export')}}
check('export');
</script>"""


def options_html():
    all_m = methods()
    groups = {
        "Recommande": BEST,
        "GGUF - conversion directe": [f"gguf:{t}" for t in CONVERT],
        "GGUF - requantification (llama-quantize)": [f"gguf:{t}" for t in QUANTIZE],
        "Safetensors quantifies": [f"hf:{k}" for k in list(TORCHAO) + list(BNB)],
        "Sans quantification": ["bf16", "adapter"],
    }
    return "".join(
        f"<optgroup label='{g}'>"
        + "".join(f'<option value="{k}">{k} - {all_m[k]}</option>' for k in ks)
        + "</optgroup>"
        for g, ks in groups.items()
    )


def spawn(job, lines, state):
    """Relance ce script en worker et stream sa sortie. Un seul travail a la fois :
    le sous-processus isole les 15 Go de torch du serveur web."""
    if state["running"]:
        return False
    lines.clear()
    state["running"] = True
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(job)]

    def worker():
        # utf-8 explicite : sinon la banniere unicode d'abliterix arrive en mojibake
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1,
                             env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        for line in p.stdout:
            lines.append(line.rstrip())
        p.wait()
        lines.append(f"--- termine (code {p.returncode}) ---")
        state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return True


def serve(port, adapter, device):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn

    lines, state = [], {"running": False}
    html = (PAGE.replace("OPTIONS", options_html()).replace("ADAPTER", adapter or "")
            .replace("TRIALS", str(ABL_TRIALS)))
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return html

    @app.get("/check")
    def check(act: str = "export", adapter: str = "", m: str = "bf16", src: str = "auto"):
        """Etape 0 : le preflight, affiche dans la carte 1 et rejoue avant chaque lancement."""
        r = preflight(act, adapter, m if m in methods() else "bf16", src, device)
        r["abliterated"] = (ABLITERATED / "config.json").exists()
        return r

    @app.post("/run")
    def start(adapter: str, m: str, dest: str = "", src: str = "auto"):
        if m not in methods() or src not in ("auto", "merged", "abliterated"):
            return {"ok": False, "why": "parametres invalides"}
        return {"ok": spawn({"action": "export", "adapter": adapter, "method": m,
                             "dest": dest.strip(), "source": src, "device": device}, lines, state)}

    @app.post("/abliterate")
    def start_abl(adapter: str, trials: int = ABL_TRIALS):
        return {"ok": spawn({"action": "abliterate", "adapter": adapter,
                             "trials": max(1, trials), "device": device}, lines, state)}

    @app.get("/log")
    def get_log(o: int = 0):
        return {"lines": lines[o:], "next": len(lines), "running": state["running"]}

    log(f"http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# --------------------------------------------------------------------------- entrees


def worker(job):
    """Appel interne du serveur : preflight, puis la phase demandee."""
    adapter = Path(job["adapter"]).expanduser()
    action = job["action"]
    r = preflight(action, adapter, job.get("method", "bf16"), job.get("source", "auto"), job["device"])
    for m in r["warnings"]:
        log(f"[!] {m}")
    if r["errors"]:
        raise SystemExit("\n".join(f"[x] {m}" for m in r["errors"]))
    if action == "abliterate":
        return abliterate(adapter, job["device"], job["trials"])
    return export(job["method"], adapter, job["device"], job["dest"] or None, job["source"])


def setup():
    """Installe tout : venv principal (exports) puis venv isole d'abliterix."""
    req = HERE / "requirements.txt"
    if not req.exists():
        req.write_text(REQUIREMENTS)
    venv.EnvBuilder(with_pip=True, upgrade_deps=True).create(HERE / ".venv")
    py = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip_torch(py)  # avant requirements.txt : sinon pip resout torch sur PyPI, donc en CPU
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)], check=True)
    if not ABL_BIN.exists():
        log(f"[setup] venv isole pour abliterix -> {ABL_VENV} (ses pins casseraient .venv)")
        venv.EnvBuilder(with_pip=True).create(ABL_VENV)
        abl_py = ABL_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip_torch(abl_py)  # idem : abliterix tirerait sinon un torch CPU en dependance
        subprocess.run([str(abl_py), "-m", "pip", "install", "abliterix"], check=True)
    log(f"pret : {py} export.py --web")


def selftest():
    m = methods()
    assert len(m) == len(CONVERT) + len(QUANTIZE) + len(TORCHAO) + len(BNB) + 2, len(m)
    assert set(BEST) <= set(m)
    assert not (set(CONVERT) & set(QUANTIZE)), "type GGUF en double"
    html = PAGE.replace("OPTIONS", options_html())
    assert all(f'value="{k}"' in html for k in m), "methode absente de l'UI"
    for line in PAGE.split("<script>")[1].splitlines():  # un \n aplati casse tout le script
        assert line.count("'") % 2 == 0, f"chaine JS non fermee : {line}"
    for field in ("id=ad", "id=ck", "id=chk", "id=tr", "id=ab", "id=m", "id=out", "id=src",
                  "id=b", "id=o", "id=s"):
        assert field in PAGE, f"champ {field} absent de l'UI"
    assert "id=st" not in PAGE, "le bouton Setup ne doit plus etre dans l'UI"

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        default = Path(d) / "model-q4_k_m.gguf"
        assert dest_path(None, default, gguf=True) == default
        assert dest_path(f"{d}/perso.gguf", default, gguf=True) == Path(d) / "perso.gguf"
        assert dest_path(d, default, gguf=True) == Path(d) / "model-q4_k_m.gguf"  # dossier
        assert dest_path(f"{d}/dossier", Path(d) / "model-fp8") == Path(d) / "dossier"
        # preflight : un adaptateur bidon doit etre refuse avant toute fusion
        bad = preflight("export", d)
        assert bad["errors"] and "adaptateur introuvable" in bad["errors"][0]
    assert human(1536) == "1.5 Ko" and human(0) == "0 o", human(1536)
    print(f"selftest ok ({len(m)} methodes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Deux commandes : --setup puis --web.")
    ap.add_argument("--setup", action="store_true", help="installe .venv et .venv-abl")
    ap.add_argument("--web", action="store_true", help="ouvre l'interface (tout le travail se pilote la)")
    ap.add_argument("--port", type=int, default=7801)
    ap.add_argument("--adapter", default=ADAPTER, help="pre-remplit le champ Adaptateur de l'interface")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 ou cpu si la VRAM manque")
    ap.add_argument("--selftest", action="store_true", help="verifications internes")
    ap.add_argument("--worker", help=argparse.SUPPRESS)  # appel interne du serveur
    a = ap.parse_args()

    if a.setup:
        setup()
    elif a.selftest:
        selftest()
    elif a.worker:
        worker(json.loads(a.worker))
    elif a.web:
        serve(a.port, a.adapter, a.device)
    else:
        ap.print_help()
