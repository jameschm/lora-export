# lora-export

Exporte un adaptateur LoRA (PEFT / Unsloth) en **GGUF** ou en **safetensors quantifies**,
y compris quand le modele de base embarque du code distant (`trust_remote_code`) — le cas
ou l'export integre d'Unsloth Studio echoue avec `Cannot determine model type for config file: None`.

Un seul fichier, une interface web locale, 45 formats de sortie.

## Installation

```bash
python export.py --setup          # cree .venv/ et .venv-abl/, installe requirements.txt
```

`--setup` regarde si `nvidia-smi` existe : si oui il installe torch depuis l'index CUDA
(`https://download.pytorch.org/whl/cu130`) dans les deux venvs, sinon la roue CPU. C'est
necessaire parce que `pip install torch` depuis PyPI donne une build **sans CUDA** sur Windows —
le GPU est alors invisible pour torch comme pour abliterix, sans le moindre message d'erreur.
Pour une autre version de CUDA, change `CUDA_INDEX` en tete de `export.py`.

La fusion et toute la chaine GGUF marchent en CPU (il faut surtout de la RAM : ~2x la taille du
modele en bf16). L'abliteration et les sorties `hf:fp8` / `nf4` / `fp4` / `int8bnb`, elles,
veulent un vrai GPU — le preflight le signale au lieu de laisser deviner.

## Utilisation

Deux commandes, pas une de plus :

```bash
python export.py --setup     # installe .venv (exports) + .venv-abl (abliterix)
python export.py --web       # http://127.0.0.1:7801 — tout le travail se pilote la
```

Options d'appoint : `--port`, `--adapter` (pre-remplit le champ de l'interface), `--device`,
`--selftest`. Il n'y a **pas** de mode ligne de commande pour exporter ou ablitérer :
l'interface est le seul pilote.

### Les etapes, dans l'ordre

**0. Preflight** — a chaque ouverture et avant chaque lancement, la carte 1 verifie ce qui peut
echouer : adaptateur lisible, modele de base resolu et en cache, CUDA present, llama.cpp et
`llama-quantize` disponibles *si* le format choisi en a besoin, modele abliterate present *si*
la source l'exige, place disque estimee. Une erreur bloque le lancement — avant la fusion de
15 Go, pas apres.

**1. Adaptateur** — le dossier LoRA, et le bouton *Verifier*.

**2. Abliteration** *(optionnelle)* — fusionne le LoRA puis fait tourner
[abliterix](https://github.com/wuwangzhang1216/abliterix) en mode non-interactif ; resultat et
manifest de reproductibilite dans `out/abliterated-bf16`. Le champ *Trials* regle le budget de
recherche. Tres lent sans GPU, reseau requis (datasets HF).

**3. Export** — *Format* parmi 45 cibles, *Destination* (fichier `.gguf` ou dossier ; vide =
`out/model-<type>`), *Modele a exporter* : `auto` (l'abliterate s'il existe), `merged` (fusion
simple) ou `abliterated` (echoue si l'etape 2 n'a pas tourne).

**4. Verification** — chaque phase se termine par `OK -> chemin (taille)` : le script regarde ce
qu'il a ecrit au lieu de l'annoncer sur parole.

La fusion bf16 est mise en cache dans `out/merged-bf16` avec l'empreinte de l'adaptateur
(`.source.json`) : changer de LoRA la reconstruit au lieu de reexporter l'ancien en silence.

## Methodes

| Famille | Exemples | Remarques |
|---|---|---|
| `gguf:` direct | `f32 f16 bf16 q8_0 tq1_0 tq2_0` | `convert_hf_to_gguf.py`, Python pur, rien a compiler |
| `gguf:` requantifie | `q4_k_m q5_k_m q6_k iq4_xs mxfp4_moe` … | passe par un f16 intermediaire puis le binaire `llama-quantize` |
| `hf:` torchao | `fp8 fp8wo int8 int8wo int4wo` | safetensors relisibles par transformers / vLLM |
| `hf:` bitsandbytes | `nf4 fp4 int8bnb` | |
| autres | `bf16`, `adapter` | fusion non quantifiee, ou adaptateur seul pour `vllm --enable-lora` |

### llama.cpp

Le script **n'installe rien** : il reutilise le llama.cpp deja present sur le poste.
Il cherche, dans l'ordre : `$LLAMA_CPP`, `./llama.cpp`, `~/llama.cpp`, `~/src/llama.cpp`,
`~/.unsloth/llama.cpp`, `~/.cache/llama.cpp`, `C:/llama.cpp`, `/opt/llama.cpp`,
`/usr/local/share/llama.cpp`, puis les dossiers parents de `llama-cli` / `llama-server` /
`llama-quantize` trouves dans le `PATH`.

S'il ne trouve rien, il s'arrete en indiquant quoi faire :

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
export LLAMA_CPP=/chemin/vers/llama.cpp
```

Les types `gguf:` directs n'ont besoin que du depot (Python pur). Les types requantifies
demandent en plus le binaire `llama-quantize` compile, ou une
[release](https://github.com/ggml-org/llama.cpp/releases).

## Quel format choisir

- **Latence mono-requete sur un poste Windows** : GGUF `q4_k_m` (ou `q8_0` si la VRAM le permet),
  llama.cpp a le CUDA natif la ou vLLM demande WSL2.
- **Debit en batch sur Linux/WSL2** : `hf:fp8` sur GPU Ada/Hopper/Blackwell.
- **Reference qualite** : `bf16`.

## Licence

MIT. [abliterix](https://github.com/wuwangzhang1216/abliterix) est sous AGPL-3.0-or-later :
il est appele comme programme externe dans son propre venv, jamais importe, donc ce projet
reste MIT. Un `import abliterix` changerait la donne.
