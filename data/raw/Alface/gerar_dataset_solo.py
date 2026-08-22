"""
=============================================================================
AgroLab AI - Gerador do Dataset Sintetico: ALFACE EM SOLO CONVENCIONAL (v2)
=============================================================================
Projeto : IC AgroLab AI - Alface Crespa (Lactuca sativa var. crispa)
Base    : documento cientifico "alface_agrolab_V2.pdf" (schema Tab.11, params Tab.9)
Sistema : SOLO CONVENCIONAL

O QUE MUDOU DA v1 -> v2 (alinhamento ao documento cientifico V2)
  + CULTIVAR como variavel categorica que DESLOCA o limiar de pendoamento
      (V2 sec.2.3: 'Vera','Veronica','Vanda' e linhagens Embrapa sao tolerantes
       ao calor; cultivares padrao pendoam mais cedo).
  + FASE fenologica via dias_apos_transplante (V2 schema, Tab.11).
  + TEMPERATURA DO SOLO (V2 Tab.7/9: ideal ~15-22C).
  + NITRATO foliar (proxy de qualidade; V2 sec.2.8) e SAUDE (score derivado).
  + nomes de coluna padronizados ao schema da Tabela 11.
  + volume ampliado (CENARIOS_POR_DIA alto).
  + rotulagem VETORIZADA (np.select) para aguentar o volume.

CRITERIOS MANTIDOS (desde o inicio do projeto)
  - foco exclusivo em alface crespa;
  - features ambientais ancoradas no clima REAL corrigido (clima/clima_corrigido.csv);
  - faixas validadas: pH solo 6,0-6,8 (V2 Tab.1); P e K pelo IAC Boletim 100
    (grupo hortalicas); gatilho de classe 3 refinado (calor cronico/agudo + frio);
  - rotulagem por regras de prioridade (V2 / info.md sec.6):
      P1 extremos climaticos -> classe 3 | P2 excesso -> classe 1
      P3 escassez -> classe 2 | senao -> classe 0.

PONTOS QUE AINDA DEPENDEM DE VALIDACAO (sinalizados)
  - N em mg/dm3 no solo e um PROXY (o Boletim 100 nao interpreta N por analise;
    N e manejado por dose kg/ha). Validar com o orientador.
  - nitrato_mg_kg e saude_pct sao DERIVADOS por proxy, nao medidos.
  - biomassa/num_folhas (targets de regressao do V2) exigem modelo de crescimento
    e NAO sao gerados aqui (etapa futura) para nao fabricar dado sem base.

SAIDA (subpasta solo/)
  solo/dataset_solo.csv
  solo/relatorio_dataset_solo.txt

Dependencias: pip install pandas numpy
=============================================================================
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
CLIMA_CSV = DIR / "clima" / "clima_corrigido.csv"
SAIDA_DIR = DIR / "solo"   # dataset do sistema SOLO nesta subpasta

SEED = 42
CENARIOS_POR_DIA = 25   # 8036 dias x 25 = ~200 mil amostras

# --- Cultivares crespas e tolerancia ao calor (V2 sec.2.3) -------------------
# 'bolting_media': temperatura MEDIA (movel 3 dias) que dispara pendoamento.
CULTIVARES = {
    "Vera":         {"tolerancia": "tolerante", "bolting_media": 27.0},
    "Veronica":     {"tolerancia": "tolerante", "bolting_media": 27.0},
    "Vanda":        {"tolerancia": "tolerante", "bolting_media": 26.0},
    "Grand Rapids": {"tolerancia": "padrao",    "bolting_media": 24.0},
    "Simpson":      {"tolerancia": "padrao",    "bolting_media": 24.0},
}

# --- Limiares agronomicos (fontes no cabecalho) ------------------------------
PH_IDEAL = (6.0, 6.8)                 # V2 Tab.1
PH_CRIT_BAIXO, PH_CRIT_ALTO = 5.5, 7.5

# P resina mg/dm3 - IAC B100 hortalicas (deficit <26; excesso >120)
P_DEFICIT, P_EXCESSO = 26, 120
# K trocavel mg/dm3 - IAC B100 (mmolc x 39.1): deficit <60; excesso >235
K_DEFICIT, K_EXCESSO = 60, 235
# N mineral mg/dm3 - PROXY (nao consta no B100) - VALIDAR
N_DEFICIT, N_EXCESSO = 15, 50

# Umidade do solo (% da agua disponivel: 100=cap. de campo, 0=ponto de murcha)
UMID_IRRIGAR, UMID_SATURADO = 50, 100

# Extremos climaticos (V2 Tab.3; info.md sec.4.3/6)
T_FRIO = 7.0            # T_min < 7 -> necrose/geada
T_CALOR_AGUDO = 32.0    # T_max > 32 -> estresse agudo severo

# --- Situacoes de manejo (controlam o equilibrio de classes; info.md 3.2) ----
SITUACOES = {
    "ideal":      0.38,   # -> classe 0
    "def_nutri":  0.17,   # -> classe 2
    "seco":       0.13,   # -> classe 2
    "exc_nutri":  0.14,   # -> classe 1
    "encharcado": 0.10,   # -> classe 1
    "ph_fora":    0.08,   # -> classe 1/2
}


def preparar_clima(clima: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a media movel de 3 dias (para calor cronico e temp. do solo)."""
    partes = []
    for polo, d in clima.groupby("polo"):
        d = d.sort_values("data").copy()
        d["T2M_3D"] = d["T2M"].rolling(3, min_periods=1).mean()
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def amostra_manejo(rng, n):
    """N, P, K, pH, umidade do solo dirigidos por situacao (+ vetor de situacao)."""
    sit = rng.choice(list(SITUACOES), size=n, p=list(SITUACOES.values()))
    N = rng.uniform(N_DEFICIT, N_EXCESSO, n)
    P = rng.uniform(P_DEFICIT, P_EXCESSO, n)
    K = rng.uniform(K_DEFICIT, K_EXCESSO, n)
    U = rng.uniform(UMID_IRRIGAR, UMID_SATURADO, n)
    pH = rng.uniform(PH_IDEAL[0], PH_IDEAL[1], n)

    m = sit == "def_nutri"
    if m.any():
        q = rng.integers(0, 3, m.sum()); idx = np.where(m)[0]
        N[idx[q == 0]] = rng.uniform(0.2*N_DEFICIT, N_DEFICIT, (q == 0).sum())
        P[idx[q == 1]] = rng.uniform(0.2*P_DEFICIT, P_DEFICIT, (q == 1).sum())
        K[idx[q == 2]] = rng.uniform(0.2*K_DEFICIT, K_DEFICIT, (q == 2).sum())
    m = sit == "exc_nutri"
    if m.any():
        q = rng.integers(0, 3, m.sum()); idx = np.where(m)[0]
        N[idx[q == 0]] = rng.uniform(N_EXCESSO, N_EXCESSO*2.2, (q == 0).sum())
        P[idx[q == 1]] = rng.uniform(P_EXCESSO, P_EXCESSO*2.2, (q == 1).sum())
        K[idx[q == 2]] = rng.uniform(K_EXCESSO, K_EXCESSO*2.2, (q == 2).sum())
    m = sit == "seco"
    U[m] = rng.uniform(5, UMID_IRRIGAR, m.sum())
    m = sit == "encharcado"
    U[m] = rng.uniform(UMID_SATURADO, 130, m.sum())
    m = sit == "ph_fora"
    if m.any():
        idx = np.where(m)[0]; baixo = rng.random(m.sum()) < 0.5
        pH[idx[baixo]] = rng.uniform(4.8, PH_CRIT_BAIXO, baixo.sum())
        pH[idx[~baixo]] = rng.uniform(PH_CRIT_ALTO, 8.2, (~baixo).sum())

    return (np.round(N, 1), np.round(P, 1), np.round(K, 1),
            np.round(pH, 2), np.round(U, 1), sit)


def rotular_vetorizado(df):
    """Regras de prioridade -> classe + motivo (vetorizado, np.select)."""
    frost   = df["temp_min_c"].to_numpy() < T_FRIO
    agudo   = df["temp_max_c"].to_numpy() > T_CALOR_AGUDO
    cronico = df["temp_ar_3d_c"].to_numpy() > df["bolting_lim"].to_numpy()
    umid_sat = df["umidade_solo_pct"].to_numpy() > UMID_SATURADO
    nut_exc  = ((df["N"] > N_EXCESSO) | (df["P"] > P_EXCESSO) | (df["K"] > K_EXCESSO)).to_numpy()
    ph_alto  = df["ph"].to_numpy() > PH_CRIT_ALTO
    umid_sec = df["umidade_solo_pct"].to_numpy() < UMID_IRRIGAR
    nut_def  = ((df["N"] < N_DEFICIT) | (df["P"] < P_DEFICIT) | (df["K"] < K_DEFICIT)).to_numpy()
    ph_baixo = df["ph"].to_numpy() < PH_CRIT_BAIXO

    conds = [frost, cronico, agudo, umid_sat, nut_exc, ph_alto, umid_sec, nut_def, ph_baixo]
    classes = [3, 3, 3, 1, 1, 1, 2, 2, 2]
    motivos = [
        "proteger: frio/geada (T_min<7C)",
        "proteger: calor sustentado (pendoamento)",
        "proteger: calor agudo (T_max>32C)",
        "travar irrigacao: solo encharcado",
        "corrigir excesso de nutrientes (risco salino)",
        "corrigir pH alto",
        "irrigar: umidade do solo baixa",
        "adubar: deficiencia nutricional",
        "corrigir pH baixo (calagem)",
    ]
    classe = np.select(conds, classes, default=0).astype(int)
    motivo = np.select(conds, motivos, default="condicoes dentro do ideal")
    return classe, motivo


def nitrato_proxy(N, dli, rng):
    """
    Proxy de nitrato foliar (mg/kg). V2 sec.2.8: acumulo ligado a manejo de N
    (mais N -> mais nitrato) e a luz (mais DLI -> menos nitrato). NAO e medido.
    """
    val = 2500 + 15*(N - 30) - 30*(dli - 20) + rng.normal(0, 250, len(N))
    return np.clip(val, 500, 6000).round(0)


def saude_proxy(df):
    """Score de saude 0-100: penaliza condicoes fora do ideal. Derivado."""
    s = np.full(len(df), 100.0)
    s -= 18 * (df["classe_acao"] == 3)
    s -= 12 * (df["classe_acao"] == 1)
    s -= 12 * (df["classe_acao"] == 2)
    s -= 6 * (~df["ph"].between(*PH_IDEAL))
    s -= 5 * (~df["umidade_relativa_pct"].between(60, 75))
    s -= 5 * (~df["vpd_kpa"].between(0.8, 1.2))
    return np.clip(s, 0, 100).round(0)


def main():
    SAIDA_DIR.mkdir(exist_ok=True)
    if not CLIMA_CSV.exists():
        raise SystemExit(f"[ERRO] Nao achei {CLIMA_CSV}. Rode o pipeline_clima.py antes.")

    rng = np.random.default_rng(SEED)
    clima = preparar_clima(pd.read_csv(CLIMA_CSV, parse_dates=["data"]))

    base = clima.loc[clima.index.repeat(CENARIOS_POR_DIA)].reset_index(drop=True)
    n = len(base)

    N, P, K, pH, U, sit = amostra_manejo(rng, n)

    cults = rng.choice(list(CULTIVARES), size=n)
    bolting = np.array([CULTIVARES[c]["bolting_media"] for c in cults])
    toler = np.array([CULTIVARES[c]["tolerancia"] for c in cults])
    dias = rng.integers(1, 46, n)
    fase = np.select([dias <= 10, dias <= 25, dias <= 40],
                     ["muda", "crescimento", "desenvolvimento"], default="colheita")
    temp_solo = np.clip(base["T2M_3D"].to_numpy() + rng.normal(0, 0.8, n), 3, 35).round(1)

    ds = pd.DataFrame({
        "timestamp": base["data"].dt.strftime("%Y-%m-%d"),
        "polo": base["polo"],
        "sistema": "solo",
        "cultivar": cults,
        "tolerancia_calor": toler,
        "dias_apos_transplante": dias,
        "fase": fase,
        "N": N, "P": P, "K": K,
        "ph": pH,
        "umidade_solo_pct": U,
        "temp_ar_c": base["T2M"].round(1),
        "temp_max_c": base["T2M_MAX"].round(1),
        "temp_min_c": base["T2M_MIN"].round(1),
        "temp_ar_3d_c": base["T2M_3D"].round(1),
        "temp_solo_c": temp_solo,
        "umidade_relativa_pct": base["RH2M"].round(1),
        "vpd_kpa": base["VPD"].round(2),
        "dli_mol_m2_d": base["DLI"].round(1),
        "bolting_lim": bolting,
    })

    classe, motivo = rotular_vetorizado(ds)
    ds["classe_acao"] = classe
    ds["motivo"] = motivo
    ds["nitrato_mg_kg"] = nitrato_proxy(ds["N"].to_numpy(), ds["dli_mol_m2_d"].to_numpy(), rng)
    ds["saude_pct"] = saude_proxy(ds)

    ds = ds.drop(columns=["bolting_lim", "temp_ar_3d_c"])
    ds = ds.sample(frac=1, random_state=SEED).reset_index(drop=True)

    saida = SAIDA_DIR / "dataset_solo.csv"
    ds.to_csv(saida, index=False, encoding="utf-8")

    dist = ds["classe_acao"].value_counts().sort_index()
    nomes = {0: "nao fazer nada", 1: "travar/corrigir excesso",
             2: "irrigar/corrigir falta", 3: "proteger (extremos)"}
    rel = ["=" * 70, "AgroLab AI - DATASET SOLO CONVENCIONAL (v2, alinhado ao doc V2)",
           "=" * 70,
           f"Gerado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | seed={SEED}",
           f"Dias de clima: {len(clima)} | cenarios/dia: {CENARIOS_POR_DIA}",
           f"TOTAL DE AMOSTRAS: {len(ds)}", "",
           "DISTRIBUICAO DAS CLASSES:"]
    for c in range(4):
        q = int(dist.get(c, 0))
        rel.append(f"  Classe {c} ({nomes[c]:24s}): {q:7d}  ({100*q/len(ds):5.1f}%)")
    rel += ["", f"CULTIVARES ({len(CULTIVARES)}): " + ", ".join(CULTIVARES),
            "  classe 3 por tolerancia (tolerante deve pendoar MENOS):",
            "   " + str(ds.groupby("tolerancia_calor")["classe_acao"]
                        .apply(lambda x: round((x == 3).mean()*100, 1)).to_dict()),
            "", "FASES: " + str(ds["fase"].value_counts().to_dict()),
            "", f"COLUNAS ({len(ds.columns)}): " + ", ".join(ds.columns), "",
            "ESTATISTICAS (manejo + ambiente + targets):",
            ds[["N", "P", "K", "ph", "umidade_solo_pct", "temp_ar_c", "temp_solo_c",
                "umidade_relativa_pct", "vpd_kpa", "dli_mol_m2_d",
                "nitrato_mg_kg", "saude_pct"]].describe().round(1).to_string(),
            "", "COERENCIA motivo x classe:",
            ds.groupby(["classe_acao", "motivo"]).size().to_string()]
    (SAIDA_DIR / "relatorio_dataset_solo.txt").write_text("\n".join(rel), encoding="utf-8")

    print("\n".join(rel))
    print(f"\nOK -> solo/dataset_solo.csv ({len(ds)} linhas, {len(ds.columns)} colunas)")


if __name__ == "__main__":
    main()