"""
=============================================================================
AgroLab AI - Gerador do Dataset Sintetico: ALFACE EM HIDROPONIA NFT
=============================================================================
Projeto : IC AgroLab AI - Alface Crespa (Lactuca sativa var. crispa)
Base    : documento cientifico "alface_agrolab_V2.pdf" + info.md
Sistema : HIDROPONICO NFT (Nutrient Film Technique)

DIFERENCAS EM RELACAO AO GERADOR DE SOLO (e por que existem)
  1) N, P, K deixam de ser PROXY e viram CONCENTRACAO REAL DA SOLUCAO (mg/L),
     pela solucao de referencia Furlani/IAC (V2 Tab.4). No solo, N nao e
     interpretado por analise; aqui ele e uma variavel medida/controlada.
  2) A variavel de "umidade" vira CE DA SOLUCAO (mS/cm), cujo alvo MUDA COM A
     FASE (V2 Tab.2: bercario 0,5-0,8 -> pleno 1,2-2,0). Isso da um sinal rico
     ao modelo: CE 0,7 e adequada na muda, mas e deficit no crescimento pleno.
  3) Nutrientes ESCALAM COM A CE (a CE e proxy da concentracao ionica total),
     criando a correlacao realista CE<->N,P,K pedida no info.md sec.3.3.
  4) Entram variaveis exclusivas do NFT: TEMPERATURA DA SOLUCAO, OXIGENIO
     DISSOLVIDO (fisicamente derivado da temperatura) e NIVEL DO RESERVATORIO.
  5) AMBIENTE PROTEGIDO: hidroponia comercial de folhosas ocorre sob cobertura.
     O clima externo e atenuado (estufa aquece de dia, protege do frio a noite,
     eleva UR e transmite ~70% da luz). Fatores declarados e ajustaveis abaixo.

MANTIDO DO GERADOR DE SOLO (mesmos criterios do projeto)
  - foco exclusivo em alface crespa;
  - features ambientais ancoradas no clima REAL corrigido (clima/clima_corrigido.csv);
  - CULTIVAR deslocando o limiar de pendoamento (V2 sec.2.3);
  - fase fenologica (dias apos transplante);
  - rotulagem por REGRAS DE PRIORIDADE (info.md sec.6):
      P1 extremos -> classe 3 | P2 excesso -> classe 1 | P3 escassez -> classe 2
      | senao -> classe 0;
  - rotulagem vetorizada; volume alto; targets derivados sinalizados como proxy.

PONTOS SINALIZADOS (honestidade)
  - nitrato_mg_kg e saude_pct sao DERIVADOS por proxy, nao medidos.
  - biomassa/num_folhas NAO sao gerados (exigem modelo de crescimento).
  - fatores de ambiente protegido sao valores tipicos de literatura/pratica,
    ajustaveis; declarar no artigo.

SAIDA (subpasta nft/)
  nft/dataset_nft.csv
  nft/relatorio_dataset_nft.txt

Dependencias: pip install pandas numpy
=============================================================================
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
CLIMA_CSV = DIR / "clima" / "clima_corrigido.csv"
SAIDA_DIR = DIR / "nft"

SEED = 42
CENARIOS_POR_DIA = 25   # 8036 dias x 25 = ~200 mil amostras

# --- Cultivares (mesmo esquema do gerador de solo; V2 sec.2.3) --------------
CULTIVARES = {
    "Vera":         {"tolerancia": "tolerante", "bolting_media": 27.0},
    "Veronica":     {"tolerancia": "tolerante", "bolting_media": 27.0},
    "Vanda":        {"tolerancia": "tolerante", "bolting_media": 26.0},
    "Grand Rapids": {"tolerancia": "padrao",    "bolting_media": 24.0},
    "Simpson":      {"tolerancia": "padrao",    "bolting_media": 24.0},
}

# --- AMBIENTE PROTEGIDO (estufa) - fatores declarados ----------------------
# Efeito tipico de cobertura plastica sobre o clima externo.
ESTUFA_TMAX_OFFSET = 1.5    # C a mais no pico diurno (efeito estufa JA COM
                            # ventilacao lateral/zenital e sombrite, praxe em
                            # folhosas; sem manejo o offset seria ~3-5C)
ESTUFA_TMIN_OFFSET = 1.5    # C a mais a noite (protecao contra geada)
ESTUFA_UR_OFFSET = 5.0      # pontos percentuais a mais (menos ventilacao)
ESTUFA_TRANSMISSAO_LUZ = 0.70   # filme transmite ~70% da luz -> DLI interno

# --- pH da solucao (V2 Tab.1: NFT ideal 5,5-6,5; aceitavel 5,5-6,8) --------
PH_IDEAL = (5.5, 6.5)
PH_OPER_ALTO = 6.8      # acima disso: corrigir (precipita micronutrientes)
PH_OPER_BAIXO = 5.5     # abaixo disso: corrigir
PH_CRIT_BAIXO, PH_CRIT_ALTO = 4.5, 7.5

# --- CE por fase (V2 Tab.2), em mS/cm --------------------------------------
CE_POR_FASE = {
    "muda":            (0.5, 0.8),   # bercario
    "crescimento":     (0.8, 1.2),   # crescimento inicial
    "desenvolvimento": (1.2, 2.0),   # crescimento pleno
    "colheita":        (1.2, 2.0),
}
CE_FORCA_PLENA = 2.0   # CE da solucao de Furlani em forca plena (V2 Tab.4)

# --- Solucao de referencia Furlani/IAC em forca plena, mg/L (V2 Tab.4) -----
FURLANI = {"N": 196.0, "P": 43.0, "K": 186.0, "Ca": 146.0, "Mg": 33.0, "S": 49.0}

# --- Solucao / raiz (V2 Tab.7) ---------------------------------------------
TSOL_IDEAL = (18.0, 24.0)
TSOL_CRITICA = 27.0     # acima: queda de O2 e do crescimento
OD_MINIMO = 5.0         # mg/L (ideal ~8)
OD_CRITICO = 4.0        # mg/L
RESERV_MINIMO = 30.0    # % do reservatorio -> repor

# --- Extremos do ar (V2 Tab.3) ---------------------------------------------
T_FRIO = 7.0
T_CALOR_AGUDO = 32.0

# --- Situacoes de manejo (equilibram as classes; info.md sec.3.2) ----------
SITUACOES = {
    "ideal":             0.44,
    "ce_baixa":          0.14,   # diluicao / falha de fertirrigacao -> classe 2
    "ce_alta":           0.13,   # salinizacao / superdosagem       -> classe 1
    "ph_fora":           0.10,   # -> classe 1 ou 2
    "reservatorio_baixo": 0.10,  # -> classe 2
    "solucao_quente":    0.05,   # T solucao > 27C                  -> classe 3
    "falha_bomba":       0.04,   # aeracao cai -> hipoxia (OD baixo)-> classe 3
}
# Nota: solucao_quente e falha_bomba sao FALHAS DE SISTEMA - eventos raros por
# definicao. Probabilidades baixas evitam inflar a classe 3 artificialmente;
# o restante da classe 3 emerge do clima real (info.md sec.3.2).


def preparar_clima(clima: pd.DataFrame) -> pd.DataFrame:
    """Media movel de 3 dias (calor cronico) por polo."""
    partes = []
    for _, d in clima.groupby("polo"):
        d = d.sort_values("data").copy()
        d["T2M_3D"] = d["T2M"].rolling(3, min_periods=1).mean()
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def od_saturacao(temp_c):
    """
    O2 dissolvido na saturacao (mg/L) em agua doce, em funcao da temperatura.
    Aproximacao classica: quanto mais quente, menos O2 dissolve.
    ~9,0 mg/L a 20C | ~8,2 a 25C | ~7,4 a 30C
    """
    t = np.asarray(temp_c, dtype=float)
    return 14.652 - 0.41022*t + 0.007991*t**2 - 0.000077774*t**3


def amostra_manejo(rng, n, fase, temp_ar_3d):
    """
    Gera CE, pH, temperatura da solucao, aeracao e nivel do reservatorio,
    dirigidos por situacao. Retorna tambem os limites de CE da fase.
    """
    sit = rng.choice(list(SITUACOES), size=n, p=list(SITUACOES.values()))

    ce_min = np.array([CE_POR_FASE[f][0] for f in fase])
    ce_max = np.array([CE_POR_FASE[f][1] for f in fase])

    # CE dentro do alvo da fase por padrao
    CE = rng.uniform(ce_min, ce_max)
    m = sit == "ce_baixa"
    CE[m] = rng.uniform(0.15, 1.0, m.sum()) * ce_min[m]      # abaixo do alvo
    m = sit == "ce_alta"
    CE[m] = ce_max[m] * rng.uniform(1.05, 1.9, m.sum())      # acima do alvo

    # pH da solucao
    pH = rng.uniform(PH_IDEAL[0], PH_IDEAL[1], n)
    m = sit == "ph_fora"
    if m.any():
        idx = np.where(m)[0]
        baixo = rng.random(m.sum()) < 0.5
        pH[idx[baixo]] = rng.uniform(4.2, PH_OPER_BAIXO, baixo.sum())
        pH[idx[~baixo]] = rng.uniform(PH_OPER_ALTO, 7.9, (~baixo).sum())

    # Temperatura da solucao: segue o ar amortecido (inercia termica da agua)
    t_sol = 0.55 * temp_ar_3d + 0.45 * 21.0 + rng.normal(0, 1.0, n)
    m = sit == "solucao_quente"
    t_sol[m] = rng.uniform(TSOL_CRITICA, 32.0, m.sum())
    t_sol = np.clip(t_sol, 8.0, 34.0)

    # Aeracao (bomba). Falha -> queda drastica -> hipoxia
    aeracao = rng.uniform(0.70, 0.98, n)
    m = sit == "falha_bomba"
    aeracao[m] = rng.uniform(0.20, 0.45, m.sum())

    # Nivel do reservatorio
    nivel = rng.uniform(RESERV_MINIMO, 100.0, n)
    m = sit == "reservatorio_baixo"
    nivel[m] = rng.uniform(5.0, RESERV_MINIMO, m.sum())

    return CE, pH, t_sol, aeracao, nivel, sit, ce_min, ce_max


def rotular_vetorizado(df):
    """Regras de prioridade -> classe + motivo (vetorizado)."""
    # P1 - extremos (classe 3)
    frio = df["temp_min_c"].to_numpy() < T_FRIO
    agudo = df["temp_max_c"].to_numpy() > T_CALOR_AGUDO
    cronico = df["temp_ar_3d_c"].to_numpy() > df["bolting_lim"].to_numpy()
    sol_quente = df["temp_solucao_c"].to_numpy() > TSOL_CRITICA
    hipoxia = df["od_mg_l"].to_numpy() < OD_CRITICO
    # P2 - excesso (classe 1)
    ce_alta = df["ce_ms_cm"].to_numpy() > df["ce_alvo_max"].to_numpy()
    ph_alto = df["ph"].to_numpy() > PH_OPER_ALTO
    # P3 - escassez (classe 2)
    ce_baixa = df["ce_ms_cm"].to_numpy() < df["ce_alvo_min"].to_numpy()
    nivel_baixo = df["nivel_reservatorio_pct"].to_numpy() < RESERV_MINIMO
    ph_baixo = df["ph"].to_numpy() < PH_OPER_BAIXO

    conds = [frio, cronico, agudo, sol_quente, hipoxia,
             ce_alta, ph_alto,
             ce_baixa, nivel_baixo, ph_baixo]
    classes = [3, 3, 3, 3, 3, 1, 1, 2, 2, 2]
    motivos = [
        "proteger: frio/geada (T_min<7C)",
        "proteger: calor sustentado (pendoamento)",
        "proteger: calor agudo (T_max>32C)",
        "proteger: solucao quente (>27C, queda de O2)",
        "proteger: hipoxia radicular (OD<4 mg/L)",
        "travar fertirrigacao: CE acima do alvo da fase",
        "corrigir pH alto (precipita micronutrientes)",
        "repor nutrientes: CE abaixo do alvo da fase",
        "repor agua: nivel do reservatorio baixo",
        "corrigir pH baixo",
    ]
    classe = np.select(conds, classes, default=0).astype(int)
    motivo = np.select(conds, motivos, default="condicoes dentro do ideal")
    return classe, motivo


def nitrato_proxy(N_mg_l, dli, rng):
    """
    Proxy de nitrato foliar (mg/kg). V2 sec.2.8 e Tab.5: hidroponia acumula
    MAIS nitrato que o convencional; mais N na solucao -> mais nitrato;
    mais luz (DLI) -> menos nitrato (a luz dirige a reducao do nitrato).
    NAO e medido.
    """
    val = 3000 + 6.0*(N_mg_l - 150) - 35*(dli - 18) + rng.normal(0, 280, len(N_mg_l))
    return np.clip(val, 600, 6500).round(0)


def saude_proxy(df):
    """Score 0-100 penalizando desvios. Derivado (proxy)."""
    s = np.full(len(df), 100.0)
    s -= 18 * (df["classe_acao"] == 3)
    s -= 12 * (df["classe_acao"] == 1)
    s -= 12 * (df["classe_acao"] == 2)
    s -= 6 * (~df["ph"].between(*PH_IDEAL))
    s -= 6 * (df["od_mg_l"] < OD_MINIMO)
    s -= 5 * (~df["temp_solucao_c"].between(*TSOL_IDEAL))
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

    # --- AMBIENTE PROTEGIDO: atenua o clima externo ------------------------
    t_med_ext = base["T2M"].to_numpy()
    t_max_int = base["T2M_MAX"].to_numpy() + ESTUFA_TMAX_OFFSET
    t_min_int = base["T2M_MIN"].to_numpy() + ESTUFA_TMIN_OFFSET
    t_med_int = (t_max_int + t_min_int) / 2
    t_3d_int = base["T2M_3D"].to_numpy() + (ESTUFA_TMAX_OFFSET + ESTUFA_TMIN_OFFSET) / 2
    ur_int = np.clip(base["RH2M"].to_numpy() + ESTUFA_UR_OFFSET, 0, 100)
    dli_int = base["DLI"].to_numpy() * ESTUFA_TRANSMISSAO_LUZ

    # VPD recalculado para o ambiente interno (T e UR mudaram)
    es = 0.6108 * np.exp((17.27 * t_med_int) / (t_med_int + 237.3))
    vpd_int = np.clip(es * (1 - ur_int / 100.0), 0, None)

    # --- cultivar e fenologia ---------------------------------------------
    cults = rng.choice(list(CULTIVARES), size=n)
    bolting = np.array([CULTIVARES[c]["bolting_media"] for c in cults])
    toler = np.array([CULTIVARES[c]["tolerancia"] for c in cults])
    dias = rng.integers(1, 46, n)
    fase = np.select([dias <= 10, dias <= 25, dias <= 40],
                     ["muda", "crescimento", "desenvolvimento"], default="colheita")

    # --- manejo da solucao -------------------------------------------------
    CE, pH, t_sol, aeracao, nivel, sit, ce_min, ce_max = amostra_manejo(
        rng, n, fase, t_3d_int)

    # nutrientes ESCALAM com a CE (CE = proxy da concentracao ionica total)
    fator = CE / CE_FORCA_PLENA
    ruido = lambda: rng.normal(1.0, 0.06, n)   # consumo diferencial da planta
    N = np.clip(FURLANI["N"] * fator * ruido(), 5, None)
    P = np.clip(FURLANI["P"] * fator * ruido(), 1, None)
    K = np.clip(FURLANI["K"] * fator * ruido(), 5, None)
    Ca = np.clip(FURLANI["Ca"] * fator * ruido(), 5, None)
    Mg = np.clip(FURLANI["Mg"] * fator * ruido(), 1, None)

    # oxigenio dissolvido: saturacao (funcao da temperatura) x aeracao
    od = np.clip(od_saturacao(t_sol) * aeracao, 0.5, 14.0)

    ds = pd.DataFrame({
        "timestamp": base["data"].dt.strftime("%Y-%m-%d"),
        "polo": base["polo"],
        "sistema": "hidroponia_nft",
        "cultivar": cults,
        "tolerancia_calor": toler,
        "dias_apos_transplante": dias,
        "fase": fase,
        # --- solucao nutritiva (manejo/decisao) ---
        "ce_ms_cm": CE.round(2),
        "ph": pH.round(2),
        "N": N.round(1), "P": P.round(1), "K": K.round(1),
        "Ca": Ca.round(1), "Mg": Mg.round(1),
        "temp_solucao_c": t_sol.round(1),
        "od_mg_l": od.round(2),
        "nivel_reservatorio_pct": nivel.round(1),
        # --- ambiente interno (estufa, derivado do clima real) ---
        "temp_ar_c": t_med_int.round(1),
        "temp_max_c": t_max_int.round(1),
        "temp_min_c": t_min_int.round(1),
        "temp_ar_3d_c": t_3d_int.round(1),
        "umidade_relativa_pct": ur_int.round(1),
        "vpd_kpa": vpd_int.round(2),
        "dli_mol_m2_d": dli_int.round(1),
        # --- referencia externa (rastreabilidade) ---
        "temp_ar_externa_c": t_med_ext.round(1),
        # --- auxiliares para rotulagem ---
        "ce_alvo_min": ce_min.round(2),
        "ce_alvo_max": ce_max.round(2),
        "bolting_lim": bolting,
    })

    classe, motivo = rotular_vetorizado(ds)
    ds["classe_acao"] = classe
    ds["motivo"] = motivo
    ds["nitrato_mg_kg"] = nitrato_proxy(ds["N"].to_numpy(), ds["dli_mol_m2_d"].to_numpy(), rng)
    ds["saude_pct"] = saude_proxy(ds)

    ds = ds.drop(columns=["bolting_lim", "temp_ar_3d_c"])
    ds = ds.sample(frac=1, random_state=SEED).reset_index(drop=True)

    saida = SAIDA_DIR / "dataset_nft.csv"
    ds.to_csv(saida, index=False, encoding="utf-8")

    # --- relatorio ---------------------------------------------------------
    dist = ds["classe_acao"].value_counts().sort_index()
    nomes = {0: "nao fazer nada", 1: "travar/corrigir excesso",
             2: "repor/corrigir falta", 3: "proteger (extremos)"}
    rel = ["=" * 70, "AgroLab AI - DATASET HIDROPONICO NFT", "=" * 70,
           f"Gerado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} | seed={SEED}",
           f"Dias de clima: {len(clima)} | cenarios/dia: {CENARIOS_POR_DIA}",
           f"TOTAL DE AMOSTRAS: {len(ds)}", "",
           "AMBIENTE PROTEGIDO (fatores aplicados ao clima externo):",
           f"  T_max +{ESTUFA_TMAX_OFFSET}C | T_min +{ESTUFA_TMIN_OFFSET}C | "
           f"UR +{ESTUFA_UR_OFFSET}pp | luz x{ESTUFA_TRANSMISSAO_LUZ}", "",
           "DISTRIBUICAO DAS CLASSES:"]
    for c in range(4):
        q = int(dist.get(c, 0))
        rel.append(f"  Classe {c} ({nomes[c]:24s}): {q:7d}  ({100*q/len(ds):5.1f}%)")

    rel += ["", "CE MEDIA POR FASE (deve subir da muda ate o pleno):",
            ds.groupby("fase")["ce_ms_cm"].mean().round(2).to_string(),
            "", "CORRELACAO CE x nutrientes (deve ser ~1.0, pois escalam juntos):",
            f"  CE-N: {ds['ce_ms_cm'].corr(ds['N']):.3f} | "
            f"CE-K: {ds['ce_ms_cm'].corr(ds['K']):.3f}",
            "", "CORRELACAO temp_solucao x OD (deve ser NEGATIVA - fisica):",
            f"  {ds['temp_solucao_c'].corr(ds['od_mg_l']):.3f}",
            "", "DLI interno vs externo (cobertura transmite ~70%):",
            f"  interno medio: {ds['dli_mol_m2_d'].mean():.1f} mol/m2/dia",
            "", "classe 3 por tolerancia da cultivar (tolerante deve pendoar MENOS):",
            "  " + str(ds.groupby("tolerancia_calor")["classe_acao"]
                       .apply(lambda x: round((x == 3).mean()*100, 1)).to_dict()),
            "", f"COLUNAS ({len(ds.columns)}): " + ", ".join(ds.columns), "",
            "ESTATISTICAS:",
            ds[["ce_ms_cm", "ph", "N", "P", "K", "temp_solucao_c", "od_mg_l",
                "nivel_reservatorio_pct", "temp_ar_c", "umidade_relativa_pct",
                "vpd_kpa", "dli_mol_m2_d", "nitrato_mg_kg", "saude_pct"]]
            .describe().round(1).to_string(),
            "", "COERENCIA motivo x classe:",
            ds.groupby(["classe_acao", "motivo"]).size().to_string()]
    (SAIDA_DIR / "relatorio_dataset_nft.txt").write_text("\n".join(rel), encoding="utf-8")

    print("\n".join(rel))
    print(f"\nOK -> nft/dataset_nft.csv ({len(ds)} linhas, {len(ds.columns)} colunas)")


if __name__ == "__main__":
    main()