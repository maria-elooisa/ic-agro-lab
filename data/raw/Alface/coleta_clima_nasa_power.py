"""
=============================================================================
AgroLab AI - Coleta de Dados Meteorologicos (NASA POWER)
=============================================================================
Projeto  : IC AgroLab AI - Alface Crespa (Lactuca sativa var. crispa)
Etapa    : Ancoragem climatica do dataset sintetico (ver info.md, secao 5)
Local    : data/raw/Alface/

Fonte    : NASA POWER - Prediction of Worldwide Energy Resources
           NASA Langley Research Center (LaRC)
           Radiacao: satelite (CERES / GEWEX SRB)
           Meteorologia: modelo de assimilacao MERRA-2 (NASA/GMAO)
           Licenca: CC BY 4.0 - uso aberto, sem chave de API
           Docs: https://power.larc.nasa.gov/docs/services/api/temporal/daily/

AGRADECIMENTO OBRIGATORIO NO ARTIGO:
    Creditar o NASA Langley Research Center (LaRC) POWER Project,
    financiado pelo NASA Earth Science / Applied Science Program.

Escopo   : 3 coordenadas / 2 polos climaticos (caminho A)
           - Polo Sudoeste : Ibiuna + Piedade  (mesma celula da grade)
           - Polo Alto Tiete: Mogi das Cruzes
Periodo  : 01/01/2015 a 31/12/2025 (11 anos completos = 4018 dias)
Grade    : ~0.5 x 0.625 (meteorologia) | 1 x 1 (radiacao) -> celulas ~50x60 km

Saidas (no mesmo diretorio do script):
    clima_bruto_ibiuna.csv
    clima_bruto_piedade.csv
    clima_bruto_mogi_das_cruzes.csv
    clima_processado.csv          <- serie consolidada dos 2 polos (+ DLI, VPD)
    relatorio_clima.txt           <- perfil climatico + frequencia de edge cases

Dependencias: pip install requests pandas numpy
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# =============================================================================
# CONFIGURACAO
# =============================================================================

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DATA_INICIO = "20150101"
DATA_FIM = "20251231"

# Sentinela de ausencia da POWER. NUNCA converter para zero.
SENTINELA_NULO = -999.0


@dataclass(frozen=True)
class Local:
    nome: str
    slug: str
    lat: float
    lon: float
    polo: str


# Coordenadas validadas (centroides municipais).
# Ibiuna e Piedade distam ~22 km -> abaixo do tamanho da celula da grade.
# O script BAIXA AS TRES e testa empiricamente se sao identicas (secao 5.3 do info.md).
LOCAIS: list[Local] = [
    Local("Ibiuna",          "ibiuna",          -23.6583, -47.2113, "Sudoeste"),
    Local("Piedade",         "piedade",         -23.7170, -47.4142, "Sudoeste"),
    Local("Mogi das Cruzes", "mogi_das_cruzes", -23.5217, -46.1860, "Alto Tiete"),
]

# Parametros POWER -> features do AgroLab (ver info.md, secao 5.5)
PARAMETROS = [
    "T2M",                # Temperatura do ar (media)      -> feature temperatura
    "T2M_MAX",            # Temperatura maxima             -> pico diurno / onda de calor
    "T2M_MIN",            # Temperatura minima             -> noite / geada
    "RH2M",               # Umidade relativa               -> feature umidade relativa
    "ALLSKY_SFC_SW_DWN",  # Radiacao solar incidente       -> feature radiacao -> DLI
    "ALLSKY_SFC_PAR_TOT", # PAR (luz fotossintetica)       -> DLI direto (se disponivel)
    "T2MDEW",             # Ponto de orvalho               -> calculo do VPD
    "PRECTOTCORR",        # Precipitacao                   -> contexto (UR alta / fungos)
    "WS2M",               # Vento a 2 m                    -> evapotranspiracao (FAO-56)
]

# Parametro que pode nao existir no catalogo AG; se a API recusar, tentamos sem ele.
PARAM_OPCIONAL = "ALLSKY_SFC_PAR_TOT"

# --- Limiares agronomicos (info.md, secao 4.3 e 4.4) -------------------------
T_GEADA = 7.0          # T_min < 7 C  -> necrose marginal / geada    (classe 3)
T_CALOR = 28.0         # T_max > 28 C -> risco de pendoamento        (classe 3)
T_IDEAL_MIN = 15.0     # faixa diurna ideal
T_IDEAL_MAX = 20.0
DLI_IDEAL_MIN = 12.0   # mol/m2/dia
DLI_IDEAL_MAX = 16.0
UR_IDEAL_MIN = 60.0    # %
UR_IDEAL_MAX = 75.0
UR_RISCO_FUNGO = 85.0  # %
VPD_IDEAL_MIN = 0.8    # kPa
VPD_IDEAL_MAX = 1.2

# Constantes fisicas
FRACAO_PAR = 0.45      # fracao da radiacao de onda curta que e PAR
UMOL_POR_JOULE = 4.57  # umol/J - conversao energia -> fotons na faixa PAR

DIR_SAIDA = Path(__file__).resolve().parent


# =============================================================================
# 1. DOWNLOAD
# =============================================================================

def baixar_local(local: Local, parametros: list[str], tentativa: int = 1) -> dict:
    """Requisita a serie diaria de um ponto na API da NASA POWER."""
    params = {
        "parameters": ",".join(parametros),
        "community": "AG",          # comunidade Agroclimatology
        "latitude": local.lat,
        "longitude": local.lon,
        "start": DATA_INICIO,
        "end": DATA_FIM,
        "format": "JSON",
    }

    print(f"  -> Requisitando {local.nome} ({local.lat}, {local.lon})...")
    try:
        r = requests.get(BASE_URL, params=params, timeout=120)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"\n[ERRO DE REDE] Falha ao acessar a NASA POWER: {e}")

    # Se a API recusou o parametro opcional, tenta novamente sem ele.
    if r.status_code in (400, 422) and PARAM_OPCIONAL in parametros and tentativa == 1:
        print(f"     [AVISO] API recusou '{PARAM_OPCIONAL}'. Repetindo sem ele.")
        print(f"             O DLI sera derivado de ALLSKY_SFC_SW_DWN.")
        restantes = [p for p in parametros if p != PARAM_OPCIONAL]
        return baixar_local(local, restantes, tentativa=2)

    if r.status_code != 200:
        raise SystemExit(
            f"\n[ERRO HTTP {r.status_code}] {local.nome}\n"
            f"Resposta: {r.text[:500]}"
        )

    return r.json()


def json_para_dataframe(payload: dict, local: Local) -> tuple[pd.DataFrame, dict]:
    """Converte o JSON da POWER em DataFrame e extrai as unidades declaradas."""
    try:
        blocos = payload["properties"]["parameter"]
    except KeyError:
        raise SystemExit(f"\n[ERRO] Resposta inesperada da API para {local.nome}.")

    # Unidades vem declaradas pela propria API - nao assumimos nada.
    unidades = {
        nome: meta.get("units", "?")
        for nome, meta in payload.get("parameters", {}).items()
    }

    df = pd.DataFrame(blocos)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df = df.sort_index()

    # Sentinela -999 -> NaN (JAMAIS zero).
    df = df.replace(SENTINELA_NULO, np.nan)

    df.insert(0, "cidade", local.nome)
    df.insert(1, "polo", local.polo)
    df.insert(2, "latitude", local.lat)
    df.insert(3, "longitude", local.lon)

    return df, unidades


# =============================================================================
# 2. VARIAVEIS DERIVADAS (DLI e VPD)
# =============================================================================

def radiacao_para_mj(valores: pd.Series, unidade: str) -> pd.Series:
    """
    Normaliza a radiacao para MJ/m2/dia, lendo a unidade declarada pela API.
    A POWER ja mudou a unidade default desta variavel entre versoes,
    entao nao assumimos - convertemos conforme o que ela informar.
    """
    u = (unidade or "").lower()

    if "mj" in u:
        return valores                     # ja esta em MJ/m^2/dia
    if "kw" in u:                          # kW-hr/m^2/dia
        return valores * 3.6
    if "w/m" in u or "w m" in u:           # W/m^2 (media diaria)
        return valores * 0.0864            # W/m2 * 86400 s / 1e6
    print(f"     [AVISO] Unidade de radiacao nao reconhecida: '{unidade}'. "
          f"Assumindo MJ/m^2/dia.")
    return valores


def calcular_dli(df: pd.DataFrame, unidades: dict) -> pd.Series:
    """
    DLI (mol/m2/dia) - integral diaria de luz.
    Preferencia: PAR direto. Fallback: radiacao de onda curta * fracao PAR.
    """
    if "ALLSKY_SFC_PAR_TOT" in df.columns and df["ALLSKY_SFC_PAR_TOT"].notna().any():
        u = (unidades.get("ALLSKY_SFC_PAR_TOT") or "").lower()
        par = df["ALLSKY_SFC_PAR_TOT"]
        if "w/m" in u or "w m" in u:
            # PAR em W/m2 (media diaria) -> mol/m2/dia
            dli = par * 86400 * UMOL_POR_JOULE / 1e6
        else:
            # PAR em MJ/m2/dia -> mol/m2/dia
            dli = par * UMOL_POR_JOULE
        print(f"     DLI derivado de PAR direto (unidade: {unidades.get('ALLSKY_SFC_PAR_TOT')})")
        return dli

    # Fallback: radiacao global -> fracao PAR -> fotons
    u = unidades.get("ALLSKY_SFC_SW_DWN", "")
    mj = radiacao_para_mj(df["ALLSKY_SFC_SW_DWN"], u)
    dli = mj * FRACAO_PAR * UMOL_POR_JOULE
    print(f"     DLI derivado de ALLSKY_SFC_SW_DWN (unidade: {u}) "
          f"[fracao PAR={FRACAO_PAR}, {UMOL_POR_JOULE} umol/J]")
    return dli


def pressao_saturacao(temp_c: pd.Series) -> pd.Series:
    """Pressao de vapor de saturacao (kPa) - equacao de Tetens."""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def calcular_vpd(df: pd.DataFrame) -> pd.Series:
    """
    VPD (kPa) - Deficit de Pressao de Vapor.
    Usa a media de es(Tmax) e es(Tmin) (padrao FAO-56), mais fiel que es(Tmedia),
    e a pressao real de vapor a partir do ponto de orvalho.
    """
    if "T2MDEW" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        es = (pressao_saturacao(df["T2M_MAX"]) + pressao_saturacao(df["T2M_MIN"])) / 2
    else:
        es = pressao_saturacao(df["T2M"])

    ea = pressao_saturacao(df["T2MDEW"])   # es no ponto de orvalho = pressao real
    return (es - ea).clip(lower=0)


# =============================================================================
# 3. CONTROLE DE QUALIDADE
# =============================================================================

def controle_qualidade(df: pd.DataFrame, local: Local, linhas: list[str]) -> None:
    """Verifica continuidade, ausencias e sanidade fisica da serie."""
    linhas.append(f"\n--- QC: {local.nome} ---")

    esperado = (pd.Timestamp(DATA_FIM) - pd.Timestamp(DATA_INICIO)).days + 1
    linhas.append(f"Dias esperados : {esperado}")
    linhas.append(f"Dias recebidos : {len(df)}")
    if len(df) != esperado:
        linhas.append("  [ALERTA] Numero de dias diverge do esperado.")

    # Lacunas no calendario
    faltantes = pd.date_range(df.index.min(), df.index.max(), freq="D").difference(df.index)
    linhas.append(f"Datas ausentes : {len(faltantes)}")

    # Valores nulos por coluna
    numericas = df.select_dtypes(include=[np.number])
    nulos = numericas.isna().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) == 0:
        linhas.append("Valores nulos  : nenhum")
    else:
        linhas.append("Valores nulos  :")
        for col, n in nulos.items():
            linhas.append(f"   {col}: {n} ({100*n/len(df):.2f}%)")

    # Sanidade fisica
    problemas = []
    if {"T2M_MIN", "T2M", "T2M_MAX"}.issubset(df.columns):
        viol = (df["T2M_MIN"] > df["T2M"]) | (df["T2M"] > df["T2M_MAX"])
        if viol.any():
            problemas.append(f"T_min <= T_med <= T_max violado em {viol.sum()} dias")
    if "RH2M" in df.columns:
        viol = (df["RH2M"] < 0) | (df["RH2M"] > 100)
        if viol.any():
            problemas.append(f"UR fora de [0,100] em {viol.sum()} dias")
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        viol = df["ALLSKY_SFC_SW_DWN"] < 0
        if viol.any():
            problemas.append(f"Radiacao negativa em {viol.sum()} dias")

    if problemas:
        for p in problemas:
            linhas.append(f"  [ALERTA] {p}")
    else:
        linhas.append("Sanidade fisica: OK")

    # Confronto com a climatologia publicada (info.md, secao 5.10)
    if "T2M" in df.columns:
        media = df["T2M"].mean()
        linhas.append(f"Temp. media anual: {media:.1f} C")
        ref = {"Ibiuna": 18.0, "Piedade": 18.7, "Mogi das Cruzes": 19.5}
        alvo = ref.get(local.nome)
        if alvo is not None:
            desvio = abs(media - alvo)
            status = "OK" if desvio <= 2.0 else "[ALERTA] divergencia alta"
            linhas.append(f"  Climatologia publicada: ~{alvo} C | desvio: {desvio:.1f} C -> {status}")


# =============================================================================
# 4. TESTE DE EQUIVALENCIA (Ibiuna x Piedade)
# =============================================================================

def testar_equivalencia(dfs: dict[str, pd.DataFrame], linhas: list[str]) -> bool:
    """
    Valida empiricamente a fusao do polo Sudoeste (info.md, secao 5.3).
    Se as series forem identicas, as duas cidades caem na mesma celula da grade
    e trata-las como pontos independentes seria PSEUDORREPLICACAO.
    """
    linhas.append("\n" + "=" * 70)
    linhas.append("TESTE DE EQUIVALENCIA - Ibiuna x Piedade (polo Sudoeste)")
    linhas.append("=" * 70)

    a, b = dfs.get("Ibiuna"), dfs.get("Piedade")
    if a is None or b is None:
        linhas.append("Series indisponiveis - teste nao realizado.")
        return False

    cols = [c for c in PARAMETROS if c in a.columns and c in b.columns]
    idx = a.index.intersection(b.index)
    identicas = True

    linhas.append(f"{'Parametro':<22} {'Dif. media':>12} {'Dif. maxima':>12}   Situacao")
    linhas.append("-" * 70)
    for c in cols:
        dif = (a.loc[idx, c] - b.loc[idx, c]).abs()
        med, mx = dif.mean(), dif.max()
        igual = mx < 1e-6
        identicas &= igual
        linhas.append(f"{c:<22} {med:>12.6f} {mx:>12.6f}   "
                      f"{'IDENTICO' if igual else 'DIFERENTE'}")

    linhas.append("")
    if identicas:
        linhas.append(">> CONFIRMADO: as series sao IDENTICAS.")
        linhas.append("   Ibiuna e Piedade caem na MESMA celula da grade NASA POWER.")
        linhas.append("   A fusao no polo 'Sudoeste' esta empiricamente comprovada.")
        linhas.append("   Usar as duas como pontos independentes seria pseudorreplicacao.")
    else:
        linhas.append(">> ATENCAO: as series DIFEREM.")
        linhas.append("   As cidades caem em celulas distintas. A decisao de fundi-las")
        linhas.append("   (caminho A) deve ser REVISTA - ha informacao independente aqui.")

    return identicas


# =============================================================================
# 5. PERFIL CLIMATICO E EDGE CASES
# =============================================================================

def perfil_climatico(df: pd.DataFrame, polo: str, linhas: list[str]) -> None:
    """Estatisticas descritivas + frequencia dos eventos-limite (classe 3)."""
    linhas.append("\n" + "=" * 70)
    linhas.append(f"PERFIL CLIMATICO - Polo {polo}")
    linhas.append("=" * 70)

    n = len(df)
    linhas.append(f"Periodo: {df.index.min().date()} a {df.index.max().date()}  ({n} dias)")

    # Descritivas
    cols = [c for c in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "DLI", "VPD",
                        "PRECTOTCORR"] if c in df.columns]
    linhas.append("\nEstatisticas descritivas:")
    desc = df[cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.columns = ["media", "desvio", "min", "mediana", "max"]
    linhas.append(desc.round(2).to_string())

    # Sazonalidade
    linhas.append("\nSazonalidade (media mensal):")
    mensal = df.groupby(df.index.month)[cols].mean().round(1)
    mensal.index.name = "mes"
    linhas.append(mensal.to_string())

    # ---- EDGE CASES: a frequencia REAL da classe 3 --------------------------
    linhas.append("\n" + "-" * 70)
    linhas.append("EVENTOS-LIMITE (calibram a classe 3 - 'Proteger')")
    linhas.append("-" * 70)

    if "T2M_MIN" in df.columns:
        frio = df["T2M_MIN"] < T_GEADA
        geada = df["T2M_MIN"] < 0
        linhas.append(f"Dias com T_min < {T_GEADA} C (necrose/geada) : "
                      f"{frio.sum():>5} ({100*frio.mean():.2f}% dos dias)")
        linhas.append(f"Dias com T_min < 0 C (geada severa)      : "
                      f"{geada.sum():>5} ({100*geada.mean():.2f}%)")
        if frio.any():
            por_mes = df[frio].groupby(df[frio].index.month).size()
            linhas.append(f"  Distribuicao mensal: {por_mes.to_dict()}")

    if "T2M_MAX" in df.columns:
        calor = df["T2M_MAX"] > T_CALOR
        calor30 = df["T2M_MAX"] > 30
        linhas.append(f"Dias com T_max > {T_CALOR} C (pendoamento)   : "
                      f"{calor.sum():>5} ({100*calor.mean():.2f}%)")
        linhas.append(f"Dias com T_max > 30 C (estresse severo)  : "
                      f"{calor30.sum():>5} ({100*calor30.mean():.2f}%)")
        if calor.any():
            por_mes = df[calor].groupby(df[calor].index.month).size()
            linhas.append(f"  Distribuicao mensal: {por_mes.to_dict()}")

    if "RH2M" in df.columns:
        umido = df["RH2M"] > UR_RISCO_FUNGO
        linhas.append(f"Dias com UR > {UR_RISCO_FUNGO}% (risco fungico)  : "
                      f"{umido.sum():>5} ({100*umido.mean():.2f}%)")

    # Quantos dias sao "classe 3" por criterio climatico (frio OU calor)
    if {"T2M_MIN", "T2M_MAX"}.issubset(df.columns):
        classe3 = (df["T2M_MIN"] < T_GEADA) | (df["T2M_MAX"] > T_CALOR)
        linhas.append("")
        linhas.append(f">> CLASSE 3 por criterio climatico: {classe3.sum()} dias "
                      f"({100*classe3.mean():.1f}%)")
        linhas.append(f">> CLASSE 0/1/2 (clima dentro do aceitavel): "
                      f"{(~classe3).sum()} dias ({100*(~classe3).mean():.1f}%)")
        linhas.append("   -> Esta e a proporcao NATURAL. O balanceamento sintetico")
        linhas.append("      (info.md, secao 3.2) deve superamostrar a classe 3.")

    # ---- Aderencia as faixas ideais da alface -------------------------------
    linhas.append("\n" + "-" * 70)
    linhas.append("ADERENCIA AS FAIXAS IDEAIS DA ALFACE (info.md, secao 4)")
    linhas.append("-" * 70)

    if "T2M" in df.columns:
        ok = df["T2M"].between(T_IDEAL_MIN, T_IDEAL_MAX)
        linhas.append(f"T media em 15-20 C (ideal)     : {100*ok.mean():.1f}% dos dias")
    if "DLI" in df.columns and df["DLI"].notna().any():
        ok = df["DLI"].between(DLI_IDEAL_MIN, DLI_IDEAL_MAX)
        baixo = df["DLI"] < DLI_IDEAL_MIN
        alto = df["DLI"] > DLI_IDEAL_MAX
        linhas.append(f"DLI em 12-16 mol/m2/dia (ideal): {100*ok.mean():.1f}% dos dias")
        linhas.append(f"  DLI abaixo de 12 (luz insuf.): {100*baixo.mean():.1f}%")
        linhas.append(f"  DLI acima de 16 (satura/tipburn): {100*alto.mean():.1f}%")
        linhas.append("")
        linhas.append("  [NOTA IMPORTANTE] O DLI de CAMPO ABERTO sera muito maior que")
        linhas.append("  a faixa 12-16 mol/m2/dia (tipicamente 25-50 em dias de sol).")
        linhas.append("  Isso NAO e erro de conversao: a faixa 12-16 e o otimo de")
        linhas.append("  AMBIENTE CONTROLADO (indoor/NFT com lampada), onde se busca")
        linhas.append("  eficiencia energetica e se evita tipburn. Em campo, a alface")
        linhas.append("  recebe luz em excesso e o manejo usa sombrite/telado.")
        linhas.append("  -> Para a bancada NFT do projeto, o alvo continua sendo 12-16.")
        linhas.append("  -> Para o dataset de campo, o DLI real (alto) e o valor correto.")
    if "RH2M" in df.columns:
        ok = df["RH2M"].between(UR_IDEAL_MIN, UR_IDEAL_MAX)
        linhas.append(f"UR em 60-75% (ideal)           : {100*ok.mean():.1f}% dos dias")
    if "VPD" in df.columns and df["VPD"].notna().any():
        ok = df["VPD"].between(VPD_IDEAL_MIN, VPD_IDEAL_MAX)
        linhas.append(f"VPD em 0,8-1,2 kPa (ideal)     : {100*ok.mean():.1f}% dos dias")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("AgroLab AI - Coleta de Dados Meteorologicos (NASA POWER)")
    print("=" * 70)
    print(f"Periodo : {DATA_INICIO} a {DATA_FIM}")
    print(f"Locais  : {len(LOCAIS)} coordenadas / 2 polos climaticos")
    print(f"Saida   : {DIR_SAIDA}")
    print()

    relatorio: list[str] = []
    relatorio.append("=" * 70)
    relatorio.append("AgroLab AI - RELATORIO DE COLETA CLIMATICA")
    relatorio.append("=" * 70)
    relatorio.append(f"Fonte   : NASA POWER (LaRC) - API diaria - CC BY 4.0")
    relatorio.append(f"Periodo : {DATA_INICIO} a {DATA_FIM}")
    relatorio.append(f"Gerado  : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")

    dfs: dict[str, pd.DataFrame] = {}
    unidades_globais: dict = {}

    # --- 1. Download ---------------------------------------------------------
    print("[1/5] Baixando series...")
    for local in LOCAIS:
        payload = baixar_local(local, PARAMETROS)
        df, unidades = json_para_dataframe(payload, local)
        unidades_globais.update(unidades)

        caminho = DIR_SAIDA / f"clima_bruto_{local.slug}.csv"
        df.to_csv(caminho, encoding="utf-8")
        print(f"     OK: {len(df)} dias -> {caminho.name}")

        dfs[local.nome] = df
        time.sleep(1)  # cortesia com a API

    relatorio.append("\nUnidades declaradas pela API:")
    for k, v in unidades_globais.items():
        relatorio.append(f"   {k}: {v}")

    # --- 2. Controle de qualidade -------------------------------------------
    print("\n[2/5] Controle de qualidade...")
    relatorio.append("\n" + "=" * 70)
    relatorio.append("CONTROLE DE QUALIDADE")
    relatorio.append("=" * 70)
    for local in LOCAIS:
        controle_qualidade(dfs[local.nome], local, relatorio)
    print("     OK")

    # --- 3. Teste de equivalencia -------------------------------------------
    print("\n[3/5] Testando equivalencia Ibiuna x Piedade...")
    identicas = testar_equivalencia(dfs, relatorio)
    print(f"     Resultado: {'IDENTICAS (fusao confirmada)' if identicas else 'DIFERENTES (rever fusao)'}")

    # --- 4. Variaveis derivadas + consolidacao ------------------------------
    print("\n[4/5] Calculando DLI e VPD...")

    # Polo Sudoeste: usa Ibiuna como representante (se identicas) ou mantem ambas.
    if identicas:
        representantes = [dfs["Ibiuna"], dfs["Mogi das Cruzes"]]
        relatorio.append("\n>> Consolidacao: polo Sudoeste representado por Ibiuna "
                         "(Piedade e identica).")
    else:
        representantes = [dfs["Ibiuna"], dfs["Piedade"], dfs["Mogi das Cruzes"]]
        relatorio.append("\n>> Consolidacao: as 3 series foram mantidas (nao sao identicas).")

    processados = []
    for df in representantes:
        d = df.copy()
        d["DLI"] = calcular_dli(d, unidades_globais)
        d["VPD"] = calcular_vpd(d)
        processados.append(d)

    consolidado = pd.concat(processados).sort_values(["polo", "cidade"]).sort_index()
    caminho = DIR_SAIDA / "clima_processado.csv"
    consolidado.to_csv(caminho, encoding="utf-8")
    print(f"     OK: {len(consolidado)} linhas -> {caminho.name}")

    # --- 5. Perfil climatico -------------------------------------------------
    print("\n[5/5] Gerando perfil climatico e edge cases...")
    for polo, grupo in consolidado.groupby("polo"):
        perfil_climatico(grupo, str(polo), relatorio)

    relatorio.append("\n" + "=" * 70)
    relatorio.append("FIM DO RELATORIO")
    relatorio.append("=" * 70)

    caminho_rel = DIR_SAIDA / "relatorio_clima.txt"
    caminho_rel.write_text("\n".join(relatorio), encoding="utf-8")
    print(f"     OK -> {caminho_rel.name}")

    print("\n" + "=" * 70)
    print("CONCLUIDO")
    print("=" * 70)
    for f in ["clima_bruto_ibiuna.csv", "clima_bruto_piedade.csv",
              "clima_bruto_mogi_das_cruzes.csv", "clima_processado.csv",
              "relatorio_clima.txt"]:
        p = DIR_SAIDA / f
        if p.exists():
            print(f"  {f:<32} {p.stat().st_size/1024:>8.1f} KB")
    print("\nLeia relatorio_clima.txt: e la que estao a confirmacao da fusao")
    print("Ibiuna/Piedade e a frequencia real dos eventos-limite (classe 3).")


if __name__ == "__main__":
    main()