"""
=============================================================================
AgroLab AI - Pipeline de Dados Climaticos (coleta + correcao, unificado)
=============================================================================
Projeto : IC AgroLab AI - Alface Crespa (Lactuca sativa var. crispa)
Etapa   : Ancoragem climatica do dataset sintetico (ver info.md, secao 5)
Local   : data/raw/Alface/

Este UNICO script substitui os antigos 'coleta_clima_nasa_power.py' e
'corrige_vies_clima.py'. Faz tudo em sequencia e gera apenas 2 saidas:

    clima/clima_corrigido.csv   -> dataset climatico final (usar na geracao)
    clima/relatorio_clima.txt   -> relatorio unico (coleta + QC + correcao)
    clima/clima_bruto_*.csv     -> cache bruto do download (rastreabilidade)

Todas as saidas vao para a subpasta 'clima/' (criada automaticamente).
Os arquivos de referencia INMET (opcionais) ficam na pasta do script.

Cache: se os arquivos brutos 'clima_bruto_*.csv' ja existirem, eles sao
reaproveitados e a NASA NAO e consultada de novo. Para forcar novo download,
apague os brutos (ou rode com --recoletar).

Fonte NASA POWER (CC BY 4.0). Agradecer no artigo o NASA LaRC POWER Project,
financiado pelo NASA Earth Science / Applied Science Program.

Dependencias: pip install requests pandas numpy
=============================================================================
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # so e necessario se for baixar da NASA

DIR = Path(__file__).resolve().parent
SAIDA_DIR = DIR / "clima"   # todos os arquivos gerados vao para esta subpasta
FORCAR_RECOLETA = "--recoletar" in sys.argv

# --- parametros gerais -------------------------------------------------------
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
DATA_INICIO, DATA_FIM = "20150101", "20251231"
SENTINELA_NULO = -999.0

# --- constantes fisicas ------------------------------------------------------
FRACAO_PAR = 0.45
UMOL_POR_JOULE = 4.57
LAPSE_RATE = 6.5 / 1000.0

# --- limiares agronomicos (info.md secao 4 e 6) ------------------------------
T_FRIO = 7.0            # T_min < 7 C  -> necrose/geada        (classe 3)
T_CALOR_AGUDO = 32.0    # T_max > 32 C -> estresse agudo raro  (classe 3)
T_MEDIA_BOLTING = 24.0  # media sustentada -> pendoamento      (classe 3)
DIAS_SUSTENTADO = 3
T_IDEAL_MIN, T_IDEAL_MAX = 15.0, 20.0
DLI_IDEAL_MIN, DLI_IDEAL_MAX = 12.0, 16.0
UR_IDEAL_MIN, UR_IDEAL_MAX = 60.0, 75.0
UR_RISCO_FUNGO = 85.0
VPD_IDEAL_MIN, VPD_IDEAL_MAX = 0.8, 1.2

PARAMETROS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN",
              "ALLSKY_SFC_PAR_TOT", "T2MDEW", "PRECTOTCORR", "WS2M"]
PARAM_OPCIONAL = "ALLSKY_SFC_PAR_TOT"


@dataclass(frozen=True)
class Local:
    nome: str
    slug: str
    lat: float
    lon: float
    polo: str


LOCAIS = [
    Local("Ibiuna",          "ibiuna",          -23.6583, -47.2113, "Sudoeste"),
    Local("Piedade",         "piedade",         -23.7170, -47.4142, "Sudoeste"),
    Local("Mogi das Cruzes", "mogi_das_cruzes", -23.5217, -46.1860, "Alto Tiete"),
]

# --- config dos polos para correcao de vies ----------------------------------
POLOS = {
    "Sudoeste":   {"representante": "Ibiuna",          "altitude_alvo": 996.0,
                   "arquivo_inmet": DIR / "referencia_inmet_sudoeste.csv",
                   "altitude_estacao": 600.0},
    "Alto Tiete": {"representante": "Mogi das Cruzes", "altitude_alvo": 780.0,
                   "arquivo_inmet": DIR / "referencia_inmet_alto_tiete.csv",
                   "altitude_estacao": 780.0},
}

# Normais mensais (C) - FONTE: climate-data.org (WorldClim, 1991-2021).
# Climatologia MODELADA por municipio (Ibiuna/Piedade/Mogi nao tem estacao INMET
# propria de 30 anos). Rota 2 (INMET real) e a referencia rigorosa quando houver.
NORMAIS_MENSAIS = {
    "Sudoeste":   {1: 21.1, 2: 21.4, 3: 20.5, 4: 19.1, 5: 16.5, 6: 15.7,
                   7: 15.3, 8: 16.3, 9: 17.7, 10: 19.0, 11: 19.2, 12: 20.5},
    "Alto Tiete": {1: 22.2, 2: 22.4, 3: 21.6, 4: 20.1, 5: 17.4, 6: 16.5,
                   7: 16.0, 8: 17.0, 9: 18.5, 10: 19.8, 11: 20.2, 12: 21.5},
}


# ============================================================================
# UTILIDADES
# ============================================================================
def _norm(txt):
    return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().lower().strip()


def pressao_saturacao(t):
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def calcular_vpd(df):
    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        es = (pressao_saturacao(df["T2M_MAX"]) + pressao_saturacao(df["T2M_MIN"])) / 2
    else:
        es = pressao_saturacao(df["T2M"])
    return (es - pressao_saturacao(df["T2MDEW"])).clip(lower=0)


def radiacao_para_mj(valores, unidade):
    u = (unidade or "").lower()
    if "mj" in u:
        return valores
    if "kw" in u:
        return valores * 3.6
    if "w/m" in u or "w m" in u:
        return valores * 0.0864
    return valores


def calcular_dli(df, unidades):
    if "ALLSKY_SFC_PAR_TOT" in df.columns and df["ALLSKY_SFC_PAR_TOT"].notna().any():
        u = (unidades.get("ALLSKY_SFC_PAR_TOT") or "").lower()
        par = df["ALLSKY_SFC_PAR_TOT"]
        return par * 86400 * UMOL_POR_JOULE / 1e6 if ("w/m" in u or "w m" in u) else par * UMOL_POR_JOULE
    mj = radiacao_para_mj(df["ALLSKY_SFC_SW_DWN"], unidades.get("ALLSKY_SFC_SW_DWN", ""))
    return mj * FRACAO_PAR * UMOL_POR_JOULE


# ============================================================================
# 1. COLETA (com cache)
# ============================================================================
def baixar_local(local, parametros, tentativa=1):
    if requests is None:
        raise SystemExit("[ERRO] 'requests' nao instalado e nao ha cache. "
                         "Rode: pip install requests")
    params = {"parameters": ",".join(parametros), "community": "AG",
              "latitude": local.lat, "longitude": local.lon,
              "start": DATA_INICIO, "end": DATA_FIM, "format": "JSON"}
    print(f"  baixando {local.nome}...")
    r = requests.get(BASE_URL, params=params, timeout=120)
    if r.status_code in (400, 422) and PARAM_OPCIONAL in parametros and tentativa == 1:
        return baixar_local(local, [p for p in parametros if p != PARAM_OPCIONAL], 2)
    if r.status_code != 200:
        raise SystemExit(f"[ERRO HTTP {r.status_code}] {local.nome}: {r.text[:300]}")
    return r.json()


def json_para_df(payload, local):
    blocos = payload["properties"]["parameter"]
    unidades = {k: v.get("units", "?") for k, v in payload.get("parameters", {}).items()}
    df = pd.DataFrame(blocos)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "data"
    df = df.sort_index().replace(SENTINELA_NULO, np.nan)
    df.insert(0, "cidade", local.nome)
    df.insert(1, "polo", local.polo)
    df.insert(2, "latitude", local.lat)
    df.insert(3, "longitude", local.lon)
    return df, unidades


def coletar(rel):
    """Coleta as 3 coordenadas. Usa cache clima_bruto_<slug>.csv se existir."""
    rel.append("=" * 70 + "\n1) COLETA (NASA POWER)\n" + "=" * 70)
    dfs, unidades = {}, {}
    for local in LOCAIS:
        cache = SAIDA_DIR / f"clima_bruto_{local.slug}.csv"
        if cache.exists() and not FORCAR_RECOLETA:
            df = pd.read_csv(cache, parse_dates=["data"]).set_index("data")
            rel.append(f"  {local.nome}: cache reaproveitado ({len(df)} dias)")
        else:
            payload = baixar_local(local, PARAMETROS)
            df, u = json_para_df(payload, local)
            unidades.update(u)
            df.to_csv(cache, encoding="utf-8")
            rel.append(f"  {local.nome}: baixado da NASA ({len(df)} dias)")
            time.sleep(1)
        dfs[local.nome] = df
    if not unidades:  # veio tudo do cache -> assume unidades padrao POWER
        unidades = {"ALLSKY_SFC_SW_DWN": "MJ/m^2/day", "ALLSKY_SFC_PAR_TOT": "MJ/m^2/day"}
    return dfs, unidades


# ============================================================================
# 2. QC + EQUIVALENCIA + CONSOLIDACAO
# ============================================================================
def qc_e_equivalencia(dfs, rel):
    rel.append("\n" + "=" * 70 + "\n2) CONTROLE DE QUALIDADE + FUSAO DE POLOS\n" + "=" * 70)
    esperado = (pd.Timestamp(DATA_FIM) - pd.Timestamp(DATA_INICIO)).days + 1
    ref_clim = {"Ibiuna": 18.5, "Piedade": 18.7, "Mogi das Cruzes": 19.5}
    for nome, df in dfs.items():
        nulos = df.select_dtypes("number").isna().sum().sum()
        media = df["T2M"].mean()
        alvo = ref_clim.get(nome)
        status = "OK" if (alvo and abs(media - alvo) <= 2.5) else "verificar"
        rel.append(f"  {nome:16s}: {len(df)}/{esperado} dias | nulos={nulos} | "
                   f"T media={media:.1f}C (ref~{alvo}) -> {status}")

    # equivalencia Ibiuna x Piedade
    a, b = dfs.get("Ibiuna"), dfs.get("Piedade")
    if a is not None and b is not None:
        idx = a.index.intersection(b.index)
        cols = [c for c in PARAMETROS if c in a.columns and c in b.columns]
        difmax = max((a.loc[idx, c] - b.loc[idx, c]).abs().max() for c in cols)
        ident = difmax < 1e-6
        rel.append(f"\n  Teste Ibiuna x Piedade: dif. maxima = {difmax:.6f} -> "
                   f"{'IDENTICAS (fusao confirmada)' if ident else 'DIFERENTES (rever)'}")
        if ident:
            rel.append("  => Piedade cai na mesma celula da grade que Ibiuna;")
            rel.append("     polo Sudoeste representado por Ibiuna (evita pseudorreplicacao).")
    return ["Ibiuna", "Mogi das Cruzes"]  # representantes dos 2 polos


# ============================================================================
# 3. CORRECAO DE VIES
# ============================================================================
def _parse_datas(serie):
    amostra = serie.dropna().astype(str).str.strip()
    amostra = amostra[amostra != ""]
    if amostra.empty:
        return pd.to_datetime(serie, errors="coerce")
    p = amostra.iloc[0]
    if re.match(r"^\d{4}-\d{2}-\d{2}", p):
        return pd.to_datetime(serie, errors="coerce", format="%Y-%m-%d")
    if re.match(r"^\d{2}/\d{2}/\d{4}", p):
        return pd.to_datetime(serie, errors="coerce", format="%d/%m/%Y")
    return pd.to_datetime(serie, errors="coerce")


def ler_inmet(caminho, rel):
    if not caminho.exists():
        return None
    texto = caminho.read_text(encoding="latin-1", errors="ignore").splitlines()
    inicio = 0
    for i, l in enumerate(texto):
        if re.search(r"data", _norm(l)) and (l.count(";") + l.count(",")) >= 2:
            inicio = i
            break
    sep = ";" if texto[inicio].count(";") >= texto[inicio].count(",") else ","
    df = pd.read_csv(StringIO("\n".join(texto[inicio:])), sep=sep, decimal=",", engine="python")
    cd = cm = cx = cn = None
    for c in df.columns:
        n = _norm(c)
        if cd is None and "data" in n:
            cd = c
        if "temp" in n:
            if "maxim" in n:
                cx = c
            elif "minim" in n:
                cn = c
            elif "medi" in n:
                cm = c
    if cd is None or (cm is None and cx is None and cn is None):
        rel.append(f"  [ERRO] colunas nao reconhecidas em {caminho.name}")
        return None
    out = pd.DataFrame({"data": _parse_datas(df[cd])})
    if cm is not None:
        out["T2M"] = pd.to_numeric(df[cm], errors="coerce")
    if cx is not None:
        out["T2M_MAX"] = pd.to_numeric(df[cx], errors="coerce")
    if cn is not None:
        out["T2M_MIN"] = pd.to_numeric(df[cn], errors="coerce")
    if "T2M" not in out and {"T2M_MAX", "T2M_MIN"}.issubset(out.columns):
        out["T2M"] = (out["T2M_MAX"] + out["T2M_MIN"]) / 2
    return out.dropna(subset=["data"]).set_index("data").sort_index()


def quantile_mapping(power, ref):
    corr = power.copy().astype(float)
    for mes in range(1, 13):
        p = power[power.index.month == mes].dropna()
        r = ref[ref.index.month == mes].dropna()
        if len(p) < 30 or len(r) < 30:
            continue
        ranks = p.rank(pct=True).to_numpy()
        corr.loc[p.index] = np.quantile(r.to_numpy(), np.clip(ranks, 0.001, 0.999))
    return corr


def ajusta_lapse(ref, alt_est, alt_alvo, rel):
    dz = alt_alvo - alt_est
    if abs(dz) < 1:
        return ref
    aj = -LAPSE_RATE * dz
    rel.append(f"  lapse rate: {alt_est:.0f}m -> {alt_alvo:.0f}m ({aj:+.2f}C na referencia)")
    out = ref.copy()
    for c in ["T2M", "T2M_MAX", "T2M_MIN"]:
        if c in out.columns:
            out[c] = out[c] + aj
    return out


def corrigir_vies(consolidado, unidades, rel):
    rel.append("\n" + "=" * 70 + "\n3) CORRECAO DE VIES\n" + "=" * 70)
    saida = []
    for polo, cfg in POLOS.items():
        sub = consolidado[consolidado["polo"] == polo].copy().set_index("data").sort_index()
        if sub.empty:
            continue
        orig = sub["T2M"].copy()
        ref = ler_inmet(cfg["arquivo_inmet"], rel)
        if ref is not None:
            ref = ajusta_lapse(ref, cfg["altitude_estacao"], cfg["altitude_alvo"], rel)
            for c in ["T2M", "T2M_MAX", "T2M_MIN"]:
                if c in sub.columns and c in ref.columns:
                    sub[c] = quantile_mapping(sub[c], ref[c])
            metodo = "ROTA 2 (quantile mapping INMET)"
        else:
            normais = NORMAIS_MENSAIS[polo]
            delta = {m: normais[m] - sub.loc[sub.index.month == m, "T2M"].mean() for m in range(1, 13)}
            for c in ["T2M", "T2M_MAX", "T2M_MIN"]:
                for m in range(1, 13):
                    msk = sub.index.month == m
                    sub.loc[msk, c] = sub.loc[msk, c] + delta[m]
            metodo = "ROTA 1 (delta sazonal vs normais climate-data.org)"
        sub["VPD"] = calcular_vpd(sub)
        rel.append(f"\n  Polo {polo} ({cfg['representante']}): {metodo}")
        rel.append(f"    T media anual: {orig.mean():.1f}C -> {sub['T2M'].mean():.1f}C "
                   f"({sub['T2M'].mean()-orig.mean():+.1f}C)")
        saida.append(sub.reset_index())
    return pd.concat(saida).sort_values(["polo", "data"])


# ============================================================================
# 4. PERFIL + EDGE CASES (classe 3)
# ============================================================================
def perfil(final, rel):
    rel.append("\n" + "=" * 70 + "\n4) PERFIL CLIMATICO + EVENTOS-LIMITE (classe 3)\n" + "=" * 70)
    for polo, s in final.groupby("polo"):
        s = s.sort_values("data").reset_index(drop=True)
        quente = s["T2M"] > T_MEDIA_BOLTING
        sust = quente & quente.shift(1, fill_value=False) & quente.shift(2, fill_value=False)
        cronico = sust | sust.shift(-1, fill_value=False) | sust.shift(-2, fill_value=False)
        agudo = s["T2M_MAX"] > T_CALOR_AGUDO
        frio = s["T2M_MIN"] < T_FRIO
        c3 = cronico | agudo | frio
        rel.append(f"\n  Polo {polo}:")
        rel.append(f"    T media anual={s['T2M'].mean():.1f}C | DLI medio={s['DLI'].mean():.1f} mol/m2/dia | VPD medio={s['VPD'].mean():.2f} kPa")
        rel.append(f"    calor cronico={cronico.mean()*100:.1f}% | calor agudo={agudo.mean()*100:.1f}% | frio={frio.mean()*100:.1f}%")
        rel.append(f"    >> CLASSE 3 (proteger)={c3.mean()*100:.1f}% | CLASSE 0/1/2={(~c3).mean()*100:.1f}%")
        rel.append(f"    dias em faixa ideal T(15-20C)={s['T2M'].between(T_IDEAL_MIN,T_IDEAL_MAX).mean()*100:.0f}% | "
                   f"UR(60-75%)={s['RH2M'].between(UR_IDEAL_MIN,UR_IDEAL_MAX).mean()*100:.0f}%")


# ============================================================================
# MAIN
# ============================================================================
def main():
    SAIDA_DIR.mkdir(exist_ok=True)   # garante a subpasta clima/
    rel = ["=" * 70, "AgroLab AI - PIPELINE CLIMATICO (coleta + correcao)",
           "=" * 70, f"Gerado: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
           f"Periodo: {DATA_INICIO} a {DATA_FIM} | Fonte: NASA POWER (CC BY 4.0)"]

    dfs, unidades = coletar(rel)
    representantes = qc_e_equivalencia(dfs, rel)

    # consolida os 2 polos e calcula DLI/VPD
    partes = []
    for nome in representantes:
        d = dfs[nome].copy()
        d["DLI"] = calcular_dli(d, unidades)
        d["VPD"] = calcular_vpd(d)
        partes.append(d.reset_index())
    consolidado = pd.concat(partes).sort_values(["polo", "data"])

    # corrige vies
    final = corrigir_vies(consolidado, unidades, rel)
    # recalcula DLI no final (nao muda com temperatura, mas garante a coluna)
    if "DLI" not in final.columns:
        final["DLI"] = calcular_dli(final, unidades)

    perfil(final, rel)

    # --- SAIDAS (apenas 2 arquivos) -----------------------------------------
    saida_csv = SAIDA_DIR / "clima_corrigido.csv"
    final.to_csv(saida_csv, index=False, encoding="utf-8")
    rel.append("\n" + "=" * 70)
    rel.append(f"SAIDA: clima_corrigido.csv ({len(final)} linhas) + relatorio_clima.txt")
    rel.append("=" * 70)
    (SAIDA_DIR / "relatorio_clima.txt").write_text("\n".join(rel), encoding="utf-8")

    print("\n".join(rel))
    print(f"\nOK -> clima_corrigido.csv | relatorio_clima.txt")


if __name__ == "__main__":
    main()