"""
Montagem sistemática de carteira B3:
- Universo e fundamentos via Fundamentus
- Preços/volumes via yfinance em lotes (com tqdm)
- Filtros de liquidez/volatilidade
- Scores (momentum, qualidade, valuation)
- Limites por papel e setor (contagem + cap de peso)
- Pesos ~ Score/Vol_20d, caps e normalização
- Gera JSON "lock" com holdings e regras
- >>> Adequado à execução real: shares inteiras (fracionário ou lote), caixa residual e pesos efetivos
"""
import os
import json
import argparse
from datetime import date, timedelta
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import fundamentus
from tqdm.auto import tqdm

# ------------------ Parâmetros do TRABALHO ------------------
CAPITAL = 1_000_000.00
N_MIN, N_MAX = 8, 10
CAP_PAPEL = 0.12            # teto de peso por papel
CAP_SETOR = 0.35            # teto de peso por setor
MAX_POR_SETOR = 2           # nº máx. de papéis por setor
LIQ_MIN_RS = 25_000_000     # piso de liquidez (mediana 60d) em R$/dia
CUT_VOL_TOP = 0.10          # corta top 10% mais voláteis (Vol_20d)
EXCLUIR_ULTIMA_SEMANA = 5
EXCLUIR_ULTIMOS_3_DIAS = 3
START_HIST = "2023-01-01"   # p/ sinais e vol

# Silencia barulhos de log das libs (opcional)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("detalhes").setLevel(logging.WARNING)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def window_return(series, start_offset, end_offset):
    s = series.dropna()
    if len(s) < start_offset + 2:
        return np.nan
    p_start = s.iloc[-start_offset]
    p_end = s.iloc[-end_offset] if end_offset > 0 else s.iloc[-1]
    return (p_end / p_start) - 1

def rank_pct(x):
    r = x.rank(method="average", na_option="keep")
    return (r - r.min()) / (r.max() - r.min())

def winsorize(series, p1=0.05, p2=0.95):
    lo, hi = series.quantile(p1), series.quantile(p2)
    return series.clip(lo, hi)

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def _discretize_shares(qtd: pd.Series, mode: str, lot_size: int, rounding: str) -> pd.Series:
    """
    Converte 'shares' teóricas (float) em inteiras, respeitando:
      - mode: 'fracionario' (passo = 1) ou 'lote' (passo = lot_size)
      - rounding: 'floor' (não excede capital), 'round' ou 'ceil'
    Retorna pd.Series de inteiros.
    """
    step = 1 if mode == "fracionario" else max(1, int(lot_size))
    lots = qtd / step
    if rounding == "floor":
        lots_i = np.floor(lots)
    elif rounding == "ceil":
        lots_i = np.ceil(lots)
    else:  # "round"
        lots_i = np.rint(lots)  # arred. do numpy
    return (lots_i * step).astype(int)

def main(
    output="data/portfolio_lock.json",
    use_cache=False,
    max_por_setor: int = MAX_POR_SETOR,
    exec_modo: str = "fracionario",     # 'fracionario' ou 'lote'
    lote_size: int = 100,               # usado se exec_modo == 'lote'
    rounding: str = "floor"             # 'floor' (recomendado), 'round', 'ceil'
):
    # Garante pasta de saída
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # ---------- 1) Universo via Fundamentus ----------
    if use_cache:
        try:
            df_res = pd.read_csv("inputs/universo_cache.csv", index_col=0)
        except Exception:
            df_res = fundamentus.get_resultado()
            os.makedirs("inputs", exist_ok=True)
            df_res.to_csv("inputs/universo_cache.csv")
    else:
        df_res = fundamentus.get_resultado()

    # Garantir coluna 'papel' (ticker B3) e 'ticker_yf'
    df_res = df_res.rename_axis("papel").reset_index()
    df_res["papel"] = df_res["papel"].astype(str).str.strip().str.upper()
    df_res["ticker_yf"] = df_res["papel"] + ".SA"

    # Pré-filtro de liquidez
    if "liq2m" in df_res.columns:
        df_res = df_res[df_res["liq2m"] >= LIQ_MIN_RS].copy()

    # ---------- 2) Preços & Volumes (yfinance) ----------
    tickers = sorted(df_res["ticker_yf"].unique().tolist())

    px_parts, vol_parts = [], []
    for batch in tqdm(list(chunked(tickers, 50)), desc="Baixando preços (yfinance)"):
        try:
            data = yf.download(
                batch,
                start=START_HIST,
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if data is None or len(data) == 0:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                if ("Close" in data.columns.get_level_values(0) and
                    "Volume" in data.columns.get_level_values(0)):
                    px_parts.append(data["Close"])
                    vol_parts.append(data["Volume"])
            else:
                cols = data.columns
                if "Close" in cols and "Volume" in cols:
                    name = batch[0] if len(batch) == 1 else None
                    if name is not None:
                        px_parts.append(data[["Close"]].rename(columns={"Close": name}))
                        vol_parts.append(data[["Volume"]].rename(columns={"Volume": name}))
        except Exception:
            continue

    if not px_parts:
        raise RuntimeError("Não foi possível baixar preços de nenhum ticker válido.")

    px = pd.concat(px_parts, axis=1).sort_index()
    vol = pd.concat(vol_parts, axis=1).sort_index()

    # Limpa ativos sem histórico suficiente
    valid_cols = px.columns[px.notna().sum() > 120]
    px = px[valid_cols]
    vol = vol[valid_cols]
    df_res = df_res[df_res["ticker_yf"].isin(valid_cols)]

    # ---------- 3) Liquidez e Volatilidade ----------
    liq = (px * vol)  # R$
    liq_med60 = liq.tail(60).median()
    ret = px.pct_change(fill_method=None)
    vol20 = ret.rolling(20).std().tail(1).T.squeeze()

    mask_liq = liq_med60 >= LIQ_MIN_RS
    lim_vol = vol20.quantile(1 - CUT_VOL_TOP)
    mask_vol = vol20 <= lim_vol

    universe_ok = pd.Index(mask_liq[mask_liq].index).intersection(mask_vol[mask_vol].index)
    px = px[universe_ok]
    vol = vol[universe_ok]
    vol20 = vol20[universe_ok]
    base = df_res[df_res["ticker_yf"].isin(universe_ok)].set_index("ticker_yf")

    # ---------- 4) Momentum curto ----------
    mom3m_ex1w = px.apply(lambda s: window_return(s, 63, EXCLUIR_ULTIMA_SEMANA), axis=0)
    mom1m_ex3d = px.apply(lambda s: window_return(s, 21, EXCLUIR_ULTIMOS_3_DIAS), axis=0)
    score_mom = 0.65*rank_pct(mom3m_ex1w) + 0.35*rank_pct(mom1m_ex3d)

    # ---------- 5) Qualidade & Valuation ----------
    fund = base[["pl","pvp","evebitda","mrgliq","roe","liqc","c5y","divbpatr"]].copy()
    for c in fund.columns:
        fund[c] = winsorize(fund[c])

    z_ROE = zscore(fund["roe"])
    z_ML  = zscore(fund["mrgliq"])
    z_CR  = zscore(fund["c5y"])
    z_LC  = zscore(fund["liqc"])
    z_DB  = (-1) * zscore(fund["divbpatr"])   # menor melhor

    z_PL  = (-1) * zscore(fund["pl"])
    z_EV  = (-1) * zscore(fund["evebitda"])
    z_PVP = (-1) * zscore(fund["pvp"])

    score_qual = pd.concat([z_ROE, z_ML, z_CR, z_LC, z_DB], axis=1).mean(axis=1)
    score_val  = pd.concat([z_PL, z_EV, z_PVP], axis=1).mean(axis=1)

    qcut = score_qual.quantile(0.35)
    vivos = score_qual[score_qual >= qcut].index
    score_mom = score_mom[vivos].dropna()
    vol20 = vol20[vivos]
    score_qual = score_qual[vivos]
    score_val  = score_val[vivos]

    # ---------- 6) Score Final ----------
    sf_mom  = zscore(score_mom)
    sf_qual = zscore(score_qual)
    sf_val  = zscore(score_val)
    Score_Final = (0.60*sf_mom + 0.25*sf_qual + 0.15*sf_val).sort_values(ascending=False)

    # ---------- 7) Limite setorial (nº máx. por setor) ----------
    sel, cont_setor, setor_map = [], {}, {}
    candidatos = Score_Final.index.tolist()

    for t in tqdm(candidatos, desc="Consultando setor (Fundamentus)"):
        b3 = t.replace(".SA", "")
        try:
            info = fundamentus.get_papel(b3)
            setor = str(info.loc[b3, "Setor"])
        except Exception:
            setor = "NA"
        setor_map[t] = setor
        if cont_setor.get(setor, 0) >= max_por_setor:
            continue
        sel.append(t)
        cont_setor[setor] = cont_setor.get(setor, 0) + 1
        if len(sel) == N_MAX:
            break
    if len(sel) < N_MIN:
        sel = candidatos[:N_MIN]

    Score_Final = Score_Final.loc[sel]
    vol20_sel = vol20.loc[sel]
    setores = pd.Series({t: setor_map.get(t, "NA") for t in sel})

    # ---------- 8) Pesos ∝ Score/Vol_20d + caps ----------
    alpha = (Score_Final / vol20_sel).replace([np.inf,-np.inf], np.nan).clip(lower=0).fillna(0)
    w = (alpha / alpha.sum()).clip(upper=CAP_PAPEL)
    df_w = pd.DataFrame({"w": w, "setor": setores})
    for s, grp in df_w.groupby("setor"):
        soma = grp["w"].sum()
        if soma > CAP_SETOR:
            df_w.loc[grp.index, "w"] *= (CAP_SETOR / soma)
    w = df_w["w"]; w = w / w.sum()

    # ---------- 9) Quantidades (REAL: shares inteiras) ----------
    # Preço de referência (fechamento ajustado mais recente, com ffill)
    precos = px[sel].ffill().iloc[-1]
    aloc_R = w * CAPITAL
    qtd_teor = aloc_R / precos  # teórica (float)

    # Discretiza para inteiros conforme modo/lote e arredondamento
    shares_int = _discretize_shares(qtd_teor, mode=exec_modo, lot_size=lote_size, rounding=rounding)

    # Valor investido e caixa residual
    invested = (shares_int * precos).astype(float)
    invested_total = float(invested.sum())
    cash_residual = float(CAPITAL - invested_total)

    # Pesos efetivos após discretização (com base no investido total)
    if invested_total > 0:
        weight_effective = invested / invested_total
    else:
        weight_effective = invested * 0.0

    # ---------- 10) Salva LOCK ----------
    lock_date = str(date.today())
    out = {
        "locked_at": lock_date,
        "capital": CAPITAL,
        "rules": {
            "n_min": N_MIN, "n_max": N_MAX,
            "cap_papel": CAP_PAPEL, "cap_setor": CAP_SETOR,
            "max_por_setor": max_por_setor,
            "liq_min_rs": LIQ_MIN_RS, "cut_vol_top": CUT_VOL_TOP,
            "mom_ex": {"3m_ex1w": EXCLUIR_ULTIMA_SEMANA, "1m_ex3d": EXCLUIR_ULTIMOS_3_DIAS},
            "score_weights": {"momentum": 0.60, "qual": 0.25, "val": 0.15},
            "weights_formula": "weight ∝ Score_Final / Vol_20d (depois caps e renormalização)"
        },
        # Metadados de execução real (discretização)
        "execution": {
            "mode": exec_modo,                 # 'fracionario' ou 'lote'
            "lot_size": lote_size if exec_modo == "lote" else 1,
            "rounding": rounding,              # 'floor', 'round' ou 'ceil'
            "invested_total": invested_total,
            "cash_residual": cash_residual
            # dica: em muitas corretoras, para fracionário use sufixo 'F' (ex.: PETR4F)
        },
        "holdings": []
    }

    for t in sel:
        out["holdings"].append({
            "ticker_b3": t.replace(".SA",""),
            "ticker_yf": t,
            "setor": setores.get(t, "NA"),
            "score_final": float(Score_Final[t]),
            "vol20d": float(vol20_sel[t]),
            "alpha": float(alpha[t]),
            "weight_target": float(w[t]),                 # peso alvo teórico (contínuo)
            "last_price_ref": float(precos[t]),
            "shares_target": float(qtd_teor[t]),          # teórica (float)
            "shares": int(shares_int[t]),                 # EXECUÇÃO: inteira (respeita modo/lote)
            "invested": float(invested[t]),               # R$ investido após discretização
            "weight_effective": float(weight_effective[t])# peso efetivo pós-discretização
        })

    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Resumo amigável
    df_out = pd.DataFrame(out["holdings"]).sort_values("weight_effective", ascending=False)
    print(f"Carteira TRAVADA em {lock_date} → {output}")
    print(f"Investido: R$ {invested_total:,.2f} | Caixa residual: R$ {cash_residual:,.2f}")
    print(df_out[["ticker_b3","setor","shares","last_price_ref","invested","weight_target","weight_effective"]])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/portfolio_lock.json")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--max-por-setor", type=int, default=MAX_POR_SETOR,
                   help="Número máximo de papéis do mesmo setor na carteira (default: %(default)s)")
    # >>> NOVO: parâmetros de execução real
    p.add_argument("--exec-modo", choices=["fracionario","lote"], default="fracionario",
                   help="Modo de execução real: 'fracionario' (passo=1) ou 'lote' (passo=lote-size).")
    p.add_argument("--lote-size", type=int, default=100,
                   help="Tamanho do lote padrão quando exec-modo = 'lote' (default: 100).")
    p.add_argument("--rounding", choices=["floor","round","ceil"], default="floor",
                   help="Política de arredondamento para discretização das shares.")
    args = p.parse_args()
    main(
        output=args.output,
        use_cache=args.use_cache,
        max_por_setor=args.max_por_setor,
        exec_modo=args.exec_modo,
        lote_size=args.lote_size,
        rounding=args.rounding
    )
