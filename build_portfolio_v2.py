"""
Montagem sistemática de carteira B3 (one-shot p/ concurso):
- Universo e fundamentos via Fundamentus
- Preços/volumes via yfinance em lotes (com tqdm)
- Filtros de liquidez (R$) e penalização de risco
- Scores (momentum curto x-1, qualidade, valuation) com z-score POR SETOR
- EV/EBITDA desativado para financeiros
- Seleção com limite de contagem por setor
- Pesos ~ Score/Vol_20d + caps (papel e setor) por projeção iterativa
- Discretização real (fracionário ou lote)
- Greedy fill (fracionário) para gastar ~100% mantendo caps
- Gera JSON "lock" com holdings, regras e metadados de execução
"""
import os
import json
import argparse
from datetime import date
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
CUT_VOL_TOP = 0.00          # NÃO cortar por vol; vamos penalizar nos pesos
EXCLUIR_ULTIMA_SEMANA = 5
EXCLUIR_ULTIMOS_3_DIAS = 3
START_HIST = "2023-01-01"   # p/ sinais e vol

# Silencia barulhos de log das libs (opcional)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("detalhes").setLevel(logging.WARNING)

# ------------------ Helpers ------------------
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

def rank_pct_safe(x: pd.Series) -> pd.Series:
    r = x.rank(method="average", na_option="keep")
    rng = (r.max() - r.min())
    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=x.index)
    return (r - r.min()) / rng

def winsorize(series, p1=0.05, p2=0.95):
    lo, hi = series.quantile(p1), series.quantile(p2)
    return series.clip(lo, hi)

def z_by_group(series: pd.Series, groups: pd.Series) -> pd.Series:
    df = pd.DataFrame({"x": series})
    g = groups.reindex(series.index)
    def _z(s):
        std = s.std(ddof=0)
        return (s - s.mean()) / (std if std and std > 0 else 1.0)
    return df.join(g.rename("g")).groupby("g")["x"].transform(_z)

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
        lots_i = np.rint(lots)
    return (lots_i * step).astype(int)

def enforce_caps_iterative(w: pd.Series, setores: pd.Series,
                           cap_p=CAP_PAPEL, cap_s=CAP_SETOR,
                           tol=1e-8, maxit=80) -> pd.Series:
    """Projeção iterativa p/ respeitar caps por papel e setor mantendo soma=1."""
    w = w.clip(lower=0).copy()
    if w.sum() == 0:
        return w
    w /= w.sum()
    for _ in range(maxit):
        changed = False
        # Cap por papel
        over_p = w > cap_p + tol
        if over_p.any():
            excess = (w[over_p] - cap_p).sum()
            w[over_p] = cap_p
            space = (cap_p - w).clip(lower=0)
            if space.sum() > 0 and excess > 0:
                w += (space / space.sum()) * excess
            changed = True
        # Cap por setor
        ws = w.groupby(setores).sum()
        over_s = ws[ws > cap_s + tol]
        if len(over_s) > 0:
            for s in over_s.index:
                idx = (setores == s)
                total = w[idx].sum()
                if total > 0:
                    w[idx] *= (cap_s / total)
            if w.sum() > 0:
                w /= w.sum()
            changed = True
        # Normaliza
        if w.sum() > 0:
            w /= w.sum()
        if not changed:
            break
    # Clamp final (numérico)
    w = w.clip(lower=0)
    if w.sum() > 0:
        w /= w.sum()
    return w

# ------------------ Main ------------------
def main(
    output="data/portfolio_lock.json",
    use_cache=False,
    max_por_setor: int = MAX_POR_SETOR,
    exec_modo: str = "fracionario",     # 'fracionario' ou 'lote'
    lote_size: int = 100,               # usado se exec_modo == 'lote'
    rounding: str = "floor",            # 'floor' (recomendado), 'round', 'ceil'
    form_date: str = None               # "YYYY-MM-DD" → congela dados até D-1 para formação
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

    # Pré-filtro de liquidez (do Fundamentus, se houver)
    if "liq2m" in df_res.columns:
        df_res = df_res[df_res["liq2m"] >= LIQ_MIN_RS].copy()

    # ---------- 2) Preços & Volumes (yfinance) ----------
    tickers = sorted(df_res["ticker_yf"].unique().tolist())
    if len(tickers) == 0:
        raise RuntimeError("Universo vazio após filtro de liquidez do Fundamentus.")

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

    # Congelar dados até a data de formação (D-1) se fornecida
    if form_date is not None:
        px = px.loc[:form_date]
        vol = vol.loc[:form_date]

    # Limpa ativos sem histórico suficiente
    valid_cols = px.columns[px.notna().sum() > 120]
    px = px[valid_cols]
    vol = vol[valid_cols]
    df_res = df_res[df_res["ticker_yf"].isin(valid_cols)]

    # ---------- 3) Liquidez e Volatilidade (baseados em preço*volume) ----------
    liq = (px * vol)  # R$
    liq_med60 = liq.tail(60).median()
    ret = px.pct_change(fill_method=None)
    vol20 = ret.rolling(20).std().tail(1).T.squeeze()

    mask_liq = liq_med60 >= LIQ_MIN_RS
    if CUT_VOL_TOP and CUT_VOL_TOP > 0:
        lim_vol = vol20.quantile(1 - CUT_VOL_TOP)
        mask_vol = vol20 <= lim_vol
    else:
        mask_vol = pd.Series(True, index=vol20.index)

    universe_ok = pd.Index(mask_liq[mask_liq].index).intersection(mask_vol[mask_vol].index)
    px = px[universe_ok]
    vol = vol[universe_ok]
    vol20 = vol20[universe_ok]
    base = df_res[df_res["ticker_yf"].isin(universe_ok)].set_index("ticker_yf")

    # ---------- 4) Mapeamento de setor (para TODO o universo atual) ----------
    setor_map_all = {}
    for t in tqdm(px.columns.tolist(), desc="Mapeando setor (Fundamentus)"):
        b3 = t.replace(".SA", "")
        try:
            info = fundamentus.get_papel(b3)
            setor_map_all[t] = str(info.loc[b3, "Setor"])
        except Exception:
            setor_map_all[t] = "NA"
    setores_all = pd.Series(setor_map_all)

    # ---------- 5) Momentum curto (x-1) ----------
    mom3m_ex1w = px.apply(lambda s: window_return(s, 63, EXCLUIR_ULTIMA_SEMANA), axis=0)
    mom1m_ex3d = px.apply(lambda s: window_return(s, 21, EXCLUIR_ULTIMOS_3_DIAS), axis=0)
    score_mom = 0.65*rank_pct_safe(mom3m_ex1w) + 0.35*rank_pct_safe(mom1m_ex3d)

    # ---------- 6) Qualidade & Valuation (winsor + z POR SETOR) ----------
    # normaliza aliases de colunas que podem variar
    fund = base.copy()
    aliases = {
        "pl": ["pl","P/L"],
        "pvp": ["pvp","P/VP"],
        "evebitda": ["evebitda","EV/EBITDA","evebitda_aj"],
        "mrgliq": ["mrgliq","marg_liq","margem_liquida","mrg_liq"],
        "roe": ["roe","ROE"],
        "liqc": ["liqc","liq_corrente","liquidez_corrente"],
        "c5y": ["c5y","cres5a","cres_5a","cres_5y"],
        "divbpatr": ["divbpatr","divbrpatr","div_br_patr"]
    }
    # cria colunas-alvo
    for dst, opts in aliases.items():
        if dst not in fund.columns:
            for o in opts:
                if o in fund.columns:
                    fund[dst] = fund[o]
                    break
        if dst not in fund.columns:
            fund[dst] = np.nan

    # winsorize
    for c in ["pl","pvp","evebitda","mrgliq","roe","liqc","c5y","divbpatr"]:
        fund[c] = winsorize(pd.to_numeric(fund[c], errors="coerce"))

    # desativar EV/EBITDA para financeiros
    mask_fin = setores_all.reindex(fund.index).fillna("NA").str.contains("Finan|Banco|Segur", case=False, regex=True)
    fund.loc[mask_fin, "evebitda"] = np.nan

    # z por setor
    z_ROE = z_by_group(fund["roe"], setores_all)
    z_ML  = z_by_group(fund["mrgliq"], setores_all)
    z_CR  = z_by_group(fund["c5y"], setores_all)
    z_LC  = z_by_group(fund["liqc"], setores_all)
    z_DB  = -z_by_group(fund["divbpatr"], setores_all)

    z_PL  = -z_by_group(fund["pl"], setores_all)
    z_EV  = -z_by_group(fund["evebitda"], setores_all)
    z_PVP = -z_by_group(fund["pvp"], setores_all)

    score_qual = pd.concat([z_ROE, z_ML, z_CR, z_LC, z_DB], axis=1).mean(axis=1, skipna=True)
    score_val  = pd.concat([z_PL,  z_EV,  z_PVP],                 axis=1).mean(axis=1, skipna=True)

    # Barreira de qualidade: corta o pior 35%
    qcut = score_qual.quantile(0.35)
    vivos_idx = score_qual[score_qual >= qcut].index

    # alinhar séries do universo filtrado por qualidade
    score_mom = score_mom.reindex(vivos_idx).dropna()
    vol20 = vol20.reindex(vivos_idx)
    score_qual = score_qual.reindex(vivos_idx)
    score_val  = score_val.reindex(vivos_idx)

    # ---------- 7) Score Final e ranking ----------
    sf_mom  = (score_mom - score_mom.mean()) / (score_mom.std(ddof=0) if score_mom.std(ddof=0) > 0 else 1.0)
    sf_qual = (score_qual - score_qual.mean()) / (score_qual.std(ddof=0) if score_qual.std(ddof=0) > 0 else 1.0)
    sf_val  = (score_val  - score_val.mean())  / (score_val.std(ddof=0)  if score_val.std(ddof=0)  > 0 else 1.0)

    Score_Final = (0.60*sf_mom + 0.25*sf_qual + 0.15*sf_val).sort_values(ascending=False)
    candidatos = Score_Final.index.tolist()

    # ---------- 8) Limite setorial (nº máx. por setor) ----------
    sel, cont_setor = [], {}
    for t in candidatos:
        setor = setores_all.get(t, "NA")
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
    setores = setores_all.reindex(sel).fillna("NA")

    # ---------- 9) Pesos ∝ Score/Vol_20d + caps (iterativo) ----------
    alpha = (Score_Final / vol20_sel).replace([np.inf, -np.inf], np.nan).clip(lower=0).fillna(0.0)
    if alpha.sum() == 0:
        # fallback: equal-weight
        w0 = pd.Series(1.0 / len(alpha), index=alpha.index)
    else:
        w0 = alpha / alpha.sum()

    w = enforce_caps_iterative(w0, setores, cap_p=CAP_PAPEL, cap_s=CAP_SETOR)

    # ---------- 10) Quantidades (REAL: shares inteiras) ----------
    # Preço de referência (último fechamento disponível na formação)
    precos = px[sel].ffill().iloc[-1]
    aloc_R = w * CAPITAL
    qtd_teor = aloc_R / precos  # teórica (float)

    shares_int = _discretize_shares(qtd_teor, mode=exec_modo, lot_size=lote_size, rounding=rounding)

    # Valor investido inicial e caixa residual
    invested = (shares_int * precos).astype(float)
    invested_total = float(invested.sum())
    cash_residual = float(CAPITAL - invested_total)

    # ---------- 11) Greedy fill (fracionário) para gastar ~100% mantendo caps ----------
    if exec_modo == "fracionario" and cash_residual > 0 and len(sel) > 0:
        # ordem de prioridade: maior 'alpha' e preço acessível
        order = alpha.sort_values(ascending=False).index.tolist()
        # mapa setor -> índice dos papéis
        idx_setor = {s: setores[setores == s].index.tolist() for s in setores.unique()}

        # calcula pesos efetivos atuais
        def current_weights(inv):
            tot = float(inv.sum())
            return inv / tot if tot > 0 else inv * 0.0

        weight_eff = current_weights(invested)
        # loop até não conseguir alocar mais 1 ação em nenhum papel
        changed = True
        while changed:
            changed = False
            for t in order:
                p = precos[t]
                if p <= 0 or cash_residual + 1e-9 < p:
                    continue
                # checa caps prospectivos após comprar 1 cota
                new_invested_total = invested_total + p
                new_inv_t = invested[t] + p
                new_w_t = new_inv_t / new_invested_total
                if new_w_t > CAP_PAPEL + 1e-9:
                    continue
                s = setores[t]
                setor_names = idx_setor.get(s, [])
                new_sector_sum = (invested.loc[setor_names].sum() + p) / new_invested_total
                if new_sector_sum > CAP_SETOR + 1e-9:
                    continue
                # aplica compra
                shares_int[t] += 1
                invested[t] = float(new_inv_t)
                invested_total = float(new_invested_total)
                cash_residual = float(CAPITAL - invested_total)
                weight_eff = current_weights(invested)
                changed = True
                # se caixa remanescente não compra mais nada, sai
                if cash_residual < precos.min():
                    break

    # Pesos efetivos finais
    invested_total = float(invested.sum())
    cash_residual = float(CAPITAL - invested_total)
    weight_effective = invested / invested_total if invested_total > 0 else invested * 0.0

    # ---------- 12) Salva LOCK ----------
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
            "weights_formula": "weight ∝ Score_Final / Vol_20d (caps iterativos)",
            "form_date": form_date
        },
        "execution": {
            "mode": exec_modo,                 # 'fracionario' ou 'lote'
            "lot_size": lote_size if exec_modo == "lote" else 1,
            "rounding": rounding,              # 'floor', 'round' ou 'ceil'
            "invested_total": invested_total,
            "cash_residual": cash_residual
            # dica: no fracionário, muitas corretoras usam sufixo 'F' (ex.: PETR4F)
        },
        "holdings": []
    }

    for t in sel:
        out["holdings"].append({
            "ticker_b3": t.replace(".SA",""),
            "ticker_yf": t,
            "setor": setores.get(t, "NA"),
            "score_final": float(Score_Final.get(t, np.nan)),
            "vol20d": float(vol20_sel.get(t, np.nan)),
            "alpha": float(alpha.get(t, 0.0)),
            "weight_target": float(w.get(t, 0.0)),            # peso alvo teórico (contínuo)
            "last_price_ref": float(precos.get(t, np.nan)),
            "shares_target": float((w.get(t, 0.0) * CAPITAL) / max(precos.get(t, np.nan), 1e-9)),
            "shares": int(shares_int.get(t, 0)),
            "invested": float(invested.get(t, 0.0)),
            "weight_effective": float(weight_effective.get(t, 0.0))
        })

    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Resumo amigável
    df_out = pd.DataFrame(out["holdings"]).sort_values("weight_effective", ascending=False)
    print(f"Carteira TRAVADA em {lock_date} → {output}")
    print(f"Investido: R$ {invested_total:,.2f} | Caixa residual: R$ {cash_residual:,.2f}")
    cols_print = ["ticker_b3","setor","shares","last_price_ref","invested","weight_target","weight_effective"]
    print(df_out[cols_print].to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/portfolio_lock.json")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--max-por-setor", type=int, default=MAX_POR_SETOR,
                   help="Número máximo de papéis do mesmo setor na carteira (default: %(default)s)")
    # Parâmetros de execução real
    p.add_argument("--exec-modo", choices=["fracionario","lote"], default="fracionario",
                   help="Modo de execução real: 'fracionario' (passo=1) ou 'lote' (passo=lote-size).")
    p.add_argument("--lote-size", type=int, default=100,
                   help="Tamanho do lote padrão quando exec-modo = 'lote' (default: 100).")
    p.add_argument("--rounding", choices=["floor","round","ceil"], default="floor",
                   help="Política de arredondamento para discretização das shares.")
    p.add_argument("--form-date", default=None,
                   help="YYYY-MM-DD para congelar dados de formação (use a QUINTA se vai executar na sexta).")
    args = p.parse_args()
    main(
        output=args.output,
        use_cache=args.use_cache,
        max_por_setor=args.max_por_setor,
        exec_modo=args.exec_modo,
        lote_size=args.lote_size,
        rounding=args.rounding,
        form_date=args.form_date
    )