import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import altair as alt
import requests
import yfinance as yf
import streamlit as st
from bcb import sgs  # python-bcb

# NEW: calendário B3
import pandas_market_calendars as mcal

st.set_page_config(page_title="Carteira B3 - Momentum+Qual/Val", layout="wide")

# ---------------- Opções ----------------
# False = Preço-only (padrão, não embute dividendos no histórico)
# True  = Total Return (soma dividendos ao caixa ao longo do tempo)
MODE_TOTAL_RETURN = False

# -------------- Calendário B3 --------------
def b3_trading_days(start, end):
    """
    Retorna somente dias com pregão na B3 (sem feriados/finais de semana).
    Índice de datas normalizado e sem timezone.
    """
    b3 = mcal.get_calendar("BVMF")
    sched = b3.schedule(start_date=start, end_date=end)
    # 1D aqui gera um ponto por sessão (pregão). Não cria dias sem pregão.
    idx = mcal.date_range(sched, frequency="1D").tz_localize(None).normalize()
    # segurança: garante que o primeiro dia >= start e o último <= end
    idx = idx[(idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))]
    return idx

# ---------------- Helpers ----------------
@st.cache_data(ttl=60*60)
def load_lock(path="data/portfolio_lock.json"):
    """
    Lê o lock (execução real) e retorna:
      lock (dict), df_hold (DataFrame com pesos e execução).
    Colunas esperadas por holding:
      ticker_b3, ticker_yf, setor, score_final, vol20d, alpha,
      weight_target, weight_effective, last_price_ref,
      shares_target, shares, invested.
    """
    with open(path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    df = pd.DataFrame(lock["holdings"])

    # Normaliza tipos numéricos
    num_cols = [
        "score_final","vol20d","alpha","weight_target","weight_effective",
        "last_price_ref","shares_target","shares","invested"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Campo 'weight' padronizado
    if "weight_effective" in df.columns and df["weight_effective"].notna().any():
        df["weight"] = df["weight_effective"]
    else:
        df["weight"] = df.get("weight_target", np.nan)

    # Garante colunas
    for col in ["ticker_yf", "ticker_b3", "setor"]:
        if col not in df.columns:
            df[col] = np.nan

    # Metadados de execução
    exec_meta = lock.get("execution", {}) or {}
    exec_mode  = exec_meta.get("mode", "fracionario")
    lot_size   = exec_meta.get("lot_size", 1 if exec_mode=="fracionario" else 100)
    rounding   = exec_meta.get("rounding", "floor")
    invested_total = float(exec_meta.get("invested_total", float(df["invested"].sum() if "invested" in df else 0.0)))
    cash_residual  = float(exec_meta.get("cash_residual", float(lock.get("capital", 0.0) - invested_total)))

    # Consistência
    if "invested" not in df.columns or df["invested"].isna().all():
        df["invested"] = df["shares"] * df["last_price_ref"]
        invested_total = float(df["invested"].sum())
        cash_residual = float(lock.get("capital", 0.0) - invested_total)

    lock["execution"] = {
        "mode": exec_mode,
        "lot_size": lot_size,
        "rounding": rounding,
        "invested_total": invested_total,
        "cash_residual": cash_residual,
    }
    return lock, df

@st.cache_data(ttl=60*60)
def get_prices_yf(tickers, start, end=None, auto_adjust=False):
    """
    Baixa Close desde 'start' até 'end' (inclusive) e reindexa no calendário da B3.
    auto_adjust=False => preço 'puro' (recomendado p/ consistência com NAV e dividendos explícitos)
    """
    end = end or date.today()
    data = yf.download(
        tickers, start=start, end=end + timedelta(days=1),
        auto_adjust=auto_adjust, progress=False, threads=True
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].ffill()
    else:
        # único ticker
        t = tickers[0] if isinstance(tickers, (list, tuple)) and len(tickers) == 1 else "Close"
        close = data[["Close"]].rename(columns={"Close": t}).ffill()

    # Reindexa apenas em dias de pregão da B3
    idx = b3_trading_days(start, end)
    close = close.reindex(idx).ffill()
    return close

@st.cache_data(ttl=24*60*60)
def get_company_names_yf(tickers):
    """
    Mapeia ticker_yf -> nome curto da empresa (Yahoo).
    Tenta shortName > longName > displayName; fallback: próprio ticker.
    """
    names = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            name = info.get("shortName") or info.get("longName") or info.get("displayName") or t
        except Exception:
            name = t
        names[t] = name
    return pd.Series(names, name="nome")

@st.cache_data(ttl=60*60)
def get_dividends_yf(tickers, start, end=None):
    """
    Dividendos por dia (R$ por ação) para cada ticker (preço não ajustado).
    Retorna DataFrame alinhado ao calendário da B3.
    """
    end = end or date.today()
    frames = {}
    idx = b3_trading_days(start, end)
    for t in tickers:
        try:
            s = yf.Ticker(t).dividends  # Series com datas (UTC) e valores
            if s is None or s.empty:
                frames[t] = pd.Series(0.0, index=idx)
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s.reindex(idx).fillna(0.0)  # somente dias de pregão
            frames[t] = s
        except Exception:
            frames[t] = pd.Series(0.0, index=idx)
    if not frames:
        return pd.DataFrame(index=idx)
    return pd.DataFrame(frames).reindex(idx).fillna(0.0)

# Funções para cdi e selic (sem uso)
@st.cache_data(ttl=60*60)
def get_cdi_daily(start):
    """
    CDI diário (% a.d.) via Ipeadata: SGS366_CDI366.
    Saída: DataFrame com índice em dias úteis e coluna 'CDI_DAILY' em DECIMAL ao dia.
    """
    try:
        start_dt = pd.to_datetime(start).normalize()
    except Exception:
        start_dt = (pd.Timestamp.today() - pd.Timedelta(days=30)).normalize()

    fetch_from = (start_dt - pd.Timedelta(days=15)).normalize()
    # índice de saída: só datas, sem hora
    idx = pd.bdate_range(start_dt, date.today())

    try:
        url = "https://ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='SGS366_CDI366')"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        arr = r.json().get("value", [])
        if not arr:
            return pd.DataFrame(index=idx, data={"CDI_DAILY": np.nan})

        df = pd.DataFrame(arr)

        # 1) Parse datas -> UTC -> remove fuso
        df["VALDATA"] = pd.to_datetime(df["VALDATA"], utc=True, errors="coerce").dt.tz_convert(None)

        # 2) Indexa e **NORMALIZA** para remover a hora (ESSENCIAL!)
        df = df.dropna(subset=["VALDATA"]).set_index("VALDATA").sort_index()
        df.index = df.index.normalize()  # <-- sem hora

        # 3) Converte valor p/ numérico e renomeia
        df["VALVALOR"] = pd.to_numeric(df["VALVALOR"], errors="coerce")
        df = df.rename(columns={"VALVALOR": "CDI_DAILY_PCT"})

        # 4) Se houver mais de 1 registro no mesmo dia, fica com o último
        #df = df[~df.index.duplicated(keep="last")]

        # 5) Filtra desde 'fetch_from' (também normalizado)
        df = df.loc[df.index >= fetch_from]

        # 6) % a.d. -> decimal a.d.
        df["CDI_DAILY"] = df["CDI_DAILY_PCT"] / 100.0

        # 7) Reindex em dias úteis (datas puras) e preencher
        out = df[["CDI_DAILY"]].reindex(idx).ffill().bfill()
        return out

    except Exception as e:
        # Se preferir silencioso, troque por retorno com NaN.
        raise RuntimeError(f"Falha ao consultar/parsear Ipeadata: {e}")


@st.cache_data(ttl=60*60)
def get_selic_daily(start):
    """
    SGS 11 — SELIC diária (% a.d.). BCB publica em D+1.
    Usamos end = hoje-1 para evitar 'Value(s) not found'.
    Retorna DatetimeIndex para compatibilidade com reindex em idx úteis.
    """
    try:
        start_dt = pd.to_datetime(start)
    except Exception:
        start_dt = pd.Timestamp.today() - pd.Timedelta(days=10)

    end_dt = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)  # D+1
    if start_dt > end_dt:
        start_dt = end_dt - pd.Timedelta(days=10)

    try:
        df = sgs.get({"selic": 11}, start=start_dt.date(), end=end_dt.date())
        df = df.rename(columns={"selic": "SELIC_DAILY"})
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame(columns=["SELIC_DAILY"])

def to_index(series_daily_returns, base=100.0):
    s = pd.Series(series_daily_returns).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return base * (1.0 + s).cumprod()

def compute_drawdown(index_series):
    s = pd.Series(index_series).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    peak = s.cummax()
    return s / peak - 1.0

def annualized_vol(ret_daily, trading_days=252):
    s = pd.Series(ret_daily).dropna()
    if s.empty or s.std(ddof=0) == 0:
        return 0.0
    return float(s.std(ddof=0) * np.sqrt(trading_days))

def safe_total_return(index_series):
    s = pd.Series(index_series).dropna()
    if s.size < 2 or s.iloc[0] == 0 or np.isnan(s.iloc[0]):
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1.0)

# ----------------- App -----------------
st.title("Painel Quant B3 — Momentum & Value")
st.caption("por Joel Victor")

lock, df_hold = load_lock()
lock_date = pd.to_datetime(lock["locked_at"]).date()
today = date.today()

exec_meta = lock.get("execution", {})
capital = float(lock.get("capital", 0.0))
invested_total = float(exec_meta.get("invested_total", 0.0))
cash_residual_start = float(exec_meta.get("cash_residual", capital - invested_total))

colA, colB, colC, colD = st.columns(4)
colA.metric("Data de Travamento", str(lock_date))
colB.metric("Capital Inicial", f"R$ {capital:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
colC.metric("N de Ativos", f"{len(df_hold)}")
colD.metric(
    "Execução",
    f"{exec_meta.get('mode','—')} (lote={exec_meta.get('lot_size','—')}, {exec_meta.get('rounding','—')})"
)

colE, colF = st.columns(2)
colE.metric("Investido (no lock)", f"R$ {invested_total:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))
colF.metric("Caixa residual (no lock)", f"R$ {cash_residual_start:,.2f}".replace(",", "X").replace(".", ",").replace("X","."))

with st.expander("Ver parâmetros do lock (rules)"):
    st.json(lock.get("rules", {}), expanded=False)
with st.expander("Ver metadados de execução (execution)"):
    st.json(lock.get("execution", {}), expanded=False)

st.markdown("---")

# ---------- Preços & Benchmarks ----------
tickers_yf = df_hold["ticker_yf"].dropna().tolist()
bench_candidates = ["^BVSP", "BOVA11.SA"]  # preferência índice oficial
start_prices = lock_date - timedelta(days=7)  # margem

# preços (preço puro por padrão)
close_all = get_prices_yf(tickers_yf + bench_candidates, start=start_prices, end=today, auto_adjust=False)
if close_all.empty:
    st.error("Falha ao obter cotações do Yahoo Finance.")
    st.stop()

# Janela do lock até o último pregão disponível
idx = b3_trading_days(lock_date, today)
close = close_all.reindex(idx).ffill()

# Benchmark disponível (ordem de preferência)
bench = next((t for t in bench_candidates if t in close.columns and close[t].notna().any()), None)

# Separa benchmark e ações
close_bench = close[bench] if bench else pd.Series(index=idx, dtype=float)
close_sel = close.drop(columns=[c for c in bench_candidates if c in close.columns], errors="ignore")
if close_sel.empty or close_sel.shape[0] == 0:
    st.warning("Sem cotações válidas no período para os papéis da carteira.")
    st.stop()

# ---------- Nomes ----------
names_series = get_company_names_yf(tickers_yf)  # index: ticker_yf -> nome
b3_series = df_hold.set_index("ticker_yf")["ticker_b3"].astype(str)

# ---------- NAV (preço-only/total return) ----------
shares = df_hold.set_index("ticker_yf")["shares"].reindex(close_sel.columns).fillna(0.0)

# NAV investido (preço puro)
nav_invested = (close_sel * shares).sum(axis=1)  # só ações (preço)

# Dividendos (se Total Return)
div_cash_cum = pd.Series(0.0, index=nav_invested.index)
if MODE_TOTAL_RETURN:
    div_df = get_dividends_yf(close_sel.columns.tolist(), start=lock_date, end=today)
    # fluxo de dividendos por dia (R$/ação * nº de ações)
    div_flow = (div_df * shares.reindex(div_df.columns)).sum(axis=1).reindex(nav_invested.index).fillna(0.0)
    div_cash_cum = div_flow.cumsum()

# Caixa: começa no cash_residual do lock e (se TR) soma dividendos ao longo do tempo
cash_series = pd.Series(cash_residual_start, index=nav_invested.index) + div_cash_cum

nav_total = nav_invested + cash_series

if nav_total.empty or nav_total.iloc[0] == 0:
    st.warning("NAV insuficiente para calcular índices. Verifique o lock e as cotações.")
    st.stop()

nav_invested_index = 100 * nav_invested / nav_invested.iloc[0]
nav_total_index    = 100 * nav_total    / nav_total.iloc[0]

# Benchmark índice
if bench and not close_bench.dropna().empty:
    bench_index = 100 * close_bench / close_bench.iloc[0]
else:
    bench_index = pd.Series(dtype=float)

# ---------- KPIs (sem CDI/SELIC) ----------
def fmt_pct(x): return "—" if (x is None or np.isnan(x)) else f"{x*100:,.2f}%"
def fmt_pp(x):  return "—" if (x is None or np.isnan(x)) else f"{x*100:,.2f} p.p."

# retornos totais no período
ret_price_invested = safe_total_return(nav_invested_index)
ret_price_total    = safe_total_return(nav_total_index)
ret_bench_total    = safe_total_return(bench_index)

# risco (investido)
ret_daily_invested = nav_invested.pct_change()
vol_ann_invested   = annualized_vol(ret_daily_invested)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Retorno (investido)", fmt_pct(ret_price_invested))
    with st.popover("O que é?"):
        st.write("Variação percentual do valor **somente investido em ações** desde o lock.")
        st.latex(r"""
R_{\text{investido}}(t)
= \frac{\sum_i p_{i,t}\,q_i}{\sum_i p_{i,0}\,q_i} - 1
""")
        st.caption("p_{i,t} = preço do papel i no dia t; q_i = número de ações do papel i; 0 = dia do lock.")

with c2:
    st.metric("Retorno (com caixa)", fmt_pct(ret_price_total))
    with st.popover("O que é?"):
        st.write("Inclui **caixa residual** do lock e (opcionalmente) **dividendos** acumulados se Total Return estiver habilitado.")
        st.latex(r"""
R_{\text{com\,caixa}}(t)
= \frac{\sum_i p_{i,t}\,q_i + C_0 + \mathbf{1}_{TR}\!\sum_{\tau \le t} D_{\tau}}
       {\sum_i p_{i,0}\,q_i + C_0} - 1
""")
        st.caption("C_0 = caixa no lock; D_τ = dividendos recebidos até t; 𝟙_TR = 1 se MODE_TOTAL_RETURN=True, senão 0.")

with c3:
    st.metric("Vol. Anualizada (investido)", fmt_pct(vol_ann_invested))
    with st.popover("O que é?"):
        st.write("Volatilidade anualizada do retorno diário da carteira **investida**.")
        st.latex(r"""
\sigma_{\text{anual}}
= \sqrt{252}\;\cdot\; \operatorname{std}\!\left(r_d\right),
\quad
r_d = \frac{\text{NAV}_d}{\text{NAV}_{d-1}} - 1
""")
        st.caption("Usa 252 pregões por ano. std = desvio-padrão populacional (ddof=0).")

with c4:
    exc_bench = ret_price_invested - ret_bench_total if not np.isnan(ret_bench_total) and not np.isnan(ret_price_invested) else np.nan
    st.metric("Excesso vs Benchmark", fmt_pp(exc_bench))
    with st.popover("O que é?"):
        st.write("Diferença entre o retorno da carteira (investido) e o IBOV no período.")
        st.latex(r"""
\text{Excesso}(t)
= R_{\text{investido}}(t) - R_{\text{IBOV}}(t),
\quad
R_{\text{IBOV}}(t)=\frac{P^{\text{IBOV}}_t}{P^{\text{IBOV}}_0}-1
""")

# ---------- Gráfico principal (Altair) ----------
st.subheader("Desempenho acumulado — Carteira vs IBOV")
st.caption("Retorno acumulado desde o dia do lock; base 0%. Carteira considera apenas as ações.")

# Mesmas datas nas duas séries
common = nav_invested_index.dropna().index.intersection(bench_index.dropna().index)
carteira_idx = nav_invested_index.loc[common]
ibov_idx     = bench_index.loc[common]

# Retorno acumulado em DECIMAL (p/ formatar como %)
ret_carteira = carteira_idx / carteira_idx.iloc[0] - 1.0
ret_ibov     = ibov_idx     / ibov_idx.iloc[0]     - 1.0

perf_long = (
    pd.DataFrame({
        "date": common,
        "Carteira (investido)": ret_carteira.values,
        "IBOV": ret_ibov.values
    })
    .melt("date", var_name="Série", value_name="Ret")
    .dropna(how="any")
)

# Desempenho acumulado
chart = (
    alt.Chart(perf_long)  # <-- sem title=
      .mark_line(point=True)
      .encode(
          x=alt.X("date:T", title="Data", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0)),
          y=alt.Y("Ret:Q", title=None, axis=alt.Axis(format=".2%"), scale=alt.Scale(zero=False)),
          color=alt.Color("Série:N", title=""),
          tooltip=[alt.Tooltip("date:T", format="%d/%m/%Y"), "Série:N", alt.Tooltip("Ret:Q", format=".2%")]
      )
      .interactive()
)
st.altair_chart(chart, use_container_width=True)

# Excesso acumulado (investido vs IBOV)
st.subheader("Excesso acumulado — Carteira vs IBOV (investido)")

excesso = (ret_carteira - ret_ibov)  # decimal
df_exc = pd.DataFrame({"date": common, "Excesso": excesso}).dropna()

chart_exc = (
    alt.Chart(df_exc)  # <-- sem title=
      .mark_line()
      .encode(
          x=alt.X("date:T", title="Data", axis=alt.Axis(format="%d/%m/%Y", labelAngle=0)),
          y=alt.Y("Excesso:Q", title=None, axis=alt.Axis(format=".2%"), scale=alt.Scale(zero=False)),
          tooltip=[alt.Tooltip("date:T", format="%d/%m/%Y"), alt.Tooltip("Excesso:Q", format=".2%")]
      )
      .interactive()
)
st.altair_chart(chart_exc, use_container_width=True)

# ---------- Rendimento por papel (preço) ----------
st.subheader("Rendimento por papel (preço{})".format(" + dividendos no caixa" if MODE_TOTAL_RETURN else ""))
first_prices = close_sel.iloc[0]
last_prices  = close_sel.iloc[-1]

price_change_per_share = (last_prices - first_prices)                         # R$/ação
pnl_price = ((last_prices - first_prices) * shares).fillna(0.0)               # R$ por papel

df_rend = pd.DataFrame({
    "ticker_yf": close_sel.columns,
    "nome": names_series.reindex(close_sel.columns).values,
    "ticker_b3": b3_series.reindex(close_sel.columns).values,
    "setor": df_hold.set_index("ticker_yf").reindex(close_sel.columns)["setor"].values,
    "shares": shares.values.astype(int),
    "preco_inicial": first_prices.values,
    "preco_hoje": last_prices.values,
    "Variação preço/ação (R$)": price_change_per_share.values,
    "PnL de preço (R$)": pnl_price.values,
}).set_index("ticker_b3").sort_values("PnL de preço (R$)", ascending=False)

st.dataframe(df_rend[
    ["nome","setor","shares","preco_inicial","preco_hoje","Variação preço/ação (R$)","PnL de preço (R$)"]
], width='stretch')

st.write("**Contribuição por papel (PnL de preço em R$)**")
st.caption("Quanto cada papel contribuiu em **R$** para o resultado desde o lock, considerando apenas variação de preço (sem caixa).")
st.latex(r"""
\text{PnL}^{\text{preço}}_i(t) \;=\; \big(p_{i,t}-p_{i,0}\big)\,\cdot\,q_i
\quad\Rightarrow\quad
\sum_i \text{PnL}^{\text{preço}}_i(t)
= \sum_i p_{i,t}q_i \;-\; \sum_i p_{i,0}q_i
= \text{NAV}_{\text{investido}}(t) - \text{NAV}_{\text{investido}}(0)
""")
labels_contrib = (b3_series + " — " + names_series.reindex(b3_series.index).fillna("")).reindex(pnl_price.index)
st.bar_chart(pnl_price.sort_values(ascending=False).rename(index=lambda t: labels_contrib.get(t, t)))

# ---------- Alocação: alvo (lock) x atual (preço) ----------
st.subheader("Alocação alvo x alocação atual")
st.caption("Compara o **peso teórico** salvo no lock (alvo) com o **peso de mercado** hoje calculado sobre o capital investido (sem caixa).")
st.latex(r"""
w^{\text{alvo}}_i \;=\; \text{weight\_target}_i
\qquad\qquad
w^{\text{atual}}_i(t) \;=\; \frac{p_{i,t}\,q_i}{\sum_j p_{j,t}\,q_j}
\quad\text{e}\quad
w^{\text{atual}}_{\text{setor}}(t) \;=\; \sum_{i\in \text{setor}} w^{\text{atual}}_i(t)
""")

df_alloc = df_hold.set_index("ticker_yf")[["ticker_b3","setor","weight","shares"]].copy()
alloc_target = df_alloc["weight"].copy()  # weight_effective (ou target) do lock
alloc_current_invested = (last_prices * df_alloc["shares"]) / nav_invested.iloc[-1]
alloc_current_invested = alloc_current_invested.fillna(0.0)

# Por papel (labels com nome)
idx_labels = (df_alloc["ticker_b3"].astype(str) + " — " + names_series.reindex(df_alloc.index).fillna("")).reindex(close_sel.columns)
df_alloc_plot = pd.DataFrame({
    "%_alvo (lock)": (alloc_target.reindex(close_sel.columns).values * 100),
    "%_atual (investido)": (alloc_current_invested.reindex(close_sel.columns).values * 100)
}, index=idx_labels)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Por papel**")
    st.bar_chart(df_alloc_plot, width='stretch')

with c2:
    st.markdown("**Por setor (alvo do lock)**")
    setor_target = df_hold.groupby("setor")["weight"].sum().sort_values(ascending=False) * 100
    st.bar_chart(setor_target, width='stretch')

# ---------- Posições detalhadas ----------
st.subheader("Posições — resumo do lock + mercado")
df_pos = df_hold.set_index("ticker_yf")[[
    "ticker_b3","setor","weight_target","weight_effective",
    "shares_target","shares","last_price_ref","invested","score_final","vol20d","alpha"
]].copy()
df_pos["nome"] = names_series.reindex(df_pos.index).values
df_pos["preco_inicial"] = first_prices
df_pos["preco_hoje"] = last_prices
df_pos["retorno_preco_%"] = (df_pos["preco_hoje"]/df_pos["preco_inicial"] - 1.0) * 100.0
df_pos["peso_atual (investido)"] = (df_pos["preco_hoje"]*df_pos["shares"]) / nav_invested.iloc[-1]
df_pos = df_pos.reset_index().rename(columns={"ticker_yf":"ticker_yf (yfinance)"})
st.dataframe(df_pos, width='stretch')

# ---------- Metodologia (com LaTeX) ----------
with st.expander("Metodologia — como esta carteira é construída e travada (explicação completa)"):
    # para interpolar números do lock se existirem
    rules = lock.get("rules", {}) or {}
    liq_min_rs_val = int(rules.get("liq_min_rs", 25_000_000))
    liq_min_rs_txt = f"R$ {liq_min_rs_val:,.0f}".replace(",", "X").replace(".", ",").replace("X",".")
    cut_vol_top = float(rules.get("cut_vol_top", 0.00))

    st.markdown("### 0) Formação D-1 (anti-look-ahead)")
    st.write("Rodamos os sinais **congelando os dados até a quinta** via `--form-date YYYY-MM-DD`. "
             "Isso evita usar informação de sexta para formar a carteira executada no leilão de sexta.")

    st.markdown("### 1) Universo (Fundamentus)")
    st.write(f"Partimos do universo B3 via `fundamentus.get_resultado()`. Usamos colunas como `pl`, `pvp`, "
             f"`evebitda`, `mrgliq`, `roe`, `liqc`, `liq2m`, `c5y`, `divbpatr`. "
             f"Aplicamos **pré-filtro de liquidez**: `liq2m ≥ {liq_min_rs_txt}/dia`. "
             "Mapeamos os tickers para Yahoo adicionando `.SA`.")

    st.markdown("### 2) Preços & Volumes (Yahoo Finance)")
    st.write("Baixamos **Close** e **Volume** no Yahoo para todos os `.SA` desde `2023-01-01`. "
             "Removemos ativos com histórico insuficiente (≤120 pregões). Se `--form-date` for passado, "
             "**cortamos a série até D-1**.")

    st.markdown("### 3) Filtros de execução (na série de preços)")
    st.write(f"- **Liquidez atual**: mediana 60d de *(Preço × Volume)*; exigimos `≥ {liq_min_rs_txt}`.\n"
             f"- **Volatilidade**: **não** cortamos topo por vol (`CUT_VOL_TOP = {cut_vol_top:.0%}`); "
             "o risco é penalizado nos **pesos** (passo 8).")

    st.markdown("### 4) Setor para todo o universo")
    st.write("Mapeamos **setor** (via `fundamentus.get_papel`) para cada ticker — habilita padronização por setor.")

    st.markdown("### 5) Sinais")
    st.write("**Momentum curto (regra x-1)** — para reduzir ruído/reversões:")
    st.write("- **3 meses excluindo 1 semana**: retorno de [-63, -5] dias úteis.\n"
             "- **1 mês excluindo 3 dias**: retorno de [-21, -3] dias úteis.\n"
             "Cada janela vira **percent-rank** e combinamos:")
    st.latex(r"""
\text{Score}_{\text{Momentum}}
= 0.65\cdot \text{rank}\_{\%}\big(MOM_{3m\setminus 1w}\big)
+ 0.35\cdot \text{rank}\_{\%}\big(MOM_{1m\setminus 3d}\big)
""")
    st.write("**Qualidade & Valuation** — winsorize + **z-score por setor**. Para financeiros, desativamos **EV/EBITDA**.")
    st.latex(r"""
\text{Qualidade}
= \tfrac{1}{5}\big[z(\text{ROE}) + z(\text{Margem Líq}) + z(\text{Crescimento 5y})
+ z(\text{Liq. Corrente}) - z(\text{Div/Patrim})\big]
""")
    st.latex(r"""
\text{Valuation}
= \tfrac{1}{3}\big[-z(P/L) - z(EV/EBITDA) - z(P/VP)\big]
""")
    st.write("Aplicamos **barreira de qualidade**: cortamos o **pior 35%** em *Qualidade*.")

    st.markdown("### 6) Score Final")
    st.latex(r"""
\text{Score\_Final}
= 0.60\cdot z(\text{Momentum})
+ 0.25\cdot z(\text{Qualidade})
+ 0.15\cdot z(\text{Valuation})
""")

    st.markdown("### 7) Limites por setor (contagem)")
    st.write("Selecionamos até **10** papéis respeitando **máx. 2 por setor**. "
             "Se ficar restritivo, garantimos pelo menos **8** papéis.")

    st.markdown("### 8) Pesos-alvo (contínuos) com caps iterativos")
    st.write("Definimos o *alpha* e projetamos os pesos com caps simultâneos por papel e setor:")
    st.latex(r"""
\alpha_i \;=\; \frac{\text{Score\_Final}_i}{\text{Vol}_{20d,i}}
\quad\Rightarrow\quad
w \propto \alpha
""")
    st.latex(r"""
\text{sujeito a:}\quad
w_i \le 0.12,\qquad
\sum_{i\in s} w_i \le 0.35,\qquad
\sum_i w_i = 1
""")

    st.markdown("### 9) Execução real — discretização de shares e greedy fill")
    st.write("Com o **preço de referência D-1** (`last_price_ref`):")
    st.latex(r"""
\text{shares\_target}_i
= \frac{w_i \cdot \text{capital}}{\text{preço\_ref}_i}
""")
    st.write("Discretizamos por modo **fracionário** (passo=1) ou **lote** (passo=`lot_size`). "
             "Arredondamento **floor**. No fracionário, aplicamos **greedy fill** comprando unidades adicionais "
             "enquanto houver caixa **sem violar** caps de papel e setor, visando **cash ≈ 0**.")

    st.markdown("### 10) “Lock” gerado (JSON)")
    st.write("Gravamos `data/portfolio_lock.json` com `rules`, `execution` e `holdings` (tickers, scores, pesos, "
             "shares, invested etc.).")

    st.markdown("### Observações")
    st.write("- **Execução**: formar na **quinta (D-1)** e enviar ordens no **fechamento de sexta**.\n"
             "- **Risco**: o controle é via **Score/Vol_20d** nos pesos.\n"
             "- **Padronização por setor** evita viés entre bancos e indústrias.\n"
             "- **Financeiros**: ignoramos EV/EBITDA para não distorcer *Valuation*.")