# -*- coding: utf-8 -*-
# =========================================================
# Fuzzy FMEA – Versão Completa (Robusta para CSV) – v3
# =========================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import spearmanr

# -------------------- Config Streamlit/Matplotlib --------------------
st.set_page_config(
    page_title="Classificação de Riscos Geotécnicos em Taludes (FMEA/Fuzzy)",
    layout="wide"
)

plt.rcParams["axes.grid"] = True

# -------------------- Parâmetros/Funções base ------------------------
CONFIG = {
    "id_col": "ID", "s_col": "S", "o_col": "O", "d_col": "D", "npr_col": "NPR",
    "D_REF": 5.0,
    "NPR_BANDS_NORM": [(0.80,"Green"),(1.60,"Yellow"),(2.00,"Orange"),(float("inf"),"Red")],
    "FUZZY_BANDS": [(4.0,"Green"),(6.5,"Yellow"),(8.0,"Orange"),(float("inf"),"Red")],
    "BAND_ORDER": ["Green","Yellow","Orange","Red"],
    "BAND_COLORS": {"Green":"#2ca02c","Yellow":"#ffd23f","Orange":"#ff7f0e","Red":"#d62728"},
    "MF_S_O": {
        "Baixa":("left",1.0,3.5),
        "Média":("trap",2.5,4.5,6.5,7.5),
        "Alta":("trap",6.5,7.5,8.5,9.5),
        "Crítica":("right",9.0,10.0)
    },
    "MF_D": {
        "Excelente":("left",1.0,2.5),
        "Bom":("trap",2.0,3.5,5.0,6.0),
        "Moderado":("trap",5.0,6.0,7.5,8.5),
        "Ruim":("right",8.0,10.0)
    },
    "MF_RISK": {
        "Baixo":("left",1.0,3.0),
        "Médio":("trap",2.0,4.0,5.5,7.0),
        "Alto":("trap",6.0,7.0,8.0,8.5),
        "Crítico":("right",8.0,10.0)
    },
    "RULE_SO":"default_monotonic",
    "RULE_SD":"default_d_adjust"
}
D_REF=CONFIG["D_REF"]
NPR_BANDS_NORM=CONFIG["NPR_BANDS_NORM"]
FUZZY_BANDS=CONFIG["FUZZY_BANDS"]
BAND_ORDER=CONFIG["BAND_ORDER"]
BAND_COLORS=CONFIG["BAND_COLORS"]

def to_band(x,bands):
    try:
        if np.isnan(x): return None
    except TypeError:
        return None
    for thr,name in bands:
        if x < thr: return name
    return bands[-1][1]

def left_shoulder(x,a,b): return np.clip(np.where(x<=a,1,np.where(x>=b,0,(b-x)/(b-a+1e-9))),0,1)
def right_shoulder(x,c,d): return np.clip(np.where(x<=c,0,np.where(x>=d,1,(x-c)/(d-c+1e-9))),0,1)
def trapmf_safe(x,a,b,c,d):
    up=np.where(x<=a,0,np.where(x<b,(x-a)/(b-a+1e-9),1))
    down=np.where(x<=c,1,np.where(x<d,(d-x)/(d-c+1e-9),0))
    return np.clip(np.minimum(up,down),0,1)

def coerce_numeric(s):
    # aceita decimal com vírgula
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

# -------------------- Leitura robusta do CSV -------------------------
st.title("Classificação de Riscos Geotécnicos em Taludes Rodoviários Utilizando a Metodologia FMEA")
st.write("## PUC-Rio")
st.set_page_config(page_title="Classificação de Riscos Geotécnicos em Taludes Rodoviários Utilizando a Metodologia FMEA", layout="wide")
st.write("### Antônio Augusto Zucchi Leite, Fernanda Defilippi Bittar, Antonio Krishnamurti Beleño de Oliveira, Márcio Leão")


uploaded = st.file_uploader("📁 Envie seu CSV (ID, S, O, D, NPR). Aceita ; ou , e decimal com vírgula.", type=["csv"])

def read_csv_robust(file):
    # tenta utf-8-sig / latin-1; autodetecta separador; se vier 1 coluna, tenta sep=';'
    last_err = None
    for enc in ["utf-8-sig", "latin-1"]:
        try:
            df = pd.read_csv(file, sep=None, engine="python", encoding=enc)
            if df.shape[1] == 1:
                file.seek(0)
                df = pd.read_csv(file, sep=";", encoding=enc)
            return df
        except Exception as e:
            last_err = e
            file.seek(0)
    raise last_err

if uploaded:
    try:
        df_raw = read_csv_robust(uploaded)
        st.success(f"✅ CSV carregado: {uploaded.name}")
    except Exception as e:
        st.error(f"Falha ao ler CSV: {e}")
        st.stop()
else:
    st.info("Nenhum arquivo enviado. Usando dados de exemplo.")
    df_raw = pd.DataFrame([
        [1,6,3,3,54],[2,9,3,6,162],[3,6,3,3,54],[4,8,3,7,168],[5,6,6,5,180],
        [6,10,3,8,240],[7,5,3,3,45],[8,7,5,4,140],[9,8,5,7,280],[10,6,4,6,144],
        [11,9,4,7,252],[12,5,7,4,140],[13,7,3,6,126],[14,8,3,5,120],[15,8,4,6,192],
        [16,7,5,5,175],[17,4,3,3,36],[18,5,5,6,150],[19,9,2,4,72],[20,8,2,8,128]
    ], columns=["ID","S","O","D","NPR"])

st.write("### 📄 Pré-visualização do arquivo")
st.dataframe(df_raw.head(30), use_container_width=True)

# -------------------- Mapeamento de colunas (sidebar) ----------------
st.sidebar.header("⚙️ Mapeamento de colunas")

cols = list(df_raw.columns)
lower = {c: c.lower() for c in cols}

def guess(syns):
    for c in cols:
        for k in syns:
            if lower[c] == k: return c
    for c in cols:
        for k in syns:
            if k in lower[c]: return c
    return cols[0] if cols else None

id_guess = guess(["id","item"])
s_guess  = guess(["s","sev","severidade","severity"])
o_guess  = guess(["o","oco","ocorrencia","ocorrência","occurrence"])
d_guess  = guess(["d","det","detec","detecção","deteccao","detection"])
npr_guess= guess(["npr","rpn"])

sel_id  = st.sidebar.selectbox("Coluna ID",   cols, index=cols.index(id_guess)  if id_guess  in cols else 0)
sel_s   = st.sidebar.selectbox("Coluna S",    cols, index=cols.index(s_guess)   if s_guess   in cols else 0)
sel_o   = st.sidebar.selectbox("Coluna O",    cols, index=cols.index(o_guess)   if o_guess   in cols else 0)
sel_d   = st.sidebar.selectbox("Coluna D",    cols, index=cols.index(d_guess)   if d_guess   in cols else 0)
sel_npr = st.sidebar.selectbox("Coluna NPR",  cols, index=cols.index(npr_guess) if npr_guess in cols else 0)

df = df_raw.rename(columns={
    sel_id:"ID", sel_s:"S", sel_o:"O", sel_d:"D", sel_npr:"NPR"
}).copy()

# força numérico
for c in ["ID","S","O","D","NPR"]:
    if c in df.columns:
        df[c] = coerce_numeric(df[c])

# preenche NPR ausente
if df["NPR"].isna().any():
    df.loc[df["NPR"].isna(), "NPR"] = df.loc[df["NPR"].isna(), "S"] * df.loc[df["NPR"].isna(), "O"] * df.loc[df["NPR"].isna(), "D"]

# valida presença dos campos
missing = [c for c in ["ID","S","O","D","NPR"] if c not in df.columns]
if missing:
    st.error(f"Colunas essenciais ausentes após mapeamento: {missing}")
    st.stop()

st.write("### 📊 Dados padronizados (após mapeamento)")
st.dataframe(df, use_container_width=True)

# -------------------- NPR Numérico ----------------------
res_num = df[["ID","S","O","D","NPR"]].copy()
res_num["R_SO_num"]      = (res_num["S"] * res_num["O"]) / 10.0
res_num["R_final_num"]   = (res_num["S"] * res_num["O"] * res_num["D"]) / 100.0
res_num["R_neutral_num"] = (res_num["S"] * res_num["O"] * D_REF) / 100.0
res_num["Δ_num"]         = res_num["R_final_num"] - res_num["R_neutral_num"]
res_num["U_num"]         = res_num["R_final_num"] / res_num["R_SO_num"]
res_num["U_adj"]         = res_num["U_num"] - (D_REF/10)
res_num["NPR_norm"]      = res_num["R_final_num"]
res_num["NPR_band"]      = res_num["NPR_norm"].apply(lambda v: to_band(v, NPR_BANDS_NORM))
res_num["NPR_rank"]      = res_num["R_final_num"].rank(method="min", ascending=False).astype(int)
res_num = res_num.sort_values("ID").reset_index(drop=True)

# -------------------- FUZZY (Stage 1 & 2) ---------------
Z = np.linspace(0, 10, 1001)
def _build_terms(spec):
    terms={}
    for name,tup in spec.items():
        kind=tup[0]
        if kind=="left":
            a,b=tup[1],tup[2]; terms[name]=lambda z,a=a,b=b:left_shoulder(z,a,b)
        elif kind=="right":
            c,d=tup[1],tup[2]; terms[name]=lambda z,c=c,d=d:right_shoulder(z,c,d)
        elif kind=="trap":
            a,b,c,d=tup[1],tup[2],tup[3],tup[4]
            terms[name]=lambda z,a=a,b=b,c=c,d=d:trapmf_safe(z,a,b,c,d)
    return terms

MF_S=_build_terms(CONFIG["MF_S_O"])
MF_O=_build_terms(CONFIG["MF_S_O"])
MF_D=_build_terms(CONFIG["MF_D"])
RISK=_build_terms(CONFIG["MF_RISK"])
risk_sets={k:(lambda f:(lambda z,f=f:f(z)))(fn) for k,fn in RISK.items()}

def get_rules(_):
    return {
      ("Baixa","Baixa"):"Baixo", ("Baixa","Média"):"Médio", ("Baixa","Alta"):"Alto", ("Baixa","Crítica"):"Crítico",
      ("Média","Baixa"):"Médio", ("Média","Média"):"Médio", ("Média","Alta"):"Alto", ("Média","Crítica"):"Crítico",
      ("Alta","Baixa"):"Alto", ("Alta","Média"):"Alto", ("Alta","Alta"):"Alto", ("Alta","Crítica"):"Crítico",
      ("Crítica","Baixa"):"Crítico", ("Crítica","Média"):"Crítico", ("Crítica","Alta"):"Crítico", ("Crítica","Crítica"):"Crítico",
    }

def get_rules_sd(_):
    return {
      ("Baixo","Excelente"):"Baixo", ("Baixo","Bom"):"Baixo", ("Baixo","Moderado"):"Médio", ("Baixo","Ruim"):"Médio",
      ("Médio","Excelente"):"Baixo", ("Médio","Bom"):"Médio", ("Médio","Moderado"):"Médio", ("Médio","Ruim"):"Alto",
      ("Alto","Excelente"):"Médio", ("Alto","Bom"):"Alto", ("Alto","Moderado"):"Alto", ("Alto","Ruim"):"Crítico",
      ("Crítico","Excelente"):"Crítico", ("Crítico","Bom"):"Crítico", ("Crítico","Moderado"):"Crítico", ("Crítico","Ruim"):"Crítico",
    }

out_map1=get_rules(CONFIG["RULE_SO"])
out_map2=get_rules_sd(CONFIG["RULE_SD"])

def defuzz_centroid(agg):
    num=np.trapz(agg*Z, Z); den=np.trapz(agg, Z)
    return float(num/den) if den>1e-9 else 0.0

def stage1_rso_fuzzy(s,o):
    mS={k:f(s) for k,f in MF_S.items()}
    mO={k:f(o) for k,f in MF_O.items()}
    firing={}
    for a,va in mS.items():
        for b,vb in mO.items():
            w=min(va,vb)
            if w>0:
                t=out_map1[(a,b)]
                firing[t]=max(firing.get(t,0),w)
    agg=np.zeros_like(Z)
    for term,w in firing.items():
        agg=np.maximum(agg,np.minimum(w,risk_sets[term](Z)))
    return defuzz_centroid(agg)

def stage2_final_fuzzy(rso,d):
    mu_r={k:risk_sets[k](rso) for k in risk_sets}
    mD={k:f(d) for k,f in MF_D.items()}
    firing={}
    for r_term,vr in mu_r.items():
        if vr<=0: continue
        for d_term,vd in mD.items():
            w=min(vr,vd)
            if w>0:
                t=out_map2[(r_term,d_term)]
                firing[t]=max(firing.get(t,0),w)
    agg=np.zeros_like(Z)
    for term,w in firing.items():
        agg=np.maximum(agg,np.minimum(w,risk_sets[term](Z)))
    return defuzz_centroid(agg)

rows=[]
for r in df[["ID","S","O","D"]].itertuples(index=False):
    ID,S,O,D = r
    rso  = stage1_rso_fuzzy(S,O)
    rfin = stage2_final_fuzzy(rso,D)
    rows.append((ID,S,O,D,rso,rfin))

res_fuzzy = pd.DataFrame(rows, columns=["ID","S","O","D","R_SO_fuzzy","R_final_fuzzy"])
res_fuzzy["Δ_fuzzy"]   = res_fuzzy["R_final_fuzzy"] - res_fuzzy["R_SO_fuzzy"]
res_fuzzy["Fuzzy_band"]= res_fuzzy["R_final_fuzzy"].apply(lambda v: to_band(v, FUZZY_BANDS))
res_fuzzy["Fuzzy_rank"]= res_fuzzy["R_final_fuzzy"].rank(method="min", ascending=False).astype(int)
res_fuzzy = res_fuzzy.sort_values("ID").reset_index(drop=True)

# -------------------- Consolidação -----------------------
res_cmp = res_num.merge(
    res_fuzzy[["ID","S","O","D","R_SO_fuzzy","R_final_fuzzy","Δ_fuzzy","Fuzzy_band","Fuzzy_rank"]],
    on=["ID","S","O","D"], how="left", validate="one_to_one"
).sort_values("ID").reset_index(drop=True)

res_cmp["band_change"] = np.where(
    res_cmp["NPR_band"].astype(str) == res_cmp["Fuzzy_band"].astype(str),
    "—",
    res_cmp["NPR_band"].astype(str) + " → " + res_cmp["Fuzzy_band"].astype(str)
)
res_cmp["Δ_flag"] = np.where(res_cmp["Δ_fuzzy"] > 1e-9, "↑D",
                      np.where(res_cmp["Δ_fuzzy"] < -1e-9, "↓D", "="))

# -------------------- UI: Abas e Gráficos ----------------
st.markdown("---")
st.subheader("Análises (abas abaixo)")

st.markdown("""
<style>
.stTabs [role="tab"] { font-size: 1.25rem !important; padding: .6rem 1rem !important; }
.stTabs [role="tab"][aria-selected="true"] { font-weight: 700 !important; }
.stTabs [role="tablist"] { gap: .25rem !important; }
</style>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Dados", "📊 NPR", "🔮 Fuzzy", "🧩 Comparação", "📈 Relatório"]
)

with tab0:
    st.subheader("📄 Dados padronizados")
    st.dataframe(df, use_container_width=True)

with tab1:
    st.subheader("📊 NPR Numérico")
    st.dataframe(res_num, use_container_width=True)

    counts = res_num["NPR_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(BAND_ORDER, counts.values, color=[BAND_COLORS[b] for b in BAND_ORDER], edgecolor="black")
    ax.set_title("Distribuição por faixas — NPR_band (NPR normalizado)")
    ax.set_xlabel("Faixas (NPR_norm)"); ax.set_ylabel("Quantidade")
    for i, v in enumerate(counts.values):
        ax.annotate(str(int(v)), (i, v), xytext=(0, 5), textcoords="offset points", ha="center")
    st.pyplot(fig)

    vmin = float(res_num["R_final_num"].min()); vmax = float(res_num["R_final_num"].max())
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.2))
    sc = ax2.scatter(res_num["R_SO_num"], res_num["D"], c=res_num["R_final_num"],
                     cmap="YlOrRd", vmin=vmin, vmax=vmax, s=60, edgecolor="k", linewidth=0.5, alpha=0.95)
    ax2.set_title("NPR (numérico): R_SO_num × D (cor = R_final_num)")
    ax2.set_xlabel("R_SO_num = (S × O) / 10  [0–10]")
    ax2.set_ylabel("D (Detecção) [1–10]")
    ax2.set_xlim(0,10); ax2.set_ylim(1,10); ax2.grid(True, alpha=0.25)
    cb = plt.colorbar(sc, ax=ax2); cb.set_label(f"R_final_num\n({vmin:.2f}–{vmax:.2f})", rotation=0, ha="left", labelpad=20)
    st.pyplot(fig2)

with tab2:
    st.subheader("🔮 Fuzzy-FMEA")
    st.dataframe(res_fuzzy, use_container_width=True)

    counts = res_fuzzy["Fuzzy_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(BAND_ORDER, counts.values, color=[BAND_COLORS[b] for b in BAND_ORDER], edgecolor="black")
    ax.set_title("Distribuição por faixas — Fuzzy_band (R_final)")
    ax.set_xlabel("Faixas Fuzzy"); ax.set_ylabel("Quantidade")
    for i, v in enumerate(counts.values):
        ax.annotate(str(int(v)), (i, v), xytext=(0, 5), textcoords="offset points", ha="center")
    st.pyplot(fig)

    vmin_fz = float(res_fuzzy["R_final_fuzzy"].min()); vmax_fz = float(res_fuzzy["R_final_fuzzy"].max())
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.2))
    sc = ax2.scatter(res_fuzzy["R_SO_fuzzy"], res_fuzzy["D"], c=res_fuzzy["R_final_fuzzy"],
                     cmap="YlOrRd", vmin=vmin_fz, vmax=vmax_fz, s=60, edgecolor="k", linewidth=0.5, alpha=0.95)
    ax2.set_title("Fuzzy: R_SO_fuzzy × D (cor = R_final_fuzzy)")
    ax2.set_xlabel("R_SO_fuzzy (Estágio 1) [0–10]")
    ax2.set_ylabel("D (Detecção) [1–10]")
    ax2.set_xlim(0,10); ax2.set_ylim(1,10); ax2.grid(True, alpha=0.25)
    cb = plt.colorbar(sc, ax=ax2); cb.set_label(f"R_final_fuzzy\n({vmin_fz:.2f}–{vmax_fz:.2f})", rotation=0, ha="left", labelpad=20)
    st.pyplot(fig2)

with tab3:
    st.subheader("🧩 Matriz NPR × Fuzzy (4×4) e Consolidação")
    st.dataframe(res_cmp, use_container_width=True)
    xtab = pd.crosstab(res_cmp["NPR_band"], res_cmp["Fuzzy_band"]).reindex(
        index=BAND_ORDER, columns=BAND_ORDER, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(xtab.values, cmap="YlOrRd")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(BAND_ORDER); ax.set_yticklabels(BAND_ORDER)
    ax.set_xlabel("Fuzzy_band"); ax.set_ylabel("NPR_band")
    ax.set_title("Matriz 4×4 — NPR × Fuzzy (contagem)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(xtab.iloc[i, j]), ha="center", va="center")
    cb = plt.colorbar(im, ax=ax); cb.set_label("Contagem")
    st.pyplot(fig)

with tab4:
    st.subheader("📈 Relatório Final (Spearman / Overlap / Export)")
    N = min(10, len(res_cmp))
    top_npr   = set(res_cmp.sort_values("R_final_num", ascending=False).head(N)["ID"])
    top_fuzzy = set(res_cmp.sort_values("R_final_fuzzy", ascending=False).head(N)["ID"])
    overlap   = len(top_npr & top_fuzzy)
    rho, pval = spearmanr(res_cmp["R_final_num"], res_cmp["R_final_fuzzy"])

    fuzzy_dist = res_fuzzy["Fuzzy_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    vmin_fz = float(res_fuzzy["R_final_fuzzy"].min()); vmax_fz = float(res_fuzzy["R_final_fuzzy"].max())

    st.write(f"- Spearman ρ: **{rho:.3f}** (p={pval:.3g})")
    st.write(f"- Top-{N} overlap: **{overlap}/{N} ({overlap/N:.0%})**")
    st.write(f"- R_final_fuzzy range: **{vmin_fz:.2f} – {vmax_fz:.2f}**")
    st.write(f"- 4 faixas populadas: **{'SIM ✅' if all(fuzzy_dist > 0) else 'NÃO ❌'}**")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ NPR (numérico)", res_num.to_csv(index=False).encode("utf-8"),
                           "FMEA_NPR_numerico.csv", "text/csv")
    with c2:
        st.download_button("⬇️ Fuzzy", res_fuzzy.to_csv(index=False).encode("utf-8"),
                           "FMEA_Fuzzy.csv", "text/csv")
    with c3:
        st.download_button("⬇️ Consolidação", res_cmp.to_csv(index=False).encode("utf-8"),
                           "FMEA_Consolidado.csv", "text/csv")
    
    
    
    # --- Botão para executar cálculos e gráficos ---
    st.sidebar.markdown("## Execução")
    go = st.sidebar.button("🚀 Gerar análise")

    if not go:
        st.info("Clique em **Gerar análise** na barra lateral para calcular e mostrar os gráficos.")
        # Mostra apenas pré-visualização e mapeamento
        st.write("### 📄 Pré-visualização do arquivo")
        st.dataframe(df_raw.head(30), use_container_width=True)
        st.stop()
