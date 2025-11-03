# -*- coding: utf-8 -*-
# =========================================================
# Fuzzy FMEA – Classificação de Riscos Geotécnicos
# PUC-Rio - Versão Otimizada
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import spearmanr

# -------------------- Config Streamlit (DEVE SER PRIMEIRO) --------------------
st.set_page_config(
    page_title="Fuzzy FMEA - Riscos Geotécnicos em Taludes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Configurações Matplotlib --------------------
plt.rcParams["axes.grid"] = True
plt.rcParams['figure.dpi'] = 100

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

# -------------------- Funções Auxiliares --------------------
def to_band(x,bands):
    try:
        if np.isnan(x): return None
    except TypeError:
        return None
    for thr,name in bands:
        if x < thr: return name
    return bands[-1][1]

def left_shoulder(x,a,b): 
    return np.clip(np.where(x<=a,1,np.where(x>=b,0,(b-x)/(b-a+1e-9))),0,1)

def right_shoulder(x,c,d): 
    return np.clip(np.where(x<=c,0,np.where(x>=d,1,(x-c)/(d-c+1e-9))),0,1)

def trapmf_safe(x,a,b,c,d):
    up=np.where(x<=a,0,np.where(x<b,(x-a)/(b-a+1e-9),1))
    down=np.where(x<=c,1,np.where(x<d,(d-x)/(d-c+1e-9),0))
    return np.clip(np.minimum(up,down),0,1)

def coerce_numeric(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def read_csv_robust(file):
    """Leitura robusta de CSV com detecção automática de separador e encoding"""
    last_err = None
    for enc in ["utf-8-sig", "latin-1", "iso-8859-1"]:
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

# -------------------- Funções Fuzzy --------------------
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

# ==================== INTERFACE STREAMLIT ====================

# Cabeçalho
st.title("🏔️ Classificação de Riscos Geotécnicos em Taludes Rodoviários")
st.markdown("### Metodologia FMEA com Lógica Fuzzy")
st.markdown("**PUC-Rio** | Antônio Augusto Zucchi Leite, Fernanda Defilippi Bittar, Antonio Krishnamurti Beleño de Oliveira, Márcio Leão")
st.markdown("---")

# -------------------- Upload de arquivo --------------------
st.sidebar.header("📁 Upload de Dados")
uploaded = st.sidebar.file_uploader(
    "Envie seu arquivo CSV", 
    type=["csv"],
    help="O arquivo deve conter colunas: ID, S, O, D, NPR (opcional)"
)

# Carregamento de dados
if uploaded:
    try:
        df_raw = read_csv_robust(uploaded)
        st.sidebar.success(f"✅ Arquivo carregado: {uploaded.name}")
    except Exception as e:
        st.error(f"❌ Erro ao ler arquivo: {e}")
        st.stop()
else:
    st.sidebar.info("📌 Usando dados de exemplo")
    df_raw = pd.DataFrame([
        [1,6,3,3,54],[2,9,3,6,162],[3,6,3,3,54],[4,8,3,7,168],[5,6,6,5,180],
        [6,10,3,8,240],[7,5,3,3,45],[8,7,5,4,140],[9,8,5,7,280],[10,6,4,6,144],
        [11,9,4,7,252],[12,5,7,4,140],[13,7,3,6,126],[14,8,3,5,120],[15,8,4,6,192],
        [16,7,5,5,175],[17,4,3,3,36],[18,5,5,6,150],[19,9,2,4,72],[20,8,2,8,128]
    ], columns=["ID","S","O","D","NPR"])

# -------------------- Mapeamento de colunas --------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Mapeamento de Colunas")

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

sel_id  = st.sidebar.selectbox("Coluna ID",   cols, index=cols.index(id_guess)  if id_guess  in cols else 0, key="id")
sel_s   = st.sidebar.selectbox("Coluna S (Severidade)",    cols, index=cols.index(s_guess)   if s_guess   in cols else 0, key="s")
sel_o   = st.sidebar.selectbox("Coluna O (Ocorrência)",    cols, index=cols.index(o_guess)   if o_guess   in cols else 0, key="o")
sel_d   = st.sidebar.selectbox("Coluna D (Detecção)",    cols, index=cols.index(d_guess)   if d_guess   in cols else 0, key="d")
sel_npr = st.sidebar.selectbox("Coluna NPR",  cols, index=cols.index(npr_guess) if npr_guess in cols else 0, key="npr")

# Renomear colunas
df = df_raw.rename(columns={
    sel_id:"ID", sel_s:"S", sel_o:"O", sel_d:"D", sel_npr:"NPR"
}).copy()

# Forçar numérico
for c in ["ID","S","O","D","NPR"]:
    if c in df.columns:
        df[c] = coerce_numeric(df[c])

# Preencher NPR ausente
if "NPR" in df.columns and df["NPR"].isna().any():
    df.loc[df["NPR"].isna(), "NPR"] = df.loc[df["NPR"].isna(), "S"] * df.loc[df["NPR"].isna(), "O"] * df.loc[df["NPR"].isna(), "D"]

# Validar colunas essenciais
missing = [c for c in ["ID","S","O","D"] if c not in df.columns]
if missing:
    st.error(f"❌ Colunas essenciais ausentes: {missing}")
    st.stop()

# -------------------- Pré-visualização --------------------
with st.expander("📄 Visualizar dados carregados", expanded=False):
    st.dataframe(df, use_container_width=True, height=300)
    st.caption(f"Total de registros: {len(df)}")

# -------------------- Botão para processar --------------------
st.sidebar.markdown("---")
process_button = st.sidebar.button("🚀 **PROCESSAR ANÁLISE**", type="primary", use_container_width=True)

if not process_button:
    st.info("👈 Configure as colunas na barra lateral e clique em **PROCESSAR ANÁLISE** para iniciar os cálculos.")
    st.stop()

# ==================== PROCESSAMENTO ====================
with st.spinner("⚙️ Processando análise..."):
    
    # -------------------- NPR Numérico --------------------
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

    # -------------------- FUZZY (Stage 1 & 2) --------------------
    rows=[]
    progress_bar = st.progress(0)
    for idx, r in enumerate(df[["ID","S","O","D"]].itertuples(index=False)):
        ID,S,O,D = r
        rso  = stage1_rso_fuzzy(S,O)
        rfin = stage2_final_fuzzy(rso,D)
        rows.append((ID,S,O,D,rso,rfin))
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()

    res_fuzzy = pd.DataFrame(rows, columns=["ID","S","O","D","R_SO_fuzzy","R_final_fuzzy"])
    res_fuzzy["Δ_fuzzy"]   = res_fuzzy["R_final_fuzzy"] - res_fuzzy["R_SO_fuzzy"]
    res_fuzzy["Fuzzy_band"]= res_fuzzy["R_final_fuzzy"].apply(lambda v: to_band(v, FUZZY_BANDS))
    res_fuzzy["Fuzzy_rank"]= res_fuzzy["R_final_fuzzy"].rank(method="min", ascending=False).astype(int)
    res_fuzzy = res_fuzzy.sort_values("ID").reset_index(drop=True)

    # -------------------- Consolidação --------------------
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

st.success("✅ Análise concluída!")

# ==================== VISUALIZAÇÃO DE RESULTADOS ====================
st.markdown("---")
st.header("📊 Resultados da Análise")

# CSS para estilizar abas
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 8px 24px;
    background-color: #f0f2f6;
    border-radius: 5px 5px 0 0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 NPR Tradicional", "🔮 Análise Fuzzy", "🧩 Comparação", "📈 Relatório Final"]
)

# -------------------- ABA 1: NPR --------------------
with tab1:
    st.subheader("📊 Análise NPR Tradicional")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            res_num.style.background_gradient(subset=["R_final_num"], cmap="YlOrRd"),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.metric("Risco Mínimo", f"{res_num['R_final_num'].min():.2f}")
        st.metric("Risco Máximo", f"{res_num['R_final_num'].max():.2f}")
        st.metric("Risco Médio", f"{res_num['R_final_num'].mean():.2f}")
        
        # Distribuição por faixa
        counts = res_num["NPR_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
        for band in BAND_ORDER:
            st.markdown(f"**{band}**: {counts[band]} itens")
    
    # Gráficos
    st.markdown("#### Visualizações")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Distribuição por faixas
        counts = res_num["NPR_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(BAND_ORDER, counts.values, color=[BAND_COLORS[b] for b in BAND_ORDER], 
                      edgecolor="black", linewidth=1.5)
        ax.set_title("Distribuição por Faixas de Risco (NPR)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Faixas de Risco"); ax.set_ylabel("Quantidade")
        for i, v in enumerate(counts.values):
            ax.annotate(str(int(v)), (i, v), xytext=(0, 5), textcoords="offset points", 
                       ha="center", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col_b:
        # Scatter plot
        vmin = float(res_num["R_final_num"].min()); vmax = float(res_num["R_final_num"].max())
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(res_num["R_SO_num"], res_num["D"], c=res_num["R_final_num"],
                       cmap="YlOrRd", vmin=vmin, vmax=vmax, s=80, edgecolor="k", 
                       linewidth=0.8, alpha=0.9)
        ax.set_title("NPR: R_SO × Detecção", fontsize=12, fontweight='bold')
        ax.set_xlabel("R_SO_num (S × O / 10)")
        ax.set_ylabel("D (Detecção)")
        ax.set_xlim(0,10); ax.set_ylim(0,11); ax.grid(True, alpha=0.3)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("R_final", rotation=270, labelpad=20)
        st.pyplot(fig)
        plt.close()

# -------------------- ABA 2: FUZZY --------------------
with tab2:
    st.subheader("🔮 Análise com Lógica Fuzzy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            res_fuzzy.style.background_gradient(subset=["R_final_fuzzy"], cmap="YlOrRd"),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.metric("Risco Mínimo", f"{res_fuzzy['R_final_fuzzy'].min():.2f}")
        st.metric("Risco Máximo", f"{res_fuzzy['R_final_fuzzy'].max():.2f}")
        st.metric("Risco Médio", f"{res_fuzzy['R_final_fuzzy'].mean():.2f}")
        
        # Distribuição por faixa
        counts = res_fuzzy["Fuzzy_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
        for band in BAND_ORDER:
            st.markdown(f"**{band}**: {counts[band]} itens")
    
    # Gráficos
    st.markdown("#### Visualizações")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Distribuição por faixas
        counts = res_fuzzy["Fuzzy_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(BAND_ORDER, counts.values, color=[BAND_COLORS[b] for b in BAND_ORDER], 
                      edgecolor="black", linewidth=1.5)
        ax.set_title("Distribuição por Faixas de Risco (Fuzzy)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Faixas de Risco"); ax.set_ylabel("Quantidade")
        for i, v in enumerate(counts.values):
            ax.annotate(str(int(v)), (i, v), xytext=(0, 5), textcoords="offset points", 
                       ha="center", fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col_b:
        # Scatter plot
        vmin_fz = float(res_fuzzy["R_final_fuzzy"].min())
        vmax_fz = float(res_fuzzy["R_final_fuzzy"].max())
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(res_fuzzy["R_SO_fuzzy"], res_fuzzy["D"], c=res_fuzzy["R_final_fuzzy"],
                       cmap="YlOrRd", vmin=vmin_fz, vmax=vmax_fz, s=80, edgecolor="k", 
                       linewidth=0.8, alpha=0.9)
        ax.set_title("Fuzzy: R_SO × Detecção", fontsize=12, fontweight='bold')
        ax.set_xlabel("R_SO_fuzzy (Estágio 1)")
        ax.set_ylabel("D (Detecção)")
        ax.set_xlim(0,10); ax.set_ylim(0,11); ax.grid(True, alpha=0.3)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("R_final_fuzzy", rotation=270, labelpad=20)
        st.pyplot(fig)
        plt.close()

# -------------------- ABA 3: COMPARAÇÃO --------------------
with tab3:
    st.subheader("🧩 Comparação: NPR vs Fuzzy")
    
    # Tabela consolidada
    st.dataframe(
        res_cmp.style.background_gradient(subset=["R_final_num", "R_final_fuzzy"], cmap="YlOrRd"),
        use_container_width=True,
        height=400
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Matriz de confusão
        st.markdown("#### Matriz de Concordância 4×4")
        xtab = pd.crosstab(res_cmp["NPR_band"], res_cmp["Fuzzy_band"]).reindex(
            index=BAND_ORDER, columns=BAND_ORDER, fill_value=0
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(xtab.values, cmap="YlOrRd", aspect='auto')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(BAND_ORDER); ax.set_yticklabels(BAND_ORDER)
        ax.set_xlabel("Fuzzy (Colunas)", fontweight='bold')
        ax.set_ylabel("NPR (Linhas)", fontweight='bold')
        ax.set_title("Matriz NPR × Fuzzy", fontweight='bold')
        
        # Adicionar valores nas células
        for i in range(4):
            for j in range(4):
                text_color = "white" if xtab.iloc[i, j] > xtab.values.max()/2 else "black"
                ax.text(j, i, int(xtab.iloc[i, j]), ha="center", va="center",
                       color=text_color, fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label="Contagem")
        st.pyplot(fig)
        plt.close()
    
    with col_b:
        # Scatter comparativo
        st.markdown("#### Correlação NPR vs Fuzzy")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(res_cmp["R_final_num"], res_cmp["R_final_fuzzy"], 
                  alpha=0.6, s=80, edgecolor='k', linewidth=0.5)
        
        # Linha de identidade
        max_val = max(res_cmp["R_final_num"].max(), res_cmp["R_final_fuzzy"].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Identidade')
        
        ax.set_xlabel("R_final_num (NPR)", fontweight='bold')
        ax.set_ylabel("R_final_fuzzy", fontweight='bold')
        ax.set_title("Correlação entre Métodos", fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()

# -------------------- ABA 4: RELATÓRIO --------------------
with tab4:
    st.subheader("📈 Relatório Final e Métricas")
    
    # Calcular métricas
    N = min(10, len(res_cmp))
    top_npr   = set(res_cmp.sort_values("R_final_num", ascending=False).head(N)["ID"])
    top_fuzzy = set(res_cmp.sort_values("R_final_fuzzy", ascending=False).head(N)["ID"])
    overlap   = len(top_npr & top_fuzzy)
    rho, pval = spearmanr(res_cmp["R_final_num"], res_cmp["R_final_fuzzy"])
    
    fuzzy_dist = res_fuzzy["Fuzzy_band"].value_counts().reindex(BAND_ORDER, fill_value=0)
    vmin_fz = float(res_fuzzy["R_final_fuzzy"].min())
    vmax_fz = float(res_fuzzy["R_final_fuzzy"].max())
    all_bands_populated = all(fuzzy_dist > 0)
    
    # Métricas em cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Correlação Spearman",
            f"{rho:.3f}",
            help="Correlação de ranking entre NPR e Fuzzy"
        )
    
    with col2:
        st.metric(
            f"Overlap Top-{N}",
            f"{overlap}/{N}",
            f"{overlap/N:.0%}",
            help="Concordância nos itens mais críticos"
        )
    
    with col3:
        st.metric(
            "Range Fuzzy",
            f"{vmin_fz:.2f} - {vmax_fz:.2f}",
            help="Amplitude dos valores fuzzy"
        )
    
    with col4:
        if all_bands_populated:
            st.metric("Cobertura de Faixas", "✅ Completa", help="Todas as 4 faixas possuem itens")
        else:
            st.metric("Cobertura de Faixas", "⚠️ Incompleta", help="Algumas faixas estão vazias")
    
    # Resumo textual
    st.markdown("#### 📝 Resumo Executivo")
    st.write(f"""
    - **Total de itens analisados**: {len(res_cmp)}
    - **Correlação entre métodos**: ρ = {rho:.3f} (p-value = {pval:.4g})
    - **Concordância Top-{N}**: {overlap} de {N} itens ({overlap/N:.0%})
    - **Distribuição Fuzzy**: Green={fuzzy_dist['Green']}, Yellow={fuzzy_dist['Yellow']}, 
      Orange={fuzzy_dist['Orange']}, Red={fuzzy_dist['Red']}
    - **Interpretação**: {'Alta concordância entre métodos' if rho > 0.8 else 'Diferenças significativas detectadas'}
    """)
    
    # Downloads
    st.markdown("#### 💾 Exportar Resultados")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ NPR Numérico",
            res_num.to_csv(index=False).encode("utf-8"),
            "FMEA_NPR_numerico.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            "⬇️ Análise Fuzzy",
            res_fuzzy.to_csv(index=False).encode("utf-8"),
            "FMEA_Fuzzy.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        st.download_button(
            "⬇️ Consolidação Completa",
            res_cmp.to_csv(index=False).encode("utf-8"),
            "FMEA_Consolidado.csv",
            "text/csv",
            use_container_width=True
        )

# -------------------- Rodapé --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Fuzzy FMEA - Análise de Riscos Geotécnicos</strong></p>
    <p>Desenvolvido na PUC-Rio | © 2024</p>
</div>
""", unsafe_allow_html=True)
