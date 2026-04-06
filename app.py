import streamlit as st
import pandas as pd
import plotly.express as px
from analise import SentimentAnalyzer
import time

# Configuração da página para um visual premium
st.set_page_config(
    page_title="Sentify Analytics - Monitor de Marca IA",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.5em;
        background: linear-gradient(45deg, #00c6ff 0%, #0072ff 100%);
        color: white; font-weight: bold; border: none;
    }
    .sentiment-card {
        padding: 25px; border-radius: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    .metric-box {
        background: rgba(40, 167, 69, 0.1); padding: 15px;
        border-radius: 10px; border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

def update_usage_state(usage, model_type):
    if not usage: return
    st.session_state.total_tokens += usage.get('total_tokens', 0)
    
    # Preços aproximados (US$ -> BRL x5.0)
    # Gemini 1.5 Flash: $0.125/$0.375 per 1M
    # GPT-4o mini: $0.15/$0.60 per 1M
    usd_to_brl = 5.0
    if model_type == "Gemini":
        c_in, c_out = 0.125, 0.375
    else: # OpenAI (GPT-4o mini)
        c_in, c_out = 0.15, 0.60
        
    cost = (usage.get('prompt_tokens', 0) * (c_in/1e6) + usage.get('candidates_tokens', 0) * (c_out/1e6)) * usd_to_brl
    st.session_state.acc_cost += cost

def show_result(container, result):
    label = result.get('label', 'NEUTRO')
    score = result.get('score', 0.0)
    explanation = result.get('explanation', '')
    color = "#28a745" if label == "POSITIVO" else "#dc3545" if label == "NEGATIVO" else "#ffc107"
    container.markdown(f"""
        <div class="sentiment-card">
            <h2 style="color: {color}; text-align: center; font-size: 2.2em;">{label}</h2>
            <p style="text-align: center; opacity: 0.8;">Confiança: {score:.1%}</p>
        </div>
    """, unsafe_allow_html=True)
    if explanation: container.info(explanation)

def main():
    st.sidebar.title("💎 Sentify Dashboard")
    st.sidebar.markdown("---")
    
    local_ok = SentimentAnalyzer.is_local_available()
    
    st.sidebar.subheader("🛠️ Motor de IA")
    engine_choice = st.sidebar.selectbox(
        "Selecione o Provedor",
        ["Gemini (Google)", "OpenAI (GPT-4o)", "Local (Transformers)"],
        index=0
    )
    
    engine = "Gemini"
    if "OpenAI" in engine_choice: engine = "OpenAI"
    elif "Local" in engine_choice:
        if not local_ok:
            st.sidebar.error("❌ Motor local não encontrado. Mudando para Gemini.")
            engine = "Gemini"
        else: engine = "Local"
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Gestão de Custos")
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens, st.session_state.acc_cost = 0, 0.0

    st.sidebar.markdown(f"""
    <div class="metric-box">
        <p style="margin:0; font-size: 0.8em; opacity:0.7;">Tokens / Custo Acumulado</p>
        <h3 style="margin:0;">{st.session_state.total_tokens:,} | R$ {st.session_state.acc_cost:.5f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Resetar Metas"):
        st.session_state.total_tokens, st.session_state.acc_cost = 0, 0.0
        st.rerun()

    st.title("Sentify Intelligence")
    st.write(f"Motor Ativo: **{engine}**")

    t1, t2, t3 = st.tabs(["🎯 Express", "📂 Lote", "📡 Radar"])

    with t1:
        c1, c2 = st.columns([2, 1])
        with c1:
            txt = st.text_area("Texto para análise:", height=150)
            if st.button("Analisar") and txt:
                az = SentimentAnalyzer(engine=engine)
                res = az.analyze(txt)
                update_usage_state(res.get('usage'), engine)
                show_result(c2, res)

    with t2:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            if 'texto' in df.columns:
                if st.button("Processar Lote"):
                    pb = st.progress(0)
                    az = SentimentAnalyzer(engine=engine)
                    res_list = []
                    for i, r in df.iterrows():
                        res = az.analyze(str(r['texto']))
                        res_list.append(res.get('label', 'NEUTRO'))
                        update_usage_state(res.get('usage'), engine)
                        pb.progress((i+1)/len(df))
                    df['sentimento'] = res_list
                    st.plotly_chart(px.pie(df, names='sentimento', hole=0.4))
                    st.dataframe(df)

    with t3:
        cp_name = st.text_input("Nome da Empresa ou Marca", key="comp_radar")
        plt_choice = st.multiselect("Selecionar Redes Sociais", ["instagram", "facebook", "twitter", "tiktok"], default=["instagram", "twitter"])
        
        if st.button("📡 Lançar Radar e Ver Tópicos"):
            if cp_name and plt_choice:
                with st.spinner(f"Varrendo redes sociais para '{cp_name}'..."):
                    from analise import SocialMonitor
                    monitor = SocialMonitor()
                    lista_mentions = []
                    
                    for p in plt_choice:
                        m = monitor.search_mentions(cp_name, p)
                        if m:
                            for item in m: item['plataforma'] = p
                            lista_mentions.extend(m)
                    
                    if lista_mentions:
                        st.success(f"Radar concluído! {len(lista_mentions)} menções encontradas.")
                        df_m = pd.DataFrame(lista_mentions)
                        az = SentimentAnalyzer(engine=engine)
                        
                        sents = []
                        for t in df_m['texto']:
                            res = az.analyze(t)
                            sents.append(res.get('label', 'NEUTRO'))
                            update_usage_state(res.get('usage'), engine)
                        df_m['sentimento'] = sents
                        
                        c_a, c_b = st.columns([1, 1])
                        with c_a:
                            fig = px.pie(df_m, names='sentimento', title="Sentimento do Radar",
                                         color='sentimento', 
                                         color_discrete_map={"POSITIVO": "#28a745", "NEUTRO": "#ffc107", "NEGATIVO": "#dc3545"},
                                         hole=0.4, template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with c_b:
                            st.markdown("### 🤖 Inteligência de Tópicos")
                            if engine != "Local":
                                st.markdown("#### 🏆 Top 20 Assuntos Detalhados")
                                temas = monitor.analyze_social_trends(lista_mentions, az)
                                if temas:
                                    st.write(pd.DataFrame(temas))
                                else:
                                    st.info("A IA não conseguiu estruturar a tabela de tópicos desta vez.")
                            else:
                                st.warning("Ative Gemini ou OpenAI para ver os tópicos.")
                        
                        st.markdown("### 📑 Menções Brutas Encontradas")
                        st.dataframe(df_m[['plataforma', 'sentimento', 'texto', 'fonte']], use_container_width=True)
                    else:
                        st.warning("O Radar Social não encontrou menções públicas recentes para esta busca nas plataformas selecionadas.")
            else:
                st.error("Preencha o nome da empresa e escolha ao menos uma rede.")

if __name__ == "__main__":
    main()
