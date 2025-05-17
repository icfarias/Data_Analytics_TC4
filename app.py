import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    from prophet import Prophet
    from prophet.plot import plot_plotly
    prophet_ok = True
except:
    prophet_ok = False

st.set_page_config(layout="wide", page_title="FIAP PÓS TECH – DATA ANALYTICS", page_icon="📊")

# --------- LOGO DA FIAP CENTRALIZADO ----------
st.image("fiap_logo.png", width=240)  # Ajuste o tamanho se desejar
st.markdown("<h1 style='text-align: center; color: #0e4c92; font-size: 2.6rem; margin-bottom:0.2em;'>FIAP PÓS TECH – DATA ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #545454; margin-top:0px; font-size: 1.5rem;'>TECH CHALLENGE 4</h2>", unsafe_allow_html=True)
st.markdown("---")

# --------- MENU LATERAL ---------
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Navegação",
    ["🏠 Introdução", "📈 Exploração e Insights", "🔮 Forecasting"],
    index=0
)

# ---- INTEGRANTES NO MENU LATERAL ----
integrantes = [
    "Alexandre Barbosa",
    "Igor Calheiros de Farias",
    "João Paulo Machado",
    "Suellen dos Santos Rocha Godoi",
    "Thiago Moreira Dobbns"
]
integrantes = sorted(integrantes)
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Integrantes do Grupo")
integrantes_md = "".join([f"<div style='margin-bottom: -6px;'>👤 {nome}</div>" for nome in integrantes])
st.sidebar.markdown(integrantes_md, unsafe_allow_html=True)

# --------- INTRODUÇÃO ---------
if page == "🏠 Introdução":
    st.subheader("Introdução")
    st.markdown("""
O mercado global de petróleo caracteriza-se por sua alta volatilidade e importância estratégica, sendo diretamente influenciado por uma multiplicidade de fatores que impactam economias e cadeias produtivas ao redor do mundo. Entender as dinâmicas por trás da variação dos preços do petróleo é fundamental não apenas para empresas do setor, mas também para investidores, formuladores de políticas públicas e demais agentes econômicos que buscam antecipar tendências, minimizar riscos e identificar oportunidades em um ambiente de constantes mudanças.

Este estudo tem como objetivo analisar os principais fatores que determinam a variabilidade dos preços do petróleo, abordando aspectos como eventos geopolíticos, oscilações econômicas globais, padrões de demanda energética e o papel das inovações tecnológicas. Ao investigar como esses elementos interagem e influenciam o mercado, busca-se oferecer uma visão abrangente sobre a formação dos preços desse importante recurso energético.

A partir de uma análise detalhada dos processos que moldam o mercado petrolífero, este artigo visa contribuir para uma melhor compreensão das forças que governam as flutuações de preço e destacar a relevância desses fatores para a tomada de decisão estratégica. Assim, serão discutidos exemplos práticos e tendências recentes, enfatizando as implicações para o futuro do setor energético global.
    """)
    st.markdown("---")
    st.info("Use o menu lateral para avançar para a análise exploratória e previsão do mercado de petróleo Brent.")

# --------- EXPLORAÇÃO E INSIGHTS ---------
elif page == "📈 Exploração e Insights":
    st.title("📈 Previsão do Preço do Petróleo Brent com Eventos Relevantes")

    @st.cache_data
    def load_data():
        try:
            df = pd.read_excel("brent_com_eventos.xlsx")
            df["Data"] = pd.to_datetime(df["Data"])
            df = df.sort_values("Data")
            return df
        except Exception as e:
            st.error(f'Erro ao carregar o arquivo: {e}')
            return None

    df = load_data()
    if df is None:
        st.stop()

    eventos_unicos = df["Evento"].unique().tolist()
    evento_selecionado = st.sidebar.multiselect(
        "Filtrar por evento:", eventos_unicos, default=eventos_unicos
    )

    df_filtrado = df[df["Evento"].isin(evento_selecionado)]

    fig_hist = px.line(
        df_filtrado, x="Data", y="Preço", color="Evento",
        title="Histórico do Preço do Petróleo com Eventos"
    )
    fig_hist.update_layout(xaxis_title="Data", yaxis_title="Preço (US$)")
    st.plotly_chart(fig_hist, use_container_width=True)

    if not prophet_ok:
        st.error("É necessário instalar `prophet` e `plotly` para a previsão funcionar.")
        st.stop()

    df_dummies = pd.get_dummies(df["Evento"])
    df_prophet = pd.concat([df[["Data", "Preço"]], df_dummies], axis=1)
    df_prophet = df_prophet.rename(columns={"Data": "ds", "Preço": "y"})

    modelo = Prophet()
    for col in df_dummies.columns:
        modelo.add_regressor(col)

    modelo.fit(df_prophet)

    future = modelo.make_future_dataframe(periods=90)
    for col in df_dummies.columns:
        future[col] = 0

    forecast = modelo.predict(future)

    st.subheader("🕒 Previsão para os Próximos 90 Dias")
    fig_forecast = plot_plotly(modelo, forecast)
    fig_forecast.update_layout(xaxis_title="Data", yaxis_title="Preço Previsto (US$)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📊 Métricas da Previsão")
    ultimo_preco = df["Preço"].iloc[-1]
    preco_futuro = forecast["yhat"].iloc[-1]
    variacao = ((preco_futuro - ultimo_preco) / ultimo_preco) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Preço Atual", f"${ultimo_preco:.2f}")
    col2.metric("Previsão em 90 dias", f"${preco_futuro:.2f}")
    col3.metric("Variação Estimada", f"{variacao:.2f}%")

    st.subheader("🔍 Insights Relevantes")
    st.markdown("""
    **Anos 1980 e 1990**

    • 1987-1990: O preço do petróleo foi relativamente estável após a crise do petróleo dos anos 1970. No entanto, a Guerra do Golfo em 1990 causou um aumento significativo nos preços devido à incerteza no fornecimento.  
    • 1997-1998: A crise financeira asiática levou a uma queda na demanda por petróleo, resultando em uma queda acentuada nos preços.  

    **Anos 2000**

    • 2001: Os ataques de 11 de setembro nos EUA causaram uma breve queda nos preços devido à incerteza econômica global.  
    • 2003-2008: A invasão do Iraque em 2003 e o aumento da demanda da China e da Índia contribuíram para um aumento constante nos preços, culminando em um pico em 2008, quando os preços atingiram cerca de $147 por barril.  
    • 2008: A crise financeira global resultou em uma queda abrupta nos preços do petróleo.  

    **Anos 2010**

    • 2010-2014: A recuperação econômica global e a instabilidade no Oriente Médio, incluindo a Primavera Árabe, mantiveram os preços elevados.  
    • 2014-2016: Aumento da produção de petróleo de xisto nos EUA e a decisão da OPEP de não reduzir a produção levaram a uma queda significativa nos preços.  
    • 2019-2020: A pandemia de COVID-19 causou uma queda drástica na demanda por petróleo, resultando em preços negativos temporários em abril de 2020.  

    **Anos 2020**

    • 2021-2022: A recuperação econômica pós-pandemia e as tensões geopolíticas, como a invasão da Ucrânia pela Rússia, contribuíram para a volatilidade dos preços.  
    • 2023-2025: A transição para fontes de energia mais limpas e as políticas de descarbonização global começaram a impactar a demanda por petróleo, resultando em uma tendência de preços mais baixos e voláteis.
    """)

    st.markdown("---")
    st.markdown("Desenvolvido como MVP interativo com Machine Learning para previsão do petróleo Brent.")

# --------- FORECASTING ---------
elif page == "🔮 Forecasting":
    st.title("🔮 Previsão Customizada do Brent")

    st.info("""
Bem-vindo à área de **Previsão Customizada**!  
Aqui você pode escolher um período de datas específico para obter uma análise preditiva detalhada do preço do petróleo Brent.  
Selecione a data inicial e final do período desejado. O modelo irá recalibrar a previsão baseada no intervalo escolhido, permitindo analisar tendências de curto, médio ou longo prazo.

> **Dica:** Períodos longos tendem a apresentar maior incerteza. Compare diferentes janelas temporais para insights mais ricos!
""")

    @st.cache_data
    def load_data():
        try:
            df = pd.read_excel("brent_com_eventos.xlsx")
            df["Data"] = pd.to_datetime(df["Data"])
            df = df.sort_values("Data")
            return df
        except Exception as e:
            st.error(f'Erro ao carregar o arquivo: {e}')
            return None

    df = load_data()
    if df is None:
        st.stop()

    min_data, max_data = df["Data"].min(), df["Data"].max()
    st.write(f"Período disponível nos dados: `{min_data.date()} — {max_data.date()}`")

    # Filtros de data para o intervalo de análise
    col1, col2 = st.columns(2)
    with col1:
        data_ini = st.date_input("Data inicial", min_data, min_value=min_data, max_value=max_data)
    with col2:
        data_fim = st.date_input("Data final", max_data, min_value=data_ini, max_value=max_data)

    df_periodo = df[(df["Data"] >= pd.Timestamp(data_ini)) & (df["Data"] <= pd.Timestamp(data_fim))]

    st.write(f"Total de dados selecionados: **{len(df_periodo)} registros**")

    if not prophet_ok:
        st.error("É necessário instalar `prophet` e `plotly` para a previsão funcionar.")
        st.stop()

    # Modelagem com o período customizado
    df_dummies = pd.get_dummies(df_periodo["Evento"])
    df_prophet = pd.concat([df_periodo[["Data", "Preço"]], df_dummies], axis=1)
    df_prophet = df_prophet.rename(columns={"Data": "ds", "Preço": "y"})

    modelo = Prophet()
    for col in df_dummies.columns:
        modelo.add_regressor(col)
    modelo.fit(df_prophet)

    # Escolher horizonte de previsão
    periodo_forecast = st.slider("Quantos dias quer prever à frente?", 30, 365, 90, step=30)

    future = modelo.make_future_dataframe(periods=periodo_forecast)
    for col in df_dummies.columns:
        future[col] = 0

    forecast = modelo.predict(future)

    st.subheader(f"🕒 Previsão para os Próximos {periodo_forecast} Dias")
    fig_forecast = plot_plotly(modelo, forecast)
    fig_forecast.update_layout(xaxis_title="Data", yaxis_title="Preço Previsto (US$)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📊 Métricas do Cenário Customizado")
    if len(df_periodo) > 0:
        ultimo_preco = df_periodo["Preço"].iloc[-1]
        preco_futuro = forecast["yhat"].iloc[-1]
        variacao = ((preco_futuro - ultimo_preco) / ultimo_preco) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço Último Histórico", f"${ultimo_preco:.2f}")
        col2.metric(f"Previsão em {periodo_forecast} dias", f"${preco_futuro:.2f}")
        col3.metric("Variação Estimada", f"{variacao:.2f}%")
    else:
        st.warning("Selecione um intervalo de datas válido com pelo menos um dado para análise.")