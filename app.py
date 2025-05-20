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
A história do preço do petróleo tem sido marcada por períodos de estabilidade e turbulência ao longo dos anos. No século 20, os preços globais do petróleo geralmente variavam entre \$1,00 e \$2,00 por barril até 1970, o que equivale a aproximadamente \$20 a \$40 por barril ajustados pela inflação. Naquela época, os Estados Unidos eram o maior produtor mundial de petróleo e mantinham os preços regulados, aproveitando a abundância de petróleo doméstico.

Contudo, a situação começou a mudar com o advento de eventos geopolíticos e crises econômicas. Por exemplo, os atentados de 11 de setembro de 2001, a segunda Guerra do Golfo em 2003 e a crise no Oriente Médio elevaram os preços do petróleo de forma lenta, mas constante. Além disso, em 2020, o mercado de petróleo e o preço do petróleo brent sofreu mudanças importantes devido à guerra de preços entre a Rússia e a Arábia Saudita, bem como a desaceleração econômica global provocada pela pandemia de COVID-19.

Ao longo das décadas, diversos fatores geopolíticos e econômicos moldaram a evolução dos preços do petróleo, evidenciando a natureza complexa e imprevisível desse mercado.

Ao longo da história, eventos políticos e acordos internacionais têm desempenhado um papel fundamental na evolução dos preços do petróleo. A criação da OPEP (Organização dos Países Exportadores de Petróleo) em 1960, por exemplo, permitiu maior cooperação e controle sobre a oferta de petróleo pelos principais produtores, como a Arábia Saudita e a Rússia. Eventos políticos, como conflitos no Oriente Médio envolvendo o Irã, Israel e outros países também afetaram os preços ao longo do tempo.

Os preços do petróleo são influenciados por diversos fatores econômicos e de mercado. A inflação, por exemplo, pode levar a flutuações nos preços. Durante a década de 1970, os preços globais do petróleo variaram entre \$1,00 e \$2,00 por barril, ajustados pela inflação, chegando a aproximadamente \$20/\bbl e \$40/\bbl. A volatilidade nos mercados e as mudanças na oferta e demanda também têm impacto nos preços do petróleo.
""")

    st.image("img_dash_1.png", use_container_width=True, caption="")

    st.markdown("""
Em 2 de agosto de 1990, o Exército iraquiano de Saddam Hussein invadiu o emirado do Kuwait e anexou este pequeno território rico em petróleo, mas sete meses depois foi expulso dali por uma coalizão internacional liderada pelos Estados Unidos.
""")

    st.image("img_dash_2.png", use_container_width=True, caption="")

    st.markdown("""
Os ataques promovidos pela Al-Qaeda e seu líder Osama Bin Laden ocorreram em decorrência de um longo conflito histórico entre os EUA e os países árabes do Oriente Médio.
O preço do petróleo disparou diante da possibilidade de uma guerra entre os Estados Unidos e o Oriente Médio.
Em 7 de outubro de 2001, quase um mês após os atentados ao World Trade Center, os Estados Unidos, com o apoio da Inglaterra, iniciaram uma ofensiva militar contra a milícia Talibã, que controlava o Afeganistão, e a rede Al Qaeda, comandada por Osama Bin Laden, acusadas de serem as responsáveis pelos ataques terroristas.
""")

    st.image("img_dash_3.png", use_container_width=True, caption="")

    st.markdown("""
A invasão do Iraque em 2003 foi o primeiro estágio da Guerra do Iraque. A fase de invasão começou em 19 de março de 2003 (aéreo) e 20 de março de 2003 (terrestre) e durou pouco mais de um mês, incluindo 26 dias de grandes operações de combate, nas quais uma força combinada de tropas dos Estados Unidos, Reino Unido, Austrália e a Polônia invadiu o Iraque.
""")

    st.image("img_dash_4.png", use_container_width=True, caption="")

    st.markdown("""
A crise das hipotecas nos Estados Unidos e suas consequências causaram dias de pânico nos mercados financeiros. No dia da quebra do Lehman Brothers, as bolsas em Wall Street tiveram as piores perdas desde os ataques de 11 de Setembro, em 2001.
""")

    st.image("img_dash_5.png", use_container_width=True, caption="")

    st.markdown("""
O desenvolvimento de novas tecnologias de extração, como o óleo de xisto, permitiu uma maior produção de petróleo. Os Estados Unidos, por exemplo, se tornaram o maior produtor mundial de petróleo, graças às suas empresas americanas e suas inovações. Esses avanços contribuíram para um aumento na oferta global e consequentes variações nos preços do petróleo.
""")

    st.image("img_dash_6.png", use_container_width=True, caption="")

    st.markdown("""
A Primavera Árabe foi um conjunto de manifestações populares que aconteceram em países do Norte da África e em parte da região do Oriente Médio entre o final de 2010 e o ano de 2013, com consequências que reverberam até o presente em muitos dos países envolvidos.
A onda de protestos se espalhou rapidamente pelo chamado mundo árabe, que compreende as nações falantes do idioma, com o auxílio das redes sociais, e tinha como principal objetivo a retomada democrática e a melhoria da qualidade de vida em seus respectivos países, muitos dos quais estavam sendo governado por lideranças autoritárias e/ou corruptas, além de uma grave crise econômica.
""")

    st.image("img_dash_7.png", use_container_width=True, caption="")

    st.markdown("""
O mais recente tratado internacional é o Acordo de Paris, adotado em 2015, durante a 21ª Conferência das Partes ocorreu, em Paris.
O acordo de Paris tem como objetivo fortalecer a resposta global à ameaça das mudanças climáticas. Ele foi aprovado pelos 195 países participantes que se comprometeram em reduzir emissões de gases de efeito estufa.
""")

    st.image("img_dash_8.png", use_container_width=True, caption="")

    st.markdown("""
Uma série de sanções econômicas impostas pelos Estados Unidos ao Irã. Para Donald Trump, o presidente norte-americano, o regime de Teerã não cumpre os termos do acordo nuclear assinado em 2015.
""")

    st.image("img_dash_9.png", use_container_width=True, caption="")

    st.markdown("""
Crises globais e pandemias, como a COVID-19, impactam a demanda mundial de petróleo e, consequentemente, os preços. Em 2020, os preços do marcador Brent registraram o menor valor desde 2004 devido à redução da atividade econômica e restrições impostas pelos países para controlar a propagação do vírus.
""")

    st.image("img_dash_10.png", use_container_width=True, caption="")

    st.markdown("""
Os integrantes da Organização dos Países Exportadores de Petróleo e aliados (Opep+) confirmaram o corte de produção de petróleo em 2 milhões de barris por dia (bpd) a partir de novembro 2022. Esse é o maior corte desde abril de 2020, quando a pandemia começou. O grupo ainda informa que o acordo de cooperação atual foi estendido até 31 de dezembro de 2023.
""")

    st.image("img_dash_11.png", use_container_width=True, caption="")

    st.markdown("""
Dois navios comerciais foram atacados no Mar Vermelho. A ação terrorista aconteceu nesta segunda-feira (18) e teve a autoria reclamada pelo movimento Houthi, do Iêmen. Drones foram usados para o ataque que gerou reação de empresas petroleiras, que começam a desviar embarcações da região.
A notícia gerou apreensão nos investidores, e houve impacto no mercado de petróleo. Nesta segunda-feira, o barril do tipo brent — principal referência para o mercado brasileiro — subiu 1,80%, e fechou o dia a US$ 77,95. Já o contrato do tipo WTI teve alta de 1,50%, para US$ 72,47.
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