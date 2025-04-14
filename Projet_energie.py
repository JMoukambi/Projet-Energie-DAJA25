import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datetime
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os

st.set_page_config(page_title="Projet Energie", layout="centered")


@st.cache_data
def load_data(chemin,separateur):
  return pd.read_csv(chemin,sep = separateur)
#df = load_data(r"C:\Users\maxdo\csv_nettoye.csv", ';')
df = load_data("csv_nettoye.csv", ';')


@st.cache_data
def load_data_2():
    df = pd.read_csv("csv_nettoye.csv", sep=";")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Jour"] = df["Date"].dt.weekday
    df["Mois"] = df["Date"].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return "Hiver"
        elif month in [3, 4, 5]:
            return "Printemps"
        elif month in [6, 7, 8]:
            return "Été"
        else:
            return "Automne"

    df["Saison"] = df["Mois"].apply(get_season)
    df_daily = df.groupby("Date").agg({
        "Consommation (MW)": "sum",
        "Hydraulique (MW)": "sum",
        "Eolien (MW)": "sum",
        "Solaire (MW)": "sum",
        "Jour": "first",
        "Saison": "first"
    }).reset_index()
    df_daily = pd.get_dummies(df_daily, columns=["Saison"], drop_first=True)
    return df_daily

df_daily = load_data_2()

X = df_daily[["Jour", "Hydraulique (MW)", "Eolien (MW)", "Solaire (MW)"] + 
             [col for col in df_daily.columns if "Saison_" in col]]
y = df_daily["Consommation (MW)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Régression Linéaire": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}




#st.title("Projet Energie")
st.sidebar.title("Sommaire")
pages=["Page de couverture",
    "Équilibre Énergétique en France : Données, Visualisation et Prédictions", 
    "Nettoyage des Données", 
    "Visualisation des Données",
  "Visualisation interactive", "Modélisation au mois", "Modélisation au jour/à la demi-heure",
  "Introduction Machine Learning",
    "Régression Linéaire",
    "Random Forest",
    "Gradient Boosting",
    "Conclusion Machine Learning",
    "Conclusion et ouverture",
    "Annexe"]
page=st.sidebar.radio("Aller vers", pages)

# st.sidebar.title("Participants")
# st.sidebar.write("Jeanne MOUKAMBI KOYO\n\nLeïla DAOUD\n\nGaston LACHAIZE\n\nMaximilien DOURSTER")




if page == "Page de couverture":
    st.markdown("""

        ## Cohorte Data Analyst - Bootcamp - Janvier 2025
        ### Groupe Énergie
    """, unsafe_allow_html=True)
    
    logo_path = "logo-2021 DS.png"
    if os.path.exists(logo_path):
        st.image(logo_path, caption="Logo DataScientest", 
                 #use_container_width=True
                 )
    else:
        st.error(f"L'image du logo n'a pas été trouvée à cet emplacement : {logo_path}")
    
    st.markdown(""" 
        ### **Jeanne Moukambi Koyo**  
        ### **Leila Daoud**  
        ### **Gaston Lachaize**  
        ### **Maximilien Dourster** 
    """, unsafe_allow_html=True)

elif page == "Équilibre Énergétique en France : Données, Visualisation et Prédictions":
    st.title("⚡ Équilibre Énergétique en France : Données, Visualisation et Prédictions ⚡")
    st.write("🔸 Pourquoi ? Assurer l’équilibre du réseau face aux fluctuations saisonnières et aux énergies renouvelables")
    st.write("🔸 Exemple récent : En 2022, la France a dû importer de l’électricité (arrêt des centrales nucléaires)")
    
    st.subheader("🎯 Objectifs du projet :")
    st.write("✅ Analyser la relation entre production & consommation")
    st.write("✅ Visualiser les tendances avec des graphiques interactifs")
    st.write("✅ Prédire la consommation future avec du Machine Learning")
    
    st.subheader("📂 Sources des données :")
    st.write("✅ ODRE (Open Data Réseaux Énergies)")
    st.write("✅ Période : 2013-2023 | Granularité : 30 minutes")
    st.write("✅ Données clés : Consommation, Production par filière, Échanges inter-régions")

elif page == "Nettoyage des Données":
    st.title("🔹 Nettoyage des Données (Problèmes & Solutions)")
    st.write("⚠️ Problèmes rencontrés :")
    st.write("✅ Données incomplètes (NaN)")
    st.write("✅ Formats hétérogènes")
    st.write("✅ Colonnes inutiles")
    
    st.subheader("🛠 Solutions appliquées :")
    st.write("✅ Suppression/Remplacement des valeurs manquantes")
    st.write("✅ Conversion des formats pour homogénéiser les données")
    st.write("✅ Suppression des colonnes redondantes (Code INSEE, stockage batterie)")

elif page == "Visualisation des Données":
    st.title("🔹 Visualisation des Données")
    
    if df is not None:
        st.markdown("""### Consommation selon l'heure de la journée (médiane sur 2013-2022)""")
        # fig, ax = plt.subplots(figsize=(10, 5))
        # for jour in df['Jour'].unique():
        #     subset = df[df['Jour'] == jour].groupby('Heure')['Consommation (MW)'].median()
        #     ax.plot(subset, label=jour)
        # ax.set_ylabel("MW")
        # ax.set_xlabel("Heure")
        # ax.set_xticks(range(0, 24))
        # ax.set_xticklabels([f"{h}:00" for h in range(24)], rotation=45, ha="right")
        # ax.legend()
        # ax.grid(True)
        # st.pyplot(fig)

        conso_pays = df[df['Année']<2023].groupby('DateHeure').agg({'Consommation (MW)' : 'sum' , 'Heure' : 'first' , 'Jour' : 'first'})
        conso_journee = conso_pays.groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_lundi = conso_pays[conso_pays['Jour'] == 'lundi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_mardi = conso_pays[conso_pays['Jour'] == 'mardi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_mercredi = conso_pays[conso_pays['Jour'] == 'mercredi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_jeudi = conso_pays[conso_pays['Jour'] == 'jeudi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_vendredi = conso_pays[conso_pays['Jour'] == 'vendredi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_samedi = conso_pays[conso_pays['Jour'] == 'samedi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
        conso_dimanche = conso_pays[conso_pays['Jour'] == 'dimanche'].groupby('Heure').agg({'Consommation (MW)' : 'median'})

        fig = plt.figure(figsize = (9,7))

        plt.plot(conso_journee.index , conso_journee['Consommation (MW)'], label = 'jour moyen' , linewidth = 4)
        plt.plot(conso_journee.index , conso_lundi['Consommation (MW)'], label = 'lundi')
        plt.plot(conso_journee.index , conso_mardi['Consommation (MW)'], label = 'mardi')
        plt.plot(conso_journee.index , conso_mercredi['Consommation (MW)'], label = 'mercredi')
        plt.plot(conso_journee.index , conso_jeudi['Consommation (MW)'], label = 'jeudi')
        plt.plot(conso_journee.index , conso_vendredi['Consommation (MW)'], label = 'vendredi')
        plt.plot(conso_journee.index , conso_samedi['Consommation (MW)'], label = 'samedi')
        plt.plot(conso_journee.index , conso_dimanche['Consommation (MW)'], label = 'dimanche')


        plt.title('''Consommation du pays selon l'heure de la journée (médiane sur 2013-2022)''')
        plt.xticks(conso_journee.index[::2],rotation = 90)
        plt.xlabel('Heure')
        plt.ylabel('MW')
        plt.grid()
        plt.legend(loc = 'lower right')

        st.pyplot(fig)

        st.markdown("""### Répartition de la production d'électricité par filière (entre 2013 et 2023)""")
        production_cols = ["Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)", "Hydraulique (MW)", "Bioénergies (MW)"]
        production_sorted = df[production_cols].sum().sort_values()
        fig, ax = plt.subplots()
        production_sorted.plot(kind="pie", autopct='%1.1f%%', ax=ax, colors=["#FFDDC1", "#FFABAB", "#FFC3A0", "#D5AAFF", "#85E3FF", "#B9FBC0"])
        ax.set_ylabel("")
        st.pyplot(fig)

        st.markdown("""### Evolution de la production de renouvelables (entre 2013 et 2023)""")
        st.image('renouvelables.png')    

        st.markdown("""### Évolution de la consommation et de la production énergétique (entre 2013 et 2023)""")
        image_path = "evolution_energie_france_10_ans_TWh.png"
        if os.path.exists(image_path):
            st.image(image_path, caption="Évolution de la consommation et de la production énergétique en France (2013 - 2023)")
        else:
            st.error(f"L'image n'a pas été trouvée à cet emplacement : {image_path}")







if page == pages[4] :
  
  st.header("DataViz dynamique")

#   conso_pays = df[df['Année']<2023].groupby('DateHeure').agg({'Consommation (MW)' : 'sum' , 'Heure' : 'first' , 'Jour' : 'first'})
#   conso_journee = conso_pays.groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_lundi = conso_pays[conso_pays['Jour'] == 'lundi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_mardi = conso_pays[conso_pays['Jour'] == 'mardi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_mercredi = conso_pays[conso_pays['Jour'] == 'mercredi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_jeudi = conso_pays[conso_pays['Jour'] == 'jeudi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_vendredi = conso_pays[conso_pays['Jour'] == 'vendredi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_samedi = conso_pays[conso_pays['Jour'] == 'samedi'].groupby('Heure').agg({'Consommation (MW)' : 'median'})
#   conso_dimanche = conso_pays[conso_pays['Jour'] == 'dimanche'].groupby('Heure').agg({'Consommation (MW)' : 'median'})

#   fig = plt.figure(figsize = (9,7))

#   plt.plot(conso_journee.index , conso_journee['Consommation (MW)'], label = 'jour moyen' , linewidth = 4)
#   plt.plot(conso_journee.index , conso_lundi['Consommation (MW)'], label = 'lundi')
#   plt.plot(conso_journee.index , conso_mardi['Consommation (MW)'], label = 'mardi')
#   plt.plot(conso_journee.index , conso_mercredi['Consommation (MW)'], label = 'mercredi')
#   plt.plot(conso_journee.index , conso_jeudi['Consommation (MW)'], label = 'jeudi')
#   plt.plot(conso_journee.index , conso_vendredi['Consommation (MW)'], label = 'vendredi')
#   plt.plot(conso_journee.index , conso_samedi['Consommation (MW)'], label = 'samedi')
#   plt.plot(conso_journee.index , conso_dimanche['Consommation (MW)'], label = 'dimanche')


#   plt.title('''Consommation du pays selon l'heure de la journée (médiane sur 2013-2022)''')
#   plt.xticks(conso_journee.index[::2],rotation = 90)
#   plt.xlabel('Heure')
#   plt.ylabel('MW')
#   plt.grid()
#   plt.legend(loc = 'lower right')

#   st.pyplot(fig)

  col1, col2 = st.columns([1, 5])

  df['Prod tot (MW)'] = df['Thermique (MW)'] + df['Nucléaire (MW)'] + df['Eolien (MW)'] + df['Solaire (MW)'] + df['Hydraulique (MW)'] + df['Bioénergies (MW)']
  df_temp = df.groupby('DateHeure').agg({
                                                      'Thermique (MW)' : 'sum' , 'Nucléaire (MW)' : 'sum' ,
                                                      'Eolien (MW)' : 'sum' , 'Solaire (MW)' : 'sum' ,
                                                      'Hydraulique (MW)' : 'sum' , 'Bioénergies (MW)' : 'sum',
                                                        'Consommation (MW)' : 'sum' , 'Pompage (MW)': 'sum',
                                                        'Ech. physiques (MW)': 'sum' , 'Prod tot (MW)' : 'sum'
  })


  df_temp_jour = df_temp.reset_index()
  df_temp_jour['DateHeure'] = df_temp_jour['DateHeure'].apply(lambda x:x[:10])
  df_temp_jour = df_temp_jour.groupby('DateHeure').agg({
                                                      'Thermique (MW)' : 'mean' , 'Nucléaire (MW)' : 'mean' ,
                                                      'Eolien (MW)' : 'mean' , 'Solaire (MW)' : 'mean' ,
                                                      'Hydraulique (MW)' : 'mean' , 'Bioénergies (MW)' : 'mean',
                                                        'Consommation (MW)' : 'mean' , 'Pompage (MW)': 'mean',
                                                        'Ech. physiques (MW)': 'mean' , 'Prod tot (MW)' : 'mean'
  })
  
  debut = col1.date_input("date de début",datetime.date(2023, 1, 1))
  fin = col1.date_input("date de fin",datetime.date(2023, 1,31))
  granularite = col1.selectbox(
    'Granularité souhaitée',
     ('Demi-heure', 'Jour'))
  
  df_temp.index = pd.to_datetime(df_temp.index)
  df_temp_jour.index = pd.to_datetime(df_temp_jour.index)

  df_temp = df_temp.loc[debut:fin]
  df_temp_jour = df_temp_jour.loc[debut:fin]


  df_temp['-Pompage (MW)'] = -df_temp['Pompage (MW)']
  df_temp['Conso avec pomp (MW)'] = df_temp['Consommation (MW)'] - df_temp['Pompage (MW)']
  df_temp['Ech. phy recalc (MW)'] = df_temp['Conso avec pomp (MW)'] - df_temp['Prod tot (MW)']
  df_temp['couleur Ech. phy'] = df_temp['Ech. physiques (MW)'].apply(lambda x:'red' if x>0 else 'green')

  df_temp_jour['-Pompage (MW)'] = -df_temp_jour['Pompage (MW)']
  df_temp_jour['Conso avec pomp (MW)'] = df_temp_jour['Consommation (MW)'] - df_temp_jour['Pompage (MW)']
  df_temp_jour['Ech. phy recalc (MW)'] = df_temp_jour['Conso avec pomp (MW)'] - df_temp_jour['Prod tot (MW)']
  df_temp_jour['couleur Ech. phy'] = df_temp_jour['Ech. physiques (MW)'].apply(lambda x:'red' if x>0 else 'green')

  df_temp_prod = df_temp.loc[:,[ 'Nucléaire (MW)','Thermique (MW)' ,'Eolien (MW)',
                                                                'Solaire (MW)', 'Hydraulique (MW)',
                                                                'Bioénergies (MW)']]
  
  df_temp_conso = df_temp.loc[:,[ 'Consommation (MW)' , 'Conso avec pomp (MW)','Ech. physiques (MW)','Ech. phy recalc (MW)','couleur Ech. phy']]

  df_temp_jour_prod = df_temp_jour.loc[:,[ 'Nucléaire (MW)','Thermique (MW)' ,'Eolien (MW)',
                                                                'Solaire (MW)', 'Hydraulique (MW)',
                                                                'Bioénergies (MW)']]
  
  df_temp_jour_conso = df_temp_jour.loc[:,[ 'Consommation (MW)' , 'Conso avec pomp (MW)','Ech. physiques (MW)','Ech. phy recalc (MW)','couleur Ech. phy']]
  
  if granularite == "Demi-heure":
    
    fig = px.area(df_temp_prod)
    fig.update_traces(line_width=0)
    #fig = go.Figure()
    #fig.add_traces(go.Scatter( x = df_temp_conso.index , y = df_temp_conso['Consommation (MW)'] , line = dict(width=3), opacity = 0.4, name = 'Conso hors pompage (MW)'))
    fig.add_traces(go.Scatter( x = df_temp_conso.index , y = df_temp_conso['Conso avec pomp (MW)'] , line = dict(width=3,color='black'), opacity = 0.7, name = 'Conso (MW)'))
    fig.add_traces(go.Scatter( x = df_temp_conso.index , y = df_temp_conso['Ech. physiques (MW)'] , mode='markers', marker_color = df_temp['couleur Ech. phy']  , name = 'Echanges (MW)'))
    #fig.add_traces(go.Scatter( x = df_temp_conso.index , y = df_temp_conso['Ech. phy recalc (MW)'], name = 'Echanges recalc (MW)'))
    fig.update_layout( autosize=False, width=900, height=700)
    fig.update_layout(
        title=dict(
            text="Puissance produite et consommée"
        ),
        xaxis=dict(
            title=dict(
                text="Temps"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Puissance (MW)"
            )
        ),
        legend=dict(
            title=dict(
                text="Type de puissance"
            )))



    col2.plotly_chart(fig)
  
  else:

    
    fig = px.area(df_temp_jour_prod)
    fig.update_traces(line_width=0)
    #fig = go.Figure()
    #fig.add_traces(go.Scatter( x = df_temp_jour_conso.index , y = df_temp_jour_conso['Consommation (MW)'] , line = dict(width=3), opacity = 0.4, name = 'Conso hors pompage (MW)'))
    fig.add_traces(go.Scatter( x = df_temp_jour_conso.index , y = df_temp_jour_conso['Conso avec pomp (MW)'] , line = dict(width=3,color='black'), opacity = 0.7, name = 'Conso (MW)'))
    fig.add_traces(go.Scatter( x = df_temp_jour_conso.index , y = df_temp_jour_conso['Ech. physiques (MW)'] , mode='markers', marker_color = df_temp_jour['couleur Ech. phy']  , name = 'Echanges (MW)'))
    #fig.add_traces(go.Scatter( x = df_temp_conso.index , y = df_temp_conso['Ech. phy recalc (MW)'], name = 'Echanges recalc (MW)'))
    fig.update_layout( autosize=False, width=900, height=700)
    fig.update_layout(
        title=dict(
            text="Puissance produite et consommée moyennée par jour"
        ),
        xaxis=dict(
            title=dict(
                text="Temps"
            )
        ),
        yaxis=dict(
            title=dict(
                text="Puissance (MW)"
            )
        ),
        legend=dict(
            title=dict(
                text="Type de puissance"
            )))



    col2.plotly_chart(fig)
  
  








if page == pages[5] :
  st.header("Modélisation ARIMA")
  #st.write("### Modélisation ARIMA")
  st.write("L'optimisation des coefficients nous amène au choix (p,d,q) = (2,0,3)")

  col1, col2 = st.columns([1, 7])

  coef_p = col1.selectbox('Coefficient p',(0,1,2,3,4,5),2)
  coef_d = col1.selectbox('Coefficient d',(0,1,2,3,4,5),0)
  coef_q = col1.selectbox('Coefficient q',(0,1,2,3,4,5),3)


  df_arima=df.groupby('DateHeure').agg({'Consommation (MW)' : 'sum'})
  df_arima=df_arima.reset_index()
  df_mois = df_arima
  df_mois['DateHeure']=df_mois['DateHeure'].apply(lambda x:x[:7])
  df_mois = df_mois.groupby('DateHeure').mean()
  df_mois.index = pd.to_datetime(df_mois.index)
  df_train = df_mois.loc[df_mois.index.year<2021]
  df_test = df_mois.loc[df_mois.index.year>=2021]


  model=sm.tsa.ARIMA(df_train,order=(coef_p,coef_d,coef_q))
  arima=model.fit()
  pred_train = arima.predict(0, 95)
  pred_test = arima.predict(96, 120)
  pred_futur = arima.predict(121, 150)

  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_train.index , y = df_train['Consommation (MW)'] , name = 'df_train', line = dict(color='blue') ))
  fig.add_traces(go.Scatter( x = df_test.index , y = df_test['Consommation (MW)'] , name = 'df_test', line = dict(color='red') ))
  fig.add_traces(go.Scatter( x = pred_train.index , y = pred_train.values , name = 'pred_train',line = dict(color='blue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_test.index , y = pred_test.values , name = 'pred_test',line = dict(color='red',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_futur.index , y = pred_futur.values , name = 'pred_futur',line = dict(color='green',dash='dot')))

  fig.update_layout(
    title=dict(
        text="Puissance moyenne mensuelle"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))


  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  col2.plotly_chart(fig)


  st.header("Modélisation SARIMA")
  st.write("L'optimisation des coefficients nous amène au choix (p,d,q)(P,D,Q,S) = (2,0,3)(0,1,1)12")
  model=sm.tsa.SARIMAX(df_train,order=(2,0,3),seasonal_order=(0,1,1,12))
  sarima=model.fit()

  pred_train_ = sarima.predict(0, 95)
  pred_test_ = sarima.predict(96, 120)
  pred_futur_ = sarima.predict(121, 150)

  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_train.index , y = df_train['Consommation (MW)'] , name = 'df_train', line = dict(color='blue') ))
  fig.add_traces(go.Scatter( x = df_test.index , y = df_test['Consommation (MW)'] , name = 'df_test', line = dict(color='red') ))
  fig.add_traces(go.Scatter( x = pred_train_.index , y = pred_train_.values , name = 'pred_train_',line = dict(color='blue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_test_.index , y = pred_test_.values , name = 'pred_test_',line = dict(color='red',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_futur_.index , y = pred_futur_.values , name = 'pred_futur_',line = dict(color='green',dash='dot')))

  fig.update_layout(
    title=dict(
        text="Puissance moyenne mensuelle"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))


  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  st.plotly_chart(fig)




  st.header("Modélisation Prophet")
  st.write("Nous réglons les paramètres 'fixes' (aspect non multiplicatif et type de saisonnalité) et nous optimisons les autres paramètres")
  st.write("Nous faisons plusieurs essais avec des horizons d'apprentissage différents")


  pred_prophet_2020_ = load_data('pred_prophet_fin_2019.csv',',')
  train_prophet_2020_ = load_data('train_prophet_fin_2019.csv',',')
  pred_prophet_2019_ = load_data('pred_prophet_fin_2018.csv',',')
  pred_prophet_2020_['ds'] = pd.to_datetime(pred_prophet_2020_['ds'])
  pred_prophet_2019_['ds'] = pd.to_datetime(pred_prophet_2019_['ds'])
  
  train_prophet = train_prophet_2020_
  pred_prophet = pred_prophet_2020_.loc[pred_prophet_2020_['ds'].dt.year>=2020]

  pred_prophet_2020_ = pred_prophet_2020_.loc[pred_prophet_2020_['ds'].dt.year>2020]
  pred_prophet_2019_ = pred_prophet_2019_.loc[pred_prophet_2019_['ds'].dt.year>2020]
  
  df_train = df_mois.loc[df_mois.index.year<2020]
  df_test = df_mois.loc[df_mois.index.year>=2020]
 
  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_train.index, y = df_train['Consommation (MW)'] , name = 'df_train', line = dict(color='blue') ))
  fig.add_traces(go.Scatter( x = df_test.index, y = df_test['Consommation (MW)'] , name = 'df_test', line = dict(color='red') ))

  fig.add_traces(go.Scatter( x = train_prophet['ds'] , y = train_prophet['yhat'] , name = 'pred_train', line = dict(color='blue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_prophet['ds'] , y = pred_prophet['yhat'] , name = 'pred_futur', line = dict(color='red',dash='dot')))

  fig.update_layout(
    title=dict(
        text="Puissance moyenne mensuelle"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))


  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  st.plotly_chart(fig)


  st.header("Comparaison des performances")

  df_test = df_mois.loc[df_mois.index.year>=2021]


  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_test.index , y = df_test['Consommation (MW)'] , name = 'df_test', line = dict(color='red') ))
  fig.add_traces(go.Scatter( x = pred_test.index , y = pred_test.values , name = 'pred_test ARIMA',line = dict(color='lightblue')))
  fig.add_traces(go.Scatter( x = pred_futur.index , y = pred_futur.values , name = 'pred_futur ARIMA',line = dict(color='lightblue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_test_.index , y = pred_test_.values , name = 'pred_test SARIMA',line = dict(color='green')))
  fig.add_traces(go.Scatter( x = pred_futur_.index , y = pred_futur_.values , name = 'pred_futur SARIMA',line = dict(color='green',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_prophet_2020_['ds'] , y = pred_prophet_2020_['yhat'] , name = 'pred_futur Prophet fin appr 2019',line = dict(color='yellow',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_prophet_2019_['ds'] , y = pred_prophet_2019_['yhat'] , name = 'pred_futur Prophet fin appr 2018',line = dict(color='orange',dash='dot')))


  fig.update_layout(
    title=dict(
        text="Puissance moyenne mensuelle"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))


  fig.update_layout( autosize=False, width=900, height=500, plot_bgcolor='black')
  fig.update_xaxes(gridcolor='grey',showgrid=True)
  fig.update_yaxes(gridcolor='grey')
  st.plotly_chart(fig)


  arima_mae = mean_absolute_error(df_test, pred_test)
  arima_mse = mean_squared_error(df_test, pred_test)
  arima_rmse = mean_squared_error(df_test, pred_test,squared = False)
  arima_r2 = r2_score(df_test, pred_test)

  sarima_mae = mean_absolute_error(df_test, pred_test_)
  sarima_mse = mean_squared_error(df_test, pred_test_)
  sarima_rmse = mean_squared_error(df_test, pred_test_,squared = False)
  sarima_r2 = r2_score(df_test, pred_test_)

  pred_prophet_2020_ = pred_prophet_2020_.iloc[:25]['yhat']
  pred_prophet_2019_ = pred_prophet_2019_.iloc[:25]['yhat']
 
  pro20_mae = mean_absolute_error(df_test, pred_prophet_2020_)
  pro20_mse = mean_squared_error(df_test, pred_prophet_2020_)
  pro20_rmse = mean_squared_error(df_test, pred_prophet_2020_,squared = False)
  pro20_r2 = r2_score(df_test, pred_prophet_2020_)

  pro19_mae = mean_absolute_error(df_test, pred_prophet_2019_)
  pro19_mse = mean_squared_error(df_test, pred_prophet_2019_)
  pro19_rmse = mean_squared_error(df_test, pred_prophet_2019_,squared = False)
  pro19_r2 = r2_score(df_test, pred_prophet_2019_)

  perf = pd.DataFrame([[arima_mae,arima_mse,arima_rmse,arima_r2],
                     [sarima_mae,sarima_mse,sarima_rmse,sarima_r2],
                     [pro20_mae,pro20_mse,pro20_rmse,pro20_r2],
                     [pro19_mae,pro19_mse,pro19_rmse,pro19_r2]],
                   #columns = ,
                   #index : {'ARIMA','SARIMA','Prophet fin 2019','Prophet fin 2018'}
                   )
  perf = perf.set_axis(['ARIMA','SARIMA','Prophet fin 2019','Prophet fin 2018'], axis=0)
  perf = perf.set_axis(['mae','mse','rmse','R2-score'], axis=1)
  st.write("Nous comparons les performances sur la période où les 4 algorithmes ont prédit.")
  st.write("Cette période est de janvier 2021 à janvier 2023 inclus (25 points).")
  st.dataframe(perf)

  st.header("Prédictions au niveau régional")
  st.write("Un travail de prédiction de la consommation a également été réalisé au niveau régional ce qui a permis \
           d'identifier des différences entre régions (ex : PACA vs IdF)")
  st.image('prophet_regions.png')



if page == pages[6] :
  
  st.write("Comme pour la série mensuelle, nous réglons et optimisons les paramètres")
  st.write("Nous entrainons le modèle jusque fin 2021 et nous évaluons les prévisions sur janvier 2022 à janvier 2023")

  expander = st.expander("Modélisation Prophet données quotidiennes")
  expander.header("Modélisation Prophet données quotidiennes")

  df_=df.groupby('DateHeure').agg({'Consommation (MW)' : 'sum'})
  df_.reset_index(inplace=True)
  df_['DateHeure']=df_['DateHeure'].apply(lambda x:x[:10])
  df2 = df_.groupby('DateHeure').mean()
  df2=df2.reset_index()
  df2 = df2.rename(columns = {"DateHeure": "ds", "Consommation (MW)": "y"})
  df2['ds'] = pd.to_datetime(df2['ds'])
  df_train = df2.loc[(df2['ds'].dt.year < 2022)]
  df_test = df2.loc[(df2['ds'].dt.year >= 2022)]

  pred_train = load_data('train_prophet_quotidien.csv',',')
  pred_futur = load_data('futur_prophet_quotidien.csv',',')

  
  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_train['ds'], y = df_train['y'] , name = 'df_train', line = dict(color='blue') ))
  fig.add_traces(go.Scatter( x = df_test['ds'], y = df_test['y'] , name = 'df_test', line = dict(color='red') ))

  fig.add_traces(go.Scatter( x = pred_train['ds'] , y = pred_train['yhat'] , name = 'pred_train', line = dict(color='blue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_futur['ds'] , y = pred_futur['yhat'] , name = 'pred_futur', line = dict(color='red',dash='dot')))

  fig.update_layout(
    title=dict(
        text="Puissance moyenne quotidienne"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))

  fig.update_xaxes(showgrid=True)
  fig.update_layout( autosize=False, width=800, height=500)
  expander.plotly_chart(fig)




  st.header("Modélisation Prophet données à la demi-heure")

  st.image('saisonnalites.png',caption='Tendance et saisonnalités identifiées par Prophet')

  df_=df.groupby('DateHeure').agg({'Consommation (MW)' : 'sum'})
  df_.reset_index(inplace=True)
  df_ = df_.rename(columns = {"DateHeure": "ds", "Consommation (MW)": "y"})
  df_['ds'] = pd.to_datetime(df_['ds'])
  df_train = df_.loc[(df_['ds'].dt.year < 2022)]
  df_test = df_.loc[(df_['ds'].dt.year >= 2022)]

  pred_train = load_data('train_prophet_demiheure.csv',',')
  pred_futur = load_data('futur_prophet_demiheure.csv',',')

  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_train['ds'], y = df_train['y'] , name = 'df_train', line = dict(color='blue') ))
  fig.add_traces(go.Scatter( x = df_test['ds'], y = df_test['y'] , name = 'df_test', line = dict(color='red') ))

  fig.add_traces(go.Scatter( x = pred_train['ds'] , y = pred_train['yhat'] , name = 'pred_train', line = dict(color='blue',dash='dot')))
  fig.add_traces(go.Scatter( x = pred_futur['ds'] , y = pred_futur['yhat'] , name = 'pred_futur', line = dict(color='red',dash='dot')))

  fig.update_layout(
    title=dict(
        text="Puissance mesurée toutes les demi-heures"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))


  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  st.plotly_chart(fig)


  st.write("Il y a des périodes pour lesquelles la consommation prédite matche parfaitement avec la réalité. Par exemple la période estivale")



  pred_futur['ds'] = pd.to_datetime(pred_futur['ds'])
  df_ete = df_test.loc[(df_test['ds'].dt.date>datetime.date(year=2022,month=9,day=8)) & (df_test['ds'].dt.date<datetime.date(year=2022,month=9,day=25))]
  pred_ete = pred_futur.loc[(pred_futur['ds'].dt.date>datetime.date(year=2022,month=9,day=8)) & (pred_futur['ds'].dt.date<datetime.date(year=2022,month=9,day=25))]


  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_ete['ds'], y = df_ete['y'] , name = 'df_test', line = dict(color='red') ))
  fig.add_traces(go.Scatter( x = pred_ete['ds'] , y = pred_ete['yhat'] , name = 'pred_futur', line = dict(color='red',dash='dot') , fill = 'tonexty'))

  fig.update_layout(
    title=dict(
        text="Puissance mesurée toutes les demi-heures"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))
  
  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  st.plotly_chart(fig)

  st.write("Pour d'autres périodes l'écart est plus important. \
           Ici à l'automne, on voit que l'algorithme avait prévu une croissance continue alors que \
           la consommation stagne au niveau estival \
           Les prévisions ne prennent pas en compte les événements exogènes comme météo plus clémente que \
           d'habitude ou ici peut-être le changement de comportement dû au conflit en Ukraine.\n \
           La prise en compte des jours fériés/ponts est également perfectible (exemple ici 1er novembre)")

  df_automne = df_test.loc[(df_test['ds'].dt.date>datetime.date(year=2022,month=10,day=9)) & (df_test['ds'].dt.date<datetime.date(year=2022,month=11,day=20))]
  pred_automne = pred_futur.loc[(pred_futur['ds'].dt.date>datetime.date(year=2022,month=10,day=9)) & (pred_futur['ds'].dt.date<datetime.date(year=2022,month=11,day=20))]

  
  fig = go.Figure()
  fig.add_traces(go.Scatter( x = df_automne['ds'], y = df_automne['y'] , name = 'df_test', line = dict(color='red') ))
  fig.add_traces(go.Scatter( x = pred_automne['ds'] , y = pred_automne['yhat'] , name = 'pred_futur', line = dict(color='red',dash='dot') , fill = 'tonexty'))

  fig.update_layout(
    title=dict(
        text="Puissance mesurée toutes les demi-heures"
    ),
    xaxis=dict(
        title=dict(
            text="Temps"
        )
    ),
    yaxis=dict(
        title=dict(
            text="Puissance (MW)"
        )
    ),
    legend=dict(
        title=dict(
            text="Type de donnée"
        )))
  
  fig.update_layout( autosize=False, width=800, height=500)
  fig.update_xaxes(showgrid=True)
  st.plotly_chart(fig)





if page == "Introduction Machine Learning":
    st.title("Introduction Machine Learning")
    st.write("""
    La consommation énergétique en France fluctue selon plusieurs facteurs : saisonnalité, production d’énergies renouvelables,
    activité économique, etc. Pour mieux comprendre ces tendances et anticiper la consommation, nous avons utilisé des **modèles de Machine Learning**.
    
    ## Pourquoi le Machine Learning ?
    - Identifier les **facteurs clés** influençant la consommation.
    - Prédire les besoins énergétiques pour **optimiser la gestion du réseau**.
    - Comparer plusieurs modèles pour trouver le **plus performant**.
    
    ## Modèles utilisés
    Trois modèles de régression ont été testés :
    - **Régression Linéaire** : Modèle simple capturant des tendances linéaires.
    - **Random Forest** : Modèle basé sur des arbres de décision, efficace pour les relations complexes.
    - **Gradient Boosting** : Modèle avancé qui améliore la précision des prédictions.
    """)

if page in models.keys():
    model_name = page
    st.title(f"Modèle : {model_name}")
    
    if model_name == "Random Forest":
        n_estimators = st.slider("Nombre d'arbres", 10, 200, 100, 10)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    elif model_name == "Gradient Boosting":
        learning_rate = st.slider("Taux d'apprentissage", 0.01, 0.3, 0.1, 0.01)
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=learning_rate, random_state=42)
    else:
        model = models[model_name]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Évaluation du modèle :**")
    st.write(f"- MAE : {mae:.2f}")
    st.write(f"- MSE : {mse:.2f}")
    st.write(f"- RMSE : {rmse:.2f}")
    st.write(f"- R² : {r2:.4f}")

    st.subheader("Distribution des erreurs")
    errors = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(errors, bins=30, kde=True, ax=ax)
    ax.set_title("Histogramme des erreurs de prédiction")
    ax.set_xlabel("Erreur")
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
    ax.set_xlabel("Valeurs Réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title(f"Comparaison des Valeurs Réelles et Prédites ({model_name})")
    st.pyplot(fig)

if page == "Conclusion Machine Learning":
    st.title("Conclusion Machine Learning")
    st.write("""
    ## Synthèse des résultats
    - **La Régression Linéaire** est utile mais limitée pour des tendances non linéaires.
    - **Le Random Forest** offre une meilleure précision en capturant des interactions complexes.
    - **Le Gradient Boosting** est un modèle performant, réduisant les erreurs de prédiction.
    
    ## Comparaison des performances des modèles
    """)
    
    fig, ax = plt.subplots()
    performance = {m: r2_score(y_test, models[m].fit(X_train, y_train).predict(X_test)) for m in models.keys()}
    sns.barplot(x=list(performance.keys()), y=list(performance.values()), ax=ax)
    ax.set_title("Comparaison des scores R² des modèles")
    ax.set_ylabel("Score R²")
    st.pyplot(fig)
    
    st.write("""
    ## Limites et perspectives
    - **Données supplémentaires** : Ajouter des variables météorologiques pourrait améliorer la prédiction.
    - **Optimisation** : Ajuster les hyperparamètres pour encore affiner les modèles.
    """)

if page == "Conclusion et ouverture":
    st.title("Conclusion et ouverture")
    st.write("""
    L’analyse énergétique menée dans ce rapport met en évidence l’importance d’une prévision précise pour une **meilleure gestion du réseau électrique**.
    
    ## Points clés
    - La consommation varie fortement selon les saisons et les sources d'énergie.
    - Les modèles de Machine Learning permettent d’anticiper ces fluctuations.
    
    ## Vers l’avenir
    - **Smart Grids** : L’intégration de réseaux intelligents améliorerait l’équilibre entre production et demande.
    - **Stockage d’énergie** : Une gestion plus efficace des surplus permettrait une stabilisation accrue.
    """)

if page == "Annexe":
    st.image('renouvelables_par_heure.png')
    st.image('taux_de_charge.png')
    st.image('renouvelables_regions.png')
    st.image('prod_conso_regions.png')
    st.image('prod_par_an.png')



