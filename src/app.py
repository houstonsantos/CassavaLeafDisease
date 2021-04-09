import time
import numpy as np
import streamlit as st
import tensorflow as ts
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image


# Definindo configuraçõs da página
st.set_page_config(
    page_title = 'Doença da Folha da Mandioca',
    page_icon = ':leaves:',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
)

# Carregando modelo CNN
leaf_model = load_model('../modelo/CassavaLeafDiseaseV1_vc.h5')

# Carregando arquivo de imagem do usuário
uploaded_file = st.sidebar.file_uploader("Arraste ou faça upload de um arquivo JPG", type = ["jpg"])

# Definindo 2 colunas para o layout
image = Image.open('../img/mandioca.jpeg')
col1, col2 = st.beta_columns(2)
col1.title('Doença da Folha da Mandioca')
col1.header('Predictive Machine')
col2.image(image)

# Infomações
with st.beta_expander('Informações', expanded = True):
    st.write('''O modelo tem como finalidade classificar o tipo da doença encontrada nas folhas da mandioca. Podendo ser elas, 
                Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD)
                e também identificar uma folha saudável. Fique a vontate para testar, procure imagens das doenças e teste.''')
    st.write('O modelo atual tem uma acuracidade de 96%.')
st.write('')


def leaf_treatment(img: None) -> None:
    '''
        preparação das imagens, redimencionando-as para 456x456 e colocando no shape (1,456,456,3)
        em um numpy array, servindo de entrada para camda input do modelo.
    '''
    leaf = Image.open(uploaded_file).resize((456, 456), resample = Image.NEAREST)
    leaf = np.expand_dims(leaf, axis = 0)
    return leaf


def leaf_predict(leaf: None) -> tuple:
    '''
        realizando predição da imagem com base no array, o resultado e uma função softmax com o 
        percentual para cada doença, onde pegaremos o maior percentual indicado. Selecionando assim
        a doença e as descrições que seram enviadas para a página.
    '''
    result = leaf_model.predict(leaf).argmax(axis = 1)
    disease = ''
    description = ''
    causer = ''
    link = ''

    if result == 0:
        disease = 'Cassava Bacterial Blight (CBB) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/300039/cassava-bacterial-blight'
        description = '''Traduzido do inglês - Xanthomonas axonopodis pv. manihotis é o patógeno que causa a praga bacteriana da mandioca. 
                        Originalmente descoberta no Brasil em 1912, a doença seguiu o cultivo de mandioca em todo o mundo.'''
        causer = '''Os sintomas são causados por uma estirpe da bactéria Xanthomonas axonopodis, que prontamente infecta plantas de mandioca 
                    (Manihotis). Dentro da plantação (ou campos), as bactérias se dispersam pelo vento ou respingos de chuva. Ferramentas 
                    contaminadas também são um importante meio de disseminação, bem como a movimentação de pessoas e animais pelas plantações, 
                    especialmente durante ou depois da chuva. Contudo, o maior problema com este patógeno é a sua distribuição a longas distâncias 
                    em material plantio, manivas e sementes aparentemente sem sintomas, particularmente na África e na Ásia. O processo de 
                    contaminação e o desenvolvimento da doença requerem 12 horas com umidade relativa de 90-100%, com temperaturas ideais indo 
                    de 22 a 30 °C. As bactérias permanecem viáveis por muitos meses em caules e na goma, retomando a atividade durante períodos 
                    chuvosos. O outro hospedeiro importante desta bactéria é a planta ornamental Euphorbia pulcherrima (bico-de-papagaio).'''
    elif result == 1:
        disease = 'Cassava Brown Streak Disease (CBSD) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/200043/cassava-brown-streak-disease'
        description = '''Traduzido do inglês - A doença do vírus da raia marrom da mandioca é uma doença prejudicial para as plantas de mandioca 
                        e é especialmente problemática na África Oriental. Foi identificado pela primeira vez em 1936 na Tanzânia e se espalhou 
                        para outras áreas costeiras da África Oriental, do Quênia a Moçambique.'''
        causer = '''Os sintomas são causados pela doença do listrado castanho da mandioca, que até onde se sabe, só afeta a mandioca e maniçoba 
                    (Manihot glaziovii). A CBSV pode ser transmitida por ácaros e afídeos, bem como pela mosca-branca Bemisia tabaci. Contudo, a 
                    maneira predominante de disseminação da doença é por manivas infectadas transportadas por humanos e pela falta de limpeza de 
                    ferramentas agrícolas no campo. As variedades de mandioca diferem bastante em sua sensibilidade e resposta à contaminação, 
                    com perdas de rendimento variando de 18 a 70%, dependendo das áreas de contaminação e das condições ambientais. Medidas de 
                    quarentena são necessárias para restringir a movimentação de manivas infectadas entre países afetadas pela doença e aqueles 
                    sem registro.'''
    elif result == 2:
        disease = 'Cassava Green Mottle (CGM) :fallen_leaf:'
        link = ''
        description = 'Traduzido do inglês - O vírus da mancha verde da mandioca é um vírus patogênico da planta da família Secoviridae.'
    elif result == 3 :
        disease = 'Cassava Mosaic Disease (CMD) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/200042/cassava-mosaic-disease'
        description = '''Traduzido do inglês - O vírus do mosaico da mandioca é o nome comum usado para se referir a qualquer uma das onze 
                        espécies diferentes de vírus fitopatogênicos do gênero Begomovirus.'''
        causer = '''Os sintomas do mosaico africano da mandioca são causados por um grupo de vírus que frequentemente contaminam de forma 
                    paralela as plantas de mandioca. Estes vírus podem ser persistentemente transmitidos pela mosca-branca Bemisia tabaci, 
                    bem como por cortes derivados de material de plantio infectado. As moscas-brancas são transportadas por correntes 
                    de vento e podem disseminar o vírus a distâncias de vários quilômetros. As variedades de mandioca diferem muito em sua 
                    suscetibilidade ao vírus, mas geralmente as folhas jovens são as primeiras a apresentar sintomas, já que as moscas-brancas 
                    preferem se alimentar dos tecidos novos. A distribuição do vírus é muito dependente da população deste inseto, que por sua 
                    vez está condicionada às condições climáticas prevalecentes. Caso grandes populações de mosca-branca coincidam com condições 
                    ideais para o desenvolvimento da mandioca, o vírus se espalhará rapidamente. As temperaturas preferidas por esta praga são 
                    estimadas entre 20 a 32 °C.'''
    elif result == 4:
        disease = 'Saudável :leaves:'
        description = 'Não indentificamos a presença de doenças na imagem. :thumbsup:'
    else:
        print('Não foi possível realizar identificação')
    
    return result, disease, description, causer, link


# Página a ser exibida com os resultados
if uploaded_file is not None:
    
    leaf = leaf_treatment(uploaded_file)
    leaf = leaf_predict(leaf)
    img = Image.open(uploaded_file)
    col3, col4 = st.beta_columns(2)
    col3.write('')
    col3.image(img)
    col4.subheader(leaf[1])
    col4.write(leaf[2])
    col4.write(leaf[3])
    
    if leaf[0] != 4:
        col4.markdown('**Consulte:**')
        col4.write(leaf[4])
