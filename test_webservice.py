import pytest
from flask import json

import webservice

DEFAULT_TOP_K = 3


@pytest.fixture
def client():
    webservice.app.config['TESTING'] = True
    client = webservice.app.test_client()
    yield client


def test_prediction(client):
    """Test that prediction work."""

    rv = client.post('/temas', data=json.dumps(dict(
        text='A comissao especial que analisa a proposta (PEC 15/15) que torna permanente o Fundo de Manutencao e Desenvolvimento da Educacao Basica e Valorizacao dos Profissionais da Educacao (Fundeb) promove audiencia publica nesta manha para discutir sugestoes de aprimoramento do texto e medidas de cooperacao entre os entes federados no setor educacional.\
              Criado em 2006 para vigorar ate 2020, o Fundeb e um fundo que utiliza recursos federais, dos estados, Distrito Federal e municipios para financiar a educacao basica no Pais, incluindo a remuneracao dos professores.\
              O evento e uma iniciativa da relatora da PEC, deputada Professora Dorinha Seabra Rezende (DEM-TO). Segundo ela, com o fim da vigencia do fundo se aproximando e urgente debater o assunto.\
              Foram convidados para a reuniao: o vice-presidente da Frente Nacional de Prefeitos (FNP), Elias Dinis; e o gestor do Observatorio de Informacoes Municipais, Francois Eugene Jean de Bremaeker.\
              A audiencia sera realizada no plenario 9, a partir das 11 horas, e podera ser acompanhada ao vivo pelo WebCamara.')),
        content_type='application/json'
    )
    result = json.loads(rv.data)['prediction']
    assert 'Educação' == result[0][0]
    assert len(result) == DEFAULT_TOP_K


def test_prediction_url(client):
    """Test that prediction using URL work."""

    rv = client.post('/temas/url', data=json.dumps(dict(
        url='http://www2.camara.leg.br/camaranoticias/noticias/TRABALHO-E-PREVIDENCIA/538706-COMISSAO-DE-TRABALHO-APROVA-A-REGULAMENTACAO-DA-PROFISSAO-DE-GARCOM-COM-PISO-SALARIAL-DE-R$-2.811.html')),
        content_type='application/json'
    )
    result = json.loads(rv.data)['prediction']
    text = json.loads(rv.data)['text']
    assert 'Trabalho e Emprego' == result[0][0]
    assert 'Edvaldo Belitardo / Camara dos Deputados' in text
    assert len(result) == DEFAULT_TOP_K

