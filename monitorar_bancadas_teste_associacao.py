"""Protótipo experimental: saída e retorno a ROIs com YOLO11m + ByteTrack (teste de associação temporal).
Não infere improdutividade. Associações e eventos exigem validação manual.
"""
# 1. BIBLIOTECAS
# pathlib: caminhos dos arquivos; collections: contagem de votos por bancada.
# csv/json: leitura e gravação de dados; OpenCV: vídeo e imagens;
# Ultralytics: YOLO11m para detecção e ByteTrack para rastreamento.

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import cv2
from ultralytics import YOLO

# 2. ARQUIVOS DE ENTRADA E SAÍDA

BASE = Path(__file__).resolve().parent
VIDEO = BASE / "videos" / "gravacao_33min_h264.mp4"
ARQUIVO_ROIS = BASE / "resultados" / "rois_bancadas_15min.json"
PESOS = BASE / "yolo11m.pt"
VIDEO_SAIDA = BASE / "resultados" / "monitoramento_33min_teste_associacao.mp4"
CSV_SAIDA = BASE / "resultados" / "eventos_33min_teste_associacao.csv"

# 3. PARÂMETROS DE DETECÇÃO E DE ASSOCIAÇÃO

DEVICE = 0  # RTX 4050
CONF = 0.15
IMGSZ = 960
LIMIAR_SOBREPOSICAO = 0.10  # fração da caixa da pessoa dentro da ROI
OBSERVACOES_ASSOCIACAO = 8
TEMPO_MIN_ASSOCIACAO = 2.0  # segundos de observação na mesma bancada (teste)
MAX_GAP_ASSOCIACAO = 0.5   # tolera uma breve falha durante a associação inicial
TEMPO_CONFIRMA_SAIDA = 1.0  # segundos de detecções fora
TEMPO_CONFIRMA_RETORNO = 1.0  # segundos de detecções dentro
MAX_GAP_DETECCAO = 0.5  # segundos; acima disso, interrompe a medição

# 4. PROTEÇÃO VISUAL DOS ROSTOS

# Anonimizacao visual dos rostos
MODELO_FACE = BASE / "modelos" / "face_detection_yunet_2023mar.onnx"
CONFIANCA_FACE = 0.50
ESCALA_FACE = 1.5

# 5. VALIDAÇÃO DAS ENTRADAS

for caminho in (VIDEO, ARQUIVO_ROIS, PESOS):
    if not caminho.is_file():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho}')

# Carrega as coordenadas das ROIs previamente selecionadas manualmente.
# Espera a estrutura JSON com a chave "rois".

with ARQUIVO_ROIS.open(encoding='utf-8') as f:
    config = json.load(f)
rois = config.get('rois', [])
if not rois:
    raise ValueError('Arquivo JSON não contém ROIs.')

# 6. PREPARAÇÃO DO VÍDEO
# Obtém resolução e FPS, necessários para desenhar as regiões e medir os tempos.

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise RuntimeError(f'Não foi possível abrir o vídeo: {VIDEO}')
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0 or w <= 0 or h <= 0:
    cap.release()
    raise RuntimeError('Metadados inválidos do vídeo.')
# Confere se cada retângulo de bancada está dentro dos limites do vídeo.

for roi in rois:
    x, y, rw, rh = (int(roi[k]) for k in ('x', 'y', 'w', 'h'))
    if min(x, y) < 0 or rw <= 0 or rh <= 0 or x + rw > w or y + rh > h:
        cap.release()
        raise ValueError(f'ROI {roi.get("id")} inválida para vídeo {w}x{h}; confira enquadramento e resolução.')

# Prepara o arquivo MP4 de saída com a mesma resolução e FPS da entrada.

VIDEO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(VIDEO_SAIDA), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError('Não foi possível criar vídeo de saída.')


# 7. FUNÇÕES DE ASSOCIAÇÃO ESPACIAL

def sobreposicao_pessoa(caixa, roi):
    # A caixa detectada está em coordenadas (x1, y1, x2, y2).
    # A ROI é representada por canto superior esquerdo, largura e altura.

    x1, y1, x2, y2 = caixa
    rx1, ry1 = float(roi['x']), float(roi['y'])
    rx2, ry2 = rx1 + float(roi['w']), ry1 + float(roi['h'])
    iw = max(0.0, min(x2, rx2) - max(x1, rx1))
    ih = max(0.0, min(y2, ry2) - max(y1, ry1))
    return iw * ih / max(1.0, (x2 - x1) * (y2 - y1))

# Retorna o identificador da bancada candidata ou None, caso não haja
# sobreposição suficiente. Isso não confirma a identidade da trabalhadora.

def melhor_bancada(caixa):
    if not rois:
        return None
    roi = max(rois, key=lambda r: sobreposicao_pessoa(caixa, r))
    return int(roi['id']) if sobreposicao_pessoa(caixa, roi) >= LIMIAR_SOBREPOSICAO else None


# 8. CONTROLE DAS TRAJETÓRIAS E MODELOS
# "votos": observações de associação de cada ID temporário às bancadas;
# "candidatos_associacao": tempo e continuidade da associação inicial;
# "estados": máquina de estados dentro / fora / aguardando_reassociacao;
# "registros": eventos para exportação; "associacoes_iniciais": log diagnóstico.
# Atenção: o ID gerado pelo rastreador NÃO identifica permanentemente a pessoa.

# Estado por ID: associação inicial e máquina de estados da permanência.
votos = defaultdict(Counter)
candidatos_associacao = {}  # tempo e contagem para IDs ainda sem bancada confirmada
associacoes_iniciais = []  # log de diagnóstico, não identifica pessoas reais
estados = {}  # id -> dict
registros = []
modelo = YOLO(str(PESOS))
# O YuNet atua apenas sobre os rostos no vídeo exportado.
# Falta do arquivo ONNX interrompe o processamento.

# Carregar detector facial YuNet
if not MODELO_FACE.is_file():
    raise FileNotFoundError(
        f"Modelo facial nao encontrado: {MODELO_FACE}"
    )

detector_face = cv2.FaceDetectorYN.create(
    model=str(MODELO_FACE),
    config="",
    input_size=(320, 320),
    score_threshold=CONFIANCA_FACE,
    nms_threshold=0.3,
    top_k=5000
)

total_rostos = 0
indice = 0

# 9. FORMATAÇÃO E REGISTRO DOS EVENTOS
# Converte o tempo do vídeo para HH:MM:SS.mmm; não é horário de relógio real.

def formatar_tempo_video(segundos):
    if segundos is None or segundos == '':
        return ''

    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    secs = segundos % 60

    return f'{horas:02d}:{minutos:02d}:{secs:06.3f}'
    
# Cada registro contém o ID temporário, a bancada, saída e retorno.
# Se não existe retorno confirmado, deixa-se retorno/duração vazios,
# evitando inventar valores em casos de perda de rastreamento.

def salvar_evento(ident, st, status, fim=None, observacao=''):
    inicio = st['inicio_saida']

    registros.append({
        'id_rastreamento': ident,
        'bancada': st['bancada'],
        'saida_s': round(inicio, 3),
        'saida_video': formatar_tempo_video(inicio),
        'retorno_s': '' if fim is None else round(fim, 3),
        'retorno_video': formatar_tempo_video(fim),
        'duracao_s': '' if fim is None else round(max(0.0, fim - inicio), 3),
        'status': status,
        'observacao': observacao,
    })


try:
    while True:
        # 10. PROCESSAMENTO QUADRO A QUADRO
        # Lê cada imagem até o fim da gravação.

        ok, frame = cap.read()
        if not ok:
            break
        # Os tempos são calculados a partir do índice do quadro e do FPS.

        t = indice / fps
        indice += 1
        # Detecta apenas a classe 0 (pessoa). ByteTrack mantém IDs entre quadros;
        # persist=True preserva o estado interno do rastreador durante o vídeo.

        result = modelo.track(frame, persist=True, tracker=str(BASE / "bytetrack_tcc.yaml"),
                             classes=[0], conf=CONF, imgsz=IMGSZ, device=DEVICE,
                             verbose=False)[0]
        # Gera uma cópia para desenhar as bancadas e os IDs no vídeo de saída.
        # A detecção/rastreamento continua sendo feita sobre o quadro original.

        view = frame.copy()
        for roi in rois:
            x, y, rw, rh = (int(roi[k]) for k in ('x', 'y', 'w', 'h'))
            cv2.rectangle(view, (x, y), (x + rw, y + rh), (0, 210, 0), 2)
            cv2.putText(view, f'B{roi["id"]}', (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 210, 0), 2)

        # Extrai caixas e IDs retornados pelo rastreador; "presentes" armazena
        # os IDs que apareceram neste quadro para identificar desaparecimentos.

        deteccoes = []
        if result.boxes is not None and result.boxes.id is not None:
            for box, ident in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.id.int().cpu().tolist()):
                deteccoes.append((ident, [float(v) for v in box]))
        presentes = {ident for ident, _ in deteccoes}

        # 11. TRATAMENTO DAS INTERRUPÇÕES
        # Quando um ID some além do intervalo tolerado, o programa não supõe
        # que a pessoa saiu ou retornou; registra uma ocorrência para revisão.

        # Perda longa de detecção: não interpretar como saída nem retorno.
        for ident, st in list(estados.items()):
            if ident in presentes:
                continue
            if (t - st['ultimo_visto']) > MAX_GAP_DETECCAO and not st['interrompido']:
                st['interrompido'] = True
                if st['fase'] == 'fora':
                    salvar_evento(ident, st, 'revisar_perda_de_rastreamento',
                                  observacao='Perda de detecao durante afastamento; sem duracao confiavel')
                st['fase'] = 'aguardando_reassociacao'
                st['candidato_inicio'] = None

        # 12. ASSOCIAÇÃO À BANCADA E MÁQUINA DE ESTADOS
        # Cada detecção é comparada às ROIs. Uma associação inicial ainda não
        # confirmada é tratada de modo diferente de uma trajetória já associada.

        for ident, caixa in deteccoes:
            roi_atual = melhor_bancada(caixa)
            st = estados.get(ident)
            # ID novo ou ainda sem bancada: aplica votos e confirmação temporal.
            # Essa é a principal mudança em relação ao monitoramento-base.

            if st is None:
                # Evita associar alguém a uma bancada após apenas alguns quadros.
                # O ID precisa permanecer associado à mesma ROI durante um período
                # mínimo, além de acumular as observações da configuração inicial.
                # Recupera um candidato anterior para o mesmo ID, se existir.
                # A falta de ROI por tempo excessivo desfaz a tentativa de associação.

                cand = candidatos_associacao.get(ident)
                if roi_atual is None:
                    if cand is not None and t - cand['ultimo_visto'] > MAX_GAP_ASSOCIACAO:
                        candidatos_associacao.pop(ident, None)
                        votos.pop(ident, None)
                else:
                    # Mudança de ROI ou lacuna excessiva reinicia a contagem temporal
                    # e os votos, reduzindo associações após passagens momentâneas.

                    if (cand is None or cand['bancada'] != roi_atual
                            or t - cand['ultimo_visto'] > MAX_GAP_ASSOCIACAO):
                        candidatos_associacao[ident] = dict(
                            bancada=roi_atual, inicio=t, ultimo_visto=t
                        )
                        votos[ident].clear()
                    else:
                        cand['ultimo_visto'] = t

                    # Confirma somente após 8 observações E pelo menos 2 segundos,
                    # com maioria de votos e presença atual na bancada escolhida.

                    votos[ident][roi_atual] += 1
                    cand = candidatos_associacao[ident]
                    total_votos = sum(votos[ident].values())
                    tempo_na_bancada = t - cand['inicio']
                    if (total_votos >= OBSERVACOES_ASSOCIACAO
                            and tempo_na_bancada >= TEMPO_MIN_ASSOCIACAO):
                        bancada, contagem = votos[ident].most_common(1)[0]
                        if contagem > total_votos / 2 and roi_atual == bancada:
                            estados[ident] = dict(
                                bancada=bancada, fase='dentro',
                                candidato_inicio=None, inicio_saida=None,
                                ultimo_visto=t, interrompido=False
                            )
                            st = estados[ident]
                            # Registra um diagnóstico da associação para posterior
                            # comparação manual; não constitui identificação pessoal.

                            associacoes_iniciais.append({
                                'instante_s': round(t, 3),
                                'id_rastreamento': ident,
                                'bancada_inicial': bancada,
                                'tempo_observado_s': round(tempo_na_bancada, 3)
                            })
                            votos.pop(ident, None)
                            candidatos_associacao.pop(ident, None)
            else:
                # Com uma lacuna longa, retomar apenas após observar de novo
                # a pessoa na bancada original; não inventar trajetórias.
                # Uma trajetória interrompida só volta ao estado "dentro" após
                # reaparecer na ROI anteriormente associada.

                if st['fase'] == 'aguardando_reassociacao':
                    if roi_atual == st['bancada']:
                        st['fase'] = 'dentro'
                        st['interrompido'] = False
                        st['candidato_inicio'] = None
                else:
                    if t - st['ultimo_visto'] > MAX_GAP_DETECCAO:
                        if st['fase'] == 'fora':
                            salvar_evento(ident, st, 'revisar_perda_de_rastreamento',
                                          observacao='Lacuna longa entre duas detecções')
                        st['fase'] = 'aguardando_reassociacao'
                        st['candidato_inicio'] = None
                    # Para um ID associado, sair da ROI original por pelo menos
                    # TEMPO_CONFIRMA_SAIDA inicia um evento de afastamento.

                    elif st['fase'] == 'dentro':
                        if roi_atual != st['bancada']:
                            if st['candidato_inicio'] is None:
                                st['candidato_inicio'] = t
                            elif t - st['candidato_inicio'] >= TEMPO_CONFIRMA_SAIDA:
                                st['fase'] = 'fora'
                                st['inicio_saida'] = st['candidato_inicio']
                                st['candidato_inicio'] = None
                        else:
                            st['candidato_inicio'] = None
                    # Voltar à ROI original por TEMPO_CONFIRMA_RETORNO conclui
                    # o evento. Se o rastreamento falhou, não se reconstrói o trajeto.

                    elif st['fase'] == 'fora':
                        if roi_atual == st['bancada']:
                            if st['candidato_inicio'] is None:
                                st['candidato_inicio'] = t
                            elif t - st['candidato_inicio'] >= TEMPO_CONFIRMA_RETORNO:
                                salvar_evento(ident, st, 'completo', fim=st['candidato_inicio'])
                                st['fase'] = 'dentro'
                                st['inicio_saida'] = None
                                st['candidato_inicio'] = None
                        else:
                            st['candidato_inicio'] = None
                st['ultimo_visto'] = t

            # Desenha a caixa da pessoa e seu ID temporário com o estado atual.
            # Estar "fora" da ROI não significa, por si só, improdutividade.

            x1, y1, x2, y2 = (int(v) for v in caixa)
            st = estados.get(ident)
            texto = (f'ID {ident} | B{st["bancada"]} | {st["fase"]}'
                     if st else f'ID {ident} | analisando')
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(view, texto, (x1, max(16, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # ==========================================
        # 13. ANONIMIZAÇÃO NO VÍDEO FINAL
        # Usa o YuNet para detectar rostos e pixelizá-los.

        # DESFOQUE DOS ROSTOS NO VIDEO FINAL
        # ==========================================

        # Usa o quadro original para detectar rostos.
        # O YOLO11m e o ByteTrack ja foram executados.

        # Amplia temporariamente o quadro para facilitar a detecção facial;
        # as caixas encontradas serão convertidas à resolução original.

        imagem_face = cv2.resize(
            frame,
            None,
            fx=ESCALA_FACE,
            fy=ESCALA_FACE,
            interpolation=cv2.INTER_CUBIC
        )

        altura_face, largura_face = imagem_face.shape[:2]

        detector_face.setInputSize(
            (largura_face, altura_face)
        )

        # A saída pode ser None caso nenhum rosto seja encontrado neste quadro.

        _, rostos = detector_face.detect(imagem_face)

        if rostos is not None:
            for rosto in rostos:
                x, y, largura_rosto, altura_rosto = rosto[:4]

                # Converter coordenadas para o video original
                x = int(x / ESCALA_FACE)
                y = int(y / ESCALA_FACE)
                largura_rosto = int(largura_rosto / ESCALA_FACE)
                altura_rosto = int(altura_rosto / ESCALA_FACE)

                # Acrescenta uma margem para cobrir a região facial com folga.

                # Margem adicional ao redor do rosto
                margem_x = max(3, int(largura_rosto * 0.25))
                margem_y = max(3, int(altura_rosto * 0.25))

                x1 = max(0, x - margem_x)
                y1 = max(0, y - margem_y)
                x2 = min(w, x + largura_rosto + margem_x)
                y2 = min(h, y + altura_rosto + margem_y)

                if x2 <= x1 or y2 <= y1:
                    continue

                regiao = view[y1:y2, x1:x2]

                if regiao.size == 0:
                    continue

                # Reduz a resolucao da regiao facial
                # e amplia novamente para criar pixelizacao
                altura_r, largura_r = regiao.shape[:2]

                # Reduz e reamplia a região facial por vizinho mais próximo,
                # formando blocos grandes de pixels (pixelização).

                miniatura = cv2.resize(
                    regiao,
                    (
                        max(1, largura_r // 12),
                        max(1, altura_r // 12)
                    ),
                    interpolation=cv2.INTER_AREA
                )

                borrado = cv2.resize(
                    miniatura,
                    (largura_r, altura_r),
                    interpolation=cv2.INTER_NEAREST
                )

                view[y1:y2, x1:x2] = borrado
                total_rostos += 1

        # Salvar o quadro com as ROIs, IDs e rostos borrados
        # Escreve o quadro anotado no MP4; apresenta progresso no terminal.

        writer.write(view)
        if indice % 30 == 0:
            print(f'{indice} quadros processados ({t:.1f}s)')
# 14. LIBERAÇÃO DOS ARQUIVOS
# O finally executa mesmo que o processamento termine por erro, para liberar
# o vídeo de entrada e o gravador de saída.

finally:
    cap.release()
    writer.release()

# 15. ENCERRAMENTO DE EVENTOS INCONCLUSIVOS
# Se um ID ainda está fora da bancada ao fim do vídeo, registra saída sem retorno.

# Evento que começou, mas não foi concluído dentro do trecho: sem duração inventada.
for ident, st in estados.items():
    if st['fase'] == 'fora':
        salvar_evento(ident, st, 'sem_retorno_no_trecho',
                      observacao='Não foi observado retorno até o fim do arquivo')

# 16. EXPORTAÇÃO DO CSV DE EVENTOS
# Salva tempos, bancada, estado e observações em UTF-8 com BOM,
# usando ponto e vírgula para facilitar a abertura em planilhas no Windows.

with CSV_SAIDA.open('w', newline='', encoding='utf-8-sig') as f:
    colunas = [ 'id_rastreamento','bancada','saida_s','saida_video','retorno_s','retorno_video',
    'duracao_s','status', 'observacao']
    writer_csv = csv.DictWriter(f, fieldnames=colunas, delimiter=';')
    writer_csv.writeheader()
    writer_csv.writerows(registros)

# 17. CSV COMPLEMENTAR DE ASSOCIAÇÃO INICIAL
# Exclusivo deste script de teste: armazena os IDs que foram associados pela
# primeira vez, bancada atribuída e tempo necessário à confirmação.

with (BASE / 'resultados' / 'associacoes_iniciais_teste.csv').open(
        'w', newline='', encoding='utf-8-sig') as f:
    writer_assoc = csv.DictWriter(
        f,
        fieldnames=['instante_s', 'id_rastreamento', 'bancada_inicial',
                    'tempo_observado_s'],
        delimiter=';'
    )
    writer_assoc.writeheader()
    writer_assoc.writerows(associacoes_iniciais)

# 18. RESUMO FINAL
# Lembra que os registros demandam interpretação e verificação manual;
# o sistema não classifica automaticamente produtividade ou improdutividade.

print('\nConcluído!')
print(f"Deteccoes faciais acumuladas: {total_rostos}")
print(f'Vídeo: {VIDEO_SAIDA}')
print(f'Eventos: {CSV_SAIDA} ({len(registros)} registros)')
print('Não interprete eventos automaticamente como desperdício ou improdutividade.')