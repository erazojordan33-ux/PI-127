import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Seguimiento y Control")

# --- Carga de archivo ---
archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])
tab1, tab2, tab3, tab4 = st.tabs(["Inicio", "Diagrama Gantt", "Recursos", "Presupuesto"])

if archivo_excel:
    try:
        tareas_df_original = pd.read_excel(archivo_excel, sheet_name='Tareas')
        recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
        dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
    except:
        st.error("El archivo debe contener las hojas: Tareas, Recursos y Dependencias")
        st.stop()

    # --- Tab 1: Mostrar tablas originales ---
                with tab1:
                    st.markdown("#### Tablas importadas")
            
                    # Función para mostrar tabla editable
                    def mostrar_tabla(df, editable_cols=None, altura=300):
                        df_display = df.copy()
                        gb = GridOptionsBuilder.from_dataframe(df_display)
                        gb.configure_default_column(editable=False, resizable=True)
                        if editable_cols:
                            for col in editable_cols:
                                gb.configure_column(col, editable=True)
                        grid_options = gb.build()
                        custom_css = {
                            ".ag-header": {
                                "background-color": "#0D3B66",
                                "color": "white",
                                "font-weight": "bold",
                                "text-align": "center"
                            }
                        }
                        grid_response = AgGrid(
                            df_display,
                            gridOptions=grid_options,
                            update_mode=GridUpdateMode.MODEL_CHANGED,
                            custom_css=custom_css,
                            fit_columns_on_grid_load=True,
                            height=altura
                        )
                        return pd.DataFrame(grid_response['data'])
            
                    st.subheader("📋 Tareas")
                    tareas_df_original = mostrar_tabla(tareas_df_original, editable_cols=tareas_df_original.columns.tolist(), altura=400)
            
                    st.subheader("📋 Recursos")
                    recursos_df = mostrar_tabla(recursos_df, editable_cols=recursos_df.columns.tolist(), altura=300)
            
                    st.subheader("📋 Dependencias")
                    dependencias_df = mostrar_tabla(dependencias_df, editable_cols=dependencias_df.columns.tolist(), altura=300)
            
                # --- Normalización de variables ---
                tareas_df = tareas_df_original.copy()
                for col in ['FECHAINICIO','FECHAFIN']:
                    tareas_df[col] = pd.to_datetime(tareas_df[col], errors='coerce', dayfirst=True)
                tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.clip(lower=0)
                tareas_df['PREDECESORAS'] = tareas_df['PREDECESORAS'].fillna('').astype(str)
            
                if 'TARIFA' in recursos_df.columns:
                    recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)
            
                # --- Función para actualizar dependencias por RUTA_CRITICA ---
                def actualizar_dependencias_por_critica(tareas_df, columna_ruta='RUTA_CRITICA'):
                    if columna_ruta not in tareas_df.columns:
                        return tareas_df
                    tareas_df = tareas_df.copy()
            
                    # Inicializar prev_ruta_critica en session_state
                    if 'prev_ruta_critica' not in st.session_state:
                        st.session_state.prev_ruta_critica = tareas_df[columna_ruta].copy()
            
                    prev = st.session_state.prev_ruta_critica
                    curr = tareas_df[columna_ruta]
            
                    for idx, tarea_id in enumerate(tareas_df['IDRUBRO']):
                        fue_critica = prev.iloc[idx]
                        es_critica = curr.iloc[idx]
            
                        if not fue_critica and es_critica:
                            # No crítica -> crítica
                            for s_idx, fila in tareas_df.iterrows():
                                pre_list = str(fila['PREDECESORAS']).split(',')
                                pre_list = [p.strip() for p in pre_list]
                                if not any(str(tarea_id) in p for p in pre_list):
                                    pre_list.append(f"{tarea_id}FC")
                                    tareas_df.at[s_idx, 'PREDECESORAS'] = ', '.join(pre_list)
            
                        elif fue_critica and not es_critica:
                            # Crítica -> no crítica
                            for s_idx, fila in tareas_df.iterrows():
                                pre_list = str(fila['PREDECESORAS']).split(',')
                                pre_list = [p.strip() for p in pre_list if p != f"{tarea_id}FC"]
                                tareas_df.at[s_idx, 'PREDECESORAS'] = ', '.join(pre_list)
            
                    st.session_state.prev_ruta_critica = curr.copy()
                    return tareas_df
            
                # --- Función para calcular fechas según predecesoras ---
                def calcular_fechas(df):
                    df = df.copy()
                    df.columns = df.columns.str.strip()
                    inicio_rubro = df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
                    fin_rubro = df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
                    duracion_rubro = (df.set_index('IDRUBRO')['FECHAFIN'] - df.set_index('IDRUBRO')['FECHAINICIO']).dt.days.to_dict()
                    dependencias = defaultdict(list)
                    pre_count = defaultdict(int)
            
                    for _, row in df.iterrows():
                        tarea_id = row['IDRUBRO']
                        predecesoras_str = str(row['PREDECESORAS']).strip()
                        if predecesoras_str not in ['nan','']:
                            pre_list = predecesoras_str.split(',')
                            for pre in pre_list:
                                pre = pre.strip()
                                match = re.match(r'(\d+)', pre)
                                if match:
                                    pre_id = int(match.group(1))
                                    dependencias[pre_id].append(tarea_id)
                                    pre_count[tarea_id] += 1
            
                    queue = deque([tid for tid in df['IDRUBRO'] if pre_count[tid] == 0])
                    inicio_calc = inicio_rubro.copy()
                    fin_calc = fin_rubro.copy()
            
                    while queue:
                        tarea_id = queue.popleft()
                        row = df[df['IDRUBRO']==tarea_id].iloc[0]
                        duracion = duracion_rubro[tarea_id]
                        predecesoras_str = str(row['PREDECESORAS']).strip()
                        nueva_inicio = inicio_calc[tarea_id]
                        nueva_fin = fin_calc[tarea_id]
            
                        if predecesoras_str not in ['nan','']:
                            pre_list = predecesoras_str.split(',')
                            for pre in pre_list:
                                pre = pre.strip()
                                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre)
                                if match:
                                    pre_id = int(match.group(1))
                                    tipo = match.group(2).upper() if match.group(2) else 'FC'
                                    desfase = int(match.group(3)) if match.group(3) else 0
            
                                    if pre_id in inicio_calc and pre_id in fin_calc:
                                        inicio_pre = inicio_calc[pre_id]
                                        fin_pre = fin_calc[pre_id]
            
                                        if tipo == 'CC':
                                            nueva_inicio = inicio_pre + timedelta(days=desfase)
                                            nueva_fin = nueva_inicio + timedelta(days=duracion)
                                        elif tipo == 'FC':
                                            nueva_inicio = fin_pre + timedelta(days=desfase)
                                            nueva_fin = nueva_inicio + timedelta(days=duracion)
                                        elif tipo == 'CF':
                                            nueva_fin = inicio_pre + timedelta(days=desfase)
                                            nueva_inicio = nueva_fin - timedelta(days=duracion)
                                        elif tipo == 'FF':
                                            nueva_fin = fin_pre + timedelta(days=desfase)
                                            nueva_inicio = nueva_fin - timedelta(days=duracion)
            
                        inicio_calc[tarea_id] = nueva_inicio
                        fin_calc[tarea_id] = nueva_fin
            
                        for hijo in dependencias[tarea_id]:
                            pre_count[hijo] -= 1
                            if pre_count[hijo] == 0:
                                queue.append(hijo)
            
                    df['FECHAINICIO'] = df['IDRUBRO'].map(inicio_calc)
                    df['FECHAFIN'] = df['IDRUBRO'].map(fin_calc)
                    return df
            
                # --- Función para recalcular RUTA_CRITICA ---
                def calcular_ruta_critica(tareas_df):
                    df = tareas_df.copy()
                    df.columns = df.columns.str.strip()
                    es, ef, ls, lf, tf, ff = {}, {}, {}, {}, {}, {}
                    duracion_dict = df.set_index('IDRUBRO')['DURACION'].to_dict()
                    dependencias, predecesoras_map = defaultdict(list), defaultdict(list)
                    all_task_ids = set(df['IDRUBRO'].tolist())
            
                    for _, row in df.iterrows():
                        tarea_id = row['IDRUBRO']
                        predecesoras_str = str(row['PREDECESORAS']).strip()
                        if predecesoras_str not in ['nan', '']:
                            pre_list = predecesoras_str.split(',')
                            for pre_entry in pre_list:
                                pre_entry = pre_entry.strip()
                                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                                if match:
                                    pre_id = int(match.group(1))
                                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                                    desfase = int(match.group(3)) if match.group(3) else 0
                                    if pre_id in all_task_ids:
                                        dependencias[pre_id].append(tarea_id)
                                        predecesoras_map[tarea_id].append((pre_id, tipo_relacion, desfase))
            
                    # Pase hacia adelante (ES/EF)
                    in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids}
                    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
                    while queue:
                        u = queue.popleft()
                        es[u] = df.loc[df['IDRUBRO']==u,'FECHAINICIO'].values[0]
                        ef[u] = es[u] + timedelta(days=duracion_dict.get(u,0))
                        for v in dependencias.get(u,[]):
                            for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v,[]):
                                if pre_id_v==u:
                                    duration_v = duracion_dict.get(v,0)
                                    if tipo_v=='CC': start = es[u]+timedelta(days=desfase_v)
                                    elif tipo_v=='FC': start = ef[u]+timedelta(days=desfase_v)
                                    elif tipo_v=='CF': start = es[u]+timedelta(days=desfase_v)-timedelta(days=duration_v)
                                    elif tipo_v=='FF': start = ef[u]+timedelta(days=desfase_v)-timedelta(days=duration_v)
                                    else: start = ef[u]+timedelta(days=desfase_v)
                                    if v not in es or start>es[v]:
                                        es[v] = start
                                        ef[v] = es[v]+timedelta(days=duration_v)
                            in_degree[v]-=1
                            if in_degree[v]==0: queue.append(v)
            
                    # Pase hacia atrás y holguras
                    end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]
                    project_finish_date = max(ef.values())
                    ls, lf = {}, {}
                    for tid in end_tasks_ids:
                        lf[tid] = project_finish_date
                        ls[tid] = lf[tid]-timedelta(days=duracion_dict.get(tid,0))
            
                    # Holguras
                    tf, ff = {}, {}
                    successor_map = defaultdict(list)
                    for tid, pres in predecesoras_map.items():
                        for pre_id, tipo, lag in pres:
                            successor_map[pre_id].append((tid, tipo, lag))
                    for tid in all_task_ids:
                        tf[tid] = (lf.get(tid, pd.NA)-ef.get(tid, pd.NA)) if tid in lf and tid in ef else pd.NA
                        ff[tid] = pd.Timedelta(days=0)
            
                    df['FECHA_INICIO_TEMPRANA'] = df['IDRUBRO'].map(es)
                    df['FECHA_FIN_TEMPRANA'] = df['IDRUBRO'].map(ef)
                    df['FECHA_INICIO_TARDE'] = df['IDRUBRO'].map(ls)
                    df['FECHA_FIN_TARDE'] = df['IDRUBRO'].map(lf)
                    df['HOLGURA_TOTAL'] = df['IDRUBRO'].map(lambda x: tf.get(x, pd.NA).days if pd.notna(tf.get(x, pd.NA)) else pd.NA)
                    df['RUTA_CRITICA'] = df['HOLGURA_TOTAL'].apply(lambda x: x==0 if pd.notna(x) else False)
            
                    return df
            
                # --- Cálculo inicial de fechas y rutas críticas ---
                tareas_df = calcular_fechas(tareas_df)
                tareas_df = calcular_ruta_critica(tareas_df)
                st.session_state.tareas_df_work = tareas_df
            
                # --- Tab 2: Interacción con RUTA_CRITICA ---
                with tab2:
                    st.subheader("🧩 Control de Ruta Crítica")
            
                    tareas_df = st.session_state.get("tareas_df_work", tareas_df_original).copy()
                    df_preview = tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                                            'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                                            'FECHA_INICIO_TARDE','FECHA_FIN_TARDE',
                                            'DURACION','HOLGURA_TOTAL','RUTA_CRITICA']].copy()
            
                    gb = GridOptionsBuilder.from_dataframe(df_preview)
                    gb.configure_default_column(editable=False, resizable=True)
                    gb.configure_column("RUTA_CRITICA", editable=True)
                    grid_options = gb.build()
            
                    custom_css = {
                        ".ag-header": {
                            "background-color": "#0D3B66",
                            "color": "white",
                            "font-weight": "bold",
                            "text-align": "center"
                        }
                    }
            
                    grid_response = AgGrid(
                        df_preview,
                        gridOptions=grid_options,
                        update_mode=GridUpdateMode.MODEL_CHANGED,
                        custom_css=custom_css,
                        fit_columns_on_grid_load=True,
                        height=400
                    )
            
                    df_editado = pd.DataFrame(grid_response['data'])
            
                    # Detectar cambios en RUTA_CRITICA
                    if not df_editado['RUTA_CRITICA'].equals(df_preview['RUTA_CRITICA']):
                        st.success("🔄 Cambios detectados en RUTA_CRITICA. Recalculando...")
            
                        # Guardar versión antes y después
                        st.session_state.tareas_df_before_edit = tareas_df.copy()
                        st.session_state.tareas_df_after_edit = df_editado.copy()
            
                        tareas_df['RUTA_CRITICA'] = df_editado['RUTA_CRITICA']
            
                        # Recalcular dependencias y fechas
                        tareas_df = actualizar_dependencias_por_critica(tareas_df)
                        tareas_df = calcular_fechas(tareas_df)
                        tareas_df = calcular_ruta_critica(tareas_df)
            
                        st.session_state.tareas_df_work = tareas_df
                        st.experimental_rerun()
                         
              dependencias_df = dependencias_df.merge(recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
              dependencias_df['COSTO'] = dependencias_df['CANTIDAD'] * dependencias_df['TARIFA']
              costos_por_can = dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
              costos_por_can.rename(columns={'RUBRO': 'RUBRO', 'COSTO': 'COSTO_TOTAL'}, inplace=True)
              tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
              costos_por_can['RUBRO'] = costos_por_can['RUBRO'].str.strip()
              tareas_df = tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')
              
              import plotly.graph_objects as go
              import plotly.express as px
              from collections import defaultdict
              
              st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
              
              # Determinar columna de costo
              cost_column_name = None
              for col in ['COSTO_TOTAL_RUBRO', 'COSTO_TOTAL_x', 'COSTO_TOTAL']:
                     if col in tareas_df.columns:
                            cost_column_name = col
                            break
              
              if cost_column_name:
                     tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce').fillna(0)
              else:
                     st.warning("⚠️ No se encontró una columna de costos reconocida en el DataFrame. Se creará columna de costos en 0.")
                     tareas_df['COSTO_TOTAL_NUMERICO'] = 0
                     cost_column_name = 'COSTO_TOTAL_NUMERICO'
              
              if 'IDRUBRO' in tareas_df.columns:
                     tareas_df = tareas_df.sort_values(['IDRUBRO'])
              else:
                     st.warning("⚠️ Columna 'IDRUBRO' no encontrada para ordenar.")
              
              tareas_df['y_num'] = range(len(tareas_df))
              tareas_df['y_num_plot'] = tareas_df['y_num'] + 0.01

              dependencias = defaultdict(list)
              predecesoras_map_details = defaultdict(list)
              warnings_list = []
              
              for _, row in tareas_df.iterrows():
                  tarea_id = row['IDRUBRO']
                  predecesoras_str = str(row.get('PREDECESORAS', '')).strip()
                  if predecesoras_str not in ['nan', '']:
                      pre_list = predecesoras_str.split(',')
                      for pre_entry in pre_list:
                          pre_entry = pre_entry.strip()
                          match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                          if match:
                              pre_id = int(match.group(1))
                              tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                              desfase = int(match.group(3)) if match.group(3) else 0
                              if pre_id in tareas_df['IDRUBRO'].values:
                                  dependencias[pre_id].append(tarea_id)
                                  predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                              else:
                                  warnings_list.append(f"Predecesor ID {pre_id} para tarea {tarea_id} no encontrado.")
                          elif pre_entry != '':
                              warnings_list.append(f"Formato de predecesora '{pre_entry}' no reconocido para tarea {tarea_id}.")
              
              if warnings_list:
                  st.warning("⚠️ Advertencias detectadas:\n" + "\n".join(warnings_list))

              inicio_rubro_calc = tareas_df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
              fin_rubro_calc = tareas_df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
              is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()

              fig = go.Figure()
              shapes = []

              color_no_critica_barra = 'lightblue'
              color_critica_barra = 'rgb(255, 133, 133)'
              
              # Agregar barras
              for i, row in tareas_df.iterrows():
                  line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
                  line_width = 12
                  start_date = row['FECHAINICIO']
                  end_date = row['FECHAFIN']
                  if pd.isna(start_date) or pd.isna(end_date):
                      continue
              
                  valor_costo = float(row.get(cost_column_name, 0))
                  costo_formateado = f"S/ {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                  hover_text = (
                      f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
                      f"🗓️ <b>Capítulo:</b> {row.get('CAPÍTULO', '')}<br>"
                      f"📅 <b>Inicio:</b> {start_date.strftime('%d/%m/%Y')}<br>"
                      f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
                      f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
                      f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>"
                      f"💰 <b>Costo:</b> {costo_formateado}"
                  )
                  import numpy as np
                  x_vals = pd.date_range(start=start_date, end=end_date, freq='D')  # un punto por día
                  y_vals = np.full(len(x_vals), row['y_num_plot'])
                  fig.add_trace(go.Scatter(
                         x=x_vals,
                         y=y_vals,
                         mode='lines',
                         line=dict(color=line_color, width=line_width),
                         showlegend=False,
                         hoverinfo='text',
                         text=[hover_text]*len(x_vals),
                  ))
              
              # Función para dibujar flechas de dependencias
              def dibujar_flecha(pre_id, suc_id, tipo_relacion, offset=5):
                     y_pre = tareas_df.loc[tareas_df['IDRUBRO']==pre_id, 'y_num_plot'].values[0]
                     y_suc = tareas_df.loc[tareas_df['IDRUBRO']==suc_id, 'y_num_plot'].values[0]
                     pre_is_critical = is_critical_dict.get(pre_id, False)
                     suc_is_critical = is_critical_dict.get(suc_id, False)
                     arrow_color = 'red' if pre_is_critical and suc_is_critical else 'blue'
                     
                     x_pre_inicio = inicio_rubro_calc.get(pre_id)
                     x_pre_fin = fin_rubro_calc.get(pre_id)
                     x_suc_inicio = inicio_rubro_calc.get(suc_id)
                     x_suc_fin = fin_rubro_calc.get(suc_id)
                     
                     origin_x = x_pre_fin if tipo_relacion in ['FC', 'FF'] else x_pre_inicio
                     connection_x = x_suc_inicio if tipo_relacion in ['FC', 'CC'] else x_suc_fin
                     points_x, points_y = [origin_x], [y_pre]
                     
                     if tipo_relacion in ['CC', 'FC']:
                             elbow1_x, elbow1_y = origin_x - timedelta(days=offset), y_pre
                             elbow2_x, elbow2_y = elbow1_x, y_suc
                             points_x.extend([elbow1_x, elbow2_x, connection_x])
                             points_y.extend([elbow1_y, elbow2_y, y_suc])
                     elif tipo_relacion in ['CF', 'FF']:
                             elbow1_x, elbow1_y = origin_x, y_suc
                             points_x.extend([elbow1_x, connection_x])
                             points_y.extend([elbow1_y, y_suc])

                     arrow_symbol = 'triangle-right'  # valor por defecto
                     if tipo_relacion == 'CC':
                         arrow_symbol = 'triangle-right'
                     elif tipo_relacion == 'CF':
                         arrow_symbol = 'triangle-left'
                     elif tipo_relacion == 'FF':
                         arrow_symbol = 'triangle-left'

                     # Marcador en el predecesor (círculo)
                     fig.add_trace(go.Scattergl(
                         x=[origin_x],
                         y=[y_pre],
                         mode='markers',
                         marker=dict(symbol='circle', size=8, color=arrow_color),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                     
                     # Línea de la flecha (igual que antes)
                     fig.add_trace(go.Scatter(
                         x=points_x,
                         y=points_y,
                         mode='lines',
                         line=dict(color=arrow_color, width=1, dash='dash'),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                     
                     # Marcador en el sucesor (triángulo)
                     fig.add_trace(go.Scattergl(
                         x=[connection_x],
                         y=[y_suc],
                         mode='markers',
                         marker=dict(symbol=arrow_symbol, size=10, color=arrow_color),
                         hoverinfo='none',
                         showlegend=False,
                     ))
                                   
              # Dibujar todas las flechas
              for pre_id, sucesores in dependencias.items():
                  for suc_id in sucesores:
                      tipo_rel = 'FC'
                      for pre_tmp, type_tmp, _ in predecesoras_map_details.get(suc_id, []):
                          if pre_tmp == pre_id:
                              tipo_rel = type_tmp.upper() if type_tmp else 'FC'
                              break
                      dibujar_flecha(pre_id, suc_id, tipo_rel)
              
              # Preparar Y-ticks
              y_ticktext_styled = []
              for y_pos in range(len(tareas_df)):
                  row = tareas_df[tareas_df['y_num'] == y_pos]
                  if not row.empty:
                      rubro_text = row.iloc[0]['RUBRO']
                      y_ticktext_styled.append(rubro_text)
                  else:
                      y_ticktext_styled.append("")

              fecha_min = tareas_df['FECHAINICIO'].min()
              fecha_max = tareas_df['FECHAFIN'].max()
              years = list(range(fecha_min.year, fecha_max.year + 1))
              colors = ['rgba(200,200,200,0.2)', 'rgba(100,100,100,0.2)']  # gris claro y blanco huevo
              shapes_years = []
              
              for i, year in enumerate(years):
                  shapes_years.append(
                      dict(
                          type='rect',
                          xref='x',
                          yref='paper',  # cubre todo el eje Y
                          x0=pd.Timestamp(f'{year}-01-01'),
                          x1=pd.Timestamp(f'{year}-12-31'),
                          y0=0,
                          y1=1,
                          fillcolor=colors[i % 2],
                          opacity=0.5,
                          layer='below',
                          line_width=0,
                      )
                  )

              shapes = shapes + shapes_years
              
              # Layout
              fig.update_layout(
                  xaxis=dict(
                      title='Fechas',
                      side='bottom',
                      dtick='M1',
                      tickangle=-90,
                      showgrid=True,
                      gridcolor='rgba(128,128,128,0.3)',
                      gridwidth=0.5
                  ),
                     
                  yaxis_title='Rubro',
                  yaxis=dict(
                      autorange='reversed',
                      tickvals=tareas_df['y_num_plot'],
                      ticktext=y_ticktext_styled,
                      tickfont=dict(size=10),
                      showgrid=False
                  ),
                  shapes=shapes,
                  height=max(600, len(tareas_df)*25),
                  showlegend=False,
                  plot_bgcolor='white',
                  hovermode='closest'
              )

              fig.update_xaxes(
                  side="top",
                  overlaying="x",
                  dtick='M1',
                  tickangle=-90,
                  showgrid=True,
                  gridcolor='rgba(128,128,128,0.3)',
                  gridwidth=0.5
              )
              
              st.plotly_chart(fig, use_container_width=True)
    
       tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'])
       tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'])

       tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
       dependencias_df['RUBRO'] = dependencias_df['RUBRO'].str.strip()
        
       recursos_tareas_df = dependencias_df.merge(
              tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
              left_on='RUBRO',
              right_on='RUBRO',
              how='left'
       )

       daily_resource_usage_list = []

       for index, row in recursos_tareas_df.iterrows():
              task_id = row['IDRUBRO']
              resource_name = row['RECURSO']
              unit = row['UNIDAD']
              total_quantity = row['CANTIDAD']
              start_date = row['FECHAINICIO']
              end_date = row['FECHAFIN']
              duration_days = row['DURACION']

              if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
                     st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea ID {task_id}, recurso '{resource_name}'. Saltando.")
                     continue
    
              if duration_days <= 0:
                   daily_quantity = total_quantity
                   date_range = [start_date]
              else:
                   daily_quantity = total_quantity / (duration_days + 1)
                   date_range = pd.date_range(start=start_date, end=end_date, freq='D')
       
              temp_df = pd.DataFrame({
                   'Fecha': date_range,
                   'IDRUBRO': task_id,
                   'RECURSO': resource_name,
                   'UNIDAD': unit,
                   'Cantidad_Diaria': daily_quantity,
                   'Cantidad_Total_Tarea': total_quantity
               })
               
              daily_resource_usage_list.append(temp_df)

       if daily_resource_usage_list:
              all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
       else:
              st.warning("\nNo se generaron datos de uso diario de recursos.")
              all_daily_resource_usage_df = pd.DataFrame()
        
       daily_resource_demand_df = all_daily_resource_usage_df.groupby(
              ['Fecha', 'RECURSO', 'UNIDAD'],
              as_index=False
       )['Cantidad_Diaria'].sum()

       daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)
       daily_resource_demand_df['RECURSO'] = daily_resource_demand_df['RECURSO'].str.strip()
       recursos_df['RECURSO'] = recursos_df['RECURSO'].str.strip()
    
       resource_demand_with_details_df = daily_resource_demand_df.merge(
              recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
              on='RECURSO',
              how='left'
       )

       resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']

       daily_cost_by_type_df = resource_demand_with_details_df.groupby(
              ['Fecha', 'TYPE'],
              as_index=False
       )['Costo_Diario'].sum()

       daily_demand_by_resource_df = resource_demand_with_details_df.groupby(
              ['Fecha', 'RECURSO', 'UNIDAD'],
              as_index=False
       )['Demanda_Diaria_Total'].sum()

       with tab3:

           import pandas as pd
           import plotly.graph_objects as go
           import plotly.express as px
           import re
           from datetime import timedelta, datetime
           from collections import defaultdict
           st.subheader("📊 Distribución de Recursos")
       
           if 'RUBRO' not in recursos_tareas_df.columns:
       
               if 'tareas_df' in locals() or 'tareas_df' in globals():
       
                   tareas_df['RUBRO'] = tareas_df['RUBRO'].astype(str).str.strip()
                   recursos_tareas_df['RUBRO'] = recursos_tareas_df['RUBRO'].astype(str).str.strip()
           
                   if 'IDRUBRO' in recursos_tareas_df.columns and 'IDRUBRO' in tareas_df.columns:
                        recursos_tareas_df = recursos_tareas_df.merge(
                           tareas_df[['IDRUBRO', 'RUBRO']],
                           left_on='IDRUBRO',
                           right_on='IDRUBRO',
                           how='left'
                       )
                        st.warning("Re-merged to include 'RUBRO' column using IDRUBRO.")
                   else:
                        st.warning("❌ Error: 'IDRUBRO' column not found in one of the dataframes. Cannot re-add 'RUBRO'.")
                        raise KeyError("'IDRUBRO' column not found for re-merging.")
           
               else:
                   st.warning("❌ Error: 'tareas_df' not found. Cannot re-add 'RUBRO' column.")
                   raise NameError("'tareas_df' not found.")
           
           unique_rubros = sorted(recursos_tareas_df['RUBRO'].dropna().unique().tolist())
       
           fig_resource_timeline = go.Figure()
       
           pastel_blue = 'rgb(174, 198, 207)' 
       
           for i, row in recursos_tareas_df.iterrows():
               fig_resource_timeline.add_trace(go.Scattergl(
                   x=[row['FECHAINICIO'], row['FECHAFIN']],
                   y=[row['RECURSO'], row['RECURSO']],
                   mode='lines',
                   line=dict(color=pastel_blue, width=10), 
                   name=row['RECURSO'], 
                   showlegend=False, 
                   hoverinfo='text',
                   text=f"<b>Rubro:</b> {row['RUBRO']}<br><b>Recurso:</b> {row['RECURSO']}<br><b>Inicio:</b> {row['FECHAINICIO'].strftime('%Y-%m-%d')}<br><b>Fin:</b> {row['FECHAFIN'].strftime('%Y-%m-%d')}",
                   customdata=[row['RUBRO']] 
               ))
       
           dropdown_options = [{'label': 'All Tasks', 'method': 'update', 'args': [{'visible': [True] * len(fig_resource_timeline.data)}, {'title': 'Línea de Tiempo de Uso de Recursos'}]}]
       
           for rubro in unique_rubros:
       
               visibility = [trace.customdata[0] == rubro for trace in fig_resource_timeline.data if trace.customdata and trace.customdata[0] in unique_rubros]
       
               visibility_list = [trace.customdata[0] == rubro if trace.customdata and len(trace.customdata) > 0 else False for trace in fig_resource_timeline.data]
           
               dropdown_options.append({
                   'label': rubro,
                   'method': 'update',
                   'args': [{'visible': visibility_list}, {'title': f'Línea de Tiempo de Uso de Recursos (Filtrado por: {rubro})'}]
               })
       
           fig_resource_timeline.update_layout(
               updatemenus=[
                   go.layout.Updatemenu(
                       buttons=dropdown_options,
                       direction="down",
                       pad={"r": 10, "t": 10},
                       showactive=True,
                       x=0.01,
                       xanchor="left",
                       y=1.1,
                       yanchor="top"
                   ),
               ],
               yaxis=dict(
                   autorange="reversed",
                   title="Recurso",
                   tickfont=dict(size=10) 
               ),
               xaxis=dict(
                   title='Fechas',
                   side='bottom',
                   dtick='M1', 
                   tickangle=-90, 
                   showgrid=True,
                   gridcolor='rgba(128,128,128,0.3)',
                   gridwidth=0.5
               ),
               xaxis2=dict(
                   title='Fechas',
                   overlaying='x',
                   side='top',
                   dtick='M1',
                   tickangle=90,
                   showgrid=True,
                   gridcolor='rgba(128,128,128,0.3)',
                   gridwidth=0.5
               ),
               height=max(600, len(recursos_tareas_df['RECURSO'].unique()) * 20), 
               showlegend=False,
               plot_bgcolor='white',
               hovermode='closest'
           )
           st.plotly_chart(fig_resource_timeline, use_container_width=True)
    #__________________________________________________________________________________________________

       if 'resource_demand_with_details_df' in locals() or 'resource_demand_with_details_df' in globals():
              required_columns_and_types = {
                     'Fecha': 'datetime64[ns]',
                     'RECURSO': 'object', 
                     'UNIDAD': 'object', 
                     'Demanda_Diaria_Total': 'float64', 
                     'TYPE': 'object',
                     'TARIFA': 'float64', 
                     'Costo_Diario': 'float64' 
              }
              missing_columns = [col for col in required_columns_and_types if col not in resource_demand_with_details_df.columns]
              if not missing_columns:
                     type_issues = []
                     for col, expected_type in required_columns_and_types.items():
                            if expected_type == 'object':
                                   if pd.api.types.is_numeric_dtype(resource_demand_with_details_df[col]):
                                          type_issues.append(f"Column '{col}' is numeric but expected object (string).")
                            elif not pd.api.types.is_dtype_equal(resource_demand_with_details_df[col].dtype, expected_type):
                                   if expected_type == 'float64' and pd.api.types.is_integer_dtype(resource_demand_with_details_df[col]):
                                          pass 
                                   else:
                                          type_issues.append(f"Column '{col}' has type {resource_demand_with_details_df[col].dtype} but expected {expected_type}.")
              else:
                     st.warning(f"❌ Error: Missing required columns in resource_demand_with_details_df: {missing_columns}")
       else:
              st.warning("❌ Error: DataFrame 'resource_demand_with_details_df' not found.")

       resource_demand_with_details_df['Fecha'] = pd.to_datetime(resource_demand_with_details_df['Fecha'])
       resource_demand_with_details_df['Periodo_Mensual'] = resource_demand_with_details_df['Fecha'].dt.to_period('M')
       monthly_costs_df = resource_demand_with_details_df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
       monthly_costs_df['Periodo_Mensual'] = monthly_costs_df['Periodo_Mensual'].astype(str) 
       monthly_costs_df['Costo_Acumulado'] = monthly_costs_df['Costo_Diario'].cumsum()
     
       def format_currency(value):
              if pd.notna(value):
                     return f"S/ {value:,.2f}"  
              return "S/ 0.00"  
       
       monthly_costs_df['Costo_Mensual_Formateado'] = monthly_costs_df['Costo_Diario'].apply(format_currency)
       monthly_costs_df['Costo_Acumulado_Formateado'] = monthly_costs_df['Costo_Acumulado'].apply(format_currency)

       with tab4:
           from plotly.subplots import make_subplots
           import plotly.graph_objects as go
           
           st.subheader("📊 Cronograma Valorado")
           
           fig = make_subplots(specs=[[{"secondary_y": True}]])
           
           # Barra de Costo Mensual
           fig.add_bar(
               x=monthly_costs_df['Periodo_Mensual'],
               y=monthly_costs_df['Costo_Diario'],
               name='Costo Mensual',
               text=monthly_costs_df['Costo_Mensual_Formateado'],
               hoverinfo='text',
               hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
               secondary_y=False
           )
           
           # Línea de Costo Acumulado
           fig.add_scatter(
               x=monthly_costs_df['Periodo_Mensual'],
               y=monthly_costs_df['Costo_Acumulado'],
               mode='lines+markers',
               name='Costo Acumulado',
               text=monthly_costs_df['Costo_Acumulado_Formateado'],
               hoverinfo='text',
               hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
               line=dict(color='red'),
               secondary_y=True
           )
           
           # Configuración de ejes
           fig.update_yaxes(
               title_text="Costo Mensual",
               secondary_y=False,
               showgrid=False,  # ❌ Sin grilla horizontal para la barra
               range=[0, monthly_costs_df['Costo_Diario'].max()*1.1]
           )
           fig.update_yaxes(
               title_text="Costo Acumulado",
               secondary_y=True,
               showgrid=True,   # ✅ Solo el acumulador tiene grilla
               gridcolor='lightgrey',
               range=[0, monthly_costs_df['Costo_Acumulado'].max()*1.1]
           )
           fig.update_xaxes(title_text="Período Mensual", tickangle=-45)
           
           # Layout
           fig.update_layout(
               hovermode='x unified',
               height=600,
               legend=dict(
                   x=1.1,
                   y=1,
                   bgcolor='rgba(255, 255, 255, 0.5)',
                   bordercolor='rgba(0, 0, 0, 0.5)'
               ),
               plot_bgcolor='white'
           )
           
           st.plotly_chart(fig, use_container_width=True)


else:
       st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")















































