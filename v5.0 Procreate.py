# Declarar e importar bibliotecas___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import math
import plotly.express as px

# Definir archivo y pestañas___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Seguimiento y Control")

archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])
tab1, tab2, tab3, tab4 = st.tabs(["Inicio", "Diagrama Gantt", "Recursos", "Presupuesto"])

# Definir funciones de calculo___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
##1
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
                            else:
                                st.warning(f"⚠️ Tipo de relación '{tipo}' no reconocido en '{pre}' para tarea {tarea_id}") 

            inicio_calc[tarea_id] = nueva_inicio
            fin_calc[tarea_id] = nueva_fin

            for hijo in dependencias[tarea_id]:
                pre_count[hijo] -= 1
                if pre_count[hijo] == 0:
                    queue.append(hijo)

        df['FECHAINICIO'] = df['IDRUBRO'].map(inicio_calc)
        df['FECHAFIN'] = df['IDRUBRO'].map(fin_calc)
    
        return df

##2
def calculo_ruta_critica(tareas_df=None, archivo=None):
    try:
        if tareas_df is None:
            if archivo is not None:
                tareas_df = pd.read_excel(archivo, sheet_name='Tareas')
            else:
                raise ValueError("No se proporcionó ni tareas_df ni archivo para cargarlo.")
        for col in ['FECHAINICIO', 'FECHAFIN']:
            if col not in tareas_df.columns:
                raise KeyError(f"Columna {col} no encontrada en tareas_df")
            if not pd.api.types.is_datetime64_any_dtype(tareas_df[col]):
                tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True)
    except Exception as e:
        st.warning(f"Error cargando o procesando fechas: {e}")
        raise e
    tareas_df.columns = tareas_df.columns.str.strip()
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())
    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row['PREDECESORAS']).strip()
        if predecesoras_str not in ['nan', '']:
            for pre_entry in predecesoras_str.split(','):
                pre_entry = pre_entry.strip()
                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                if match:
                    pre_id = int(match.group(1))
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                    desfase = int(match.group(3)) if match.group(3) else 0
                    if pre_id in all_task_ids:
                        dependencias[pre_id].append(tarea_id)
                        predecesoras_map[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                        st.warning(f"Predecesor {pre_id} de tarea {tarea_id} no encontrado. Ignorado.")
                elif pre_entry != '':
                    st.warning(f"Formato de predecesora '{pre_entry}' no reconocido para tarea {tarea_id}.")
    es, ef, ls, lf, tf, ff = {}, {}, {}, {}, {}, {}
    initial_tasks_ids = [tid for tid in all_task_ids if tid not in predecesoras_map]
    queue = deque(initial_tasks_ids)
    processed_forward = set(queue)
    in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids}
    for tid in queue:
        row = tareas_df[tareas_df['IDRUBRO']==tid]
        if not row.empty and pd.notna(row.iloc[0]['FECHAINICIO']):
            es[tid] = row.iloc[0]['FECHAINICIO']
            ef[tid] = es[tid] + timedelta(days=duracion_dict.get(tid,0))
        else:
            st.warning(f"Tarea inicial {tid} inválida para ES/EF.")
    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
    processed_forward = set(queue)
    while queue:
        u = queue.popleft()
        for v in dependencias.get(u, []):
            for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v, []):
                if pre_id_v == u and u in ef and u in es:
                    duration_v = duracion_dict.get(v,0)
                    if tipo_v == 'CC': potential_es_v = es[u]+timedelta(days=desfase_v)
                    elif tipo_v == 'FC': potential_es_v = ef[u]+timedelta(days=desfase_v)
                    elif tipo_v == 'CF': potential_es_v = (es[u]+timedelta(days=desfase_v)) - timedelta(days=duration_v)
                    elif tipo_v == 'FF': potential_es_v = (ef[u]+timedelta(days=desfase_v)) - timedelta(days=duration_v)
                    else: potential_es_v = ef[u]+timedelta(days=desfase_v)
                    if v not in es or potential_es_v>es[v]: es[v]=potential_es_v
                    ef[v]=es[v]+timedelta(days=duration_v)
            in_degree[v]-=1
            if in_degree[v]==0 and v not in processed_forward:
                queue.append(v)
                processed_forward.add(v)
    project_finish_date = max(ef.values()) if ef else None
    if not project_finish_date:
        raise ValueError("No EF calculado. No se puede determinar fin del proyecto.")
    end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]
    for tid in end_tasks_ids:
        lf[tid]=project_finish_date
        ls[tid]=lf[tid]-timedelta(days=duracion_dict.get(tid,0))
    queue_backward = deque(end_tasks_ids)
    processed_backward = set(end_tasks_ids)
    while queue_backward:
        v = queue_backward.popleft()
        for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
            if v in ls and v in lf:
                dur_u = duracion_dict.get(u,0)
                if tipo_relacion_uv=='CC': potential_lf_u = (ls[v]-timedelta(days=desfase_uv))+timedelta(days=dur_u)
                elif tipo_relacion_uv=='FC': potential_lf_u = ls[v]-timedelta(days=desfase_uv)
                elif tipo_relacion_uv=='CF': potential_lf_u = (lf[v]-timedelta(days=desfase_uv))+timedelta(days=dur_u)
                elif tipo_relacion_uv=='FF': potential_lf_u = lf[v]-timedelta(days=desfase_uv)
                else: potential_lf_u = ls[v]-timedelta(days=desfase_uv)
                if u not in lf or potential_lf_u<lf[u]:
                    lf[u]=potential_lf_u
                    ls[u]=lf[u]-timedelta(days=dur_u)
    for tid in all_task_ids:
        if tid in ef and tid in lf:
            tf[tid]=lf[tid]-ef[tid]
            ff[tid]=timedelta(days=0)
        else:
            tf[tid]=pd.NA
            ff[tid]=pd.NA
    tareas_df['FECHA_INICIO_TEMPRANA'] = tareas_df['IDRUBRO'].map(es)
    tareas_df['FECHA_FIN_TEMPRANA'] = tareas_df['IDRUBRO'].map(ef)
    tareas_df['FECHA_INICIO_TARDE'] = tareas_df['IDRUBRO'].map(ls)
    tareas_df['FECHA_FIN_TARDE'] = tareas_df['IDRUBRO'].map(lf)
    tareas_df['HOLGURA_TOTAL_TD'] = tareas_df['IDRUBRO'].map(tf)
    tareas_df['HOLGURA_LIBRE_TD'] = tareas_df['IDRUBRO'].map(ff)
    tareas_df['HOLGURA_TOTAL'] = tareas_df['HOLGURA_TOTAL_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
    tareas_df['HOLGURA_LIBRE'] = tareas_df['HOLGURA_LIBRE_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
    tolerance_days = 1e-9
    tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL'].apply(lambda x: abs(x)<tolerance_days if pd.notna(x) else False)
    return tareas_df

# Definicion de variables y calculo___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

if archivo_excel:
    if 'archivo_hash' not in st.session_state or st.session_state.archivo_hash != hash(archivo_excel.getvalue()):
        st.session_state.archivo_hash = hash(archivo_excel.getvalue())
        try:
            st.session_state.tareas_df_original = pd.read_excel(archivo_excel, sheet_name='Tareas')
            st.session_state.recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
            st.session_state.dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
            st.session_state.tareas_df = st.session_state.tareas_df_original.copy()

            if 'TARIFA' in st.session_state.recursos_df.columns:
                    st.session_state.recursos_df['TARIFA'] = pd.to_numeric(st.session_state.recursos_df['TARIFA'], errors='coerce').fillna(0)

            for col in ['FECHAINICIO','FECHAFIN']:
                st.session_state.tareas_df[col] = pd.to_datetime(st.session_state.tareas_df[col], errors='coerce')
                st.session_state.tareas_df[col] = st.session_state.tareas_df[col].dt.strftime('%d/%m/%Y')
        
            for col in ['FECHAINICIO','FECHAFIN']:
                st.session_state.tareas_df[col] = pd.to_datetime(st.session_state.tareas_df[col], dayfirst=True, errors='coerce')

            st.session_state.tareas_df['DURACION'] = (st.session_state.tareas_df['FECHAFIN'] - st.session_state.tareas_df['FECHAINICIO']).dt.days
            st.session_state.tareas_df.loc[st.session_state.tareas_df['DURACION'] < 0, 'DURACION'] = 0  # prevenir negativos
            st.session_state.tareas_df['PREDECESORAS'] = st.session_state.tareas_df['PREDECESORAS'].fillna('').astype(str)

            tareas_df=calcular_fechas(tareas_df)
            tareas_df=calculo_ruta_critica(tareas_df)
            
        except:
            st.error(f"Error al leer el archivo Excel. Asegúrese de que contiene las hojas 'Tareas', 'Recursos' y 'Dependencias' y que el formato es correcto: {e}")
            st.stop()

        with tab1:
            st.markdown("#### Datos Importados:")

            st.subheader("📋 Tabla Tareas")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.tareas_df_original)
            gb.configure_default_column(editable=True)
            grid_options = gb.build()
            custom_css = {
                         ".ag-header": {  # clase del header completo
                         "background-color": "#0D3B66",  # azul oscuro
                         "color": "white",               # texto blanco
                         "font-weight": "bold",
                         "text-align": "center"
                         }
            }
            tareas_df_original_grid_response = AgGrid(st.session_state.tareas_df_original, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css, key='tareasoriginal_grid_tab1')
            st.session_state.tareas_df_original = pd.DataFrame(tareas_df_original_grid_response['data'])
                
            st.subheader("📋 Tabla Recursos")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.recursos_df)
            gb.configure_default_column(editable=True)
            grid_options = gb.build()
            custom_css = {
                   ".ag-header": {
                   "background-color": "#0D3B66",
                   "color": "white",
                   "font-weight": "bold",
                   "text-align": "center"
                   }
            }
            recursos_grid_response = AgGrid(st.session_state.recursos_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css, key='recursos_grid_tab1') # Add a unique key
            st.session_state.recursos_df = pd.DataFrame(recursos_grid_response['data'])
    
            st.subheader("📋 Tabla Dependencias")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.dependencias_df)
            gb.configure_default_column(editable=True)
            gb.configure_column(
                   "CANTIDAD",
                   type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                   precision=2,
                   editable=True
            )
            grid_options = gb.build()
            custom_css = {
                   ".ag-header": {
                   "background-color": "#0D3B66",
                   "color": "white",
                   "font-weight": "bold",
                   "text-align": "center"
                   }
            }
            dependencias_grid_response = AgGrid(st.session_state.dependencias_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED,custom_css=custom_css, key='dependencias_grid_tab1') # Add a unique key
            st.session_state.dependencias_df = pd.DataFrame(dependencias_grid_response['data'])


    tareas_df = calcular_fechas(tareas_df)

    # _________________________________________________________________________________________________
    st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
    st.dataframe(tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                            'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                            'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','RUTA_CRITICA']])
    dependencias_df = dependencias_df.merge(recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
    dependencias_df['COSTO'] = dependencias_df['CANTIDAD'] * dependencias_df['TARIFA']
    costos_por_can = dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
    costos_por_can.rename(columns={'RUBRO': 'RUBRO', 'COSTO': 'COSTO_TOTAL'}, inplace=True)
    tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
    costos_por_can['RUBRO'] = costos_por_can['RUBRO'].str.strip()
    tareas_df = tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')
    

    st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
    cost_column_name = None
    if 'COSTO_TOTAL_RUBRO' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL_RUBRO'
    elif 'COSTO_TOTAL_x' in tareas_df.columns:
         cost_column_name = 'COSTO_TOTAL_x'
    elif 'COSTO_TOTAL' in tareas_df.columns: 
         cost_column_name = 'COSTO_TOTAL'
    if cost_column_name:
        tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce')
        tareas_df[cost_column_name] = tareas_df[cost_column_name].fillna(0)
    else:
        st.warning("⚠️ Advertencia: No se encontró una columna de costos reconocida en el DataFrame.")
        tareas_df['COSTO_TOTAL_NUMERICO'] = 0
        cost_column_name = 'COSTO_TOTAL_NUMERICO'
    if 'IDRUBRO' in tareas_df.columns:
        tareas_df = tareas_df.sort_values(['IDRUBRO'])
    else:
        st.warning("⚠️ Advertencia: Columna 'IDRUBRO' no encontrada para ordenar.")

    tareas_df['y_num'] = range(len(tareas_df))

    fig = go.Figure()

    fecha_inicio_col = 'FECHAINICIO'
    fecha_fin_col = 'FECHAFIN'
    if fecha_inicio_col not in tareas_df.columns or fecha_fin_col not in tareas_df.columns:
         st.warning("❌ Error: No se encontraron columnas de fechas de inicio/fin necesarias para dibujar el Gantt.")
    
    inicio_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_inicio_col].to_dict()
    fin_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_fin_col].to_dict()
    is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()
    dependencias = defaultdict(list)
    predecesoras_map_details = defaultdict(list)
    
    for _, row in tareas_df.iterrows():
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

                    if pre_id in tareas_df['IDRUBRO'].values:
                         dependencias[pre_id].append(tarea_id)
                         predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                         st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                else:
                    if pre_entry != '': 
                        st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")

    shapes = []
    color_banda = 'rgba(220, 220, 220, 0.6)'

    for y_pos in range(len(tareas_df)):
        if y_pos % 2 == 0: 
            shapes.append(
                dict(
                    type="rect",
                    xref="paper",
                    yref="y",    
                    x0=0,        
                    x1=1,    
                    y0=y_pos - 0.5, 
                    y1=y_pos + 0.5, 
                    fillcolor=color_banda,
                    layer="below", 
                    line_width=0,
                )
            )
    color_no_critica_barra = 'lightblue' 
    color_critica_barra = 'rgb(255, 133, 133)'
    
    for i, row in tareas_df.iterrows():
        line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
        line_width = 12 
        start_date = row[fecha_inicio_col]
        end_date = row[fecha_fin_col]

        if pd.isna(start_date) or pd.isna(end_date):
             st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea {row['RUBRO']} (ID {row['IDRUBRO']}). No se dibujará la barra.")
             continue

        try:
            valor_costo = float(row.get(cost_column_name, 0))
            costo_formateado = f"S/ {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            costo_formateado = "S/ 0,00"

        hover_text = (
            f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
            f"🗓️ <b>Capítulo:</b> {row['CAPÍTULO']}<br>"
            f"📅 <b>Inicio</b> {start_date.strftime('%d/%m/%Y')}<br>"
            f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
            f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
            f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>"
            f"💰 <b>Costo:</b> {costo_formateado}"
        )
    
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[row['y_num'], row['y_num']],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            showlegend=False, 
            hoverinfo='text',
            text=hover_text,
        ))

    offset_days_horizontal = 5 

    color_no_critica_flecha = 'blue'
    color_critica_flecha = 'red'

    for pre_id, sucesores in dependencias.items():
        pre_row_df = tareas_df[tareas_df['IDRUBRO'] == pre_id]
        if pre_row_df.empty:
            st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flechas desde este ID.")
            continue
        y_pre = pre_row_df.iloc[0]['y_num']
        pre_is_critical = is_critical_dict.get(pre_id, False)
        x_pre_inicio = inicio_rubro_calc.get(pre_id)
        x_pre_fin = fin_rubro_calc.get(pre_id)

        if pd.isna(x_pre_inicio) or pd.isna(x_pre_fin):
             st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el predecesor ID {pre_id}. No se dibujarán flechas desde este ID.")
             continue
    
        for suc_id in sucesores:
            suc_row_df = tareas_df[tareas_df['IDRUBRO'] == suc_id]
            if suc_row_df.empty:
                st.warning(f"⚠️ Advertencia: Sucesor ID {suc_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flecha hacia este ID.")
                continue
            y_suc = suc_row_df.iloc[0]['y_num']
            suc_is_critical = is_critical_dict.get(suc_id, False)
            arrow_color = color_critica_flecha if pre_is_critical and suc_is_critical else color_no_critica_flecha
            line_style = dict(color=arrow_color, width=1, dash='dash')
            x_suc_inicio = inicio_rubro_calc.get(suc_id)
            x_suc_fin = fin_rubro_calc.get(suc_id)
            if pd.isna(x_suc_inicio) or pd.isna(x_suc_fin):
                 st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el sucesor ID {suc_id}. No se dibujará la flecha.")
                 continue
    
            tipo_relacion = 'FC' 
            for pre_id_suc, type_suc, desfase_suc in predecesoras_map_details.get(suc_id, []):
                if pre_id_suc == pre_id:
                     tipo_relacion = type_suc.upper() if type_suc else 'FC'
                     break 

            origin_x = x_pre_fin
            if tipo_relacion == 'CC':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'CF':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'FF':
                origin_x = x_pre_fin

            connection_x = x_suc_inicio 
            arrow_symbol = 'triangle-right' 
    
            if tipo_relacion == 'CC':
                connection_x = x_suc_inicio
                arrow_symbol = 'triangle-right'
            elif tipo_relacion == 'CF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left'
            elif tipo_relacion == 'FF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left' 

            fig.add_trace(go.Scattergl(
                x=[origin_x],
                y=[y_pre],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=8, 
                    color=arrow_color, 
                ),
                hoverinfo='none',
                showlegend=False,
            ))

            line_style = dict(color=arrow_color, width=1, dash='dash') 
            points_x = [origin_x]
            points_y = [y_pre]

            if tipo_relacion in ['CC', 'FC']:
                elbow1_x = origin_x - timedelta(days=offset_days_horizontal)
                elbow1_y = y_pre
                elbow2_x = elbow1_x 
                elbow2_y = y_suc 
                points_x.append(elbow1_x)
                points_y.append(elbow1_y)
                points_x.append(elbow2_x)
                points_y.append(elbow2_y)
                points_x.append(connection_x)
                points_y.append(y_suc)

            elif tipo_relacion in ['CF', 'FF']:
                 elbow1_x = origin_x
                 elbow1_y = y_suc 
                 points_x.append(elbow1_x)
                 points_y.append(elbow1_y)
                 points_x.append(connection_x)
                 points_y.append(y_suc)
    
            else:
                 st.warning(f"⚠️ Tipo de relación '{tipo_relacion}' no reconocido para dibujar segmentos de flecha entre ID {pre_id} y ID {suc_id}. Saltando flecha.")
                 continue
    
            fig.add_trace(go.Scatter(
                x=points_x,
                y=points_y,
                mode='lines',
                line=line_style,
                hoverinfo='none',
                showlegend=False,
            ))
    
            fig.add_trace(go.Scattergl(
                x=[connection_x],
                y=[y_suc],
                mode='markers',
                marker=dict(
                    symbol=arrow_symbol, 
                    size=10, 
                    color=arrow_color,
                ),
                hoverinfo='none',
                showlegend=False,
            ))
    
    y_ticktext_styled = []
    for y_pos in range(len(tareas_df)):
        row_for_y_pos = tareas_df[tareas_df['y_num'] == y_pos]
        if not row_for_y_pos.empty:
            rubro_text = row_for_y_pos.iloc[0]['RUBRO']
            y_ticktext_styled.append(f"<b>{rubro_text}</b>" if y_pos % 2 == 0 else rubro_text)    
        else:
             y_ticktext_styled.append("")
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
        yaxis_title='Rubro',
        yaxis=dict(
            autorange='reversed',
            tickvals=tareas_df['y_num'],
            ticktext=y_ticktext_styled, 
            tickfont=dict(size=10),
            showgrid=False
        ),
        shapes=shapes,
        height=max(600, len(tareas_df) * 25), 
        showlegend=False, 
        plot_bgcolor='white',
        hovermode='closest'
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


































































