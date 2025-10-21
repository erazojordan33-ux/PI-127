import streamlit as st
import pandas as pd
import re
from datetime import timedelta, datetime
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Cronograma Valorado y Recursos")

archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])

if archivo_excel:
    try:
        tareas_df = pd.read_excel(archivo_excel, sheet_name='Tareas')
        recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
        dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
    except Exception as e:
        st.error("El archivo debe contener las hojas: Tareas, Recursos y Dependencias")
        st.stop()

    st.subheader("📋 Tabla Tareas")
    gb = GridOptionsBuilder.from_dataframe(tareas_df)
    gb.configure_default_column(editable=True)
    tareas_grid = AgGrid(tareas_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    tareas_df = pd.DataFrame(tareas_grid['data'])

    st.subheader("📋 Tabla Recursos")
    gb = GridOptionsBuilder.from_dataframe(recursos_df)
    gb.configure_default_column(editable=True)
    recursos_grid = AgGrid(recursos_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    recursos_df = pd.DataFrame(recursos_grid['data'])

    st.subheader("📋 Tabla Dependencias")
    gb = GridOptionsBuilder.from_dataframe(dependencias_df)
    gb.configure_default_column(editable=True)
    dependencias_grid = AgGrid(dependencias_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    dependencias_df = pd.DataFrame(dependencias_grid['data'])

    for col in ['FECHAINICIO', 'FECHAFIN']:
        if col in tareas_df.columns:
            tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce')

    if 'FECHAINICIO' in tareas_df.columns and 'FECHAFIN' in tareas_df.columns:
        tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)
        tareas_df.loc[tareas_df['DURACION'] < 0, 'DURACION'] = 0
    else:
        tareas_df['DURACION'] = 0

    if 'PREDECESORAS' in tareas_df.columns:
        tareas_df['PREDECESORAS'] = tareas_df['PREDECESORAS'].fillna('').astype(str)
    else:
        tareas_df['PREDECESORAS'] = ''

    if 'TARIFA' in recursos_df.columns:
        recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)
    else:
        recursos_df['TARIFA'] = 0

    def calcular_fechas(df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        if 'IDRUBRO' not in df.columns or 'FECHAINICIO' not in df.columns or 'FECHAFIN' not in df.columns:
            return df
        inicio_rubro = df.set_index('IDRUBRO')['FECHAINICIO'].to_dict()
        fin_rubro = df.set_index('IDRUBRO')['FECHAFIN'].to_dict()
        duracion_rubro = (df.set_index('IDRUBRO')['FECHAFIN'] - df.set_index('IDRUBRO')['FECHAINICIO']).dt.days.to_dict()
        dependencias_local = defaultdict(list)
        pre_count = defaultdict(int)
        for _, row in df.iterrows():
            tarea_id = row['IDRUBRO']
            predecesoras_str = str(row.get('PREDECESORAS', '')).strip()
            if predecesoras_str not in ['nan', '']:
                pre_list = [p.strip() for p in predecesoras_str.split(',') if p.strip() != '']
                for pre in pre_list:
                    match = re.match(r'(\d+)', pre)
                    if match:
                        pre_id = int(match.group(1))
                        dependencias_local[pre_id].append(tarea_id)
                        pre_count[tarea_id] += 1

        queue = deque([tid for tid in df['IDRUBRO'] if pre_count.get(tid, 0) == 0])
        inicio_calc = inicio_rubro.copy()
        fin_calc = fin_rubro.copy()

        while queue:
            tarea_id = queue.popleft()
            row = df[df['IDRUBRO'] == tarea_id]
            if row.empty:
                continue
            row = row.iloc[0]
            duracion = duracion_rubro.get(tarea_id, 0)
            predecesoras_str = str(row.get('PREDECESORAS', '')).strip()

            nueva_inicio = inicio_calc.get(tarea_id)
            nueva_fin = fin_calc.get(tarea_id)

            if predecesoras_str not in ['nan', '']:
                pre_list = [p.strip() for p in predecesoras_str.split(',') if p.strip() != '']
                for pre in pre_list:
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

            if nueva_inicio is not None:
                inicio_calc[tarea_id] = nueva_inicio
            if nueva_fin is not None:
                fin_calc[tarea_id] = nueva_fin

            for hijo in dependencias_local.get(tarea_id, []):
                pre_count[hijo] -= 1
                if pre_count[hijo] == 0:
                    queue.append(hijo)

        df['FECHAINICIO'] = df['IDRUBRO'].map(inicio_calc)
        df['FECHAFIN'] = df['IDRUBRO'].map(fin_calc)
        return df

    tareas_df = calcular_fechas(tareas_df)

    try:
        if ('FECHAINICIO' in tareas_df.columns and not pd.api.types.is_datetime64_any_dtype(tareas_df['FECHAINICIO'])) or \
           ('FECHAFIN' in tareas_df.columns and not pd.api.types.is_datetime64_any_dtype(tareas_df['FECHAFIN'])):
            tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], dayfirst=True, errors='coerce')
            tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], dayfirst=True, errors='coerce')
    except Exception as e:
        st.warning(f"Error checking/converting fechas: {e}")

    tareas_df.columns = tareas_df.columns.str.strip()
    if 'DURACION' not in tareas_df.columns:
        tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)

    es = {}
    ef = {}
    ls = {}
    lf = {}
    tf = {}
    ff = {}
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()

    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())

    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row.get('PREDECESORAS', '')).strip()
        if predecesoras_str not in ['nan', '']:
            pre_list = [p.strip() for p in predecesoras_str.split(',') if p.strip() != '']
            for pre_entry in pre_list:
                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                if match:
                    pre_id = int(match.group(1))
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                    desfase = int(match.group(3)) if match.group(3) else 0
                    if pre_id in all_task_ids:
                        dependencias[pre_id].append(tarea_id)
                        predecesoras_map[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                        st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                else:
                    if pre_entry != '':
                        st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")

    initial_tasks_ids = [tid for tid in all_task_ids if tid not in predecesoras_map]

    for tid in initial_tasks_ids:
        task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
        if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)):
                duration = 0
            ef[tid] = es[tid] + timedelta(days=duration)
        else:
            st.warning(f"⚠️ Advertencia: Tarea ID {tid} no encontrada o FECHAINICIO inválida para inicializar ES/EF.")

    in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids}
    queue = deque([tid for tid in all_task_ids if in_degree.get(tid, 0) == 0])
    processed_forward = set(queue)

    for tid in list(queue):
        task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
        if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)):
                duration = 0
            ef[tid] = es[tid] + timedelta(days=duration)
        else:
            processed_forward.discard(tid)
            if tid in es:
                del es[tid]
            if tid in ef:
                del ef[tid]

    while queue:
        u = queue.popleft()
        for v in dependencias.get(u, []):
            for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v, []):
                if pre_id_v == u:
                    potential_es_v = None
                    duration_v = duracion_dict.get(v, 0)
                    if not isinstance(duration_v, (int, float)):
                        duration_v = 0
                    if u in ef and u in es:
                        if tipo_v == 'CC':
                            potential_es_v = es[u] + timedelta(days=desfase_v)
                        elif tipo_v == 'FC':
                            potential_es_v = ef[u] + timedelta(days=desfase_v)
                        elif tipo_v == 'CF':
                            potential_es_v = (es[u] + timedelta(days=desfase_v)) - timedelta(days=duration_v)
                        elif tipo_v == 'FF':
                            potential_es_v = (ef[u] + timedelta(days=desfase_v)) - timedelta(days=duration_v)
                        else:
                            st.warning(f"⚠️ Tipo de relación '{tipo_v}' no reconocido para calcular ES de tarea {v} basada en {u}. Usando lógica FC por defecto.")
                            potential_es_v = ef[u] + timedelta(days=desfase_v)
                        if v not in es or (potential_es_v is not None and potential_es_v > es[v]):
                            es[v] = potential_es_v
                        if v in es:
                            duration_v_calc = duracion_dict.get(v, 0)
                            if not isinstance(duration_v_calc, (int, float)):
                                duration_v_calc = 0
                            ef[v] = es[v] + timedelta(days=duration_v_calc)
                    else:
                        st.warning(f"⚠️ Advertencia: ES/EF no calculados para predecesor ID {u} al procesar sucesor ID {v}. Saltando cálculo de ES/EF para v basado en u.")
            in_degree[v] -= 1
            if in_degree[v] == 0 and v not in processed_forward:
                queue.append(v)
                processed_forward.add(v)

    unprocessed_forward = all_task_ids - processed_forward
    if unprocessed_forward:
        st.warning(f"⚠️ Advertencia: Las siguientes tareas no fueron procesadas en el pase hacia adelante (posible ciclo o grafo desconectado): {unprocessed_forward}")
        for tid in unprocessed_forward:
            if tid not in es:
                task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
                if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
                    es[tid] = task_row.iloc[0]['FECHAINICIO']
                    duration = duracion_dict.get(tid, 0)
                    if not isinstance(duration, (int, float)):
                        duration = 0
                    ef[tid] = es[tid] + timedelta(days=duration)
                    st.warning(f"Inicializando ES/EF para tarea no procesada {tid} con su fecha de inicio original.")
                else:
                    st.warning(f"❌ Error: Tarea no procesada {tid} no encontrada o FECHAINICIO inválida. No se pudo inicializar ES/EF.")

    end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]
    project_finish_date = None
    if ef:
        project_finish_date = max(ef.values())
    else:
        st.warning("❌ Error: No se calculó ninguna Fecha de Finalización Temprana (EF) en el pase hacia adelante. No se puede determinar la fecha de fin del proyecto.")
        raise ValueError("No EF calculated in forward pass.")

    for tid in end_tasks_ids:
        if tid in ef:
            lf[tid] = project_finish_date
            duration = duracion_dict.get(tid, 0)
            if not isinstance(duration, (int, float)):
                duration = 0
            ls[tid] = lf[tid] - timedelta(days=duration)
        else:
            st.warning(f"⚠️ Advertencia: Tarea final ID {tid} no encontrada en EF. No se puede inicializar LF/LS.")

    queue_backward = deque(end_tasks_ids)
    processed_backward = set(end_tasks_ids)
    successor_process_count = defaultdict(int)

    while queue_backward:
        v = queue_backward.popleft()
        for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
            potential_lf_u = None
            if v in ls and v in lf:
                if tipo_relacion_uv == 'CC':
                    potential_lf_u = (ls[v] - timedelta(days=desfase_uv)) + timedelta(days=duracion_dict.get(u, 0))
                elif tipo_relacion_uv == 'FC':
                    potential_lf_u = ls[v] - timedelta(days=desfase_uv)
                elif tipo_relacion_uv == 'CF':
                    potential_lf_u = (lf[v] - timedelta(days=desfase_uv)) + timedelta(days=duracion_dict.get(u, 0))
                elif tipo_relacion_uv == 'FF':
                    potential_lf_u = lf[v] - timedelta(days=desfase_uv)
                else:
                    st.warning(f"⚠️ Tipo de relación '{tipo_relacion_uv}' no reconocido para calcular LF de tarea {u} basada en {v}. Usando lógica FC por defecto.")
                    potential_lf_u = ls[v] - timedelta(days=desfase_uv)
                if u not in lf or (potential_lf_u is not None and potential_lf_u < lf.get(u, potential_lf_u)):
                    lf[u] = potential_lf_u
                    duration_u = duracion_dict.get(u, 0)
                    if not isinstance(duration_u, (int, float)):
                        duration_u = 0
                    ls[u] = lf[u] - timedelta(days=duration_u)
            else:
                st.warning(f"⚠️ Advertencia: LS/LF no calculados para sucesora ID {v} al procesar predecesora ID {u}. Saltando cálculo de LF/LS para u basado en v.")
            total_successors_of_u = len(dependencias.get(u, []))
            successor_process_count[u] += 1
            if successor_process_count[u] == total_successors_of_u and u not in processed_backward:
                queue_backward.append(u)
                processed_backward.add(u)

    unprocessed_backward = all_task_ids - processed_backward
    if unprocessed_backward:
        st.warning(f"⚠️ Advertencia: Las siguientes tareas no fueron procesadas en el pase hacia atrás (posible ciclo o problema en el grafo/inicialización): {unprocessed_backward}")
        for tid in unprocessed_backward:
            if tid in es and tid in ef:
                lf[tid] = ef[tid]
                ls[tid] = es[tid]
                st.warning(f"Inicializando LF/LS para tarea no procesada hacia atrás {tid} con sus fechas tempranas.")
            else:
                st.warning(f"❌ Error: Tarea no procesada hacia atrás {tid} no encontrada en ES/EF. No se pudo inicializar LF/LS.")

    for tid in all_task_ids:
        if tid in ef and tid in lf:
            tf[tid] = lf[tid] - ef[tid]
            if hasattr(tf[tid], 'total_seconds') and tf[tid].total_seconds() < -1e-9:
                tf[tid] = timedelta(days=0)
            min_successor_es = None
            for suc_id in dependencias.get(tid, []):
                for pre_id_suc, tipo_suc, desfase_suc in predecesoras_map.get(suc_id, []):
                    if pre_id_suc == tid:
                        required_start_suc = None
                        if tid in es and tid in ef:
                            if tipo_suc == 'CC':
                                required_start_suc = es[tid] + timedelta(days=desfase_suc)
                            elif tipo_suc == 'FC':
                                required_start_suc = ef[tid] + timedelta(days=desfase_suc)
                            elif tipo_suc == 'CF':
                                duration_suc = duracion_dict.get(suc_id, 0)
                                if not isinstance(duration_suc, (int, float)):
                                    duration_suc = 0
                                required_start_suc = (es[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                            elif tipo_suc == 'FF':
                                duration_suc = duracion_dict.get(suc_id, 0)
                                if not isinstance(duration_suc, (int, float)):
                                    duration_suc = 0
                                required_start_suc = (ef[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                            else:
                                st.warning(f"⚠️ Tipo de relación '{tipo_suc}' no reconocido al calcular FF para tarea {tid} basada en sucesor {suc_id}. Usando lógica FC por defecto.")
                                required_start_suc = ef[tid] + timedelta(days=desfase_suc)
                            if required_start_suc is not None:
                                if min_successor_es is None or required_start_suc < min_successor_es:
                                    min_successor_es = required_start_suc
                            break
            if min_successor_es is not None and tid in ef:
                ff[tid] = min_successor_es - ef[tid]
                if hasattr(ff[tid], 'total_seconds') and ff[tid].total_seconds() < -1e-9:
                    ff[tid] = timedelta(days=0)
            else:
                ff[tid] = timedelta(days=0)
        else:
            tf[tid] = pd.NA
            ff[tid] = pd.NA

    tareas_df['FECHA_INICIO_TEMPRANA'] = tareas_df['IDRUBRO'].map(es)
    tareas_df['FECHA_FIN_TEMPRANA'] = tareas_df['IDRUBRO'].map(ef)
    tareas_df['FECHA_INICIO_TARDE'] = tareas_df['IDRUBRO'].map(ls)
    tareas_df['FECHA_FIN_TARDE'] = tareas_df['IDRUBRO'].map(lf)
    tareas_df['HOLGURA_TOTAL_TD'] = tareas_df['IDRUBRO'].map(tf)
    tareas_df['HOLGURA_LIBRE_TD'] = tareas_df['IDRUBRO'].map(ff)
    tareas_df['HOLGURA_TOTAL'] = tareas_df['HOLGURA_TOTAL_TD'].apply(lambda x: x.days if pd.notna(x) and hasattr(x, 'days') else pd.NA)
    tareas_df['HOLGURA_LIBRE'] = tareas_df['HOLGURA_LIBRE_TD'].apply(lambda x: x.days if pd.notna(x) and hasattr(x, 'days') else pd.NA)
    tolerance_days = 1e-9
    tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL'].apply(lambda x: abs(x) < tolerance_days if pd.notna(x) else False)

    # _______________________________________
    st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
    st.dataframe(tareas_df[['IDRUBRO', 'RUBRO', 'PREDECESORAS', 'FECHAINICIO', 'FECHAFIN',
                            'FECHA_INICIO_TEMPRANA', 'FECHA_FIN_TEMPRANA',
                            'FECHA_INICIO_TARDE', 'FECHA_FIN_TARDE', 'DURACION', 'HOLGURA_TOTAL', 'RUTA_CRITICA']])

    dependencias_df = dependencias_df.merge(recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
    dependencias_df['COSTO'] = dependencias_df.get('CANTIDAD', 0) * dependencias_df.get('TARIFA', 0)
    costos_por_can = dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
    costos_por_can.rename(columns={'COSTO': 'COSTO_TOTAL'}, inplace=True)
    if 'RUBRO' in tareas_df.columns:
        tareas_df['RUBRO'] = tareas_df['RUBRO'].astype(str).str.strip()
    costos_por_can['RUBRO'] = costos_por_can['RUBRO'].astype(str).str.strip()
    tareas_df = tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')

    # _______________________________________
    st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
    cost_column_name = None
    if 'COSTO_TOTAL_RUBRO' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL_RUBRO'
    elif 'COSTO_TOTAL_x' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL_x'
    elif 'COSTO_TOTAL' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL'
    if cost_column_name:
        tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce').fillna(0)
    else:
        tareas_df['COSTO_TOTAL_NUMERICO'] = 0
        cost_column_name = 'COSTO_TOTAL_NUMERICO'

    if 'IDRUBRO' in tareas_df.columns:
        tareas_df = tareas_df.sort_values(['IDRUBRO'])
    tareas_df['y_num'] = range(len(tareas_df))

    fig = go.Figure()
    fecha_inicio_col = 'FECHAINICIO'
    fecha_fin_col = 'FECHAFIN'
    if fecha_inicio_col not in tareas_df.columns or fecha_fin_col not in tareas_df.columns:
        st.warning("❌ Error: No se encontraron columnas de fechas de inicio/fin necesarias para dibujar el Gantt.")

    inicio_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_inicio_col].to_dict() if 'IDRUBRO' in tareas_df.columns else {}
    fin_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_fin_col].to_dict() if 'IDRUBRO' in tareas_df.columns else {}
    is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict() if 'IDRUBRO' in tareas_df.columns else {}
    dependencias_plot = defaultdict(list)
    predecesoras_map_details = defaultdict(list)

    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row.get('PREDECESORAS', '')).strip()
        if predecesoras_str not in ['nan', '']:
            pre_list = [p.strip() for p in predecesoras_str.split(',') if p.strip() != '']
            for pre_entry in pre_list:
                match = re.match(r'(\d+)\s*([A-Za-z]{2})?(?:\s*([+-]?\d+)\s*días?)?', pre_entry)
                if match:
                    pre_id = int(match.group(1))
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC'
                    desfase = int(match.group(3)) if match.group(3) else 0
                    if pre_id in tareas_df['IDRUBRO'].values:
                        dependencias_plot[pre_id].append(tarea_id)
                        predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                        st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando.")

    shapes = []
    color_banda = 'rgba(220, 220, 220, 0.6)'
    for y_pos in range(len(tareas_df)):
        if y_pos % 2 == 0:
            shapes.append(dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=y_pos - 0.5, y1=y_pos + 0.5, fillcolor=color_banda, layer="below", line_width=0))

    color_no_critica_barra = 'lightblue'
    color_critica_barra = 'rgb(255, 133, 133)'

    for _, row in tareas_df.iterrows():
        line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
        line_width = 12
        start_date = row.get(fecha_inicio_col)
        end_date = row.get(fecha_fin_col)
        if pd.isna(start_date) or pd.isna(end_date):
            st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea {row.get('RUBRO')} (ID {row.get('IDRUBRO')}). No se dibujará la barra.")
            continue
        try:
            valor_costo = float(row.get(cost_column_name, 0))
            costo_formateado = f"S/ {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            costo_formateado = "S/ 0,00"

        hover_text = (
            f"📌 <b>Rubro:</b> {row.get('RUBRO')}<br>"
            f"🗓️ <b>Capítulo:</b> {row.get('CAPÍTULO')}<br>"
            f"📅 <b>Inicio</b> {start_date.strftime('%d/%m/%Y')}<br>"
            f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
            f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
            f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>"
            f"💰 <b>Costo:</b> {costo_formateado}"
        )

        fig.add_trace(go.Scatter(x=[start_date, end_date], y=[row['y_num'], row['y_num']], mode='lines', line=dict(color=line_color, width=line_width), showlegend=False, hoverinfo='text', text=hover_text))

    offset_days_horizontal = 5
    color_no_critica_flecha = 'blue'
    color_critica_flecha = 'red'

    for pre_id, sucesores in dependencias_plot.items():
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
            origin_x = x_pre_fin if tipo_relacion in ['FC', 'FF'] else x_pre_inicio
            connection_x = x_suc_inicio if tipo_relacion in ['CC', 'FC'] else x_suc_fin
            arrow_symbol = 'triangle-right' if tipo_relacion in ['CC', 'FC'] else 'triangle-left'

            fig.add_trace(go.Scattergl(x=[origin_x], y=[y_pre], mode='markers', marker=dict(symbol='circle', size=8, color=arrow_color), hoverinfo='none', showlegend=False))

            points_x = [origin_x]
            points_y = [y_pre]

            if tipo_relacion in ['CC', 'FC']:
                elbow1_x = origin_x - timedelta(days=offset_days_horizontal)
                points_x.extend([elbow1_x, elbow1_x, connection_x])
                points_y.extend([y_pre, y_suc, y_suc])
            else:
                points_x.extend([origin_x, connection_x])
                points_y.extend([y_suc, y_suc])

            fig.add_trace(go.Scatter(x=points_x, y=points_y, mode='lines', line=dict(color=arrow_color, width=1, dash='dash'), hoverinfo='none', showlegend=False))
            fig.add_trace(go.Scattergl(x=[connection_x], y=[y_suc], mode='markers', marker=dict(symbol=arrow_symbol, size=10, color=arrow_color), hoverinfo='none', showlegend=False))

    y_ticktext_styled = []
    for y_pos in range(len(tareas_df)):
        row_for_y_pos = tareas_df[tareas_df['y_num'] == y_pos]
        if not row_for_y_pos.empty:
            rubro_text = row_for_y_pos.iloc[0]['RUBRO']
            y_ticktext_styled.append(f"<b>{rubro_text}</b>" if y_pos % 2 == 0 else rubro_text)
        else:
            y_ticktext_styled.append("")

    fig.update_layout(
        xaxis=dict(title='Fechas', side='bottom', dtick='M1', tickangle=-90, showgrid=True, gridcolor='rgba(128,128,128,0.3)', gridwidth=0.5),
        xaxis2=dict(title='Fechas', overlaying='x', side='top', dtick='M1', tickangle=90, showgrid=True, gridcolor='rgba(128,128,128,0.3)', gridwidth=0.5),
        yaxis_title='Rubro',
        yaxis=dict(autorange='reversed', tickvals=tareas_df['y_num'], ticktext=y_ticktext_styled, tickfont=dict(size=10), showgrid=False),
        shapes=shapes,
        height=max(600, len(tareas_df) * 25),
        showlegend=False,
        plot_bgcolor='white',
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], errors='coerce')
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], errors='coerce')

    if 'RUBRO' in tareas_df.columns:
        tareas_df['RUBRO'] = tareas_df['RUBRO'].astype(str).str.strip()
    if 'RUBRO' in dependencias_df.columns:
        dependencias_df['RUBRO'] = dependencias_df['RUBRO'].astype(str).str.strip()

    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
        left_on='RUBRO',
        right_on='RUBRO',
        how='left'
    )

    daily_resource_usage_list = []
    for _, row in recursos_tareas_df.iterrows():
        task_id = row.get('IDRUBRO')
        resource_name = row.get('RECURSO')
        unit = row.get('UNIDAD')
        total_quantity = row.get('CANTIDAD', 0)
        start_date = row.get('FECHAINICIO')
        end_date = row.get('FECHAFIN')
        duration_days = row.get('DURACION', 0)

        if pd.isna(start_date) or pd.isna(end_date) or (isinstance(start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp) and start_date > end_date):
            st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea ID {task_id}, recurso '{resource_name}'. Saltando.")
            continue

        if duration_days is None or duration_days <= 0:
            daily_quantity = total_quantity
            date_range = [start_date] if not pd.isna(start_date) else []
        else:
            daily_quantity = total_quantity / (duration_days + 1)
            try:
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            except Exception:
                date_range = [start_date]

        if not date_range:
            continue

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
        st.warning("No se generaron datos de uso diario de recursos.")
        all_daily_resource_usage_df = pd.DataFrame()

    daily_resource_demand_df = all_daily_resource_usage_df.groupby(['Fecha', 'RECURSO', 'UNIDAD'], as_index=False)['Cantidad_Diaria'].sum()
    daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)
    daily_resource_demand_df['RECURSO'] = daily_resource_demand_df['RECURSO'].astype(str).str.strip()
    recursos_df['RECURSO'] = recursos_df['RECURSO'].astype(str).str.strip()

    resource_demand_with_details_df = daily_resource_demand_df.merge(recursos_df[['RECURSO', 'TYPE', 'TARIFA']], on='RECURSO', how='left')
    resource_demand_with_details_df['TARIFA'] = pd.to_numeric(resource_demand_with_details_df.get('TARIFA', 0), errors='coerce').fillna(0)
    resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']

    daily_cost_by_type_df = resource_demand_with_details_df.groupby(['Fecha', 'TYPE'], as_index=False)['Costo_Diario'].sum()
    daily_demand_by_resource_df = resource_demand_with_details_df.groupby(['Fecha', 'RECURSO', 'UNIDAD'], as_index=False)['Demanda_Diaria_Total'].sum()

    # _______________________________________
    st.subheader("📊 Distribución de Recursos")

    if 'RUBRO' not in recursos_tareas_df.columns:
        if 'tareas_df' in locals() or 'tareas_df' in globals():
            tareas_df['RUBRO'] = tareas_df.get('RUBRO', '').astype(str).str.strip()
            recursos_tareas_df['RUBRO'] = recursos_tareas_df.get('RUBRO', '').astype(str).str.strip()
            if 'IDRUBRO' in recursos_tareas_df.columns and 'IDRUBRO' in tareas_df.columns:
                recursos_tareas_df = recursos_tareas_df.merge(tareas_df[['IDRUBRO', 'RUBRO']], left_on='IDRUBRO', right_on='IDRUBRO', how='left')
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

    for _, row in recursos_tareas_df.iterrows():
        start = row.get('FECHAINICIO')
        end = row.get('FECHAFIN')
        recurso = row.get('RECURSO')
        rubro = row.get('RUBRO')
        if pd.isna(start) or pd.isna(end) or pd.isna(recurso):
            continue
        hover = f"<b>Rubro:</b> {rubro}<br><b>Recurso:</b> {recurso}<br><b>Inicio:</b> {start.strftime('%Y-%m-%d')}<br><b>Fin:</b> {end.strftime('%Y-%m-%d')}"
        fig_resource_timeline.add_trace(go.Scattergl(
            x=[start, end],
            y=[recurso, recurso],
            mode='lines',
            line=dict(color=pastel_blue, width=10),
            name=recurso,
            showlegend=False,
            hoverinfo='text',
            text=hover,
            customdata=[rubro]
        ))

    dropdown_options = [{'label': 'All Tasks', 'method': 'update', 'args': [{'visible': [True] * len(fig_resource_timeline.data)}, {'title': 'Línea de Tiempo de Uso de Recursos'}]}]
    for rubro in unique_rubros:
        visibility_list = [trace.customdata[0] == rubro if (hasattr(trace, 'customdata') and trace.customdata and len(trace.customdata) > 0) else False for trace in fig_resource_timeline.data]
        dropdown_options.append({'label': rubro, 'method': 'update', 'args': [{'visible': visibility_list}, {'title': f'Línea de Tiempo de Uso de Recursos (Filtrado por: {rubro})'}]})

    fig_resource_timeline.update_layout(
        updatemenus=[go.layout.Updatemenu(buttons=dropdown_options, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=1.1, yanchor="top")],
        yaxis=dict(autorange="reversed", title="Recurso", tickfont=dict(size=10)),
        xaxis=dict(title='Fechas', side='bottom', dtick='M1', tickangle=-90, showgrid=True, gridcolor='rgba(128,128,128,0.3)', gridwidth=0.5),
        xaxis2=dict(title='Fechas', overlaying='x', side='top', dtick='M1', tickangle=90, showgrid=True, gridcolor='rgba(128,128,128,0.3)', gridwidth=0.5),
        height=max(600, len(recursos_tareas_df['RECURSO'].unique()) * 20),
        showlegend=False,
        plot_bgcolor='white',
        hovermode='closest'
    )
    st.plotly_chart(fig_resource_timeline, use_container_width=True)

    resource_demand_with_details_df['Fecha'] = pd.to_datetime(resource_demand_with_details_df['Fecha'], errors='coerce')
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

    # _______________________________________
    st.subheader("📊 Cronograma Valorado")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=monthly_costs_df['Periodo_Mensual'], y=monthly_costs_df['Costo_Diario'], name='Costo Mensual', text=monthly_costs_df['Costo_Mensual_Formateado'], hoverinfo='text', hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>', secondary_y=False)
    fig.add_scatter(x=monthly_costs_df['Periodo_Mensual'], y=monthly_costs_df['Costo_Acumulado'], mode='lines+markers', name='Costo Acumulado', text=monthly_costs_df['Costo_Acumulado_Formateado'], hoverinfo='text', hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>', line=dict(color='red'), secondary_y=True)

    fig.update_yaxes(title_text="Costo Mensual", secondary_y=False, showgrid=False)
    if not monthly_costs_df['Costo_Acumulado'].empty:
        fig.update_yaxes(title_text="Costo Acumulado", secondary_y=True, showgrid=True, gridcolor='lightgrey')
    fig.update_xaxes(title_text="Período Mensual", tickangle=-45)

    fig.update_layout(hovermode='x unified', height=600, legend=dict(x=1.1, y=1, bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)'), plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")




