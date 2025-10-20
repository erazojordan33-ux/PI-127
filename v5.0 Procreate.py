import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- Configuración de la página ---
st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Cronograma Valorado y Recursos")

# --- Subir archivo ---
archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])

if archivo_excel:

    # --- Leer hojas y validar existencia ---
    try:
        tareas_df = pd.read_excel(archivo_excel, sheet_name='Tareas')
        recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
        dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
    except:
        st.error("El archivo debe contener las hojas: Tareas, Recursos y Dependencias")
        st.stop()

    # --- Mostrar tablas editables ---
    st.subheader("📋 Tabla Tareas")
    gb = GridOptionsBuilder.from_dataframe(tareas_df)
    gb.configure_default_column(editable=True)
    tareas_grid = AgGrid(tareas_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    tareas_df = tareas_grid['data']

    st.subheader("📋 Tabla Recursos")
    gb = GridOptionsBuilder.from_dataframe(recursos_df)
    gb.configure_default_column(editable=True)
    recursos_grid = AgGrid(recursos_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    recursos_df = recursos_grid['data']

    st.subheader("📋 Tabla Dependencias")
    gb = GridOptionsBuilder.from_dataframe(dependencias_df)
    gb.configure_default_column(editable=True)
    dependencias_grid = AgGrid(dependencias_df, gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED)
    dependencias_df = dependencias_grid['data']

    # -----------------------
    # LIMPIEZA Y VALIDACIÓN DE FECHAS
    # -----------------------
    for col in ['FECHAINICIO', 'FECHAFIN']:
        tareas_df[col] = tareas_df[col].astype(str).str.strip()
        tareas_df[col] = tareas_df[col].str.replace('T',' ')
        tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce')

    # Mostrar filas con fechas inválidas
    invalid_fecha = tareas_df[tareas_df[['FECHAINICIO','FECHAFIN']].isna().any(axis=1)]
    if not invalid_fecha.empty:
        st.warning("⚠️ Algunas filas tienen fechas inválidas y se ignorarán en el cálculo de ruta crítica:")
        st.dataframe(invalid_fecha)

    # Trabajar solo con filas válidas
    tareas_validas = tareas_df.dropna(subset=['FECHAINICIO','FECHAFIN']).copy()

    # Calcular DURACION
    tareas_validas['DURACION'] = (tareas_validas['FECHAFIN'] - tareas_validas['FECHAINICIO']).dt.days + 1
    if (tareas_validas['DURACION'] <= 0).any():
        st.error("Hay tareas con duración <= 0. Revisa FECHAINICIO y FECHAFIN")
        st.stop()

    # Predecesoras
    tareas_validas['PREDECESORAS'] = tareas_validas['PREDECESORAS'].fillna('').astype(str)

    # Tarifas
    if 'TARIFA' in recursos_df.columns:
        recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)

    st.success("✅ Columnas validadas correctamente: fechas, duración, numéricos y predecesoras.")

    # -----------------------
    # CÁLCULO DE RUTA CRÍTICA
    # -----------------------
    es, ef, ls, lf, tf = {}, {}, {}, {}, {}
    duracion_dict = tareas_validas.set_index('IDRUBRO')['DURACION'].to_dict()
    all_task_ids = set(tareas_validas['IDRUBRO'].tolist())

    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)
    for _, row in tareas_validas.iterrows():
        tid = row['IDRUBRO']
        pre_list = str(row.get('PREDECESORAS','')).split(',')
        for pre in pre_list:
            pre = pre.strip()
            if pre:
                match = re.match(r'(\d+)', pre)
                if match:
                    pre_id = int(match.group(1))
                    if pre_id in all_task_ids:
                        dependencias[pre_id].append(tid)
                        predecesoras_map[tid].append((pre_id,'FC',0))

    # Forward Pass
    in_degree = {tid: len(predecesoras_map.get(tid,[])) for tid in all_task_ids}
    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
    processed = set(queue)
    for tid in queue:
        task_row = tareas_validas[tareas_validas['IDRUBRO']==tid]
        if not task_row.empty:
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            ef[tid] = es[tid]+timedelta(days=duracion_dict.get(tid,0))

    while queue:
        u = queue.popleft()
        for v in dependencias.get(u,[]):
            potential_es = ef[u]
            if v not in es or potential_es > es[v]:
                es[v] = potential_es
                ef[v] = es[v] + timedelta(days=duracion_dict.get(v,0))
            in_degree[v] -= 1
            if in_degree[v]==0 and v not in processed:
                queue.append(v)
                processed.add(v)

    # Backward Pass
    end_tasks = [tid for tid in all_task_ids if tid not in dependencias]
    project_finish = max(ef.values()) if ef else pd.Timestamp.today()
    for tid in end_tasks:
        lf[tid] = project_finish
        ls[tid] = lf[tid] - timedelta(days=duracion_dict.get(tid,0))

    queue = deque(end_tasks)
    processed = set(end_tasks)
    while queue:
        v = queue.popleft()
        for u,_,_ in predecesoras_map.get(v,[]):
            potential_lf = ls[v]
            if u not in lf or potential_lf < lf.get(u, project_finish):
                lf[u] = potential_lf
                ls[u] = lf[u] - timedelta(days=duracion_dict.get(u,0))
            queue.append(u)
            processed.add(u)

    # Holguras y ruta crítica
    for tid in all_task_ids:
        if tid in ef and tid in lf:
            tf[tid] = (lf[tid] - ef[tid]).days
        else:
            tf[tid] = 0

    tareas_validas['FECHA_INICIO_TEMPRANA'] = tareas_validas['IDRUBRO'].map(es)
    tareas_validas['FECHA_FIN_TEMPRANA'] = tareas_validas['IDRUBRO'].map(ef)
    tareas_validas['FECHA_INICIO_TARDE'] = tareas_validas['IDRUBRO'].map(ls)
    tareas_validas['FECHA_FIN_TARDE'] = tareas_validas['IDRUBRO'].map(lf)
    tareas_validas['HOLGURA_TOTAL'] = tareas_validas['IDRUBRO'].map(tf)
    tareas_validas['RUTA_CRITICA'] = tareas_validas['HOLGURA_TOTAL']==0

    # -----------------------
    # MOSTRAR TABLA FINAL Y GANTT
    # -----------------------
    st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
    st.dataframe(tareas_validas[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                                 'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                                 'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','RUTA_CRITICA']])

    # Diagrama de Gantt
    fig_gantt = go.Figure()
    for _, row in tareas_validas.iterrows():
        color = 'red' if row['RUTA_CRITICA'] else 'lightblue'
        fig_gantt.add_trace(go.Scatter(
            x=[row['FECHA_INICIO_TEMPRANA'], row['FECHA_FIN_TEMPRANA']],
            y=[row['RUBRO'], row['RUBRO']],
            mode='lines',
            line=dict(color=color, width=12),
            hovertext=f"{row['RUBRO']} ({row['DURACION']} días)",
            showlegend=False
        ))
    fig_gantt.update_layout(title="📅 Diagrama de Gantt - Ruta Crítica")
    st.plotly_chart(fig_gantt, use_container_width=True)

else:
    st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")




















