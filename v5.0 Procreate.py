import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict
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


    # --- Mini-bloque: Inspección y limpieza de FECHAINICIO y FECHAFIN ---
    st.subheader("🔍 Inspección de fechas")
    
    # Mostrar tipos de cada columna
    st.write("Tipos de columnas en la tabla de tareas:")
    st.write(tareas_df.dtypes)
    
    # Mostrar los primeros 20 valores con su tipo real
    for col in ['FECHAINICIO','FECHAFIN']:
        st.write(f"Columna: {col}")
        for i, val in tareas_df[col].head(20).items():
            st.write(f"Fila {i}: Valor = {val} | Tipo real = {type(val)}")
    
    # Limpiar caracteres problemáticos (por ejemplo 'T' de ISO)
    for col in ['FECHAINICIO','FECHAFIN']:
        tareas_df[col] = tareas_df[col].apply(lambda x: str(x).replace('T',' ') if pd.notna(x) else x)
    
    # Intentar convertir a datetime nuevamente
    for col in ['FECHAINICIO','FECHAFIN']:
        tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce')
    
    # Mostrar filas con valores que siguen siendo inválidos
    invalid_rows = tareas_df[tareas_df[['FECHAINICIO','FECHAFIN']].isnull().any(axis=1)].index.tolist()
    if invalid_rows:
        st.warning(f"⚠️ Algunas fechas aún no son válidas en las filas: {invalid_rows}")
    else:
        st.success("Todas las fechas son válidas ✅")


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
    # Validación y conversión segura de FECHAS
    # -----------------------
    for col in ['FECHAINICIO', 'FECHAFIN']:
        # Convertir todo a string y quitar espacios
        tareas_df[col] = tareas_df[col].astype(str).str.strip()
        
        # Reemplazar 'nan', 'NaT' u otros textos vacíos por None
        tareas_df[col] = tareas_df[col].replace(['nan','NaT','None',''], pd.NaT)
        
        # Convertir a datetime con inferencia
        tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce', infer_datetime_format=True)
    
    # Validar
    filas_invalidas = tareas_df[tareas_df['FECHAINICIO'].isna() | tareas_df['FECHAFIN'].isna()]
    if not filas_invalidas.empty:
        st.error(f"Algunas fechas no son válidas en las filas: {filas_invalidas.index.tolist()}")
        st.stop()
    
    # Calcular DURACION
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days + 1
    if (tareas_df['DURACION'] <= 0).any():
        st.error("Hay tareas con duración <= 0. Revisa FECHAINICIO y FECHAFIN")
        st.stop()

    
    # --- Predecesoras ---
    tareas_df['PREDECESORAS'] = tareas_df['PREDECESORAS'].fillna('').astype(str)
    
    # --- Tarifas ---
    if 'TARIFA' in recursos_df.columns:
        recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)


        
    st.success("✅ Columnas validadas correctamente: fechas, duración, numéricos y predecesoras.")


    # --- Calculo ruta crítica (igual que antes) ---
    es, ef, ls, lf, tf = {}, {}, {}, {}, {}
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())
    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)
    for _, row in tareas_df.iterrows():
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
    from collections import deque
    in_degree = {tid: len(predecesoras_map.get(tid,[])) for tid in all_task_ids}
    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
    processed = set(queue)
    for tid in queue:
        task_row = tareas_df[tareas_df['IDRUBRO']==tid]
        if not task_row.empty:
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            ef[tid] = es[tid]+timedelta(days=duracion_dict.get(tid,0))
    while queue:
        u = queue.popleft()
        for v in dependencias.get(u,[]):
            potential_es = ef[u]
            if v not in es or potential_es>es[v]:
                es[v] = potential_es
                ef[v] = es[v]+timedelta(days=duracion_dict.get(v,0))
            in_degree[v]-=1
            if in_degree[v]==0 and v not in processed:
                queue.append(v)
                processed.add(v)

    # Backward Pass
    end_tasks = [tid for tid in all_task_ids if tid not in dependencias]
    project_finish = max(ef.values())
    for tid in end_tasks:
        lf[tid]=project_finish
        ls[tid]=lf[tid]-timedelta(days=duracion_dict.get(tid,0))
    queue = deque(end_tasks)
    processed = set(end_tasks)
    while queue:
        v = queue.popleft()
        for u,_,_ in predecesoras_map.get(v,[]):
            potential_lf = ls[v]
            if u not in lf or potential_lf<lf.get(u, project_finish):
                lf[u]=potential_lf
                ls[u]=lf[u]-timedelta(days=duracion_dict.get(u,0))
            queue.append(u)
            processed.add(u)

    # Holguras y ruta critica
    for tid in all_task_ids:
        if tid in ef and tid in lf:
            tf[tid]=(lf[tid]-ef[tid]).days
        else:
            tf[tid]=0
    tareas_df['FECHA_INICIO_TEMPRANA'] = tareas_df['IDRUBRO'].map(es)
    tareas_df['FECHA_FIN_TEMPRANA'] = tareas_df['IDRUBRO'].map(ef)
    tareas_df['FECHA_INICIO_TARDE'] = tareas_df['IDRUBRO'].map(ls)
    tareas_df['FECHA_FIN_TARDE'] = tareas_df['IDRUBRO'].map(lf)
    tareas_df['HOLGURA_TOTAL'] = tareas_df['IDRUBRO'].map(tf)
    tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL']==0

    # --- Mostrar tabla final con ruta critica ---
    st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
    st.dataframe(tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                            'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                            'FECHA_INICIO_TARDE','FECHA_FIN_TARDE','DURACION','HOLGURA_TOTAL','RUTA_CRITICA']])

    # --- Diagrama de Gantt ---
    fig_gantt = go.Figure()
    for _, row in tareas_df.iterrows():
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















