import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Cronograma Valorado y Recursos")

# --- Subir archivo ---
archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])

if archivo_excel:
    try:
        tareas_df = pd.read_excel(archivo_excel, sheet_name='Tareas')
        recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
        dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
    except:
        st.error("El archivo no contiene todas las hojas requeridas")
        st.stop()

    # --- Mostrar tablas editables ---
    st.subheader("📋 Tabla Tareas")
    gb = GridOptionsBuilder.from_dataframe(tareas_df)
    gb.configure_default_column(editable=True)
    grid_options = gb.build()
    tareas_grid = AgGrid(tareas_df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED)
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

    # --- Asegurar tipos ---
    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], errors='coerce', dayfirst=True)
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], errors='coerce', dayfirst=True)
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)
    recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)
    recursos_df['CANTIDAD'] = pd.to_numeric(recursos_df['CANTIDAD'], errors='coerce').fillna(0)

    st.success("Datos cargados y convertidos correctamente ✅")

    # --- Calcular ruta crítica ---
    es, ef, ls, lf, tf, ff = {}, {}, {}, {}, {}, {}
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

    # --- Forward Pass ---
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

    # --- Backward Pass ---
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

    # --- Holguras y ruta crítica ---
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

    # --- Recursos diarios y costos ---
    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO','RUBRO','FECHAINICIO','FECHAFIN','DURACION']],
        left_on='CAN', right_on='RUBRO', how='left'
    )
    daily_list=[]
    for _,row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        start = row['FECHAINICIO']
        end = row['FECHAFIN']
        total = row.get('CANTIDAD',0)
        dur = max(row.get('DURACION',1),1)
        daily_qty = total/dur
        dates = pd.date_range(start=start,end=end)
        df_temp=pd.DataFrame({'Fecha':dates,'IDRUBRO':task_id,'RECURSO':row['RECURSO'],'Cantidad_Diaria':daily_qty})
        daily_list.append(df_temp)
    all_daily = pd.concat(daily_list,ignore_index=True)
    all_daily = all_daily.merge(recursos_df[['RECURSO','TARIFA']],on='RECURSO',how='left')
    all_daily['Costo_Diario'] = all_daily['Cantidad_Diaria']*all_daily['TARIFA']
    all_daily['Periodo_Mensual'] = all_daily['Fecha'].dt.to_period('M')
    monthly_costs = all_daily.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
    monthly_costs['Costo_Acumulado'] = monthly_costs['Costo_Diario'].cumsum()

    # --- Graficar costos ---
    fig_costs = go.Figure()
    fig_costs.add_trace(go.Bar(
        x=monthly_costs['Periodo_Mensual'].astype(str),
        y=monthly_costs['Costo_Diario'],
        name='Costo Mensual'
    ))
    fig_costs.add_trace(go.Scatter(
        x=monthly_costs['Periodo_Mensual'].astype(str),
        y=monthly_costs['Costo_Acumulado'],
        mode='lines+markers',
        name='Costo Acumulado',
        line=dict(color='red')
    ))
    fig_costs.update_layout(title="💰 Cronograma Valorado - Costos Mensuales y Acumulados",
                            xaxis_title="Periodo", yaxis_title="Costo")
    st.plotly_chart(fig_costs, use_container_width=True)

else:
    st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")













