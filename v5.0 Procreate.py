import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import timedelta, datetime
from collections import defaultdict
import locale

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")

st.title("📊 Gestión de Proyectos - Cronograma Valorado y Recursos")

# --- Subir archivos ---
archivo_tareas = st.file_uploader("Subir archivo de Tareas (Excel)", type=["xlsx"])
archivo_recursos = st.file_uploader("Subir archivo de Recursos (Excel)", type=["xlsx"])
archivo_dependencias = st.file_uploader("Subir archivo de Dependencias (Excel)", type=["xlsx"])

if archivo_tareas and archivo_recursos and archivo_dependencias:
    
    # --- Cargar DataFrames ---
    tareas_df = pd.read_excel(archivo_tareas, sheet_name='Tareas')
    recursos_df = pd.read_excel(archivo_recursos, sheet_name='Recursos')
    dependencias_df = pd.read_excel(archivo_dependencias, sheet_name='Dependencias')

    # --- Configurar fechas ---
    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], dayfirst=True)
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], dayfirst=True)
    
    tareas_df.columns = tareas_df.columns.str.strip()
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days.fillna(0).astype(int)

    # --- Calcular ruta crítica ---
    es, ef, ls, lf, tf, ff = {}, {}, {}, {}, {}, {}
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
    
    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())

    for _, row in tareas_df.iterrows():
        tarea_id = row['IDRUBRO']
        predecesoras_str = str(row['PREDECESORAS']).strip()
        if predecesoras_str not in ['nan','']:
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
    
    # --- Forward Pass (ES, EF) ---
    from collections import deque
    in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids}
    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
    processed_forward = set(queue)
    
    for tid in queue:
        task_row = tareas_df[tareas_df['IDRUBRO']==tid]
        if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
            es[tid] = task_row.iloc[0]['FECHAINICIO']
            ef[tid] = es[tid] + timedelta(days=duracion_dict.get(tid,0))
    
    while queue:
        u = queue.popleft()
        for v in dependencias.get(u, []):
            for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v, []):
                if pre_id_v==u:
                    potential_es_v = ef[u] + timedelta(days=desfase_v) if tipo_v=='FC' else es[u] + timedelta(days=desfase_v)
                    if v not in es or potential_es_v>es[v]:
                        es[v]=potential_es_v
                        ef[v]=es[v]+timedelta(days=duracion_dict.get(v,0))
            in_degree[v]-=1
            if in_degree[v]==0 and v not in processed_forward:
                queue.append(v)
                processed_forward.add(v)
    
    # --- Backward Pass (LS, LF) ---
    end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]
    project_finish_date = max(ef.values())
    
    for tid in end_tasks_ids:
        lf[tid] = project_finish_date
        ls[tid] = lf[tid]-timedelta(days=duracion_dict.get(tid,0))
    
    queue_backward = deque(end_tasks_ids)
    processed_backward = set(end_tasks_ids)
    successor_details_map = defaultdict(list)
    for suc_id, pre_list in predecesoras_map.items():
        for pre_id, tipo_rel, lag in pre_list:
            successor_details_map[pre_id].append((suc_id, tipo_rel, lag))
    
    while queue_backward:
        v = queue_backward.popleft()
        for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
            potential_lf_u = ls[v] - timedelta(days=desfase_uv) if tipo_relacion_uv=='FC' else lf[v]-timedelta(days=desfase_uv)
            if u not in lf or potential_lf_u<lf.get(u, pd.Timestamp.max):
                lf[u]=potential_lf_u
                ls[u]=lf[u]-timedelta(days=duracion_dict.get(u,0))
            queue_backward.append(u)
            processed_backward.add(u)
    
    # --- Calcular holguras ---
    for tid in all_task_ids:
        if tid in ef and tid in lf:
            tf[tid] = lf[tid]-ef[tid]
            ff[tid] = timedelta(days=0) # Simplificado
        else:
            tf[tid] = pd.NA
            ff[tid] = pd.NA
    
    tareas_df['FECHA_INICIO_TEMPRANA'] = tareas_df['IDRUBRO'].map(es)
    tareas_df['FECHA_FIN_TEMPRANA'] = tareas_df['IDRUBRO'].map(ef)
    tareas_df['FECHA_INICIO_TARDE'] = tareas_df['IDRUBRO'].map(ls)
    tareas_df['FECHA_FIN_TARDE'] = tareas_df['IDRUBRO'].map(lf)
    tareas_df['HOLGURA_TOTAL'] = tareas_df['IDRUBRO'].map(lambda x: tf[x].days if pd.notna(tf[x]) else pd.NA)
    tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL'].apply(lambda x: x==0)
    
    st.subheader("📋 Tareas con Fechas y Ruta Crítica")
    st.dataframe(tareas_df[['IDRUBRO','RUBRO','PREDECESORAS','DURACION','FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA','FECHA_INICIO_TARDE','FECHA_FIN_TARDE','HOLGURA_TOTAL','RUTA_CRITICA']])
    
    # --- Diagrama de Gantt ---
    fig_gantt = go.Figure()
    for i, row in tareas_df.iterrows():
        color = 'red' if row['RUTA_CRITICA'] else 'lightblue'
        fig_gantt.add_trace(go.Scatter(
            x=[row['FECHA_INICIO_TEMPRANA'], row['FECHA_FIN_TEMPRANA']],
            y=[row['RUBRO'], row['RUBRO']],
            mode='lines',
            line=dict(color=color, width=12),
            hovertext=f"Rubro: {row['RUBRO']}<br>Duración: {row['DURACION']} días",
            showlegend=False
        ))
    fig_gantt.update_layout(title="📅 Diagrama de Gantt - Ruta Crítica")
    st.plotly_chart(fig_gantt, use_container_width=True)
    
    # --- Recursos diarios ---
    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO','RUBRO','FECHAINICIO','FECHAFIN','DURACION']],
        left_on='CAN',
        right_on='RUBRO',
        how='left'
    )
    
    daily_resource_usage_list = []
    for _, row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        start_date = row['FECHAINICIO']
        end_date = row['FECHAFIN']
        total_quantity = row['CANTIDAD']
        duration_days = row['DURACION']
        if duration_days>0:
            daily_quantity = total_quantity/(duration_days+1)
            date_range = pd.date_range(start=start_date,end=end_date)
        else:
            daily_quantity = total_quantity
            date_range = [start_date]
        temp_df = pd.DataFrame({
            'Fecha': date_range,
            'IDRUBRO': task_id,
            'RECURSO': row['RECURSO'],
            'Cantidad_Diaria': daily_quantity
        })
        daily_resource_usage_list.append(temp_df)
    all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
    
    # --- Costos ---
    resource_demand_with_details_df = all_daily_resource_usage_df.merge(
        recursos_df[['RECURSO','TYPE','TARIFA']],
        on='RECURSO',
        how='left'
    )
    resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Cantidad_Diaria']*resource_demand_with_details_df['TARIFA']
    resource_demand_with_details_df['Periodo_Mensual'] = resource_demand_with_details_df['Fecha'].dt.to_period('M')
    monthly_costs_df = resource_demand_with_details_df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
    monthly_costs_df['Costo_Acumulado'] = monthly_costs_df['Costo_Diario'].cumsum()
    
    # --- Graficar costos ---
    fig_costs = go.Figure()
    fig_costs.add_trace(go.Bar(
        x=monthly_costs_df['Periodo_Mensual'].astype(str),
        y=monthly_costs_df['Costo_Diario'],
        name='Costo Mensual'
    ))
    fig_costs.add_trace(go.Scatter(
        x=monthly_costs_df['Periodo_Mensual'].astype(str),
        y=monthly_costs_df['Costo_Acumulado'],
        mode='lines+markers',
        name='Costo Acumulado',
        line=dict(color='red')
    ))
    fig_costs.update_layout(title="💰 Cronograma Valorado - Costos Mensuales y Acumulados",
                            xaxis_title="Período Mensual", yaxis_title="Costo")
    st.plotly_chart(fig_costs, use_container_width=True)
    
else:
    st.warning("Por favor, sube los tres archivos Excel (Tareas, Recursos y Dependencias) para continuar.")
