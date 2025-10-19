# ==========================
# STREAMLIT APP - PROCREATE
# ==========================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from collections import defaultdict
import locale

st.set_page_config(layout="wide")
st.title("📊 Proyecto Procreate: Gantt y Recursos")

# --------------------------
# Locale a español
# --------------------------
try:
    locale.setlocale(locale.LC_ALL, 'es_ES.utf8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'es_CO.utf8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'spanish')
            except:
                st.warning("⚠️ No se pudo configurar el locale a español. Los formatos podrían no mostrarse correctamente.")

# --------------------------
# --- INPUT: subir archivos
# --------------------------
st.sidebar.header("Cargar archivos")
tareas_file = st.sidebar.file_uploader("Archivo tareas (Excel/CSV)", type=['xlsx', 'csv'])
dependencias_file = st.sidebar.file_uploader("Archivo dependencias (Excel/CSV)", type=['xlsx', 'csv'])
recursos_file = st.sidebar.file_uploader("Archivo recursos (Excel/CSV)", type=['xlsx', 'csv'])

if tareas_file and dependencias_file and recursos_file:

    # --------------------------
    # --- LEER DATAFRAMES
    # --------------------------
    if tareas_file.name.endswith('.csv'):
        tareas_df = pd.read_csv(tareas_file)
    else:
        tareas_df = pd.read_excel(tareas_file)

    if dependencias_file.name.endswith('.csv'):
        dependencias_df = pd.read_csv(dependencias_file)
    else:
        dependencias_df = pd.read_excel(dependencias_file)

    if recursos_file.name.endswith('.csv'):
        recursos_df = pd.read_csv(recursos_file)
    else:
        recursos_df = pd.read_excel(recursos_file)

    # --------------------------
    # --- LIMPIEZA BÁSICA
    # --------------------------
    tareas_df['RUBRO'] = tareas_df['RUBRO'].astype(str).str.strip()
    dependencias_df['CANTIDAD'] = dependencias_df['CANTIDAD'].astype(str).str.strip()
    recursos_df['RECURSO'] = recursos_df['RECURSO'].astype(str).str.strip()

    # Convertir fechas a datetime
    for col in ['FECHAINICIO','FECHAFIN']:
        if col in tareas_df.columns:
            tareas_df[col] = pd.to_datetime(tareas_df[col], errors='coerce')

    # --------------------------
    # --- MERGE: Recursos por Tarea
    # --------------------------
    try:
        recursos_tareas_df = dependencias_df.merge(
            tareas_df[['IDRUBRO','RUBRO','FECHAINICIO','FECHAFIN','DURACION']],
            left_on='CANTIDAD',
            right_on='RUBRO',
            how='left'
        )
    except KeyError as e:
        st.error(f"Error en merge de recursos y tareas: {e}")
        st.stop()

    # Asegurar que las columnas numéricas estén bien
    for col in ['DURACION','CANTIDAD']:
        if col in recursos_tareas_df.columns:
            recursos_tareas_df[col] = pd.to_numeric(recursos_tareas_df[col], errors='coerce').fillna(0)

    # --------------------------
    # --- GENERAR USO DIARIO DE RECURSOS
    # --------------------------
    daily_resource_usage_list = []

    for _, row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        resource_name = row['RECURSO']
        unit = row.get('UNIDAD','')
        total_quantity = row.get('CANTIDAD',0)
        start_date = row['FECHAINICIO']
        end_date = row['FECHAFIN']
        duration_days = row.get('DURACION',0)

        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            continue

        if duration_days <= 0:
            daily_quantity = total_quantity
            date_range = [start_date]
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            daily_quantity = total_quantity / max(len(date_range),1)

        temp_df = pd.DataFrame({
            'Fecha': date_range,
            'IDRUBRO': task_id,
            'RECURSO': resource_name,
            'UNIDAD': unit,
            'Cantidad_Diaria': daily_quantity
        })
        daily_resource_usage_list.append(temp_df)

    if daily_resource_usage_list:
        all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
    else:
        all_daily_resource_usage_df = pd.DataFrame(columns=['Fecha','IDRUBRO','RECURSO','UNIDAD','Cantidad_Diaria'])

    # --------------------------
    # --- DEMANDA DIARIA POR RECURSO
    # --------------------------
    daily_resource_demand_df = all_daily_resource_usage_df.groupby(
        ['Fecha','RECURSO','UNIDAD'], as_index=False
    )['Cantidad_Diaria'].sum()

    daily_resource_demand_df.rename(columns={'Cantidad_Diaria':'Demanda_Diaria_Total'}, inplace=True)

    # Merge con recursos para obtener tarifa y tipo
    resource_demand_with_details_df = daily_resource_demand_df.merge(
        recursos_df[['RECURSO','TYPE','TARIFA']],
        on='RECURSO',
        how='left'
    )

    # Convertir a numérico
    for col in ['Demanda_Diaria_Total','TARIFA']:
        if col in resource_demand_with_details_df.columns:
            resource_demand_with_details_df[col] = pd.to_numeric(resource_demand_with_details_df[col], errors='coerce').fillna(0)

    # Costo diario
    resource_demand_with_details_df['Costo_Diario'] = (
        resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']
    )

    st.subheader("Tabla de demanda de recursos con costos diarios")
    st.dataframe(resource_demand_with_details_df.head())

    # --------------------------
    # --- COSTOS MENSUALES ACUMULADOS
    # --------------------------
    resource_demand_with_details_df['Fecha'] = pd.to_datetime(resource_demand_with_details_df['Fecha'])
    resource_demand_with_details_df['Periodo_Mensual'] = resource_demand_with_details_df['Fecha'].dt.to_period('M')

    monthly_costs_df = resource_demand_with_details_df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
    monthly_costs_df['Periodo_Mensual'] = monthly_costs_df['Periodo_Mensual'].astype(str)
    monthly_costs_df['Costo_Acumulado'] = monthly_costs_df['Costo_Diario'].cumsum()

    # --------------------------
    # --- FORMATO MONEDA
    # --------------------------
    def format_currency(value):
        try:
            return f"S/. {locale.format_string('%.2f', value, grouping=True)}"
        except:
            return f"S/. {value:,.2f}"

    monthly_costs_df['Costo_Mensual_Formateado'] = monthly_costs_df['Costo_Diario'].apply(format_currency)
    monthly_costs_df['Costo_Acumulado_Formateado'] = monthly_costs_df['Costo_Acumulado'].apply(format_currency)

    # --------------------------
    # --- GRÁFICO COSTOS MENSUALES Y ACUMULADOS
    # --------------------------
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly_costs_df['Periodo_Mensual'],
        y=monthly_costs_df['Costo_Diario'],
        name='Costo Mensual',
        yaxis='y1',
        text=monthly_costs_df['Costo_Mensual_Formateado'],
        hoverinfo='text',
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=monthly_costs_df['Periodo_Mensual'],
        y=monthly_costs_df['Costo_Acumulado'],
        mode='lines+markers',
        name='Costo Acumulado',
        yaxis='y2',
        line=dict(color='red'),
        text=monthly_costs_df['Costo_Acumulado_Formateado'],
        hoverinfo='text',
        hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title='Cronograma Valorado - Costos Mensuales y Acumulados',
        yaxis=dict(
            title='Costo Mensual',
            side='left',
            showgrid=False
        ),
        yaxis2=dict(
            title='Costo Acumulado',
            overlaying='y',
            side='right',
            showgrid=True,
            gridcolor='lightgrey'
        ),
        xaxis=dict(
            title='Período Mensual',
            tickangle=-45
        ),
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
    st.info("⬆️ Por favor, sube los tres archivos: Tareas, Dependencias y Recursos.")




