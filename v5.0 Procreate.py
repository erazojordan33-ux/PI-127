import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Cronograma Valorado Interactivo", layout="wide")
st.title("📊 Cronograma Valorado y Recursos del Proyecto")

st.sidebar.header("Cargar archivo Excel con tres hojas")
excel_file = st.sidebar.file_uploader("Selecciona el archivo Excel", type=["xlsx"])

if excel_file:
    # Cargar todas las hojas
    xls = pd.ExcelFile(excel_file)
    hojas = xls.sheet_names
    
    # Asumimos que las hojas se llaman: Tareas, Recursos, Dependencias (ajustar si son otros nombres)
    hoja_tareas = hojas[0]
    hoja_recursos = hojas[1]
    hoja_dependencias = hojas[2]

    tareas_df = pd.read_excel(excel_file, sheet_name=hoja_tareas)
    recursos_df = pd.read_excel(excel_file, sheet_name=hoja_recursos)
    dependencias_df = pd.read_excel(excel_file, sheet_name=hoja_dependencias)

    # Normalizar nombres de columnas
    tareas_df.columns = tareas_df.columns.str.strip().str.upper()
    recursos_df.columns = recursos_df.columns.str.strip().str.upper()
    dependencias_df.columns = dependencias_df.columns.str.strip().str.upper()

    st.success("✅ Archivos cargados correctamente!")

    st.subheader("📋 Tabla de Tareas")
    st.dataframe(tareas_df)

    st.subheader("📋 Tabla de Recursos")
    st.dataframe(recursos_df)

    st.subheader("📋 Tabla de Dependencias / Asignación de Recursos")
    st.dataframe(dependencias_df)

    # Verificar columnas necesarias y renombrar si es necesario
    if 'CANTIDAD' not in dependencias_df.columns:
        st.error("❌ La columna 'CANTIDAD' no existe en dependencias_df.")
    if 'FECHAINICIO' not in tareas_df.columns or 'FECHAFIN' not in tareas_df.columns:
        st.error("❌ Las columnas 'FECHAINICIO' o 'FECHAFIN' no existen en tareas_df.")

    # Convertir fechas y normalizar strings
    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], errors='coerce')
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], errors='coerce')
    for col in ['RUBRO']:
        tareas_df[col] = tareas_df[col].astype(str).str.strip()
    dependencias_df['CANTIDAD'] = dependencias_df['CANTIDAD'].astype(float)
    for col in ['RECURSO', 'UNIDAD']:
        recursos_df[col] = recursos_df[col].astype(str).str.strip()

    # Merge recursos y tareas
    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
        left_on='CAN',  # o la columna que une dependencias con tareas
        right_on='RUBRO',
        how='left'
    )

    # Distribución diaria de recursos
    daily_resource_usage_list = []
    for _, row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        resource_name = row['RECURSO']
        unit = row['UNIDAD']
        total_quantity = row['CANTIDAD']
        start_date = row['FECHAINICIO']
        end_date = row['FECHAFIN']
        duration_days = row['DURACION']

        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
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
        all_daily_resource_usage_df = pd.DataFrame()

    st.subheader("📈 Uso Diario de Recursos")
    st.dataframe(all_daily_resource_usage_df)

    # Agrupar por recurso y calcular costo
    daily_resource_demand_df = all_daily_resource_usage_df.groupby(
        ['Fecha', 'RECURSO', 'UNIDAD'], as_index=False
    )['Cantidad_Diaria'].sum()
    daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)

    resource_demand_with_details_df = daily_resource_demand_df.merge(
        recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
        on='RECURSO', how='left'
    )
    resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']

    st.subheader("📊 Demanda Diaria de Recursos con Costos")
    st.dataframe(resource_demand_with_details_df)

    # Línea de tiempo de recursos
    fig_timeline = go.Figure()
    pastel_blue = 'rgb(174,198,207)'
    for i, row in recursos_tareas_df.iterrows():
        fig_timeline.add_trace(go.Scattergl(
            x=[row['FECHAINICIO'], row['FECHAFIN']],
            y=[row['RECURSO'], row['RECURSO']],
            mode='lines',
            line=dict(color=pastel_blue, width=10),
            hoverinfo='text',
            text=f"{row['RUBRO']} ({row['RECURSO']})"
        ))

    fig_timeline.update_layout(
        title="📊 Línea de Tiempo de Recursos",
        yaxis=dict(autorange="reversed"),
        xaxis_title="Fecha",
        yaxis_title="Recurso",
        height=500
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Costos mensuales y acumulados
    resource_demand_with_details_df['Fecha'] = pd.to_datetime(resource_demand_with_details_df['Fecha'])
    resource_demand_with_details_df['Periodo_Mensual'] = resource_demand_with_details_df['Fecha'].dt.to_period('M')
    monthly_costs_df = resource_demand_with_details_df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
    monthly_costs_df['Costo_Acumulado'] = monthly_costs_df['Costo_Diario'].cumsum()

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
    fig_costs.update_layout(
        title="💰 Costos Mensuales y Acumulados",
        xaxis_title="Período Mensual",
        yaxis_title="Costo",
        height=500
    )
    st.plotly_chart(fig_costs, use_container_width=True)



