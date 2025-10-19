import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Cronograma Valorado Interactivo", layout="wide")
st.title("📊 Cronograma Valorado y Recursos del Proyecto")

st.sidebar.header("Cargar archivo Excel")
excel_file = st.sidebar.file_uploader("Selecciona el archivo Excel", type=["xlsx"])

if excel_file:
    # Mostrar hojas disponibles
    xls = pd.ExcelFile(excel_file)
    sheet_name = st.sidebar.selectbox("Selecciona la hoja a usar", xls.sheet_names)
    
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.upper()
    
    st.subheader(f"📋 Contenido de la hoja: {sheet_name}")
    st.dataframe(df)

    # Verificar columnas necesarias
    columnas_necesarias = ['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION', 'RECURSO', 'CANTIDAD', 'UNIDAD', 'TARIFA', 'TYPE']
    for col in columnas_necesarias:
        if col not in df.columns:
            st.error(f"❌ La columna '{col}' no existe en la hoja seleccionada.")
    
    # Convertir fechas
    if 'FECHAINICIO' in df.columns:
        df['FECHAINICIO'] = pd.to_datetime(df['FECHAINICIO'], errors='coerce')
    if 'FECHAFIN' in df.columns:
        df['FECHAFIN'] = pd.to_datetime(df['FECHAFIN'], errors='coerce')
    
    # Normalizar strings
    for col in ['RUBRO', 'RECURSO', 'TYPE', 'UNIDAD']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Crear distribución diaria de recursos
    daily_resource_usage_list = []
    if all(col in df.columns for col in ['FECHAINICIO', 'FECHAFIN', 'DURACION', 'CANTIDAD', 'RECURSO', 'UNIDAD', 'IDRUBRO', 'RUBRO']):
        for _, row in df.iterrows():
            start_date = row['FECHAINICIO']
            end_date = row['FECHAFIN']
            duration_days = row['DURACION']
            total_quantity = row['CANTIDAD']
            resource_name = row['RECURSO']
            unit = row['UNIDAD']
            task_id = row['IDRUBRO']

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
                'Cantidad_Total_Tarea': total_quantity,
                'RUBRO': row['RUBRO']
            })
            daily_resource_usage_list.append(temp_df)

    if daily_resource_usage_list:
        all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
    else:
        all_daily_resource_usage_df = pd.DataFrame()

    st.subheader("📈 Uso Diario de Recursos")
    st.dataframe(all_daily_resource_usage_df)

    # Agrupar por recurso y calcular costo
    if all(col in df.columns for col in ['RECURSO', 'TARIFA', 'TYPE']):
        daily_resource_demand_df = all_daily_resource_usage_df.groupby(
            ['Fecha', 'RECURSO', 'UNIDAD'], as_index=False
        )['Cantidad_Diaria'].sum()
        daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)

        resource_demand_with_details_df = daily_resource_demand_df.merge(
            df[['RECURSO', 'TYPE', 'TARIFA']].drop_duplicates(),
            on='RECURSO', how='left'
        )
        resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']

        st.subheader("📊 Demanda Diaria de Recursos con Costos")
        st.dataframe(resource_demand_with_details_df)

        # Línea de tiempo de recursos
        fig_timeline = go.Figure()
        pastel_blue = 'rgb(174,198,207)'
        for _, row in df.iterrows():
            if pd.notna(row.get('FECHAINICIO')) and pd.notna(row.get('FECHAFIN')):
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

