import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
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

    # Transformar directamente en las columnas FECHAINICIO y FECHAFIN
    for col in ['FECHAINICIO','FECHAFIN']:
        # Convertir de string ISO a datetime
        tareas_df[col] = pd.to_datetime(tareas_df[col], errors='coerce')
        # Formatear como DD/MM/YYYY
        tareas_df[col] = tareas_df[col].dt.strftime('%d/%m/%Y')
    
    # Transformar las columnas de texto DD/MM/YYYY a datetime
    for col in ['FECHAINICIO','FECHAFIN']:
        tareas_df[col] = pd.to_datetime(tareas_df[col], dayfirst=True, errors='coerce')

    # --- Calcular duración ---
    tareas_df['DURACION'] = (tareas_df['FECHAFIN'] - tareas_df['FECHAINICIO']).dt.days
    tareas_df.loc[tareas_df['DURACION'] < 0, 'DURACION'] = 0  # prevenir negativos

    # --- Predecesoras ---
    tareas_df['PREDECESORAS'] = tareas_df['PREDECESORAS'].fillna('').astype(str)

    # --- Tarifas ---
    if 'TARIFA' in recursos_df.columns:
        recursos_df['TARIFA'] = pd.to_numeric(recursos_df['TARIFA'], errors='coerce').fillna(0)

    # --- Calculo ruta crítica ---
    es, ef, ls, lf, tf = {}, {}, {}, {}, {}
    duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
    all_task_ids = set(tareas_df['IDRUBRO'].tolist())
    dependencias = defaultdict(list)
    predecesoras_map = defaultdict(list)

    # Crear mapa de dependencias
    for _, row in tareas_df.iterrows():
        tid = row['IDRUBRO']
        pre_list = str(row['PREDECESORAS']).split(',')
        for pre in pre_list:
            pre = pre.strip()
            if pre:
                match = re.match(r'(\d+)', pre)
                if match:
                    pre_id = int(match.group(1))
                    if pre_id in all_task_ids:
                        dependencias[pre_id].append(tid)
                        predecesoras_map[tid].append((pre_id,'FC',0))  # tipo FC, desfase 0

    # --- Forward Pass ---
    # --- Forward Pass ---
    in_degree = {tid: len(predecesoras_map.get(tid,[])) for tid in all_task_ids}
    queue = deque([tid for tid in all_task_ids if in_degree[tid]==0])
    processed = set(queue)
    
    for tid in queue:
        # Tomar fecha inicial de la tabla
        start_value = tareas_df.loc[tareas_df['IDRUBRO']==tid,'FECHAINICIO'].values[0]
    
        # Controlador: si no es Timestamp, convertir
        if not isinstance(start_value, pd.Timestamp):
            try:
                start_value = pd.to_datetime(start_value, dayfirst=True)
            except:
                st.warning(f"Tarea {tid} tiene FECHAINICIO inválido. Se usará hoy como inicio temporal.")
                start_value = pd.Timestamp.today()
    
        es[tid] = start_value
        ef[tid] = es[tid] + timedelta(days=duracion_dict.get(tid,0))
    
    while queue:
        u = queue.popleft()
        # Seguridad: asegurarse que ef[u] sea Timestamp
        if u not in ef or not isinstance(ef[u], pd.Timestamp):
            st.warning(f"ef[{u}] no es una fecha válida, se salta esta tarea.")
            continue
    
        for v in dependencias.get(u,[]):
            for pre_id, tipo, desfase in predecesoras_map.get(v, []):
                if pre_id==u:
                    # Seguridad: ef[u] es Timestamp
                    potential_es = ef[u] + timedelta(days=desfase)
                    if v not in es or potential_es>es[v]:
                        es[v]=potential_es
                        ef[v]=es[v]+timedelta(days=duracion_dict.get(v,0))
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

    # --- Mostrar tabla ---
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
    
    # --- Diagrama de Gantt ---
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import re
    from datetime import timedelta, datetime
    st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
    
    cost_column_name = None
    if 'COSTO_TOTAL_RUBRO' in tareas_df.columns:
        cost_column_name = 'COSTO_TOTAL_RUBRO'
    elif 'COSTO_TOTAL_x' in tareas_df.columns: # Fallback if the merge created this
         cost_column_name = 'COSTO_TOTAL_x'
    elif 'COSTO_TOTAL' in tareas_df.columns: # Fallback for the originally calculated column
         cost_column_name = 'COSTO_TOTAL'
    
    if cost_column_name:
        # Convertir a numérico si aún no lo está. Usar errors='coerce' para convertir no-numéricos a NaN.
        tareas_df[cost_column_name] = pd.to_numeric(tareas_df[cost_column_name], errors='coerce')
        # Llenar NaN con 0 si es necesario para el cálculo o visualización
        tareas_df[cost_column_name] = tareas_df[cost_column_name].fillna(0)
    else:
        print("⚠️ Advertencia: No se encontró una columna de costos reconocida en el DataFrame.")
        # Crear una columna de costo con 0s si no se encontró ninguna
        tareas_df['COSTO_TOTAL_NUMERICO'] = 0
        cost_column_name = 'COSTO_TOTAL_NUMERICO'
    
    if 'IDRUBRO' in tareas_df.columns:
        # Ordenar por IDRUBRO de forma ascendente
        tareas_df = tareas_df.sort_values(['IDRUBRO'])
    else:
        st.warning("⚠️ Advertencia: Columna 'IDRUBRO' no encontrada para ordenar.")
    
    
    # Crear un índice numérico para el eje Y después de ordenar
    tareas_df['y_num'] = range(len(tareas_df))
    
    # --- Crear figura ---
    fig = go.Figure()
    
    fecha_inicio_col = 'FECHA_INICIO_TEMPRANA' if 'FECHA_INICIO_TEMPRANA' in tareas_df.columns else 'FECHAINICIO'
    fecha_fin_col = 'FECHA_FIN_TEMPRANA' if 'FECHA_FIN_TEMPRANA' in tareas_df.columns else 'FECHAFIN'
    
    if fecha_inicio_col not in tareas_df.columns or fecha_fin_col not in tareas_df.columns:
         st.warning("❌ Error: No se encontraron columnas de fechas de inicio/fin necesarias para dibujar el Gantt.")


    inicio_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_inicio_col].to_dict()
    fin_rubro_calc = tareas_df.set_index('IDRUBRO')[fecha_fin_col].to_dict()
    # También necesitamos saber si una tarea es crítica para colorear las flechas
    is_critical_dict = tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()
    
    
    # Reconstruir el grafo de dependencias (predecesoras a sucesores)
    # Necesitamos esto para dibujar las flechas correctamente
    dependencias = defaultdict(list)
    # We also need the dependency details for each pre->suc link (type and lag)
    predecesoras_map_details = defaultdict(list) # suc_id -> [(pre_id1, type1, lag1), (pre_id2, type2, lag2), ...]
    
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
                    tipo_relacion = match.group(2).upper() if match.group(2) else 'FC' # Default to FC
                    desfase = int(match.group(3)) if match.group(3) else 0 # Default to 0 lag
    
                    # Only add dependency if predecessor ID exists in the tasks list
                    if pre_id in tareas_df['IDRUBRO'].values:
                         dependencias[pre_id].append(tarea_id)
                         predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                    else:
                         st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                else:
                    if pre_entry != '': # Avoid warning for empty strings
                        st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")
    
    
    # --- Agregar franjas horizontales alternadas (basado en y_num) ---
    shapes = []
    # Color gris un poco más oscuro y menos transparente
    color_banda = 'rgba(220, 220, 220, 0.6)'
    
    # Iterar directamente sobre los valores del eje Y (y_num)
    for y_pos in range(len(tareas_df)):
        if y_pos % 2 == 0: # Para posiciones Y pares (0, 2, 4, ...)
            # Añadir un rectángulo que cubra la altura de esta fila
            shapes.append(
                dict(
                    type="rect",
                    xref="paper", # Referencia al área del gráfico (0 a 1)
                    yref="y",    # Referencia al eje Y de datos
                    x0=0,        # Cubre todo el ancho del área del gráfico (en coordenadas del paper)
                    x1=1,        # Cubre todo el ancho del área del gráfico (en coordenadas del paper)
                    y0=y_pos - 0.5, # Cubre desde la mitad inferior de la fila anterior (en coordenadas del eje Y de datos)
                    y1=y_pos + 0.5, # Hasta la mitad superior de la fila siguiente (en coordenadas del eje Y de datos)
                    fillcolor=color_banda,
                    layer="below", # Asegurar que esté detrás de las barras y líneas
                    line_width=0,
                )
            )

    
    color_no_critica_barra = 'lightblue' # Celeste para no críticas
    color_critica_barra = 'rgb(255, 133, 133)' # Rojo específico para críticas
    
    for i, row in tareas_df.iterrows():
        # Determinar color de la barra
        line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
        line_width = 12 # Ancho estándar para las barras
    
        # Usar las fechas calculadas para dibujar la barra
        start_date = row[fecha_inicio_col]
        end_date = row[fecha_fin_col]
    
        # Asegurarse de que las fechas son válidas antes de dibujar la barra
        if pd.isna(start_date) or pd.isna(end_date):
             st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea {row['RUBRO']} (ID {row['IDRUBRO']}). No se dibujará la barra.")
             continue
    
    
        # --- Formatear costo para el hover (sin locale, formato estándar S/. con separador de miles y 2 decimales) ---
        try:
            valor_costo = float(row.get(cost_column_name, 0))
            costo_formateado = f"S/. {valor_costo:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            # Esto convierte: 12345.67 → S/. 12.345,67
        except Exception:
            costo_formateado = "S/. 0,00"

    
        # Crear texto para hover
        hover_text = (
            f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
            f"🗓️ <b>Capítulo:</b> {row['CAPÍTULO']}<br>"
            f"📅 <b>Inicio</b> {start_date.strftime('%d/%m/%Y')}<br>"
            f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
            f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
            f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>" # Mostrar holgura si existe
            f"💰 <b>Costo:</b> {costo_formateado}"
        )
    
        fig.add_trace(go.Scatter(
            x=[start_date, end_date],
            y=[row['y_num'], row['y_num']],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            showlegend=False, # Ocultar leyenda de las barras
            hoverinfo='text',
            text=hover_text, # Usar el texto formateado con HTML
        ))
    
    # --- Flechas de dependencias (ortogonales con 1 o 2 codos) ---
    offset_days_horizontal = 5 # Número de días para el desfase horizontal en los codos (para CC/FC)
    
    # Definir colores específicos para las flechas/líneas de dependencia
    color_no_critica_flecha = 'blue'
    color_critica_flecha = 'red'
    
    # Iterar sobre las dependencias definidas (usando el grafo reconstruido)
    for pre_id, sucesores in dependencias.items():
        # Asegurarse de que la predecesora existe en el DF y tiene un y_num asignado
        pre_row_df = tareas_df[tareas_df['IDRUBRO'] == pre_id]
        if pre_row_df.empty:
            st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flechas desde este ID.")
            continue
        # Usamos .iloc[0] ya que esperamos solo una fila por ID
        y_pre = pre_row_df.iloc[0]['y_num']
        pre_is_critical = is_critical_dict.get(pre_id, False)
    
    
        # Obtener fechas calculadas de la predecesora (usando los diccionarios calculados)
        x_pre_inicio = inicio_rubro_calc.get(pre_id)
        x_pre_fin = fin_rubro_calc.get(pre_id)
    
        # Verificar si las fechas de la predecesora son válidas
        if pd.isna(x_pre_inicio) or pd.isna(x_pre_fin):
             st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el predecesor ID {pre_id}. No se dibujarán flechas desde este ID.")
             continue
    
    
        for suc_id in sucesores:
            # Asegurarse de que la sucesora existe en el DF y tiene un y_num asignado
            suc_row_df = tareas_df[tareas_df['IDRUBRO'] == suc_id]
            if suc_row_df.empty:
                st.warning(f"⚠️ Advertencia: Sucesor ID {suc_id} en el grafo de dependencias no encontrado en tareas_df. Saltando flecha hacia este ID.")
                continue
            # Usamos .iloc[0] ya que esperamos solo una fila por ID
            y_suc = suc_row_df.iloc[0]['y_num']
            suc_is_critical = is_critical_dict.get(suc_id, False)
    
            # Determinar color de la flecha: Rojo si AMBOS (predecesor y sucesor) son críticos, Azul en otro caso
            arrow_color = color_critica_flecha if pre_is_critical and suc_is_critical else color_no_critica_flecha
            line_style = dict(color=arrow_color, width=1, dash='dash') # Estilo de línea segmentada
    
            # Obtener fechas calculadas de la sucesora (usando los diccionarios calculados)
            x_suc_inicio = inicio_rubro_calc.get(suc_id)
            x_suc_fin = fin_rubro_calc.get(suc_id)
    
            # Verificar si las fechas de la sucesora son válidas
            if pd.isna(x_suc_inicio) or pd.isna(x_suc_fin):
                 st.warning(f"⚠️ Advertencia: Fechas calculadas no encontradas o inválidas para el sucesor ID {suc_id}. No se dibujará la flecha.")
                 continue
    
            tipo_relacion = 'FC' # Default if not found or parsed correctly
            for pre_id_suc, type_suc, desfase_suc in predecesoras_map_details.get(suc_id, []):
                if pre_id_suc == pre_id: # Found the dependency from pre_id to suc_id
                     tipo_relacion = type_suc.upper() if type_suc else 'FC'
                     # Note: desfase_suc is not needed for determining origin/connection points based on type
                     break # Found the specific dependency details, exit inner loop
    
    
            origin_x = x_pre_fin # Default FC for the graph visualization point
            if tipo_relacion == 'CC':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'CF':
                origin_x = x_pre_inicio
            elif tipo_relacion == 'FF':
                origin_x = x_pre_fin
            # Else handled by default
    
            # --- Determinar el punto de CONEXIÓN en la sucesora (Flecha) ---
            connection_x = x_suc_inicio # Default FC for the graph visualization point
            arrow_symbol = 'triangle-right' # Default FC for the graph visualization point
    
            if tipo_relacion == 'CC':
                connection_x = x_suc_inicio
                arrow_symbol = 'triangle-right'
            elif tipo_relacion == 'CF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left' # User indicated left arrow for CF
            elif tipo_relacion == 'FF':
                connection_x = x_suc_fin
                arrow_symbol = 'triangle-left' # User indicated left arrow for FF
            # Else handled by default
    
    
            # --- Dibujar el círculo en el punto de origen de la predecesora ---
            # Usar el mismo color que la flecha
            fig.add_trace(go.Scattergl(
                x=[origin_x],
                y=[y_pre],
                mode='markers',
                marker=dict(
                    symbol='circle', # Símbolo de círculo
                    size=8, # Tamaño del círculo
                    color=arrow_color, # Color del círculo (mismo que la flecha)
                ),
                hoverinfo='none',
                showlegend=False,
            ))
    
    
            # --- Dibujar los segmentos de línea ortogonal ---
            # Usar el color de la flecha para los segmentos
            line_style = dict(color=arrow_color, width=1, dash='dash') # Estilo de línea segmentada
    
            # Definir los puntos para los segmentos
            points_x = [origin_x]
            points_y = [y_pre]
    
            # Lógica de codos basada en el tipo de relación y el desfase horizontal
            if tipo_relacion in ['CC', 'FC']:
                # CC/FC path: Horizontal Left -> Vertical -> Horizontal Right (con 2 codos)
                # Horizontal a la izquierda desde el origen
                elbow1_x = origin_x - timedelta(days=offset_days_horizontal)
                elbow1_y = y_pre
    
                # Vertical hasta la altura de la sucesora
                elbow2_x = elbow1_x # Misma X que elbow1
                elbow2_y = y_suc   # Misma Y que sucesor
    
                # Segmento 1 (Horizontal Left)
                points_x.append(elbow1_x)
                points_y.append(elbow1_y)
    
                # Segmento 2 (Vertical)
                points_x.append(elbow2_x)
                points_y.append(elbow2_y)
    
                # Segmento 3 (Horizontal Right) hasta el punto de conexión
                points_x.append(connection_x)
                points_y.append(y_suc)
    
    
            elif tipo_relacion in ['CF', 'FF']:
                 # CF/FF path: Vertical -> Horizontal (con 1 codo)
                 # Codo: Misma X que origen, misma Y que sucesor
                 elbow1_x = origin_x # Misma X que origen
                 elbow1_y = y_suc    # Misma Y que sucesor
    
                 # Segmento 1 (Vertical)
                 points_x.append(elbow1_x)
                 points_y.append(elbow1_y)
    
                 # Segmento 2 (Horizontal) hasta el punto de conexión
                 points_x.append(connection_x)
                 points_y.append(y_suc)
    
            else:
                 # Si el tipo no es reconocido, no dibujamos los segmentos (ya se dio advertencia al determinar tipo)
                 st.warning(f"⚠️ Tipo de relación '{tipo_relacion}' no reconocido para dibujar segmentos de flecha entre ID {pre_id} y ID {suc_id}. Saltando flecha.")
                 continue
    
    
            # Dibujar los segmentos usando un solo Scatter con mode='lines'
            fig.add_trace(go.Scatter(
                x=points_x,
                y=points_y,
                mode='lines',
                line=line_style,
                hoverinfo='none', # No mostrar hover para la línea
                showlegend=False,
            ))
    
    
            # --- Añadir flecha como marcador en el punto de conexión de la sucesora ---
            # Usar el mismo color que la flecha
            fig.add_trace(go.Scattergl(
                x=[connection_x],
                y=[y_suc],
                mode='markers',
                marker=dict(
                    symbol=arrow_symbol, # Símbolo de flecha (izquierda o derecha)
                    size=10, # Tamaño de la flecha
                    color=arrow_color, # Color de la flecha (mismo que la línea)
                ),
                hoverinfo='none',
                showlegend=False,
            ))
    
    
    # --- Preparar etiquetas del eje Y con estilo alternado (Negrita) ---
    y_ticktext_styled = []
    # Iterar directamente sobre los valores de y_num para alinear estilo con posición visual
    for y_pos in range(len(tareas_df)):
        # Encontrar la fila del DataFrame que corresponde a esta posición y_num
        row_for_y_pos = tareas_df[tareas_df['y_num'] == y_pos]
        if not row_for_y_pos.empty:
            rubro_text = row_for_y_pos.iloc[0]['RUBRO']
            # Estilo alternado: negrita cada dos filas
            y_ticktext_styled.append(f"<b>{rubro_text}</b>" if y_pos % 2 == 0 else rubro_text)
    
        else:
             # Esto no debería pasar si y_num está bien asignado, pero como fallback
             y_ticktext_styled.append("") # Añadir una cadena vacía si no se encuentra el rubro
    
    
    fig.update_layout(
        xaxis=dict(
            title='Fechas',
            # tickformat=date_tick_format, # Eliminar formato explícito para usar default
            side='bottom', # Asegurar que el eje principal está abajo
            dtick='M1', # Forzar ticks principales mensualmente
            tickangle=-90, # Rotar etiquetas a -90 grados (vertical) para eje inferior
            showgrid=True, # Mostrar cuadrícula vertical
            gridcolor='rgba(128,128,128,0.3)', # Color gris semi-transparente para la cuadrícula
            gridwidth=0.5
        ),
        xaxis2=dict(
            title='Fechas',
            # tickformat=date_tick_format, # Eliminar formato explícito para usar default
            overlaying='x', # Superponer sobre el eje X principal
            side='top', # Posicionar arriba
            dtick='M1', # Forzar ticks principales mensualmente
            tickangle=90, # Rotar etiquetas a 90 grados (vertical) para eje superior
            showgrid=True, # Mostrar cuadrícula vertical
            gridcolor='rgba(128,128,128,0.3)', # Color gris semi-transparente para  cuadrícula
            gridwidth=0.5
        ),
        yaxis_title='Rubro',
        yaxis=dict(
            autorange='reversed',
            tickvals=tareas_df['y_num'],
            ticktext=y_ticktext_styled, # Usar la lista de etiquetas con estilo
            tickfont=dict(size=10),
            showgrid=False # No mostrar cuadrícula horizontal por defecto (las bandas la reemplazan)
        ),
        shapes=shapes, # Añadimos las formas de las franjas horizontales
        height=max(600, len(tareas_df) * 25), # Ajustar altura según número de tareas
        showlegend=False, # Ocultar leyenda de las barras
        plot_bgcolor='white',
        hovermode='closest'
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    
    # Recursos
    # Ensure date columns in tareas_df are datetime
    tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'])
    tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'])
    
    # Clean 'RUBRO' and 'CAN' columns for merging
    tareas_df['RUBRO'] = tareas_df['RUBRO'].str.strip()
    dependencias_df['RUBRO'] = dependencias_df['RUBRO'].str.strip()
        
    recursos_tareas_df = dependencias_df.merge(
        tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
        left_on='RUBRO',
        right_on='RUBRO',
        how='left'
    )
    
    # List to hold daily resource usage dataframes for each task/resource combination
    daily_resource_usage_list = []
    
    # Iterate over each row in the merged DataFrame
    for index, row in recursos_tareas_df.iterrows():
        task_id = row['IDRUBRO']
        resource_name = row['RECURSO']
        unit = row['UNIDAD']
        total_quantity = row['CANTIDAD']
        start_date = row['FECHAINICIO']
        end_date = row['FECHAFIN']
        duration_days = row['DURACION']
    
        # Ensure dates are valid and duration is non-negative
        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            st.warning(f"⚠️ Advertencia: Fechas inválidas para la tarea ID {task_id}, recurso '{resource_name}'. Saltando.")
            continue
    
        if duration_days <= 0:
            daily_quantity = total_quantity
            date_range = [start_date]
        else:
            # Calculate daily quantity assuming uniform distribution
            daily_quantity = total_quantity / (duration_days + 1) # Divide by number of days including start and end
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        temp_df = pd.DataFrame({
            'Fecha': date_range,
            'IDRUBRO': task_id,
            'RECURSO': resource_name,
            'UNIDAD': unit,
            'Cantidad_Diaria': daily_quantity,
            'Cantidad_Total_Tarea': total_quantity # Keep total quantity for reference
        })
    
        # Append the temporary DataFrame to the list
        daily_resource_usage_list.append(temp_df)

    # Concatenate all temporary DataFrames into one
    if daily_resource_usage_list:
        all_daily_resource_usage_df = pd.concat(daily_resource_usage_list, ignore_index=True)
    else:
        st.warning("\nNo se generaron datos de uso diario de recursos.")
        all_daily_resource_usage_df = pd.DataFrame() # Create an empty DataFrame

    # Group by Date, Resource, and Unit to sum up daily quantities
    daily_resource_demand_df = all_daily_resource_usage_df.groupby(
        ['Fecha', 'RECURSO', 'UNIDAD'],
        as_index=False
    )['Cantidad_Diaria'].sum()
    
    # Rename the aggregated quantity column for clarity
    daily_resource_demand_df.rename(columns={'Cantidad_Diaria': 'Demanda_Diaria_Total'}, inplace=True)
    
    daily_resource_demand_df['RECURSO'] = daily_resource_demand_df['RECURSO'].str.strip()
    recursos_df['RECURSO'] = recursos_df['RECURSO'].str.strip()
    
    resource_demand_with_details_df = daily_resource_demand_df.merge(
        recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
        on='RECURSO',
        how='left'
    )

    resource_demand_with_details_df['Costo_Diario'] = resource_demand_with_details_df['Demanda_Diaria_Total'] * resource_demand_with_details_df['TARIFA']
    
    # Group by Date and Type to sum daily costs
    daily_cost_by_type_df = resource_demand_with_details_df.groupby(
        ['Fecha', 'TYPE'],
        as_index=False
    )['Costo_Diario'].sum()
    
    # Group by Date and Resource to sum daily quantities
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
            line=dict(color=pastel_blue, width=10), # Use pastel blue for all bars
            name=row['RECURSO'], # Name for potential legend (though we will hide it)
            showlegend=False, # Hide default legend
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
        title='Línea de Tiempo de Uso de Recursos', # Updated title
        yaxis=dict(
            autorange="reversed",
            title="Recurso",
            tickfont=dict(size=10) # Adjust font size if needed
        ),
        xaxis=dict(
            title='Fechas',
            side='bottom',
            dtick='M1',  # Monthly ticks
            tickangle=-90, # Vertical labels
            showgrid=True,
            gridcolor='rgba(128,128,128,0.3)',
            gridwidth=0.5
        ),
        # Add a top x-axis for symmetry/clarity if desired (optional, mirroring bottom axis)
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
        height=max(600, len(recursos_tareas_df['RECURSO'].unique()) * 20), # Adjust height based on number of unique resources
        showlegend=False, # Hide default legend
        plot_bgcolor='white',
        hovermode='closest'
    )

    st.fig_resource_timeline(fig, use_container_width=True)





else:
    st.warning("Sube el archivo Excel con las hojas Tareas, Recursos y Dependencias.")



















