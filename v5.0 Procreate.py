# 1 Declarar e importar bibliotecas___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
from datetime import timedelta
from collections import defaultdict, deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import math
import plotly.express as px

# 2. Definir archivo y pestañas___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

st.set_page_config(page_title="Gestión de Proyectos - Cronograma Valorado", layout="wide")
st.title("📊 Gestión de Proyectos - Seguimiento y Control")

archivo_excel = st.file_uploader("Subir archivo Excel con hojas Tareas, Recursos y Dependencias", type=["xlsx"])
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Inicio","Calendario","Diagrama Gantt", "Recursos", "Presupuesto"])

# 3. Definir funciones de calculo___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
##1
def calculo_ruta_critica(tareas_df=None, archivo=None):
        fecha_inicio_proyecto = st.session_state.get("fecha_inicio_proyecto", None)
        if fecha_inicio_proyecto is not None:
                existe_inicio = (
                        not tareas_df[
                                (tareas_df["IDRUBRO"] == 0) &
                                (tareas_df["RUBRO"].astype(str).str.lower() == "comienzo del proyecto")
                        ].empty
                )

                if not existe_inicio:
                        fila_inicio = pd.DataFrame([{
                                "IDRUBRO": 0,
                                "RUBRO": "Comienzo del Proyecto",
                                "PREDECESORAS": "",
                                "FECHAINICIO": fecha_inicio_proyecto,
                                "FECHAFIN": fecha_inicio_proyecto,
                                "DURACION": 0,
                                "UNIDAD_RUBRO": "",
                                "RENDIMIENTO": "",
                                "HOLGURA_TOTAL": 0,
                                "RUTA_CRITICA": ""
                        }])

                        tareas_df = pd.concat([fila_inicio, tareas_df], ignore_index=True)
                        tareas_df["IDRUBRO"] = tareas_df["IDRUBRO"].astype(int)
        
                        tareas_df.loc[
                                (tareas_df["IDRUBRO"] != 0) &
                                ((tareas_df["PREDECESORAS"].isna()) | (tareas_df["PREDECESORAS"].astype(str).str.strip() == "")),
                                "PREDECESORAS"
                        ] = "0FC"
        else:
                st.error("❌ Error: No se ha definido la fecha de inicio del proyecto antes de calcular la ruta crítica.")

        tareas_df.columns = tareas_df.columns.str.strip()

        duracion_dict = tareas_df.set_index('IDRUBRO')['DURACION'].to_dict()
        dependencias = defaultdict(list)
        predecesoras_map = defaultdict(list)
        all_task_ids = set(tareas_df['IDRUBRO'].tolist())
        es = {}
        ef = {}
        ls = {}
        lf = {}
        tf = {}
        ff = {}

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
        initial_tasks_ids = [tid for tid in all_task_ids if tid not in predecesoras_map]

        for tid in initial_tasks_ids:
                task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
                duration = duracion_dict.get(tid, 0)
                if not isinstance(duration, (int, float)):
                        duration = 0
                es[tid] = fecha_inicio_proyecto
                ef[tid] = es[tid] + timedelta(days=duration)

        predecessor_process_count = defaultdict(int)
        in_degree = {tid: len(predecesoras_map.get(tid, [])) for tid in all_task_ids}
        queue = deque([tid for tid in all_task_ids if in_degree[tid] == 0])
        processed_forward = set(queue)

        for tid in list(queue):
                task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
                if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
                        es[tid] = task_row.iloc[0]['FECHAINICIO']
                        duration = duracion_dict.get(tid, 0)
                        if not isinstance(duration, (int, float)): duration = 0
                        ef[tid] = es[tid] + timedelta(days=duration)
                else:
                        pass

        while queue:
                u = queue.popleft()
                for v in dependencias.get(u, []):
                        for pre_id_v, tipo_v, desfase_v in predecesoras_map.get(v, []):
                                if pre_id_v == u:
                                        potential_es_v = None
                                        duration_v = duracion_dict.get(v, 0)
                                        if not isinstance(duration_v, (int, float)): duration_v = 0
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
                                                        st.error(f"⚠️ Tipo de relación '{tipo_v}' no reconocido para calcular ES de tarea {v} basada en {u}. Usando lógica FC por defecto.")
                                                        potential_es_v = ef[u] + timedelta(days=desfase_v)

                                                if v not in es or (potential_es_v is not None and potential_es_v > es[v]):
                                                        es[v] = potential_es_v

                                                if v in es:
                                                        duration_v_calc = duracion_dict.get(v, 0)
                                                        if not isinstance(duration_v_calc, (int, float)): duration_v_calc = 0
                                                        ef[v] = es[v] + timedelta(days=duration_v_calc)
                                        else:
                                                st.error(f"⚠️ Advertencia: ES/EF no calculados para predecesor ID {u} al procesar sucesor ID {v}. Saltando cálculo de ES/EF para v basado en u.")

                        in_degree[v] -= 1
                        if in_degree[v] == 0 and v not in processed_forward:
                                queue.append(v)
                                processed_forward.add(v)

        unprocessed_forward = all_task_ids - processed_forward
        if unprocessed_forward:
                for tid in unprocessed_forward:
                        if tid not in es:
                                task_row = tareas_df[tareas_df['IDRUBRO'] == tid]
                                if not task_row.empty and pd.notna(task_row.iloc[0]['FECHAINICIO']):
                                        es[tid] = task_row.iloc[0]['FECHAINICIO']
                                        duration = duracion_dict.get(tid, 0)
                                        if not isinstance(duration, (int, float)): duration = 0
                                        ef[tid] = es[tid] + timedelta(days=duration)
                                else:
                                        st.error(f"❌ Error: Tarea no procesada {tid} no encontrada o FECHAINICIO inválida. No se pudo inicializar ES/EF.")

        end_tasks_ids = [tid for tid in all_task_ids if tid not in dependencias]
        tasks_without_successors = [tid for tid in all_task_ids if tid not in dependencias]

        project_finish_date = max(ef.values())  
        end_tasks_ids = [tid for tid, fecha in ef.items() if fecha == project_finish_date]

        for tid in end_tasks_ids:
                lf[tid] = project_finish_date
                duration = duracion_dict.get(tid, 0)
                ls[tid] = lf[tid] - timedelta(days=duration)

        successor_map = defaultdict(list)
        for tid, pre_list in predecesoras_map.items():
                for pre_id, tipo, desfase in pre_list:
                        successor_map[pre_id].append((tid, tipo, desfase))

        project_finish_date = max(ef.values())
        end_tasks_ids = [tid for tid, fecha in ef.items() if fecha == project_finish_date]

        lf.update({tid: project_finish_date for tid in end_tasks_ids})
        for tid in end_tasks_ids:
                duration = duracion_dict.get(tid, 0)
                if not isinstance(duration, (int, float)): duration = 0
                ls[tid] = lf[tid] - timedelta(days=duration)

        queue_backward = deque(end_tasks_ids)
        processed_backward = set(end_tasks_ids)

        successor_map = defaultdict(list)
        for tid, pre_list in predecesoras_map.items():
                for pre_id, tipo, desfase in pre_list:
                        successor_map[pre_id].append((tid, tipo, desfase))

        while queue_backward:
                v = queue_backward.popleft()

        for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
                duration_u = duracion_dict.get(u, 0)
                if not isinstance(duration_u, (int, float)): duration_u = 0
                if tipo_relacion_uv == 'FC':
                        candidate_lf = ls[v] - timedelta(days=desfase_uv)
                        candidate_ls = candidate_lf - timedelta(days=duration_u)
                elif tipo_relacion_uv == 'CC':
                        candidate_ls = ls[v] - timedelta(days=desfase_uv)
                        candidate_lf = candidate_ls + timedelta(days=duration_u)
                elif tipo_relacion_uv == 'CF':
                        candidate_ls = lf[v] - timedelta(days=desfase_uv)
                        candidate_lf = candidate_ls + timedelta(days=duration_u)
                elif tipo_relacion_uv == 'FF':
                        candidate_lf = lf[v] - timedelta(days=desfase_uv)
                        candidate_ls = candidate_lf + timedelta(days=duration_u)
                else:
                        candidate_lf = ls[v] + timedelta(days=desfase_uv)
                if u not in lf:
                        lf[u] = candidate_lf
                else:
                        lf[u] = min(lf[u], candidate_lf)
                if u not in ls:
                        ls[u] = candidate_ls
                else:
                        ls[u] = min(ls[u], candidate_ls)
                if u not in processed_backward:
                        queue_backward.append(u)
                        processed_backward.add(u)

        queue_special_backward = deque(tasks_without_successors)
        processed_special_backward = set(tasks_without_successors)
        
        print("🔹 Tareas sin sucesoras:")
        for tid in tasks_without_successors:
                print(f"- {tid}")

        for tid in tasks_without_successors:
                duration = duracion_dict.get(tid, 0)
                if not isinstance(duration, (int, float)):
                        duration = 0
                lf[tid] = project_finish_date
                ls[tid] = lf[tid] - timedelta(days=duration)
                print(f"Tarea: {tid} | Duración: {duration} | LF: {lf[tid]} | LS: {ls[tid]}")

        while queue_special_backward:
                v = queue_special_backward.popleft()
                for u, tipo_relacion_uv, desfase_uv in predecesoras_map.get(v, []):
                        duration_u = duracion_dict.get(u, 0)
                        if not isinstance(duration_u, (int, float)):
                                duration_u = 0
                        if tipo_relacion_uv == 'FC':
                                candidate_lf = ls[v] - timedelta(days=desfase_uv)
                                candidate_ls = candidate_lf - timedelta(days=duration_u)
                        elif tipo_relacion_uv == 'CC':
                                candidate_ls = ls[v] - timedelta(days=desfase_uv)
                                candidate_lf = candidate_ls + timedelta(days=duration_u)
                        elif tipo_relacion_uv == 'CF':
                                candidate_ls = lf[v] - timedelta(days=desfase_uv)
                                candidate_lf = candidate_ls + timedelta(days=duration_u)
                        elif tipo_relacion_uv == 'FF':
                                candidate_lf = lf[v] - timedelta(days=desfase_uv)
                                candidate_ls = candidate_lf - timedelta(days=duration_u)
                        else:
                                candidate_lf = ls[v] + timedelta(days=desfase_uv)
                                candidate_ls = candidate_lf - timedelta(days=duration_u)
                        if u in lf:
                                if candidate_lf < lf[u]:
                                        lf[u] = candidate_lf
                                        ls[u] = lf[u] - timedelta(days=duration_u)
                        else:
                                        lf[u] = candidate_lf
                                        ls[u] = candidate_ls

                        if lf[u] > project_finish_date:
                                lf[u] = project_finish_date
                                ls[u] = lf[u] - timedelta(days=duration_u)
                                
                                queue_forward = deque([u])
                                processed_forward = set()

                                while queue_forward:
                                        x = queue_forward.popleft()
                                        if x in processed_forward:
                                                continue
                                        processed_forward.add(x)

                                for succ, tipo_relacion_xs, desfase_xs in successor_map.get(x, []):
                                        dur_succ = duracion_dict.get(succ, 0)
                                        if not isinstance(dur_succ, (int, float)):
                                                dur_succ = 0
                                        if tipo_relacion_xs == 'FC':
                                                lf[succ] = min(lf.get(succ, lf[x] + timedelta(days=desfase_xs)), lf[x] + timedelta(days=desfase_xs))
                                                ls[succ] = lf[succ] - timedelta(days=dur_succ)
                                        elif tipo_relacion_xs == 'CC':
                                                ls[succ] = ls[x] + timedelta(days=desfase_xs)
                                                lf[succ] = ls[succ] + timedelta(days=dur_succ)
                                        elif tipo_relacion_xs == 'CF':
                                                ls[succ] = lf[x] + timedelta(days=desfase_xs)
                                                lf[succ] = ls[succ] + timedelta(days=dur_succ)
                                        elif tipo_relacion_xs == 'FF':
                                                lf[succ] = lf[x] + timedelta(days=desfase_xs)
                                                ls[succ] = lf[succ] - timedelta(days=dur_succ)
                                        if lf[succ] > project_finish_date:
                                                lf[succ] = project_finish_date
                                                ls[succ] = lf[succ] - timedelta(days=dur_succ)
                                                queue_forward.append(succ)
                                                
                        if u not in processed_special_backward:
                                queue_special_backward.append(u)
                                processed_special_backward.add(u)
                
        for tid in all_task_ids:
                if tid in ef and tid in lf:
                        tf[tid] = lf[tid] - ef[tid]
                        if tf[tid].total_seconds() < -1e-9:
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
                                                                if not isinstance(duration_suc, (int, float)): duration_suc = 0
                                                                required_start_suc = (es[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                                                        elif tipo_suc == 'FF':
                                                                duration_suc = duracion_dict.get(suc_id, 0)
                                                                if not isinstance(duration_suc, (int, float)): duration_suc = 0
                                                                required_start_suc = (ef[tid] + timedelta(days=desfase_suc)) - timedelta(days=duration_suc)
                                                        else:
                                                                required_start_suc = ef[tid] + timedelta(days=desfase_suc)
                                                        if required_start_suc is not None:
                                                                if min_successor_es is None or required_start_suc < min_successor_es:
                                                                        min_successor_es = required_start_suc
                                                        break

                        if min_successor_es is not None and tid in ef:
                                ff[tid] = min_successor_es - ef[tid]
                                if ff[tid].total_seconds() < -1e-9:
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
        tareas_df['HOLGURA_TOTAL'] = tareas_df['HOLGURA_TOTAL_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
        tareas_df['HOLGURA_LIBRE'] = tareas_df['HOLGURA_LIBRE_TD'].apply(lambda x: x.days if pd.notna(x) else pd.NA)
        tolerance_days = 1e-9
        tareas_df['RUTA_CRITICA'] = tareas_df['HOLGURA_TOTAL'].apply(lambda x: abs(x) < tolerance_days if pd.notna(x) else False)

        return tareas_df
        
##2
def calculo_predecesoras(df, fila_editada):

        row = df.loc[fila_editada]
        
        if row['RUTA_CRITICA']: 
                criticas = df[(df['RUTA_CRITICA'] == True) & 
                        (df.index != fila_editada) & 
                        (df['FECHAINICIO'] > row['FECHAINICIO'])]
                if not criticas.empty:
                        fila_predecesora = criticas.loc[criticas['FECHAINICIO'].idxmin()]
                        nuevo_valor = f"{row['IDRUBRO']}FC"
                        idx_predecesora = fila_predecesora.name
                        if pd.isna(df.at[idx_predecesora, 'PREDECESORAS']) or df.at[idx_predecesora, 'PREDECESORAS'] == "":
                                df.at[idx_predecesora, 'PREDECESORAS'] = nuevo_valor
                        else:
                                df.at[idx_predecesora, 'PREDECESORAS'] += f", {nuevo_valor}"
        else: 
                id_a_eliminar = str(row['IDRUBRO'])
                for idx, pre_row in df.iterrows():
                        predecesoras = pre_row['PREDECESORAS']
                        if pd.notna(predecesoras) and predecesoras != "":
                                partes = [p.strip() for p in predecesoras.split(",")]
                                nuevas_partes = []
                                for p in partes:
                                        match = re.match(r"(\d+)(FC|CC|FF|CF)", p)
                                        if match:
                                                if match.group(1) != id_a_eliminar:
                                                        nuevas_partes.append(p)
                                                else:
                                                        nuevas_partes.append(p)
                                df.at[idx, 'PREDECESORAS'] = ", ".join(nuevas_partes)
        return df

# Definicion de variables y calculo___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

if archivo_excel:

        try:
            st.session_state.tareas_df_original = pd.read_excel(archivo_excel, sheet_name='Tareas')
            st.session_state.recursos_df = pd.read_excel(archivo_excel, sheet_name='Recursos')
            st.session_state.dependencias_df = pd.read_excel(archivo_excel, sheet_name='Dependencias')
            st.session_state.tareas_df = st.session_state.tareas_df_original.copy()
          
        except:
            st.error(f"Error al leer el archivo Excel. Asegúrese de que contiene las hojas 'Tareas', 'Recursos' y 'Dependencias' ")
            st.stop()

# Mostrar variables en la Pestaña 1___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

        with tab1:
            st.markdown("#### Datos Importados:")

            st.subheader("📋 Tabla Tareas")
            gb = GridOptionsBuilder.from_dataframe(st.session_state.tareas_df_original)
            gb.configure_default_column(editable=False)

            gb.configure_column(
                    "CANTIDAD_RUBRO",
                    type=["numericColumn", "numberColumnFilter", "customNumericFormat"],
                    precision=2
            )
                
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

# Mostrar variables en la Pestaña 2___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

        import calendar
                
        def plot_month_calendar(calendario_df: pd.DataFrame, year: int, month: int) -> go.Figure:
                    """
                    Dibuja un calendario mensual tipo cuadrícula (lun-dom) con colores
                    para días laborables / no laborables usando plotly.go.Table.
                    calendario_df debe tener columna 'fecha' (datetime) y 'no_laborable' (bool).
                    """
                    # Asegurar que 'fecha' es datetimelike y sin hora
                    df = calendario_df.copy()
                    df["fecha"] = pd.to_datetime(df["fecha"]).dt.normalize()
                    cal = calendar.monthcalendar(year, month)  # lista de semanas (listas)
                    weekday_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
                
                    # Construir filas (semanas) de valores, colores y textos hover
                    table_rows = []
                    fill_rows = []
                
                    for week in cal:
                        row_vals = []
                        row_colors = []
                        for day in week:
                            if day == 0:
                                row_vals.append("") 
                                row_colors.append("white")
                            else:
                                fecha = pd.Timestamp(year, month, day)
                                sel = df[df["fecha"] == fecha]
                                is_no = sel["no_laborable"].any() if not sel.empty else False
                                row_vals.append(str(day))
                                # Colores: rojo para no laborable, verde para laborable
                                row_colors.append("#ffb3b3" if is_no else "#d4f7d4")
                        table_rows.append(row_vals)
                        fill_rows.append(row_colors)
                
                    # Transponer rows->columns porque go.Table espera columnas en 'cells.values'
                    if len(table_rows) == 0:
                        # mes vacío (no debería pasar), crear matriz vacía de 6 semanas x 7 días
                        table_rows = [[""]*7 for _ in range(6)]
                        fill_rows = [["white"]*7 for _ in range(6)]
                
                    table_cols = list(map(list, zip(*table_rows)))
                    fill_cols = list(map(list, zip(*fill_rows)))
                
                    # Construir la tabla con colores por celda
                    fig = go.Figure(data=[go.Table(
                        header=dict(values=weekday_names, align='center', fill_color="#f2f2f2"),
                        cells=dict(values=table_cols,
                                   fill_color=fill_cols,
                                   align='center',
                                   height=55,
                                   font=dict(size=14))
                    )])
                
                    fig.update_layout(height=400, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="white")
                    return fig

        with tab2:

            st.subheader("Configuración de días laborables")
            st.markdown("<hr>", unsafe_allow_html=True)
        
            # 1️⃣ Fechas de inicio y fin del proyecto
            fecha_inicio_default = st.session_state.tareas_df['FECHAINICIO'].min()
            fecha_fin_default = st.session_state.tareas_df['FECHAFIN'].max()
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio_proyecto = st.date_input(
                    "Fecha de inicio del proyecto", value=fecha_inicio_default
                )
            with col2:
                fecha_fin_proyecto = st.date_input(
                    "Fecha de fin del proyecto", value=fecha_fin_default
                )

            st.session_state.fecha_inicio_proyecto = fecha_inicio_proyecto
            st.session_state.fecha_fin_proyecto = fecha_fin_proyecto
        
            st.markdown("---")
            st.write("Indicar días no laborables")
            opciones_no_laborables = ["Sábados y Domingos", "Sábados", "Domingos", "24/6", "22/8", "Personalizado"]
            dias_no_laborables_seleccionados = st.multiselect(
                "Selecciona los días no laborables",
                options=opciones_no_laborables,
                default=["Sábados y Domingos"]
            )
        
            # 2️⃣ Rangos personalizados múltiples
            if "Personalizado" in dias_no_laborables_seleccionados:
                st.write("Agregar/borrar rangos de días no laborables")
                if "rangos_personalizados" not in st.session_state:
                    st.session_state.rangos_personalizados = []
        
                # Botón para agregar rango
                if st.button("Agregar rango"):
                    st.session_state.rangos_personalizados.append({
                        "inicio": fecha_inicio_proyecto,
                        "fin": fecha_inicio_proyecto
                    })
        
                # Mostrar rangos existentes con opción de borrar
                for idx, rango in enumerate(st.session_state.rangos_personalizados):
                    col1, col2, col3 = st.columns([3,3,1])
                    with col1:
                        inicio = st.date_input(f"Rango {idx+1} inicio", value=rango["inicio"], key=f"inicio_{idx}")
                    with col2:
                        fin = st.date_input(f"Rango {idx+1} fin", value=rango["fin"], key=f"fin_{idx}")
                    with col3:
                        if st.button("❌", key=f"borrar_{idx}"):
                            st.session_state.rangos_personalizados.pop(idx)
                            st.experimental_rerun()
                    st.session_state.rangos_personalizados[idx] = {"inicio": inicio, "fin": fin}
        
            st.markdown("---")
            st.write("Horas laborables por día")
            horas_por_dia = st.number_input("Horas de trabajo diarias", min_value=1, max_value=24, value=8, step=1)
        
            # 3️⃣ Generar calendario visual
            if fecha_inicio_proyecto and fecha_fin_proyecto:
                fechas_proyecto = pd.date_range(fecha_inicio_proyecto, fecha_fin_proyecto)
                calendario_df = pd.DataFrame({"fecha": fechas_proyecto})
                calendario_df["no_laborable"] = False
                calendario_df["dia_semana"] = calendario_df["fecha"].dt.weekday  # lunes=0, domingo=6
                calendario_df["mes"] = calendario_df["fecha"].dt.to_period('M')
                calendario_df["dia"] = calendario_df["fecha"].dt.day
                calendario_df["semana"] = calendario_df["fecha"].dt.isocalendar().week
        
                # Días no laborables fijos
                for opcion in dias_no_laborables_seleccionados:
                    if opcion == "Sábados y Domingos":
                        calendario_df.loc[calendario_df["dia_semana"].isin([5,6]), "no_laborable"] = True
                    elif opcion == "Sábados":
                        calendario_df.loc[calendario_df["dia_semana"] == 5, "no_laborable"] = True
                    elif opcion == "Domingos":
                        calendario_df.loc[calendario_df["dia_semana"] == 6, "no_laborable"] = True
                    elif opcion == "24/6":
                        for m in calendario_df["mes"].unique():
                            ultimos6 = calendario_df[calendario_df["mes"]==m].tail(6).index
                            calendario_df.loc[ultimos6, "no_laborable"] = True
                    elif opcion == "22/8":
                        for m in calendario_df["mes"].unique():
                            ultimos8 = calendario_df[calendario_df["mes"]==m].tail(8).index
                            calendario_df.loc[ultimos8, "no_laborable"] = True
        
                # Rangos personalizados
                for rango in st.session_state.get("rangos_personalizados", []):
                    inicio_rango = pd.to_datetime(rango["inicio"])
                    fin_rango = pd.to_datetime(rango["fin"])
                    calendario_df.loc[(calendario_df["fecha"] >= inicio_rango) & (calendario_df["fecha"] <= fin_rango), "no_laborable"] = True
        
                st.session_state.calendario = calendario_df.copy()
                
                if "calendario" in st.session_state and not st.session_state.calendario.empty:
                    calendario_df = st.session_state.calendario.copy()
                    # Valores por defecto basados en fechas del proyecto si existen
                    fecha_min = calendario_df["fecha"].min()
                    fecha_max = calendario_df["fecha"].max()
                    default_year = int(fecha_min.year) if pd.notna(fecha_min) else pd.Timestamp.now().year
                    default_month = int(fecha_min.month) if pd.notna(fecha_min) else pd.Timestamp.now().month
                
                    st.markdown("---")
                    st.subheader("Vista mensual del calendario")
                    col_a, col_b = st.columns([1,2])
                    with col_a:
                        year = st.number_input("Año", min_value=1900, max_value=3000, value=default_year, step=1)
                    with col_b:
                        month = st.selectbox("Mes", list(range(1,13)), index=default_month-1,
                                             format_func=lambda m: calendar.month_name[m].capitalize())
                
                    # Renderizar figura
                    fig_month = plot_month_calendar(calendario_df, year, month)
                    st.plotly_chart(fig_month, use_container_width=True)
                
                else:
                    st.info("El calendario aún no está generado. Asegúrate de tener fechas válidas para el proyecto.")

# Definicion calculo___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
        try:
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

                tareas_df = st.session_state.tareas_df.copy()
                calendario_df = st.session_state.calendario.copy()

                tareas_df['FECHAINICIO'] = pd.to_datetime(tareas_df['FECHAINICIO'], errors='coerce')
                tareas_df['FECHAFIN'] = pd.to_datetime(tareas_df['FECHAFIN'], errors='coerce')
                calendario_df['fecha'] = pd.to_datetime(calendario_df['fecha'], errors='coerce')

                no_laborables = set(calendario_df.loc[calendario_df['no_laborable'] == True, 'fecha'].dt.date)
                
                def calcular_duracion_efectiva(row):
                        if pd.isna(row['FECHAINICIO']) or pd.isna(row['FECHAFIN']):
                                return None
                        rango_fechas = pd.date_range(start=row['FECHAINICIO'], end=row['FECHAFIN'], freq='D')
                        dias_no_lab = sum(fecha.date() in no_laborables for fecha in rango_fechas)
                        duracion_total = len(rango_fechas)
                        duracion_efectiva = duracion_total - dias_no_lab
                        return duracion_efectiva
                
                tareas_df['DURACION_EFECTIVA'] = tareas_df.apply(calcular_duracion_efectiva, axis=1)

                tareas_df['RENDIMIENTO'] = tareas_df.apply(
                        lambda x: x['CANTIDAD_RUBRO'] / x['DURACION_EFECTIVA']
                        if pd.notna(x['CANTIDAD_RUBRO']) and pd.notna(x['DURACION_EFECTIVA']) and x['DURACION_EFECTIVA'] != 0
                        else None,
                        axis=1
                )
                
                st.session_state.tareas_df = tareas_df
                
        except:
                st.error(f"Error al tratar datos. Asegúrese del contenido de la base ")
                st.stop()
        
        st.session_state.tareas_df=calculo_ruta_critica(st.session_state.tareas_df)

        if all(col in st.session_state.tareas_df.columns for col in ["FECHA_INICIO_TEMPRANA", "FECHA_FIN_TEMPRANA"]):
            st.session_state.tareas_df["FECHAINICIO"] = st.session_state.tareas_df["FECHA_INICIO_TEMPRANA"]
            st.session_state.tareas_df["FECHAFIN"] = st.session_state.tareas_df["FECHA_FIN_TEMPRANA"]
        else:
            st.warning("⚠️ No se encontraron las columnas FECHA_INICIO_TEMPRANA o FECHA_FIN_TEMPRANA en el DataFrame.")
                
        if "tareas_df_prev" not in st.session_state:
                st.session_state.tareas_df_prev = st.session_state.tareas_df.copy()

# Mostrar variables en la Pestaña 3___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________            
        with tab3:
                st.subheader("📋 Tareas con Fechas Calculadas y Ruta Crítica")
                cols1 = [
                    'IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                    'DURACION_EFECTIVA','UNIDAD_RUBRO','RENDIMIENTO','RUTA_CRITICA'
                ]

                cols = [
                    'IDRUBRO','RUBRO','PREDECESORAS','FECHAINICIO','FECHAFIN',
                    'FECHA_INICIO_TEMPRANA','FECHA_FIN_TEMPRANA',
                    'FECHA_INICIO_TARDE','FECHA_FIN_TARDE',
                    'DURACION_EFECTIVA','UNIDAD_RUBRO','RENDIMIENTO','HOLGURA_TOTAL','RUTA_CRITICA'
                ]

                columnas_editables = ['PREDECESORAS', 'FECHAINICIO', 'FECHAFIN', 'RUTA_CRITICA']
                column_config = {col: {"editable": (col in columnas_editables)} for col in cols1}
                

                column_config["RENDIMIENTO"] = st.column_config.NumberColumn(
                    "RENDIMIENTO",
                    help="Rendimiento del rubro",
                    format="%.4f",    # 4 decimales
                    step=0.0001,
                    disabled=("RENDIMIENTO" not in columnas_editables)  # respeta tu lógica editable
                )
                tareas_editadas = st.data_editor(
                    st.session_state.tareas_df[cols1],
                    key="tareas_editor",
                    use_container_width=True,
                    column_config=column_config
                )

                st.session_state.tareas_df.reset_index(drop=True, inplace=True)
                tareas_editadas.reset_index(drop=True, inplace=True)
                st.session_state.tareas_df_prev.reset_index(drop=True, inplace=True)

                columnas_validas = [c for c in columnas_editables if c in st.session_state.tareas_df_prev.columns and c in tareas_editadas.columns]
                
                prev = st.session_state.tareas_df_prev[columnas_validas]
                now = tareas_editadas[columnas_validas]
        
                cambios = now.ne(prev)
                filas_cambiadas = cambios.any(axis=1)

                if filas_cambiadas.any():
                    for idx in filas_cambiadas.index:
                        for col in columnas_validas:
                            st.session_state.tareas_df.at[idx, col] = tareas_editadas.at[idx, col]
                                
                    if 'RUTA_CRITICA' in cambios.columns:
                        filas_ruta_critica = cambios['RUTA_CRITICA']
                        for idx in filas_ruta_critica[filas_ruta_critica].index:
                            st.session_state.tareas_df = calculo_predecesoras(st.session_state.tareas_df, idx)

                    st.session_state.tareas_df = calculo_ruta_critica(st.session_state.tareas_df)
                    if all(col in st.session_state.tareas_df.columns for col in ["FECHA_INICIO_TEMPRANA", "FECHA_FIN_TEMPRANA"]):
                            st.session_state.tareas_df["FECHAINICIO"] = st.session_state.tareas_df["FECHA_INICIO_TEMPRANA"]
                            st.session_state.tareas_df["FECHAFIN"] = st.session_state.tareas_df["FECHA_FIN_TEMPRANA"]
                    else:
                            st.warning("⚠️ No se encontraron las columnas FECHA_INICIO_TEMPRANA o FECHA_FIN_TEMPRANA en el DataFrame.")

                    st.session_state.tareas_df_prev = st.session_state.tareas_df.copy()
                
                st.subheader("📊 Diagrama de Gantt - Ruta Crítica")
                
                st.markdown(
                    """
                    <style>
                    /* Encabezado azul oscuro con texto blanco */
                    [data-testid="stDataFrame"] thead tr th {
                        background-color: #0D3B66 !important;
                        color: white !important;
                        font-weight: bold !important;
                        text-align: center !important;
                    }
                
                    /* Centrar texto en celdas */
                    [data-testid="stDataFrame"] tbody td {
                        text-align: center !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                column_config["RENDIMIENTO"] = st.column_config.NumberColumn(
                    "RENDIMIENTO",
                    help="Rendimiento del rubro",
                    format="%.4f",    # 4 decimales
                    step=0.0001,
                    disabled=("RENDIMIENTO" not in columnas_editables)  # respeta tu lógica editable
                )
                st.data_editor(
                    st.session_state.tareas_df[cols],
                    key="tareas_editor_actualizada",
                    use_container_width=True,
                    disabled=True, 
                    column_config=column_config
                )

                st.session_state.dependencias_df = st.session_state.dependencias_df.merge(st.session_state.recursos_df, left_on='RECURSO', right_on='RECURSO', how='left')
                st.session_state.dependencias_df['COSTO'] = st.session_state.dependencias_df['CANTIDAD'] * st.session_state.dependencias_df['TARIFA']
                costos_por_can = st.session_state.dependencias_df.groupby('RUBRO', as_index=False)['COSTO'].sum()
                costos_por_can.rename(columns={'RUBRO': 'RUBRO', 'COSTO': 'COSTO_TOTAL'}, inplace=True)
                st.session_state.tareas_df['RUBRO'] = st.session_state.tareas_df['RUBRO'].str.strip()
                costos_por_can['RUBRO'] = costos_por_can['RUBRO'].str.strip()
                st.session_state.tareas_df = st.session_state.tareas_df.merge(costos_por_can[['RUBRO', 'COSTO_TOTAL']], on='RUBRO', how='left')

                cost_column_name = None
                if 'COSTO_TOTAL_RUBRO' in st.session_state.tareas_df.columns:
                        cost_column_name = 'COSTO_TOTAL_RUBRO'
                elif 'COSTO_TOTAL_x' in st.session_state.tareas_df.columns:
                        cost_column_name = 'COSTO_TOTAL_x'
                elif 'COSTO_TOTAL' in st.session_state.tareas_df.columns: 
                        cost_column_name = 'COSTO_TOTAL'
                if cost_column_name:
                        st.session_state.tareas_df[cost_column_name] = pd.to_numeric(st.session_state.tareas_df[cost_column_name], errors='coerce')
                        st.session_state.tareas_df[cost_column_name] = st.session_state.tareas_df[cost_column_name].fillna(0)
                else:
                        st.warning("⚠️ Advertencia: No se encontró una columna de costos reconocida en el DataFrame.")
                        tareas_df['COSTO_TOTAL_NUMERICO'] = 0
                        cost_column_name = 'COSTO_TOTAL_NUMERICO'
                if 'IDRUBRO' in st.session_state.tareas_df.columns:
                        st.session_state.tareas_df = st.session_state.tareas_df.sort_values(['IDRUBRO'])
                else:
                        st.warning("⚠️ Advertencia: Columna 'IDRUBRO' no encontrada para ordenar.")
        
                st.session_state.tareas_df['y_num'] = range(len(st.session_state.tareas_df))
                
                fig = go.Figure()
                fecha_inicio_col = 'FECHAINICIO'
                fecha_fin_col = 'FECHAFIN'
                if fecha_inicio_col not in st.session_state.tareas_df.columns or fecha_fin_col not in st.session_state.tareas_df.columns:
                     st.warning("❌ Error: No se encontraron columnas de fechas de inicio/fin necesarias para dibujar el Gantt.")
                
                inicio_rubro_calc = st.session_state.tareas_df.set_index('IDRUBRO')[fecha_inicio_col].to_dict()
                fin_rubro_calc = st.session_state.tareas_df.set_index('IDRUBRO')[fecha_fin_col].to_dict()
                is_critical_dict = st.session_state.tareas_df.set_index('IDRUBRO')['RUTA_CRITICA'].to_dict()
                dependencias = defaultdict(list)
                predecesoras_map_details = defaultdict(list)
                
                for _, row in st.session_state.tareas_df.iterrows():
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
                                if pre_id in st.session_state.tareas_df['IDRUBRO'].values:
                                     dependencias[pre_id].append(tarea_id)
                                     predecesoras_map_details[tarea_id].append((pre_id, tipo_relacion, desfase))
                                else:
                                     st.warning(f"⚠️ Advertencia: Predecesor ID {pre_id} mencionado en '{pre_entry}' para tarea {tarea_id} no encontrado en la lista de tareas. Ignorando esta dependencia.")
                            else:
                                if pre_entry != '':
                                    st.warning(f"⚠️ Advertencia: Formato de predecesora '{pre_entry}' no reconocido para la tarea {tarea_id}. Ignorando.")
                
                shapes = []
                color_banda = 'rgba(240, 240, 240, 0.1)'
                for y_pos in range(len(st.session_state.tareas_df)):
                    if y_pos % 2 == 0:
                        shapes.append(dict(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=y_pos - 0.5, y1=y_pos + 0.5, fillcolor=color_banda, layer="below", line_width=0))
                
                color_no_critica_barra = 'lightblue'
                color_critica_barra = 'rgb(255, 133, 133)'
                
                for i, row in st.session_state.tareas_df.iterrows():
                    line_color = color_critica_barra if row.get('RUTA_CRITICA', False) else color_no_critica_barra
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
                
                    hover_text = (f"📌 <b>Rubro:</b> {row['RUBRO']}<br>"
                                  f"🗓️ <b>Capítulo:</b> {row['CAPÍTULO']}<br>"
                                  f"📅 <b>Inicio:</b> {start_date.strftime('%d/%m/%Y')}<br>"
                                  f"🏁 <b>Fin:</b> {end_date.strftime('%d/%m/%Y')}<br>"
                                  f"⏱️ <b>Duración:</b> {(end_date - start_date).days} días<br>"
                                  f"⏳ <b>Holgura Total:</b> {row.get('HOLGURA_TOTAL', 'N/A')} días<br>"
                                  f"💰 <b>Costo:</b> {costo_formateado}")

                    half_height = 0.35  # controla el grosor vertical de la barra
                    y_center = row['y_num']
                    y0 = y_center - half_height
                    y1 = y_center + half_height
                
                    xs = [start_date, end_date, end_date, start_date, start_date]
                    ys = [y0, y0, y1, y1, y0]
                
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines',
                        fill='toself',
                        fillcolor=line_color,
                        line=dict(color=line_color, width=1),
                        hoverinfo='text',
                        text=hover_text,
                        showlegend=False
                    ))
                    inicio_temp = row["FECHA_INICIO_TEMPRANA"]
                    fin_tarde = row["FECHA_FIN_TARDE"]
                        
                    if pd.notna(inicio_temp) and pd.notna(fin_tarde):
                            # Barra translúcida (rango de flexibilidad)
                            y_center = row['y_num']
                            half_height_flex = 0.35  # mismo grosor que la barra principal
                            y0_flex = y_center - half_height_flex
                            y1_flex = y_center + half_height_flex
                        
                            xs_flex = [inicio_temp, fin_tarde, fin_tarde, inicio_temp, inicio_temp]
                            ys_flex = [y0_flex, y0_flex, y1_flex, y1_flex, y0_flex]
                        
                            # Color más suave (reduce saturación / más claro)
                            fill_color_soft = line_color.replace("rgb(", "rgba(").replace(")", ", 0.3)") if "rgb(" in line_color else "rgba(173,216,230,0.3)"
                        
                            fig.add_trace(go.Scatter(
                                x=xs_flex,
                                y=ys_flex,
                                mode='lines',
                                fill='toself',
                                fillcolor=fill_color_soft,
                                line=dict(color=fill_color_soft, width=0.5),
                                hoverinfo='skip',
                                showlegend=False
                            ))

                offset_days_horizontal = 5
                color_no_critica_flecha = 'blue'
                color_critica_flecha = 'red'
                
                for pre_id, sucesores in dependencias.items():
                    pre_row_df = st.session_state.tareas_df[st.session_state.tareas_df['IDRUBRO'] == pre_id]
                    if pre_row_df.empty: continue
                    y_pre = pre_row_df.iloc[0]['y_num']
                    pre_is_critical = is_critical_dict.get(pre_id, False)
                    x_pre_inicio = inicio_rubro_calc.get(pre_id)
                    x_pre_fin = fin_rubro_calc.get(pre_id)
                    if pd.isna(x_pre_inicio) or pd.isna(x_pre_fin): continue
                    for suc_id in sucesores:
                        suc_row_df = st.session_state.tareas_df[st.session_state.tareas_df['IDRUBRO'] == suc_id]
                        if suc_row_df.empty: continue
                        y_suc = suc_row_df.iloc[0]['y_num']
                        suc_is_critical = is_critical_dict.get(suc_id, False)
                        arrow_color = color_critica_flecha if pre_is_critical and suc_is_critical else color_no_critica_flecha
                        line_style = dict(color=arrow_color, width=0.5)
                        x_suc_inicio = inicio_rubro_calc.get(suc_id)
                        x_suc_fin = fin_rubro_calc.get(suc_id)
                        if pd.isna(x_suc_inicio) or pd.isna(x_suc_fin): continue
                        tipo_relacion = 'FC'
                        for pre_id_suc, type_suc, desfase_suc in predecesoras_map_details.get(suc_id, []):
                            if pre_id_suc == pre_id:
                                 tipo_relacion = type_suc.upper() if type_suc else 'FC'
                                 break
                        origin_x = x_pre_fin
                        if tipo_relacion == 'CC': origin_x = x_pre_inicio
                        elif tipo_relacion == 'CF': origin_x = x_pre_inicio
                        elif tipo_relacion == 'FF': origin_x = x_pre_fin
                        connection_x = x_suc_inicio
                        arrow_symbol = 'triangle-right'
                        if tipo_relacion == 'CC': connection_x = x_suc_inicio; arrow_symbol='triangle-right'
                        elif tipo_relacion == 'CF': connection_x = x_suc_fin; arrow_symbol='triangle-left'
                        elif tipo_relacion == 'FF': connection_x = x_suc_fin; arrow_symbol='triangle-left'
                       
                        ajuste_vertical = 0.2
                        y_pre_ajustado = y_pre - ajuste_vertical     # sale más abajo
                        y_suc_ajustado = y_suc + ajuste_vertical     # llega más arriba

                        points_x = [origin_x]; points_y = [y_pre_ajustado]
                        if tipo_relacion in ['CC','FC']:
                                if origin_x <= connection_x: 
                                        elbow1_x = origin_x - timedelta(days=offset_days_horizontal) ; elbow1_y = y_pre_ajustado
                                        elbow2_x = elbow1_x; elbow2_y = y_suc_ajustado
                                        points_x += [elbow1_x, elbow2_x, connection_x]; points_y += [elbow1_y, elbow2_y, y_suc_ajustado]
                                else:
                                        elbow1_x = connection_x  - timedelta(days=offset_days_horizontal); elbow1_y = y_pre_ajustado
                                        elbow2_x = elbow1_x; elbow2_y = y_suc_ajustado
                                        points_x += [elbow1_x, elbow2_x, connection_x]; points_y += [elbow1_y, elbow2_y, y_suc_ajustado]

                        elif tipo_relacion in ['CF','FF']:
                                if origin_x <= connection_x:
                                        elbow1_x = connection_x + timedelta(days=offset_days_horizontal) ; elbow1_y = y_pre_ajustado
                                        elbow2_x = elbow1_x; elbow2_y = y_suc_ajustado
                                        points_x += [elbow1_x, elbow2_x, connection_x]; points_y += [elbow1_y, elbow2_y, y_suc_ajustado]
                                else:
                                        elbow1_x =  origin_x + timedelta(days=offset_days_horizontal); elbow1_y = y_pre_ajustado
                                        elbow2_x = elbow1_x; elbow2_y = y_suc_ajustado
                                        points_x += [elbow1_x, elbow2_x, connection_x]; points_y += [elbow1_y, elbow2_y, y_suc_ajustado]
                        else:
                            continue
                                
                        # 🔹 Línea continua (no discontinua)
                        fig.add_trace(go.Scatter(
                            x=points_x,
                            y=points_y,
                            mode='lines',
                            line=dict(color=arrow_color, width=1),  # línea continua
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Marcador de salida (círculo) — lo colocamos ligeramente desplazado en X para que no tape la barra
                        circle_x = origin_x
                        circle_y = y_pre_ajustado
                        if not (pd.isna(circle_x) or pd.isna(circle_y)):
                            fig.add_trace(go.Scattergl(
                                x=[circle_x],
                                y=[circle_y],
                                mode='markers',
                                marker=dict(symbol='circle', size=6, color=arrow_color, line=dict(width=0)),
                                hoverinfo='none',
                                showlegend=False
                            ))
                        
                        # Marcador de llegada (triángulo) — un poco más grande para destacar llegada
                        triangle_x = connection_x
                        triangle_y = y_suc_ajustado
                        if not (pd.isna(triangle_x) or pd.isna(triangle_y)):
                            fig.add_trace(go.Scattergl(
                                x=[triangle_x],
                                y=[triangle_y],
                                mode='markers',
                                marker=dict(symbol=arrow_symbol, size=8, color=arrow_color, line=dict(width=0)),
                                hoverinfo='none',
                                showlegend=False
                            ))

                
                y_ticktext_styled = []
                for y_pos in range(len(st.session_state.tareas_df)):
                    row_for_y_pos = st.session_state.tareas_df[st.session_state.tareas_df['y_num'] == y_pos]
                    if not row_for_y_pos.empty:
                        rubro_text = row_for_y_pos.iloc[0]['RUBRO']
                        y_ticktext_styled.append(f"<b>{rubro_text}</b>" if y_pos % 2 == 0 else rubro_text)
                    else: y_ticktext_styled.append("")
                
                fig.update_layout(
                    xaxis=dict(
                        title='Fechas',
                        side='top',  # 🔼 eje X arriba
                        dtick='M1',
                        tickangle=-90,
                        showgrid=True,
                        gridcolor='rgba(128,128,128,0.3)',
                        gridwidth=0.5
                    ),
                    yaxis_title='Rubro',
                    yaxis=dict(
                        autorange='reversed',
                        tickvals=st.session_state.tareas_df['y_num'],
                        ticktext=y_ticktext_styled,
                        tickfont=dict(size=10),
                        showgrid=False
                    ),
                    shapes=shapes,
                    height=max(600, len(st.session_state.tareas_df)*25),
                    showlegend=False,
                    plot_bgcolor='white',
                    hovermode='closest'
                )

                for i, tarea in enumerate(st.session_state.tareas_df['RUBRO'].unique()):
                    if i % 2 == 0:
                        fig.add_shape(
                            type="rect",
                            x0=st.session_state.tareas_df['FECHAINICIO'].min(),  # desde el inicio
                            x1=st.session_state.tareas_df['FECHAFIN'].max(),     # hasta el fin
                            y0=i - 0.5,
                            y1=i + 0.5,
                            fillcolor="rgba(240,240,240,0.01)",  # gris muy suave
                            line_width=0,
                            layer="below"  # debajo de las barras
                        )

                st.plotly_chart(fig, use_container_width=True)

# Mostrar variables en la Pestaña 4___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________            
        
        with tab4:
                st.session_state.tareas_df['FECHAINICIO'] = pd.to_datetime(st.session_state.tareas_df['FECHAINICIO'])
                st.session_state.tareas_df['FECHAFIN'] = pd.to_datetime(st.session_state.tareas_df['FECHAFIN'])
                
                st.session_state.tareas_df['RUBRO'] = st.session_state.tareas_df['RUBRO'].str.strip()
                st.session_state.dependencias_df['RUBRO'] = st.session_state.dependencias_df['RUBRO'].str.strip()
                    
                recursos_tareas_df = st.session_state.dependencias_df.merge(
                    st.session_state.tareas_df[['IDRUBRO', 'RUBRO', 'FECHAINICIO', 'FECHAFIN', 'DURACION']],
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
                st.session_state.recursos_df['RECURSO'] = st.session_state.recursos_df['RECURSO'].str.strip()
                
                resource_demand_with_details_df = daily_resource_demand_df.merge(
                    st.session_state.recursos_df[['RECURSO', 'TYPE', 'TARIFA']],
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
                
                st.subheader("📊 Distribución de Recursos")
                
                if 'RUBRO' not in recursos_tareas_df.columns:
                
                    if 'tareas_df' in st.session_state and 'dependencias_df' in st.session_state:
                
                        st.session_state.tareas_df['RUBRO'] = st.session_state.tareas_df['RUBRO'].astype(str).str.strip()
                        recursos_tareas_df['RUBRO'] = recursos_tareas_df['RUBRO'].astype(str).str.strip()
                
                        if 'IDRUBRO' in recursos_tareas_df.columns and 'IDRUBRO' in st.session_state.tareas_df.columns:
                            recursos_tareas_df = recursos_tareas_df.merge(
                                st.session_state.tareas_df[['IDRUBRO', 'RUBRO']],
                                left_on='IDRUBRO',
                                right_on='IDRUBRO',
                                how='left'
                            )
                            st.warning("Re-merged to include 'RUBRO' column using IDRUBRO.")
                        else:
                            st.warning("❌ Error: 'IDRUBRO' column not found in one of the dataframes. Cannot re-add 'RUBRO'.")
                            raise KeyError("'IDRUBRO' column not found for re-merging.")
                
                    else:
                        st.warning("❌ Error: 'tareas_df' or 'dependencias_df' not found. Cannot re-add 'RUBRO' column.")
                        raise NameError("'tareas_df' or 'dependencias_df' not found.")
                
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
                
                dropdown_options = [{'label': 'All Tasks', 'method': 'update', 'args': [{'visible': [True]*len(fig_resource_timeline.data)}, {'title': 'Línea de Tiempo de Uso de Recursos'}]}]
                
                for rubro in unique_rubros:
                    visibility_list = [trace.customdata[0] == rubro if trace.customdata and len(trace.customdata) > 0 else False for trace in fig_resource_timeline.data]
                    dropdown_options.append({
                        'label': rubro,
                        'method': 'update',
                        'args': [{'visible': visibility_list}, {'title': f'Línea de Tiempo de Uso de Recursos (Filtrado por: {rubro})'}]
                    })
                
                fig_resource_timeline.update_layout(
                    updatemenus=[go.layout.Updatemenu(
                        buttons=dropdown_options,
                        direction="down",
                        pad={"r":10,"t":10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.1,
                        yanchor="top"
                    )],
                    yaxis=dict(autorange="reversed", title="Recurso", tickfont=dict(size=10)),
                    xaxis=dict(title='Fechas', side='top', dtick='M1', tickangle=-90, showgrid=True, gridcolor='rgba(128,128,128,0.3)', gridwidth=0.5),
                    height=max(600, len(recursos_tareas_df['RECURSO'].unique())*20),
                    showlegend=False,
                    plot_bgcolor='white',
                    hovermode='closest'
                )
                st.plotly_chart(fig_resource_timeline, use_container_width=True)


# Mostrar variables en la Pestaña 5___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________            
        with tab5:
                             
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
                        if type_issues:
                            for issue in type_issues:
                                st.warning(f"⚠️ Tipo de dato: {issue}")
                    else:
                        st.warning(f"❌ Error: Missing required columns: {missing_columns}")
                
                    df = resource_demand_with_details_df.copy()
                    df['Fecha'] = pd.to_datetime(df['Fecha'])
                    df['Periodo_Mensual'] = df['Fecha'].dt.to_period('M')
                    monthly_costs_df = df.groupby('Periodo_Mensual')['Costo_Diario'].sum().reset_index()
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
                        showgrid=False,
                        range=[0, monthly_costs_df['Costo_Diario'].max()*1.1]
                    )
                    fig.update_yaxes(
                        title_text="Costo Acumulado",
                        secondary_y=True,
                        showgrid=True,
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
































































































































































































